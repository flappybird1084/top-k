"""Whole-model GPU preparation and verification for agent-authored installations."""
import json
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

from kernel_evolution.runtime import Harness
from kernel_evolution.verifier import paired_measure, source_scan
from kernel_evolution.whole_model import capture_reference, check_reference, install_candidate


def prepare(request):
    from kernel_evolution.compiled_profile import profile_harness
    from kernel_evolution.intake import repository_context, repository_files
    from kernel_evolution.whole_prompt import CONTRACT
    cfg = dict(request['config'])
    root = Path(request['run_dir'])
    harness = Harness(request['adapter'], cfg, root)
    reference = capture_reference(harness, cfg)
    floor = check_reference(harness, reference, cfg,update_floor=True)
    cfg['update_atol']=max(1e-7,2*floor['update_max_abs'])
    if cfg['update_atol']>cfg['atol']:
        raise RuntimeError('Eager/compiled update floor exceeds absolute state tolerance; refusing to weaken update checks')
    correctness = check_reference(harness, reference, cfg)
    # Capture code on both initial and steady-state optimizer graphs.
    harness.restore()
    fn = harness.step_callable({})
    for _ in range(2): harness.capture_step(fn)
    profile = profile_harness(harness)
    (root/'profile.json').write_text(json.dumps(profile, indent=2))
    from kernel_evolution.whole_model import selftest
    cheats = selftest(harness, reference, cfg)
    samples = []
    start = time.monotonic()
    for i in range(cfg['calibration_repeats']):
        deadline = start + cfg['calibration_seconds']*i/max(1,cfg['calibration_repeats']-1)
        if deadline > time.monotonic(): time.sleep(deadline-time.monotonic())
        ms = statistics.median(harness.time_set({}))
        samples.append({'step_ms':ms})
        print(json.dumps({'event':'calibration_sample','index':i,'step_ms':ms}),flush=True)
    times = [s['step_ms'] for s in samples]
    spread = (max(times)-min(times))/statistics.median(times)
    cfg['gate4_margin'] = max(cfg['gate4_margin'],spread)
    seed = root/'seeds'/'whole_model.py'
    seed.parent.mkdir(exist_ok=True)
    seed.write_text('"""Baseline: leave original PyTorch model and AdamW unchanged. The harness compiles the full step."""\ndef install(model, optimizer):\n    return None\n')
    repo = cfg.get('input_repo') or '/marimo/top-k'
    target = dict(id='whole_model',op='whole_model',contract='whole_model',eligible=True,
                  pct_step_time=100., shapes=[{'args':[],'count':1}],
                  seed_paths=[str(seed)],eager_reference=request['adapter'],
                  installation_contract=CONTRACT,
                  model_source=repository_context(repo,repository_files(repo),max_chars=40000),
                  modules=[{'name':name,'type':type(m).__name__,'repr':repr(m)[:1000]}
                           for name,m in harness.model.named_modules()],
                  parameters=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in harness.model.named_parameters()],
                  gpu_profile=profile,
                  numerical_contract={'rtol':cfg['rtol'],'atol':cfg['atol'],'update_atol':cfg.get('update_atol',1e-7),
                                      'tf32_allowed':False,'batch_size':harness.batch_size})
    gpu = torch.cuda.get_device_name(0)
    peak = cfg.get('peak_flops') or {'NVIDIA RTX PRO 6000 Blackwell Server Edition':125e12}.get(gpu)
    if peak is None: raise RuntimeError('Configure GPU peak FLOPs for '+gpu)
    return dict(status='ready',profile=profile,targets=[target],config=cfg,
                model=dict(id=request['adapter'],name=request['adapter'],adapter_path=request['adapter'],
                           n_params=sum(p.numel() for p in harness.model.parameters()),flops_per_sample=harness.flops(),gpu_name=gpu,peak_flops=peak),
                calibration=dict(samples=samples,noise_spread=spread,full_step_correctness=correctness,update_floor=floor,cheats=cheats,
                                 protocol='Whole-model correctness then paired full-step timing; no isolated-op surrogate'),
                step_time_ms=statistics.median(times),batch_size=harness.batch_size)


def evaluate(request):
    from kernel_evolution.gpu_worker import gate_span
    from triton.runtime.jit import JITFunction
    cfg, root = request['config'], Path(request['run_dir'])
    result = dict(gate_reached=1,compile_ok=0,correct_ok=0,accepted=0,details={})
    try:
        baseline = Harness(request['adapter'],cfg,root)
        reference = capture_reference(baseline,cfg)
        incumbent = request['incumbents'].get('whole_model')
        if incumbent: install_candidate(baseline,incumbent)
        candidate = Harness(request['adapter'],cfg,root)
        result['details']['source_flags'] = source_scan(Path(request['code_path']).read_text())
        launches = []
        backward_launches = []
        phase = {'backward':False}
        original = JITFunction.run
        backward = torch.autograd.backward
        def track_backward(*args,**kwargs):
            phase['backward']=True
            try:return backward(*args,**kwargs)
            finally:phase['backward']=False
        def track(jit,*args,**kwargs):
            if not kwargs.get('warmup',False):
                filename = getattr(getattr(jit,'fn',None),'__code__',None)
                if filename and Path(filename.co_filename).resolve()==Path(request['code_path']).resolve():
                    launches.append(jit.__name__)
                    if phase['backward']: backward_launches.append(jit.__name__)
            return original(jit,*args,**kwargs)
        with gate_span(result,'compile',1) as span:
            result['details']['installation'] = install_candidate(candidate,request['code_path'])
            with patch.object(JITFunction,'run',track), patch.object(torch.autograd,'backward',track_backward):
                candidate.state_after_step({},eager=True)
            if not launches: raise TypeError('Candidate training step launched no Triton kernel defined in its source')
            if cfg.get('require_backward_kernel') and not backward_launches:
                raise TypeError('Candidate launched no new Triton kernel during backward; backward optimization is required')
            span['output'] = {'candidate_triton_launches': sorted(set(launches)),
                              'backward_triton_launches':sorted(set(backward_launches))}
            result['details']['candidate_launches']=span['output']
        result.update(compile_ok=1,gate_reached=2)
        with gate_span(result,'correctness',2) as span:
            result['details']['correctness'] = check_reference(candidate,reference,cfg)
            span['output'] = result['details']['correctness']
        result.update(correct_ok=1,gate_reached=4)
        # The candidate unit IS the whole model, so no isolated-op speed gate is meaningful.
        result['details']['gate3'] = {'status':'not_applicable','reason':'Whole-model candidate; measured directly by training-step gate'}
        with gate_span(result,'training_step',4) as span:
            timing = paired_measure(lambda h:statistics.median(h.time_set({})),baseline,candidate,cfg['gate4_margin'])
            span['output'] = timing
        result['details']['gate4'] = timing
        ms = timing['candidate']; sps = candidate.batch_size/(ms/1000)
        result.update(step_time_ms=ms,incumbent_step_time_ms=timing['baseline'],samples_per_s=sps,
                      mfu=request['flops_per_sample']*sps/request['peak_flops'],accepted=int(timing['status']=='pass'))
        if not result['accepted']: result['failure_note']='gate4_'+timing['status']
    except Exception:
        result['failure_note']=traceback.format_exc()[-14000:]
    return result
