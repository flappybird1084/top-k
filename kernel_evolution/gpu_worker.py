"""Disposable CUDA process. Orchestrator kills the whole process group on timeout."""
import json
import os
import statistics
import sys
import time
import traceback
from unittest.mock import patch
from pathlib import Path
import torch
from kernel_evolution.runtime import Harness, inductor_seed, load_source, cuda_times, summarize, clone
from kernel_evolution.verifier import gate2, gate3, gate4, selftest, source_scan


def case_list(cases,op):
    return sorted(cases[op].values(),key=lambda c:c['count'],reverse=True)


def prepare(request):
    cfg=request['config']
    root=Path(request['run_dir'])
    harness=Harness(request['adapter'],cfg,root)
    profile=harness.profile()
    if request.get('lineage'):
        for target in profile['targets']:
            target['eligible'] &= target['id']==request['lineage']
    (root/'profile.json').write_text(json.dumps(profile,indent=2))
    seeds={}
    floors={}
    cheat_results={}
    for target in profile['targets']:
        if not target['eligible']: continue
        if target.get('fusion_of'):continue
        op=target['id']
        cases=case_list(harness.cases,op)
        seed,sources=inductor_seed(op,cases,root/'seeds')
        if not sources:
            raise RuntimeError('First seed extraction did not capture source; run with a fresh inductor cache')
        target['seed_paths']=sources
        target['eager_reference']='kernel_evolution.ops.'+op
        target['inductor_triton']=True
        seeds[op]=seed
        floors[op]=gate2(op,seed,cases,cfg,floor=True)
    targets=[t for t in profile['targets'] if t['eligible']]
    for t in targets:
        if t.get('fusion_of'):
            parent=next(p for p in targets if p['id']==t['fusion_of'])
            t['seed_paths']=parent['seed_paths']
            t['eager_reference']='kernel_evolution.ops.fused_ema_update'
    if not targets:
        return dict(status='no_targets',profile=profile)
    cfg=dict(cfg)
    from triton.runtime import driver
    target=driver.active.get_current_target()
    cfg['gpu_target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
    cfg['atol']=max(cfg['atol'],2*max(v['max_abs'] for v in floors.values()))
    cfg['rtol']=max(cfg['rtol'],2*max(v['max_rel'] for v in floors.values()))
    for target in targets:
        op=target['id']
        if not target.get('fusion_of'):
            gate2(op,seeds[op],case_list(harness.cases,op),cfg)
        cheat_results[op]=selftest(op,case_list(harness.cases,op),cfg)
    # Calibrate isolated and full-step drift over the same five-minute interval.
    samples=[]
    start=time.monotonic()
    op=targets[0]['id']
    args=case_list(harness.cases,op)[0]['args']
    for i in range(cfg['calibration_repeats']):
        deadline=start+cfg['calibration_seconds']*i/max(1,cfg['calibration_repeats']-1)
        if deadline>time.monotonic(): time.sleep(deadline-time.monotonic())
        micro=statistics.median(cuda_times(lambda:seeds[op](*args),cfg['micro_reps'],10))
        step=statistics.median(harness.time_set(seeds))
        samples.append(dict(micro_ms=micro,step_ms=step))
        print(json.dumps(dict(event='calibration_sample',index=i,**samples[-1])),flush=True)
    def spread(key):
        data=[s[key] for s in samples]
        return (max(data)-min(data))/statistics.median(data)
    cfg['gate3_margin']=max(cfg['gate3_margin'],spread('micro_ms'))
    cfg['gate4_margin']=max(cfg['gate4_margin'],spread('step_ms'))
    gpu=torch.cuda.get_device_name(0)
    # FP32, TF32 disabled. Do not substitute sparse or BF16 tensor-core peak FLOPs.
    peaks={'NVIDIA RTX PRO 6000 Blackwell Server Edition':125e12}
    peak=cfg.get('peak_flops') or peaks.get(gpu)
    if peak is None: raise RuntimeError(f'Unknown peak FP32 FLOPs for {gpu}; configure peak_flops')
    flops=harness.flops()
    return dict(status='ready',profile=profile,targets=targets,config=cfg,
        model=dict(id=request['adapter'],name=request['adapter'],adapter_path=request['adapter'],
            n_params=sum(p.numel() for p in harness.model.parameters()),flops_per_sample=flops,gpu_name=gpu,peak_flops=peak),
        calibration=dict(samples=samples,noise_spread=max(spread('micro_ms'),spread('step_ms')),
            numerical_floor=floors,cheats=cheat_results),
        step_time_ms=statistics.median(s['step_ms'] for s in samples),batch_size=harness.batch_size)


def evaluate(request):
    cfg=request['config']
    root=Path(request['run_dir'])
    op=request['op']
    result=dict(gate_reached=1,compile_ok=0,correct_ok=0,accepted=0,details={})
    try:
        cases=torch.load(root/'cases.pt',weights_only=False,map_location='cuda')
        selected=case_list(cases,op)
        fn=load_source(request['code_path'])
        result['details']['source_flags']=source_scan(Path(request['code_path']).read_text())
        from triton.runtime.jit import JITFunction
        launch=JITFunction.run
        launches=[]
        def track(jit,*args,**kwargs):
            if not kwargs.get('warmup',False):launches.append(jit.__name__)
            return launch(jit,*args,**kwargs)
        with patch.object(JITFunction,'run',track):
            fn(*clone(selected[0]['args']))
        torch.cuda.synchronize()
        if not launches:raise TypeError('Candidate entry point launched no Triton JIT kernel')
        result['details']['modal_triton_launches']=launches
        result.update(compile_ok=1,gate_reached=2)
        result['details']['correctness']=gate2(op,fn,selected,cfg)
        result.update(correct_ok=1,gate_reached=3)
        if request.get('correctness_only'):
            return result
        incumbents={}
        for name,path in request['incumbents'].items():
            if path is None:
                incumbents[name],_=inductor_seed(name,case_list(cases,name),root/'seeds_runtime')
            else: incumbents[name]=load_source(path)
        baseline=incumbents.get(op)
        if op=='fused_ema_update' and baseline is None:
            from kernel_evolution.fusion import composed
            baseline=composed(incumbents['ema_update'])
        micro=gate3(op,baseline,fn,selected,cfg)
        result['details']['gate3']=micro
        result.update(latency_us=micro['candidate']*1000,incumbent_latency_us=micro['baseline']*1000)
        if micro['status']!='pass':
            result['failure_note']='gate3_'+micro['status']
            return result
        result['gate_reached']=4
        harness=Harness(request['adapter'],cfg,root)
        step=gate4(harness,incumbents,op,fn,cfg)
        result['details']['gate4']=step
        step_ms=step['candidate']
        sps=harness.batch_size/(step_ms/1000)
        result.update(step_time_ms=step_ms,incumbent_step_time_ms=step['baseline'],samples_per_s=sps,
                      mfu=request['flops_per_sample']*sps/request['peak_flops'])
        if step['status']=='pass': result['accepted']=1
        else: result['failure_note']='gate4_'+step['status']
    except Exception as exc:
        result['failure_note']=traceback.format_exc()[-14000:]
    return result


def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    try:
        result=(prepare if request['action']=='prepare' else evaluate)(request)
    except Exception:
        result={'status':'error','failure_note':traceback.format_exc()[-18000:]}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2,default=str))


if __name__=='__main__': main()
