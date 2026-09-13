import copy
import importlib
import importlib.util
import json
import os
import random
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map, tree_flatten
from kernel_evolution.ops import instrument, regions, install, REFERENCES


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move(batch, device):
    return tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)


def clone(tree):
    return tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x), tree)


def describe(args):
    return [dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype).split('.')[-1],
                 requires_grad=x.requires_grad) if isinstance(x, torch.Tensor) else x for x in args]


def cuda_times(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def summarize(values):
    a = np.array(values, dtype=float)
    med = float(np.median(a))
    return dict(median=med, p10=float(np.quantile(a,.1)), p90=float(np.quantile(a,.9)),
                relative_spread=float((np.quantile(a,.9)-np.quantile(a,.1))/max(med,1e-12)), n=len(a))


class Harness:
    def __init__(self, adapter, config, run_dir):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required: no CPU fallback for acceptance gates')
        self.config, self.run_dir = config, Path(run_dir)
        seed_all(config['seed'])
        os.environ['KE_BATCH_SIZE'] = str(config['batch_size'])
        self.adapter = importlib.import_module(adapter)
        self.model = self.adapter.build_model().cuda().train()
        self.batch = move(next(iter(self.adapter.get_dataloader('train'))), 'cuda')
        self.batch_size = next(x for x in tree_flatten(self.batch)[0] if isinstance(x,torch.Tensor)).shape[0]
        self.loss = self.adapter.loss_fn(self.model,self.batch)
        if self.loss.ndim != 0 or not self.loss.requires_grad or not torch.isfinite(self.loss):
            raise ValueError('Adapter loss must be scalar, finite, and require grad')
        self.model = instrument(self.model)
        self.discovery={'backend':'modules','regions':[],'fallback_reason':None}
        if config.get('functional_discovery',False):
            from kernel_evolution.graph import discover_functional,validate_rewrite
            original_forward=self.model.forward
            self.discovery,rollback=discover_functional(self.model)
            if self.discovery['regions']:
                try:validate_rewrite(self.model,original_forward,self.adapter.loss_fn,self.batch,config)
                except Exception as exc:
                    rollback()
                    self.discovery['regions']=[]
                    self.discovery['fallback_reason']='Rewrite verification failed: '+str(exc)
                else:self.discovery['verified']=True
        self.replacements={}
        self.fusion_issue=None
        from kernel_evolution.fusion import capture_ema_bindings
        try:self.ema_bindings=capture_ema_bindings(self.model,getattr(self.adapter,'post_optimizer_step',None))
        except ValueError as exc:
            self.ema_bindings=[]
            self.fusion_issue=str(exc)
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=.01, foreach=False)
        self.snapshot = clone(self.model.state_dict())
        self.optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        self.cases = {}

    def restore(self):
        self.model.load_state_dict(self.snapshot)
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_snapshot))
        self.optimizer.zero_grad(set_to_none=True)
        seed_all(self.config['seed'])

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.adapter.loss_fn(self.model,self.batch)
        loss.backward()
        self.optimizer.step()
        hook = getattr(self.adapter,'post_optimizer_step',None)
        if 'fused_ema_update' in self.replacements:
            if not self.ema_bindings:raise ValueError('This adapter has no verified independent EMA fusion sites')
            from kernel_evolution.fusion import apply
            apply(self.ema_bindings,self.replacements['fused_ema_update'])
        elif hook:
            hook(self.model)
        return loss.detach()

    def profile(self):
        def capture(region,args):
            key = json.dumps(describe(args), sort_keys=True)
            byshape = self.cases.setdefault(region.op_name,{})
            if key not in byshape:
                byshape[key] = {'args':clone(args), 'count':0, 'description':describe(args)}
            byshape[key]['count'] += 1
        for r in regions(self.model): r.observe=capture
        self.step()
        for r in regions(self.model): r.observe=None
        self.restore()
        for _ in range(self.config['profile_warmup']): self.step()
        torch.cuda.synchronize()
        try:
            for r in regions(self.model):r.profiling=True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                for _ in range(self.config['profile_steps']):self.step()
        finally:
            for r in regions(self.model):r.profiling=False
        torch.cuda.synchronize()
        step_ms = statistics.median(cuda_times(self.step,10,3))
        table = {e.key: float(e.device_time_total)/self.config['profile_steps']/1000
                 for e in prof.key_averages() if e.key.startswith('kernel_evolution::')}
        targets = []
        for name, cases in self.cases.items():
            op_ms = table.get('kernel_evolution::'+name,0.)
            pct = 100*op_ms/step_ms
            entries=sorted(cases.values(),key=lambda c:c['count'],reverse=True)
            targets.append(dict(id=name,op=name,pct_step_time=pct,op_time_ms=op_ms,
                eligible=name in self.config['allowed_ops'] and pct>=self.config['min_pct_step_time'],
                shapes=[dict(args=e['description'],count=e['count']) for e in entries]))
        ema=next((t for t in targets if t['id']=='ema_update' and t['eligible']),None)
        if self.ema_bindings and ema:
            from kernel_evolution.fusion import arguments
            args=clone(arguments(self.ema_bindings))
            description=describe(args)
            self.cases['fused_ema_update']={json.dumps(description,sort_keys=True):
                dict(args=args,count=1,description=description)}
            targets.append(dict(id='fused_ema_update',op='fused_ema_update',fusion_of='ema_update',
                available_from_generation=2,call_sites=len(self.ema_bindings),
                pct_step_time=ema['pct_step_time'],op_time_ms=ema['op_time_ms'],eligible=True,
                shapes=[dict(args=description,count=1)]))
        torch.save(self.cases,self.run_dir/'cases.pt')
        self.restore()
        return dict(targets=targets,step_time_ms=step_ms,fusion_issue=self.fusion_issue,discovery=self.discovery,
                    profiler_table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))

    def time_set(self, replacements):
        self.replacements=replacements
        install(self.model,replacements)
        self.restore()
        return cuda_times(self.step,self.config['timed_steps'],self.config['warmup_steps'])

    def state_after_step(self,replacements):
        self.replacements=replacements
        install(self.model,replacements)
        self.restore()
        loss=self.step()
        torch.cuda.synchronize()
        return clone((loss,self.model.state_dict(),self.optimizer.state_dict()))

    def flops(self):
        from torch.utils.flop_counter import FlopCounterMode
        self.restore()
        self.replacements={}
        install(self.model,{})
        with FlopCounterMode(display=False) as counter:
            self.adapter.loss_fn(self.model,self.batch).backward()
        return counter.get_total_flops()/self.batch_size


def load_source(path):
    from kernel_evolution.archive import source_hash
    name='candidate_'+source_hash(Path(path).read_text())
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module
    spec.loader.exec_module(module)
    if not callable(getattr(module,'kernel',None)):
        raise TypeError('Candidate must expose callable kernel(*args)')
    return module.kernel


def inductor_seed(op, cases, output_dir):
    from kernel_evolution.seeds import InductorSeed
    compiled=InductorSeed(op,output_dir)
    for case in cases:
        compiled.specialize(case['args'])
    sources=compiled.sources
    if sources and not any('triton' in Path(p).read_text() and '@triton.jit' in Path(p).read_text() for p in sources):
        raise RuntimeError(f'Inductor emitted no Triton for {op}')
    return compiled,sources
