"""Deterministic, model-free correctness and performance gates."""
import ast
import json
import math
import statistics
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten
from kernel_evolution.ops import REFERENCES
from kernel_evolution.runtime import clone, cuda_times, summarize, seed_all


class Mismatch(AssertionError):
    pass


def compare(actual, expected, rtol, atol):
    av,aspec=tree_flatten(actual)
    ev,espec=tree_flatten(expected)
    if aspec != espec:
        raise Mismatch(f'Output structure differs: {aspec} vs {espec}')
    maximum_abs=maximum_rel=0.
    for i,(a,e) in enumerate(zip(av,ev)):
        if not isinstance(e,torch.Tensor):
            if a != e: raise Mismatch(f'Leaf {i}: {a!r} != {e!r}')
            continue
        if not isinstance(a,torch.Tensor) or a.shape!=e.shape or a.dtype!=e.dtype or a.device!=e.device:
            raise Mismatch(f'Leaf {i}: exact shape/dtype/device mismatch; expected {e.shape}, {e.dtype}, {e.device}')
        if not a.is_floating_point():
            if not torch.equal(a,e): raise Mismatch(f'Nonfloating leaf {i} mismatch')
            continue
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            raise Mismatch(f'Leaf {i}: nonfinite values')
        delta=(a.detach().float()-e.detach().float()).abs()
        err=float(delta.max()) if delta.numel() else 0.
        denominator_floor=max(atol,1e-12) if math.isfinite(atol) else 1e-5
        rel=float((delta/e.detach().float().abs().clamp_min(denominator_floor)).max()) if delta.numel() else 0.
        maximum_abs=max(maximum_abs,err)
        maximum_rel=max(maximum_rel,rel)
        bad=delta>atol+rtol*e.detach().float().abs()
        if bad.any():
            indices=bad.nonzero()[:8].tolist()
            raise Mismatch(json.dumps(dict(leaf=i,max_abs=err,max_rel=rel,first_bad_indices=indices,
                                           assertion=f'abs_error <= {atol} + {rtol} * abs(reference)')))
    return dict(max_abs=maximum_abs,max_rel=maximum_rel)


def fresh_args(op, case, seed, unseen=False):
    seed_all(seed)
    args=case['args']
    if op=='layer_norm_backward':
        x,dy,w,mean,rstd=args
        shape=list(x.shape)
        if unseen: shape[0]+=3
        x=torch.randn(shape,device=x.device,dtype=x.dtype)
        dy=torch.randn_like(x)
        w=torch.randn_like(w)
        _,mean,rstd=torch.native_layer_norm(x,(shape[-1],),w,None,1e-5)
        return x,dy,w,mean,rstd
    if op=='ema_update':
        t,s,d=args
        shape=list(t.shape)
        if unseen: shape[0]+=7
        return torch.randn(shape,device=t.device,dtype=t.dtype),torch.randn(shape,device=s.device,dtype=s.dtype),d
    if op=='fused_ema_update':
        fresh=[]
        for t,s in zip(args[:-1:2],args[1:-1:2]):
            shape=list(t.shape)
            if unseen:shape[0]+=7
            fresh.extend([torch.randn(shape,device=t.device,dtype=t.dtype),
                          torch.randn(shape,device=s.device,dtype=s.dtype)])
        return (*fresh,args[-1])
    if op=='masked_gather_add':
        x,p,ix=args
        b,t,d=x.shape
        if unseen: b+=3
        x=torch.randn(b,t,d,device=x.device,dtype=x.dtype).requires_grad_(True)
        p=torch.randn_like(p).requires_grad_(True)
        ix=torch.randint(0,t,(b,ix.shape[1]),device=ix.device)
        return x,p,ix
    if op=='gelu_mlp':
        x,w,b=args
        shape=list(x.shape)
        if unseen: shape[0]+=3
        return (torch.randn(shape,device=x.device,dtype=x.dtype).requires_grad_(True),
                torch.randn_like(w).requires_grad_(True),torch.randn_like(b).requires_grad_(True))
    raise ValueError(op)


def evaluate_with_grads(fn,args):
    leaves=[a for a in args if isinstance(a,torch.Tensor) and a.requires_grad]
    out=fn(*args)
    outs=[a for a in tree_flatten(out)[0] if isinstance(a,torch.Tensor) and a.requires_grad]
    grads=()
    if leaves:
        if not outs: raise Mismatch('Candidate detached every differentiable output')
        # Nonconstant cotangents detect incorrect backward implementations hidden by sum().
        cotangents=[torch.linspace(-.7,1.3,o.numel(),device=o.device,dtype=o.dtype).reshape(o.shape) for o in outs]
        grads=torch.autograd.grad(outs,leaves,cotangents,allow_unused=False)
    return out,grads


def gate2(op, fn, cases, config, floor=False):
    errors=[]
    selected=cases[:3]
    # Retain observed shapes; use three input draws on each plus a hidden valid shape.
    for index,case in enumerate(selected+[selected[0]]):
        for draw in range(3):
            args=fresh_args(op,case,config['seed']+1009*index+37*draw,unseen=index==len(selected))
            before=clone(args)
            if op=='layer_norm_backward':
                x,dy,w,mean,rstd=args
                x0=x.detach().requires_grad_(True)
                w0=w.detach().requires_grad_(True)
                b0=torch.zeros_like(w).requires_grad_(True)
                out=torch.nn.functional.layer_norm(x0,(x.shape[-1],),w0,b0,1e-5)
                expected=torch.autograd.grad(out,(x0,w0,b0),dy)
                actual=fn(*args)
                # These outputs ARE the first-order input and parameter gradients.
            else:
                expected=evaluate_with_grads(REFERENCES[op],args)
                actual=evaluate_with_grads(fn,args)
            errors.append(compare(actual,expected,float('inf') if floor else config['rtol'],
                                  float('inf') if floor else config['atol']))
            compare(args,before,0.,0.)
    return dict(max_abs=max(x['max_abs'] for x in errors),max_rel=max(x['max_rel'] for x in errors),
                checks=len(errors),observed_shapes=len(selected),unseen_shapes=1)


def source_scan(source):
    ast.parse(source)
    flags=[]
    for marker in ['lru_cache','global ','torch.compile','open(','pathlib','subprocess','os.','socket','requests']:
        if marker in source: flags.append(marker)
    return flags


def paired_measure(measure,baseline,candidate,margin):
    """Predeclared two stages, four paired blocks each. No repeat-until-pass loop."""
    records=[]
    for stage in range(2):
        for i in range(4):
            order=[('baseline',baseline),('candidate',candidate)]
            if i%2: order.reverse()
            record={}
            for name,fn in order: record[name]=measure(fn)
            record['gain']=1-record['candidate']/record['baseline']
            records.append(record)
        gains=[r['gain'] for r in records]
        # Require every paired block to clear the calibrated margin.
        if min(gains)>margin: status='pass'; break
        if statistics.median(gains)<=0: status='slower'; break
        status='inconclusive'
    return dict(status=status,baseline=statistics.median(r['baseline'] for r in records),
                candidate=statistics.median(r['candidate'] for r in records),margin=margin,blocks=records)


def gate3(op,baseline,candidate,cases,config):
    samples=[(fresh_args(op,c,config['seed']+71+i),c['count']) for i,c in enumerate(cases)]
    def measure(fn):
        total=0.
        count=0
        for args,weight in samples:
            med=statistics.median(cuda_times(lambda:fn(*args),config['micro_reps'],10))
            total+=med*weight
            count+=weight
        return total/count
    return paired_measure(measure,baseline,candidate,config['gate3_margin'])


def gate4(harness,incumbents,op,candidate,config):
    new={**incumbents,op:candidate}
    baseline_state=harness.state_after_step(incumbents)
    candidate_state=harness.state_after_step(new)
    compare_step_state(candidate_state,baseline_state,config['rtol'],config['atol'])
    return paired_measure(lambda replacement:statistics.median(harness.time_set(replacement)),
                          incumbents,new,config['gate4_margin'])


def compare_step_state(actual,expected,rtol,atol):
    """Ignore only equivalent optimizer execution bookkeeping, not numerical state."""
    actual=clone(actual)
    normalized=[]
    for key,state in actual[2]['state'].items():
        a=state.get('step');e=expected[2]['state'][key].get('step')
        if isinstance(a,torch.Tensor) and isinstance(e,torch.Tensor) and a.device!=e.device:
            compare(a.cpu(),e.cpu(),0.,0.)
            state['step']=a.to(e.device)
            normalized.append('step_counter_device')
    for i,group in enumerate(actual[2]['param_groups']):
        other=expected[2]['param_groups'][i]
        if group.get('capturable')!=other.get('capturable'):
            group['capturable']=other['capturable']
            normalized.append('capturable_execution_flag')
    return {**compare(actual,expected,rtol,atol),'normalized_execution_metadata':normalized}


def selftest(op,cases,config):
    cached=[]
    def caching(*args):
        if not cached: cached.append(REFERENCES[op](*args))
        return cached[0]
    shape=tuple(cases[0]['args'][0].shape)
    def hardcoded(*args):
        out=REFERENCES[op](*args)
        if tuple(args[0].shape)!=shape:
            from torch.utils._pytree import tree_map
            out=tree_map(lambda x:torch.zeros_like(x) if isinstance(x,torch.Tensor) else x,out)
        return out
    rejected=[]
    for name,fn in [('cached_output',caching),('hardcoded_shape',hardcoded)]:
        try: gate2(op,fn,cases,config)
        except (AssertionError,RuntimeError) as exc: rejected.append(dict(name=name,gate=2,error=str(exc)[:2000]))
        else: raise RuntimeError(f'Verifier self-test FAILED: {name} survived gate 2')
    return rejected
