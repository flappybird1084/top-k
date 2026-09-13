"""Horizontal fusion contract for independent EMA regions in a post-step hook.

The binding trace is model-independent: it observes EMARegion calls, not parameter
names or a particular encoder architecture. A fused candidate must still pass the
same numerical and complete-step gates as a single-region candidate.
"""
from dataclasses import dataclass
import torch
from kernel_evolution.ops import EMARegion,fused_ema_update


@dataclass
class EMABinding:
    target: torch.Tensor
    source: torch.Tensor
    decay: float


@torch.no_grad()
def capture_ema_bindings(model,post_step):
    bindings=[]
    modules=[m for m in model.modules() if isinstance(m,EMARegion)]
    if not modules or post_step is None:return []
    previous=[(m,m.candidate,m.observe) for m in modules]
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    def collect(target,source,decay):
        bindings.append(EMABinding(target,source,float(decay)))
        return target
    try:
        for module in modules:module.candidate=collect;module.observe=None
        post_step(model)
        after=model.state_dict()
        if any(not torch.equal(v,after[k]) for k,v in before.items()):
            raise ValueError('Post-step hook mutates state outside the traced EMA regions; fusion is unsupported')
    finally:
        for module,candidate,observe in previous:module.candidate=candidate;module.observe=observe
        model.load_state_dict(before)
    if len(bindings)<2:return []
    if len({b.decay for b in bindings})!=1:
        raise ValueError('EMA fusion requires a shared decay value')
    # Aliased updates can depend on earlier outputs and are not independent.
    targets=[b.target.untyped_storage().data_ptr() for b in bindings]
    sources=[b.source.untyped_storage().data_ptr() for b in bindings]
    if len(set(targets))!=len(targets) or set(targets)&set(sources):
        raise ValueError('EMA regions alias or depend on each other; horizontal fusion is unsupported')
    return bindings


def arguments(bindings):
    if not bindings:raise ValueError('No EMA regions to fuse')
    return (*[value for b in bindings for value in (b.target,b.source)],bindings[0].decay)


reference=fused_ema_update


def composed(parent):
    def kernel(*args):
        return tuple(parent(t,s,args[-1]) for t,s in zip(args[:-1:2],args[1:-1:2]))
    return kernel


@torch.no_grad()
def apply(bindings,candidate):
    outputs=candidate(*arguments(bindings))
    if not isinstance(outputs,tuple) or len(outputs)!=len(bindings):
        raise ValueError('Fused EMA output must be a tuple with one tensor per region')
    for binding,out in zip(bindings,outputs):
        if out.shape!=binding.target.shape or out.dtype!=binding.target.dtype or out.device!=binding.target.device:
            raise ValueError('Fused EMA output metadata mismatch')
        binding.target.copy_(out)


FIXTURE='''import torch
import triton
import triton.language as tl

@triton.jit
def _fused(T,S,O,N:tl.constexpr,P:tl.constexpr,D:tl.constexpr,B:tl.constexpr,K:tl.constexpr):
    pid=tl.program_id(0)
    lane=tl.arange(0,B)
    for k in tl.static_range(K):
        if (pid>=P[k]) & (pid<P[k+1]):
            i=(pid-P[k])*B+lane
            t=tl.load(T[k]+i,i<N[k],0).to(tl.float32)
            s=tl.load(S[k]+i,i<N[k],0).to(tl.float32)
            tl.store(O[k]+i,t*D+s*(1.-D),i<N[k])

def kernel(*args):
    targets=args[:-1:2]
    sources=args[1:-1:2]
    decay=args[-1]
    outputs=tuple(torch.empty_like(t) for t in targets)
    sizes=tuple(t.numel() for t in targets)
    prefix=[0]
    for n in sizes: prefix.append(prefix[-1]+triton.cdiv(n,256))
    _fused[(prefix[-1],)](targets,sources,outputs,sizes,tuple(prefix),decay,256,len(targets))
    return outputs
'''
