"""Real Triton fixture plus deterministic negative candidates."""
NORM = '''import torch
import triton
import triton.language as tl

@triton.jit
def _row(X,DY,W,MEAN,RSTD,DX,PART,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    c=tl.arange(0,B)
    x=tl.load(X+row*N+c,c<N,0).to(tl.float32)
    dy=tl.load(DY+row*N+c,c<N,0).to(tl.float32)
    w=tl.load(W+c,c<N,0).to(tl.float32)
    mean=tl.load(MEAN+row).to(tl.float32)
    rs=tl.load(RSTD+row).to(tl.float32)
    xh=tl.where(c<N,(x-mean)*rs,0.)
    g=dy*w
    a=tl.sum(g,0)/N
    b=tl.sum(g*xh,0)/N
    dx=(g-a-xh*b)*rs
    tl.store(DX+row*N+c,dx,c<N)
    tl.store(PART+row*N+c,dy*xh,c<N)

@triton.jit
def _params(PART,DY,DW,DB,M:tl.constexpr,N:tl.constexpr):
    col=tl.program_id(0)*16+tl.arange(0,16)
    rr=tl.arange(0,128)
    a=tl.full((128,16),0,tl.float32)
    b=tl.full((128,16),0,tl.float32)
    for start in range(tl.cdiv(M,128)):
        row=start*128+rr
        off=row[:,None]*N+col[None,:]
        mask=(row[:,None]<M)&(col[None,:]<N)
        a+=tl.load(PART+off,mask,0).to(tl.float32)
        b+=tl.load(DY+off,mask,0).to(tl.float32)
    tl.store(DW+col,tl.sum(a,axis=0),col<N)
    tl.store(DB+col,tl.sum(b,axis=0),col<N)

def kernel(x,dy,weight,mean,rstd):
    n=x.shape[-1]
    m=x.numel()//n
    x=x.contiguous()
    dy=dy.contiguous()
    dx=torch.empty_like(x)
    partial=torch.empty((m,n),device=x.device,dtype=torch.float32)
    dw=torch.empty_like(weight)
    db=torch.empty_like(weight)
    _row[(m,)](x,dy,weight,mean,rstd,dx,partial,n,triton.next_power_of_2(n))
    _params[(triton.cdiv(n,16),)](partial,dy,dw,db,m,n)
    return dx,dw,db
'''


EMA = '''import torch
import triton
import triton.language as tl

@triton.jit
def _ema(T,S,O,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    t=tl.load(T+i,i<N,0).to(tl.float32)
    s=tl.load(S+i,i<N,0).to(tl.float32)
    tl.store(O+i,t*D+s*(1.-D),i<N)

def kernel(target,source,decay):
    out=torch.empty_like(target)
    _ema[(triton.cdiv(target.numel(),1024),)](target,source,out,target.numel(),decay,1024)
    return out
'''


def fixture(index,op):
    if op=='fused_ema_update':
        from kernel_evolution.fusion import FIXTURE
        base=FIXTURE
    elif op in {'layer_norm_backward','ema_update'}:
        base=NORM if op=='layer_norm_backward' else EMA
    else:raise ValueError('No Triton fixture is implemented for '+op+'; eager is never a candidate')
    kind=index%4
    if kind==0: return base,'fixture_triton'
    if kind==1: return base.replace('tl.load(','tl.api_does_not_exist('),'fixture_compile_failure'
    if kind==2:
        return base.replace('dx=(g-a-xh*b)*rs','dx=(g-a-xh*b)*rs+1.').replace('t*D+s*(1.-D)','t*D+s*(1.-D)+1.'),'fixture_mismatch'
    return base+'''\n_real_kernel=kernel
_cached=[]
def kernel(*args):
    if not _cached: _cached.append(_real_kernel(*args))
    return _cached[0]
''','planted_cheat'
