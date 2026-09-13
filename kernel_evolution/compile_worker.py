"""Compile captured Triton launches without creating a CUDA context.

Fake tensors execute only the candidate's host scaffolding. JIT launches are
intercepted and compiled for an explicit target; autotuning benchmarks remain
in the later GPU worker. Unsupported host scaffolding falls back to that worker.
The CPU result can reject a compilation error, but can never accept a kernel.
"""
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch


def compile_candidate(request):
    import torch
    import triton
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource,compile,make_backend
    from triton.runtime.jit import JITFunction,create_function_from_signature
    from triton.runtime.autotuner import Autotuner
    from kernel_evolution.runtime import load_source

    if torch.cuda.is_initialized():raise RuntimeError('CPU compiler unexpectedly initialized CUDA')
    target=GPUTarget(**request['gpu_target'])
    source=Path(request['code_path']).read_text()
    identity=dict(source=source,args=request['args'],target=request['gpu_target'],
                  triton=triton.__version__,torch=torch.__version__,protocol=1)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=Path(request['cache_dir'])/(digest+'.json')
    if manifest.exists():
        cached=json.loads(manifest.read_text())
        if all(Path(p).exists() for p in cached.get('artifacts',[])):
            return {**cached,'cache_hit':True}
    backend=make_backend(target)
    artifacts=[]
    compiled_hashes=[]
    compiler_error=[]

    def capture(jit,*args,grid=None,warmup=False,**kwargs):
        try:
            binder=create_function_from_signature(jit.signature,jit.params,backend)
            bound,specialization,options=binder(*args,**kwargs)
            options,signature,constants,attrs=jit._pack_args(backend,kwargs,bound,specialization,options)
            kernel=compile(ASTSource(jit,signature,constants,attrs),target=target,options=options.__dict__)
            compiled_hashes.append(kernel.hash)
            # Triton's persistent cache is shared with the later GPU subprocess.
            artifacts.extend(str(p) for p in kernel.metadata_group.values())
            return kernel
        except Exception:
            compiler_error.append(traceback.format_exc())
            raise

    def capture_autotune(tuner,*args,**kwargs):
        # Compile every legal config. Do not benchmark or select a winner on CPU.
        tuner.nargs=dict(zip(tuner.arg_names,args))
        try:
            for config in tuner.prune_configs(kwargs):
                tuner.fn.run(*args,**kwargs,**config.all_kwargs())
        finally:tuner.nargs=None

    try:
        with torch.no_grad(),FakeTensorMode(),patch.object(JITFunction,'run',capture),patch.object(Autotuner,'run',capture_autotune):
            fn=load_source(request['code_path'])
            args=[]
            for descriptor in request['args']:
                if isinstance(descriptor,dict) and 'shape' in descriptor:
                    tensor=torch.empty_strided(descriptor['shape'],descriptor['stride'],device='cuda',
                        dtype=getattr(torch,descriptor['dtype']),requires_grad=descriptor.get('requires_grad',False))
                    args.append(tensor)
                else:args.append(descriptor)
            fn(*args)
        if torch.cuda.is_initialized():raise RuntimeError('CPU compiler initialized CUDA')
    except Exception:
        if compiler_error:
            return dict(status='compile_error',compile_ok=0,failure_note=compiler_error[-1])
        return dict(status='deferred',failure_note=traceback.format_exc(),reason='Host scaffolding requires GPU verification')
    if not compiled_hashes:
        return dict(status='deferred',reason='No Triton launch observed; GPU worker must verify the contract')
    result=dict(status='compiled',compile_ok=1,cache_hit=False,source_hash=hashlib.sha256(source.encode()).hexdigest(),
                kernels=compiled_hashes,artifacts=sorted(set(artifacts)),cuda_initialized=False,
                scope='modal_entrypoint; additional autograd kernels compile in gate 2')
    manifest.parent.mkdir(parents=True,exist_ok=True)
    temporary=manifest.with_suffix('.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(manifest)
    return result


def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    try:result=compile_candidate(request)
    except Exception:result=dict(status='deferred',failure_note=traceback.format_exc())
    Path(sys.argv[2]).write_text(json.dumps(result))


if __name__=='__main__':main()
