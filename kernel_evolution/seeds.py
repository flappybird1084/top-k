"""Inductor entry points without the outer Dynamo runtime guard wrapper.

torch.compile is used to extract a specialization, never as the timed callable.
The captured Inductor backend entry preserves AOT autograd when needed.
"""
import inspect
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten,tree_unflatten
from kernel_evolution.ops import REFERENCES


def argument_key(args):
    return (torch.is_grad_enabled(),tuple(
        (tuple(a.shape),a.stride(),a.dtype,a.device.type,a.requires_grad) if isinstance(a,torch.Tensor)
        else (type(a).__name__,a) for a in args))


class InductorSeed:
    def __init__(self,op,output_dir):
        self.op=op
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.dispatch={}
        self.sources=[]
        self.reference=REFERENCES[op]
        self.parameters=list(inspect.signature(self.reference).parameters)

    def _reader(self,source):
        if hasattr(source,'local_name'):
            index=self.parameters.index(source.local_name)
            return lambda args:args[index]
        if hasattr(source,'base') and hasattr(source,'index'):
            if getattr(source.base,'local_name',None)=='args':
                index=source.index
                return lambda args:args[index]
            base=self._reader(source.base)
            index=source.index
            return lambda args:base(args)[index]
        raise RuntimeError('Unsupported Inductor placeholder source: '+str(source))

    def specialize(self,args):
        key=argument_key(args)
        if key in self.dispatch:return self.dispatch[key]
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.utils import run_and_get_code
        captured=[]
        def backend(gm,example_inputs):
            readers=[]
            for node in gm.graph.nodes:
                if node.op=='placeholder':
                    grapharg=node.meta.get('grapharg')
                    if grapharg is None:raise RuntimeError('Missing Dynamo argument provenance')
                    readers.append(self._reader(grapharg.source))
            compiled=compile_fx(gm,example_inputs)
            captured.append((compiled,readers))
            return compiled
        torch._dynamo.config.recompile_limit=128
        extract=torch.compile(self.reference,backend=backend,dynamic=False,fullgraph=True)
        expected,code=run_and_get_code(extract,*args)
        if len(captured)!=1:raise RuntimeError('Expected exactly one extracted region')
        flat_expected,spec=tree_flatten(expected)
        compiled,readers=captured[0]
        def invoke(*new_args):
            outputs=compiled(*[read(new_args) for read in readers])
            flat,_=tree_flatten(outputs)
            if len(flat)!=len(flat_expected):raise RuntimeError('Inductor output contract differs')
            return tree_unflatten(flat,spec)
        # Validate argument order and output reconstruction before timing the entry.
        actual=invoke(*args)
        for a,b in zip(tree_flatten(actual)[0],flat_expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        sequence=len(self.dispatch)
        for j,source in enumerate(code):
            p=self.output_dir/f'{self.op}_direct_{sequence}_{j}.py'
            p.write_text(source)
            self.sources.append(str(p))
        self.dispatch[key]=invoke
        return invoke

    def __call__(self,*args):
        entry=self.dispatch.get(argument_key(args))
        if entry is None:entry=self.specialize(args)
        return entry(*args)
