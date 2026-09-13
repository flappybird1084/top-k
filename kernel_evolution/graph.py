"""Discover supported functional regions while preserving the adapter's model object.

FX is a discovery backend, not an assumption that every PyTorch program is
traceable. Unsupported traces and unverified rewrites retain the original path.
"""
import copy
import types
import torch
from torch import nn
from torch.nn import functional as F
from kernel_evolution.ops import Region,NormFunction


class FunctionalNormRegion(Region):
    op_name='layer_norm_backward'
    def forward(self,x,normalized_shape,weight,bias,eps=1e-5):
        if len(normalized_shape)!=1 or weight is None:
            return F.layer_norm(x,normalized_shape,weight,bias,eps)
        return NormFunction.apply(x,weight,bias,eps,self)


class FunctionalMLPRegion(Region):
    op_name='gelu_mlp'
    def forward(self,x,weight,bias):
        with self.record():return self.invoke(x,weight,bias)


class RegionTracer(torch.fx.Tracer):
    def is_leaf_module(self,module,qualname):
        return isinstance(module,Region) or super().is_leaf_module(module,qualname)


def argument(node,index,name,default=None):
    return node.args[index] if len(node.args)>index else node.kwargs.get(name,default)


def discover_functional(model):
    """Return a reversible rewrite plus a serializable discovery report."""
    report={'backend':'torch.fx','regions':[],'fallback_reason':None}
    original_forward=model.forward
    attached=[]
    try:
        # Tracing user Python may execute ordinary Python side effects. Trace a
        # copy; no speculative trace should mutate the live model.
        traced_root=copy.deepcopy(model)
        graph=RegionTracer().trace(traced_root)
        generated=torch.fx.GraphModule(traced_root,graph)
        # FX can freeze tensor factories into constants. Do not silently replace
        # randomness or input-independent computations with a traced constant.
        for node in graph.nodes:
            if node.op=='get_attr':
                value=model
                for part in node.target.split('.'):
                    if not hasattr(value,part):raise ValueError('Trace introduced a new tensor constant: '+node.target)
                    value=getattr(value,part)

        def add_region(node,region,args):
            name='_kernel_evolution_functional_'+str(len(report['regions']))
            while hasattr(model,name) or hasattr(generated,name):name+='x'
            generated.add_module(name,region)
            with graph.inserting_before(node):replacement=graph.call_module(name,args=args)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            report['regions'].append({'op':region.op_name,'module':name})

        for node in list(graph.nodes):
            if node.op=='call_function' and node.target is F.layer_norm:
                shape=argument(node,1,'normalized_shape')
                weight=argument(node,2,'weight')
                if not isinstance(shape,(tuple,list)) or len(shape)!=1 or weight is None:continue
                add_region(node,FunctionalNormRegion(),(argument(node,0,'input'),shape,weight,
                    argument(node,3,'bias'),argument(node,4,'eps',1e-5)))
                continue
            gelu=(node.op=='call_function' and node.target is F.gelu and
                  argument(node,1,'approximate','none')=='tanh')
            if node.op=='call_module':
                module=generated.get_submodule(node.target)
                gelu=isinstance(module,nn.GELU) and module.approximate=='tanh'
            if not gelu:continue
            linear=argument(node,0,'input')
            if not isinstance(linear,torch.fx.Node) or len(linear.users)!=1:continue
            if linear.op=='call_function' and linear.target is F.linear:
                args=(argument(linear,0,'input'),argument(linear,1,'weight'),argument(linear,2,'bias'))
                if args[2] is None:continue
            elif linear.op=='call_module' and isinstance(generated.get_submodule(linear.target),nn.Linear):
                module=generated.get_submodule(linear.target)
                if module.bias is None:continue
                with graph.inserting_before(node):
                    weight=graph.get_attr(linear.target+'.weight')
                    bias=graph.get_attr(linear.target+'.bias')
                args=(argument(linear,0,'input'),weight,bias)
            else:continue
            add_region(node,FunctionalMLPRegion(),args)
            graph.erase_node(linear)
        if not report['regions']:return report,lambda:None
        graph.lint()
        generated.recompile()
        # Keep original class, parameters, hooks and adapter-visible attributes.
        for item in report['regions']:
            name=item['module']
            model.add_module(name,generated.get_submodule(name))
            attached.append(name)
        model.forward=types.MethodType(generated.forward.__func__,model)
    except Exception as exc:
        model.forward=original_forward
        for name in attached:delattr(model,name)
        report['regions']=[]
        report['fallback_reason']=f'{type(exc).__name__}: {exc}'

    def rollback():
        model.forward=original_forward
        for name in attached:
            if hasattr(model,name):delattr(model,name)
    return report,rollback


def validate_rewrite(model,original_forward,loss_fn,batch,config):
    """Compare loss and every parameter gradient for three restored RNG states."""
    from kernel_evolution.runtime import clone,seed_all
    from kernel_evolution.verifier import compare
    rewritten=model.forward
    state=clone(model.state_dict())
    parameters=[p for p in model.parameters() if p.requires_grad]
    scalar_state=[(module,name,value) for module in model.modules() for name,value in vars(module).items()
                  if isinstance(value,(bool,int,float,str)) or value is None]
    def restore_scalars():
        for module,name,value in scalar_state:setattr(module,name,value)
    def scalars():
        return tuple(getattr(module,name) for module,name,_ in scalar_state)
    try:
        for draw in range(3):
            values=[]
            for forward in (original_forward,rewritten):
                model.load_state_dict(state)
                restore_scalars()
                seed_all(config['seed']+draw*101)
                model.forward=forward
                loss=loss_fn(model,batch)
                gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
                values.append(clone((loss,gradients,model.state_dict(),scalars())))
            compare(values[1],values[0],config['rtol'],config['atol'])
    finally:
        model.forward=rewritten
        model.load_state_dict(state)
        restore_scalars()
        model.zero_grad(set_to_none=True)
        seed_all(config['seed'])
