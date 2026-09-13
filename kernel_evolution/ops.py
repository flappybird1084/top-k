"""Model-independent operation contracts and autograd bridges."""
import torch
from torch import nn
from torch.nn import functional as F


def layer_norm_backward(x, dy, weight, mean, rstd):
    return torch.ops.aten.native_layer_norm_backward.default(
        dy, x, [x.shape[-1]], mean, rstd, weight, weight, [True, True, True])


def ema_update(target, source, decay):
    return target*decay+source*(1.-decay)


def masked_gather_add(x, positions, indices):
    return torch.gather(x+positions, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def gelu_mlp(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias), approximate='tanh')


REFERENCES = dict(layer_norm_backward=layer_norm_backward, ema_update=ema_update,
                  masked_gather_add=masked_gather_add, gelu_mlp=gelu_mlp)


class Region(nn.Module):
    op_name = ''
    def __init__(self):
        super().__init__()
        self.candidate = None
        self.observe = None

    def invoke(self, *args):
        if self.observe:
            self.observe(self, args)
        return (self.candidate or REFERENCES[self.op_name])(*args)


class NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, region):
        out, mean, rstd = torch.native_layer_norm(x, (x.shape[-1],), weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.region = region
        return out

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        with torch.profiler.record_function('kernel_evolution::layer_norm_backward'):
            dx, dw, db = ctx.region.invoke(x, dy.contiguous(), weight, mean, rstd)
        return dx, dw, db, None, None


class LayerNormRegion(Region):
    op_name = 'layer_norm_backward'
    def __init__(self, original):
        super().__init__()
        self.weight, self.bias, self.eps = original.weight, original.bias, original.eps
    def forward(self, x):
        return NormFunction.apply(x, self.weight, self.bias, self.eps, self)


class EMARegion(Region):
    op_name = 'ema_update'
    def forward(self, target, source, decay):
        with torch.profiler.record_function('kernel_evolution::ema_update'):
            return self.invoke(target, source, decay)


class GatherRegion(Region):
    op_name = 'masked_gather_add'
    def forward(self, x, positions, indices):
        with torch.profiler.record_function('kernel_evolution::masked_gather_add'):
            return self.invoke(x, positions, indices)


class MLPRegion(Region):
    op_name = 'gelu_mlp'
    def __init__(self, linear):
        super().__init__()
        self.weight, self.bias = linear.weight, linear.bias
    def forward(self, x):
        with torch.profiler.record_function('kernel_evolution::gelu_mlp'):
            return self.invoke(x, self.weight, self.bias)


def instrument(model):
    """Discover standard LayerNorms recursively; explicit composite regions also work."""
    if isinstance(model,nn.Sequential):
        entries=list(model._modules.items())
        for (name,first),(next_name,second) in zip(entries,entries[1:]):
            if isinstance(first,nn.Linear) and first.bias is not None and isinstance(second,nn.GELU) and second.approximate=='tanh':
                model._modules[name]=MLPRegion(first)
                model._modules[next_name]=nn.Identity()
    for name, child in list(model.named_children()):
        if isinstance(child, nn.LayerNorm) and len(child.normalized_shape) == 1 and child.weight is not None and child.bias is not None:
            setattr(model, name, LayerNormRegion(child))
        elif not isinstance(child, Region):
            instrument(child)
    return model


def regions(model):
    return [m for m in model.modules() if isinstance(m, Region)]


def install(model, candidates):
    for r in regions(model):
        r.candidate = candidates.get(r.op_name)
