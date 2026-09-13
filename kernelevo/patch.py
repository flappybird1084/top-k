"""Route arbitrary (repo-provided) models through the kernelevo op registry
without adapter cooperation.

- install(): monkeypatch torch.nn.functional.layer_norm so every nn.LayerNorm /
  F.layer_norm call in any model dispatches through the registry (profiling
  records it; gate 4 can swap a candidate in). The op's eager reference uses the
  original functional captured at ops import, so there is no recursion. Applied
  by the harness in every process that builds models — adapters need not know.
- fuse_mlp_blocks(model): rewrite Linear→GELU(tanh) pairs inside nn.Sequential
  to a fused module calling ops.gelu_mlp. Only tanh-approximate GELUs are fused
  (fusing an exact GELU would change the repo model's math). Adapter-writer
  output calls this on the built model; hand-rolled forwards are not covered.

ema_update / masked_gather_add stay adapter-explicit — they are model-specific.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelevo import ops

_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        ns = ((normalized_shape,) if isinstance(normalized_shape, int)
              else tuple(normalized_shape))
        if (weight is not None and bias is not None and len(ns) == 1
                and input.shape[-1] == ns[0]):
            return ops._dispatch("layer_norm", input, weight, bias, eps)
        return ops._F_LAYER_NORM(input, normalized_shape, weight, bias, eps)

    F.layer_norm = layer_norm

    if ops._F_RMS_NORM is not None:
        def rms_norm(input, normalized_shape, weight=None, eps=None):
            ns = ((normalized_shape,) if isinstance(normalized_shape, int)
                  else tuple(normalized_shape))
            if len(ns) == 1 and input.shape[-1] == ns[0]:
                return ops._dispatch("rms_norm", input, weight, eps)
            return ops._F_RMS_NORM(input, normalized_shape, weight, eps)

        F.rms_norm = rms_norm

    def cross_entropy(input, target, weight=None, size_average=None,
                      ignore_index=-100, reduce=None, reduction="mean",
                      label_smoothing=0.0):
        import torch as _t
        if (weight is None and size_average is None and reduce is None
                and reduction == "mean" and label_smoothing == 0.0
                and input.dim() == 2 and isinstance(target, _t.Tensor)
                and target.dtype == _t.long and target.dim() == 1):
            return ops._dispatch("cross_entropy", input, target, ignore_index)
        return ops._F_CROSS_ENTROPY(input, target, weight=weight,
                                    size_average=size_average,
                                    ignore_index=ignore_index, reduce=reduce,
                                    reduction=reduction,
                                    label_smoothing=label_smoothing)

    F.cross_entropy = cross_entropy


class _FusedLinearGELU(nn.Module):
    def __init__(self, linear: nn.Linear, approximate: str):
        super().__init__()
        self.linear = linear
        self.approximate = approximate

    def forward(self, x):
        return ops.gelu_mlp(x, self.linear.weight, self.linear.bias, self.approximate)


def fuse_mlp_blocks(model: nn.Module) -> int:
    fused = 0
    for mod in model.modules():
        if not isinstance(mod, nn.Sequential):
            continue
        for i in range(len(mod) - 1):
            lin, act = mod[i], mod[i + 1]
            if isinstance(lin, nn.Linear) and isinstance(act, nn.GELU):
                mod[i] = _FusedLinearGELU(lin, getattr(act, "approximate", "none"))
                mod[i + 1] = nn.Identity()
                fused += 1
    return fused


# ------------------------------------------------------------ behavioral routing
# Deterministic pattern matching onto the FIXED registry: modules are classified
# by what they COMPUTE (probe with seeded inputs, compare against each op's
# eager reference within tight tolerance), never by class or attribute names.
# No vocabulary is created at runtime; unmatched modules are left untouched.

def _probe_tol(dtype):
    return (dict(rtol=2e-2, atol=2e-2)
            if dtype in (torch.bfloat16, torch.float16) else
            dict(rtol=1e-4, atol=1e-5))


def _probe(mod, x):
    was_training = mod.training
    mod.eval()
    try:
        with torch.no_grad():
            return mod(x)
    except Exception:  # noqa: BLE001 — forward may need extra args; skip module
        return None
    finally:
        mod.train(was_training)


def _seeded_input(shape, like):
    g = torch.Generator().manual_seed(0)
    dtype = like.dtype if like.dtype.is_floating_point else torch.float32
    return torch.randn(shape, generator=g).to(device=like.device, dtype=dtype)


def _eps_candidates(mod):
    found = [getattr(mod, a) for a in ("eps", "variance_epsilon", "epsilon")
             if isinstance(getattr(mod, a, None), float)]
    return found or [1e-5, 1e-6]


def _try_norm(mod):
    params = dict(mod.named_parameters(recurse=False))
    if len(list(mod.children())) > 0:
        return None
    w = params.get("weight")
    if w is None or w.dim() != 1 or set(params) - {"weight", "bias"}:
        return None
    x = _seeded_input((2, 3, w.shape[0]), w)
    out = _probe(mod, x)
    if out is None or out.shape != x.shape:
        return None
    b = params.get("bias")
    tol = _probe_tol(w.dtype)
    for eps in _eps_candidates(mod):
        with torch.no_grad():
            if b is None and torch.allclose(
                    out, ops.REGISTRY["rms_norm"].eager(x, w, eps), **tol):
                def fwd(x, _m=mod, _e=eps):
                    return ops.rms_norm(x, _m.weight, _e)
                mod.forward = fwd
                mod._kevo_routed = "rms_norm"
                return "rms_norm"
            if b is not None and b.dim() == 1 and torch.allclose(
                    out, ops.REGISTRY["layer_norm"].eager(x, w, b, eps), **tol):
                def fwd(x, _m=mod, _e=eps):
                    return ops.layer_norm(x, _m.weight, _m.bias, _e)
                mod.forward = fwd
                mod._kevo_routed = "layer_norm"
                return "layer_norm"
    return None


def _try_gated_mlp(mod):
    linears = [(n, m) for n, m in mod.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != 3 or any(m.bias is not None for _, m in linears):
        return None
    shapes = {}
    for n, m in linears:
        shapes.setdefault(tuple(m.weight.shape), []).append((n, m))
    two = [v for v in shapes.values() if len(v) == 2]
    one = [v for v in shapes.values() if len(v) == 1]
    if not (len(two) == 1 and len(one) == 1):
        return None
    (a, b), (down,) = two[0], one[0]
    Fdim, K = a[1].weight.shape
    if down[1].weight.shape != (K, Fdim):
        return None
    x = _seeded_input((2, K), a[1].weight)
    out = _probe(mod, x)
    if out is None or out.shape != (2, K):
        return None
    variants = []
    for gate, up in ((a[1], b[1]), (b[1], a[1])):
        variants.append(("swiglu_mlp", None, gate, up,
                         F.silu(F.linear(x, gate.weight)) * F.linear(x, up.weight)))
        for approx in ("none", "tanh"):
            variants.append(("geglu_mlp", approx, gate, up,
                             F.gelu(F.linear(x, gate.weight), approximate=approx)
                             * F.linear(x, up.weight)))
    tol = _probe_tol(a[1].weight.dtype)
    with torch.no_grad():
        for opname, approx, gate, up, hidden in variants:
            if torch.allclose(out, F.linear(hidden, down[1].weight), **tol):
                if opname == "swiglu_mlp":
                    def fwd(x, _g=gate, _u=up, _d=down[1]):
                        return F.linear(ops.swiglu_mlp(x, _g.weight, _u.weight),
                                        _d.weight)
                else:
                    def fwd(x, _g=gate, _u=up, _d=down[1], _a=approx):
                        return F.linear(ops.geglu_mlp(x, _g.weight, _u.weight, _a),
                                        _d.weight)
                mod.forward = fwd
                mod._kevo_routed = opname
                return opname
    return None


def auto_route(model: nn.Module) -> dict:
    """Route recognizable modules onto the registry. Returns {op_name: count}."""
    install()
    counts: dict[str, int] = {}
    for _, mod in model.named_modules():
        if getattr(mod, "_kevo_routed", None):
            continue
        hit = _try_norm(mod) or _try_gated_mlp(mod)
        if hit:
            counts[hit] = counts.get(hit, 0) + 1
    n = fuse_mlp_blocks(model)
    if n:
        counts["gelu_mlp"] = counts.get("gelu_mlp", 0) + n
    return counts
