"""Op registry and dispatch.

Adapters call these wrapper functions instead of raw torch ops. Dispatch goes
through a per-op `impl` slot, so the harness can swap a candidate kernel in for
gate 4 without any per-model code. During profiling, dispatch also records the
argspec (shapes/dtypes/scalars) of every call — that histogram is what gate 2
verifies against — and wraps calls in torch.profiler record_function regions so
step time can be attributed per op.

An argspec is a JSON-able list, one entry per positional arg:
  {"kind": "tensor", "shape": [...], "dtype": "torch.float32", "grad": bool}
  {"kind": "scalar", "value": 0.996}
  {"kind": "none"}
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch.profiler import record_function

_PROFILING = False


@dataclass
class OpDef:
    name: str
    eager: Callable                 # reference implementation; the oracle, never a candidate
    signature: str                  # human-readable, injected into subagent prompts
    differentiable: bool = True
    mutates: tuple[int, ...] = ()   # arg indices the op mutates in place
    make_inputs: Callable | None = None   # (argspec, device, seed) -> list of args
    perturb: Callable | None = None       # argspec -> unseen-shape argspec (gate 2)
    impl: Callable = None
    shape_hist: Counter = field(default_factory=Counter)

    def __post_init__(self):
        if self.impl is None:
            self.impl = self.eager


REGISTRY: dict[str, OpDef] = {}


def _register(op: OpDef):
    REGISTRY[op.name] = op
    return op


def set_profiling(on: bool):
    global _PROFILING
    _PROFILING = on


def reset_histograms():
    for op in REGISTRY.values():
        op.shape_hist = Counter()


def argspec_of(args) -> list[dict]:
    spec = []
    for a in args:
        if a is None:
            spec.append({"kind": "none"})
        elif isinstance(a, torch.Tensor):
            spec.append({"kind": "tensor", "shape": list(a.shape),
                         "dtype": str(a.dtype), "grad": bool(a.requires_grad)})
        else:
            spec.append({"kind": "scalar", "value": a})
    return spec


def _dispatch(name: str, *args):
    op = REGISTRY[name]
    if _PROFILING:
        op.shape_hist[json.dumps(argspec_of(args))] += 1
        with record_function(f"kernelevo::{name}"):
            return op.impl(*args)
    return op.impl(*args)


@contextmanager
def swapped(mapping: dict[str, Callable]):
    saved = {n: REGISTRY[n].impl for n in mapping}
    try:
        for n, fn in mapping.items():
            REGISTRY[n].impl = fn
        yield
    finally:
        for n, fn in saved.items():
            REGISTRY[n].impl = fn


def top_shapes(name: str, k: int) -> list[list[dict]]:
    hist = REGISTRY[name].shape_hist
    return [json.loads(key) for key, _ in hist.most_common(k)]


# ---------------------------------------------------------------- input builders

def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _make_tensor(entry: dict, device, g: torch.Generator, requires_grad: bool):
    dtype = getattr(torch, entry["dtype"].replace("torch.", ""))
    if dtype in (torch.int64, torch.int32):
        raise ValueError("int tensors need an op-specific make_inputs")
    t = torch.randn(entry["shape"], generator=g, dtype=torch.float32)
    t = t.to(device=device, dtype=dtype)
    if requires_grad:
        t.requires_grad_(True)
    return t


def generic_make_inputs(op: OpDef, argspec: list[dict], device, seed: int):
    g = _seeded(seed)
    args = []
    for entry in argspec:
        if entry["kind"] == "none":
            args.append(None)
        elif entry["kind"] == "scalar":
            args.append(entry["value"])
        else:
            grad = op.differentiable and entry.get("grad", False)
            args.append(_make_tensor(entry, device, g, grad))
    return args


def make_inputs(name: str, argspec: list[dict], device, seed: int):
    op = REGISTRY[name]
    if op.make_inputs is not None:
        return op.make_inputs(op, argspec, device, seed)
    return generic_make_inputs(op, argspec, device, seed)


def perturb(name: str, argspec: list[dict]) -> list[dict]:
    op = REGISTRY[name]
    return op.perturb(argspec)


# ---------------------------------------------------------------- op definitions

# 1. EMA target-encoder update (JEPA). In-place on `target`; not differentiable.
def _ema_eager(target: torch.Tensor, online: torch.Tensor, momentum: float):
    target.mul_(momentum).add_(online, alpha=1.0 - momentum)
    return target


def _ema_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    for entry in spec:
        if entry["kind"] == "tensor":
            entry["shape"] = [entry["shape"][0] + 3] + entry["shape"][1:]
            entry["grad"] = False
    return spec


_register(OpDef(
    name="ema_update",
    eager=_ema_eager,
    signature="kernel(target: Tensor, online: Tensor, momentum: float) -> Tensor  "
              "# in-place: target = momentum*target + (1-momentum)*online; returns target. "
              "Arbitrary shape (treat as flat); no autograd required.",
    differentiable=False,
    mutates=(0,),
    perturb=_ema_perturb,
))


# 2. Masked gather + positional-embed add (JEPA).
def _mga_eager(x: torch.Tensor, idx: torch.Tensor, pos: torch.Tensor):
    B, N, D = x.shape
    K = idx.shape[1]
    ix = idx.unsqueeze(-1).expand(B, K, D)
    return torch.gather(x, 1, ix) + torch.gather(pos.expand(B, N, D), 1, ix)


def _mga_make_inputs(op, argspec, device, seed):
    g = _seeded(seed)
    x_e, idx_e, pos_e = argspec
    x = _make_tensor(x_e, device, g, op.differentiable and x_e.get("grad", True))
    N = x_e["shape"][1]
    idx = torch.randint(0, N, idx_e["shape"], generator=g).to(device)
    pos = _make_tensor(pos_e, device, g, op.differentiable and pos_e.get("grad", True))
    return [x, idx, pos]


def _mga_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    x_e, idx_e, pos_e = spec
    x_e["shape"][0] += 1
    idx_e["shape"][0] += 1
    idx_e["shape"][1] = max(1, idx_e["shape"][1] // 2)
    return spec


_register(OpDef(
    name="masked_gather_add",
    eager=_mga_eager,
    signature="kernel(x: Tensor[B,N,D], idx: LongTensor[B,K], pos: Tensor[1,N,D]) -> Tensor[B,K,D]  "
              "# out[b,k,:] = x[b, idx[b,k], :] + pos[0, idx[b,k], :]. "
              "Must support autograd wrt x and pos (torch.autograd.Function with a backward).",
    differentiable=True,
    make_inputs=_mga_make_inputs,
    perturb=_mga_perturb,
))


# 3. LayerNorm over last dim (both models); gate 2 checks fwd AND bwd.
# The original functional is captured at import so kernelevo.patch can later
# monkeypatch F.layer_norm (routing arbitrary repo models through the registry)
# without the eager reference recursing into the patched version.
_F_LAYER_NORM = F.layer_norm


def _ln_eager(x: torch.Tensor, weight, bias, eps: float = 1e-5):
    return _F_LAYER_NORM(x, (x.shape[-1],), weight, bias, eps)


def _ln_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    spec[0]["shape"] = [spec[0]["shape"][0] + 1] + spec[0]["shape"][1:]
    return spec


_register(OpDef(
    name="layer_norm",
    eager=_ln_eager,
    signature="kernel(x: Tensor[..., D], weight: Tensor[D], bias: Tensor[D], eps: float) -> Tensor[..., D]  "
              "# LayerNorm over the last dim. Must support autograd wrt x, weight, bias "
              "(torch.autograd.Function; the backward may itself be Triton).",
    differentiable=True,
    perturb=_ln_perturb,
))


# 4. GELU-MLP epilogue: first MLP linear + GELU fused (both approximations).
def _gm_eager(x: torch.Tensor, w: torch.Tensor, b, approximate: str = "tanh"):
    return F.gelu(F.linear(x, w, b), approximate=approximate)


def _gm_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    spec[0]["shape"] = [spec[0]["shape"][0] + 1] + spec[0]["shape"][1:]
    return spec


_register(OpDef(
    name="gelu_mlp",
    eager=_gm_eager,
    signature="kernel(x: Tensor[..., K], w: Tensor[F, K], b: Tensor[F] | None, "
              "approximate: str) -> Tensor[..., F]  "
              "# gelu(x @ w.T + b, approximate=approximate) — nn.Linear weight layout; "
              "approximate is 'tanh' or 'none' (exact erf GELU), bias may be None. "
              "Must support autograd wrt x, w (and b when present).",
    differentiable=True,
    perturb=_gm_perturb,
))


# 5. RMSNorm over last dim (nanochat-style models). weight and eps may be None
# (parameter-free norm, default epsilon). Routed automatically by patch.install()
# wherever the model calls F.rms_norm.
_F_RMS_NORM = getattr(F, "rms_norm", None)


def _rms_eager(x: torch.Tensor, weight, eps):
    if _F_RMS_NORM is not None:
        return _F_RMS_NORM(x, (x.shape[-1],), weight, eps)
    e = 1e-6 if eps is None else eps
    out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + e)).to(x.dtype)
    return out if weight is None else out * weight


def _rms_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    spec[0]["shape"] = [spec[0]["shape"][0] + 1] + spec[0]["shape"][1:]
    return spec


_register(OpDef(
    name="rms_norm",
    eager=_rms_eager,
    signature="kernel(x: Tensor[..., D], weight: Tensor[D] | None, eps: float | None) -> Tensor[..., D]  "
              "# RMSNorm over the last dim: x * rsqrt(mean(x^2, -1) + eps) [* weight]. "
              "weight and eps may be None (parameter-free, default eps ~1e-6 per torch). "
              "Must support autograd wrt x (and weight when present) — "
              "torch.autograd.Function; the backward may itself be Triton.",
    differentiable=True,
    perturb=_rms_perturb,
))


# 6. Squared-ReLU MLP epilogue: first MLP linear + relu(x)^2 fused (nanochat-style).
def _r2_eager(x: torch.Tensor, w: torch.Tensor, b):
    return F.relu(F.linear(x, w, b)).square()


_register(OpDef(
    name="relu2_mlp",
    eager=_r2_eager,
    signature="kernel(x: Tensor[..., K], w: Tensor[F, K], b: Tensor[F] | None) -> Tensor[..., F]  "
              "# relu(x @ w.T + b) ** 2 — nn.Linear weight layout, bias may be None. "
              "Must support autograd wrt x, w (and b when present).",
    differentiable=True,
    perturb=_gm_perturb,
))


# 7. SwiGLU epilogue: dual GEMM + silu-gate fused (modern-lm / Llama-style MLPs).
def _sw_eager(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
    return F.silu(F.linear(x, w1)) * F.linear(x, w2)


_register(OpDef(
    name="swiglu_mlp",
    eager=_sw_eager,
    signature="kernel(x: Tensor[..., K], w1: Tensor[F, K], w2: Tensor[F, K]) -> Tensor[..., F]  "
              "# silu(x @ w1.T) * (x @ w2.T) — nn.Linear weight layout, no biases. "
              "Must support autograd wrt x, w1, w2.",
    differentiable=True,
    perturb=_gm_perturb,
))


# 8. GeGLU epilogue: dual GEMM + gelu-gate (Gemma/T5-style gated MLPs).
def _gg_eager(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
              approximate: str = "none"):
    return F.gelu(F.linear(x, w1), approximate=approximate) * F.linear(x, w2)


_register(OpDef(
    name="geglu_mlp",
    eager=_gg_eager,
    signature="kernel(x: Tensor[..., K], w1: Tensor[F, K], w2: Tensor[F, K], "
              "approximate: str) -> Tensor[..., F]  "
              "# gelu(x @ w1.T, approximate) * (x @ w2.T) — nn.Linear layout, no "
              "biases. Must support autograd wrt x, w1, w2.",
    differentiable=True,
    perturb=_gm_perturb,
))


# 9. Cross-entropy over the vocab (every LM's last op).
_F_CROSS_ENTROPY = F.cross_entropy  # captured so the oracle survives patching


def _ce_eager(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    return _F_CROSS_ENTROPY(logits, targets, ignore_index=ignore_index)


def _ce_make_inputs(op, argspec, device, seed):
    g = _seeded(seed)
    lg_e, tg_e, ii = argspec[0], argspec[1], argspec[2]
    logits = _make_tensor(lg_e, device, g, lg_e.get("grad", True))
    V = lg_e["shape"][-1]
    targets = torch.randint(0, V, tg_e["shape"], generator=g).to(device)
    return [logits, targets, ii["value"] if ii["kind"] == "scalar" else -100]


def _ce_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    spec[0]["shape"][0] += 8
    spec[1]["shape"][0] += 8
    return spec


_register(OpDef(
    name="cross_entropy",
    eager=_ce_eager,
    signature="kernel(logits: Tensor[N, V], targets: LongTensor[N], ignore_index: int) "
              "-> Tensor[] (scalar)  # mean cross-entropy, entries with "
              "targets==ignore_index excluded. Must support autograd wrt logits.",
    differentiable=True,
    make_inputs=_ce_make_inputs,
    perturb=_ce_perturb,
))


# 10. LM-head projection + cross-entropy as ONE boundary (loss-side fusion the
# compiler never crosses: candidates may avoid materializing [N, V] logits).
def _lce_eager(x: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
               ignore_index: int = -100):
    return _F_CROSS_ENTROPY(F.linear(x, weight), targets, ignore_index=ignore_index)


def _lce_make_inputs(op, argspec, device, seed):
    g = _seeded(seed)
    x_e, w_e, t_e, ii = argspec
    x = _make_tensor(x_e, device, g, x_e.get("grad", True))
    w = _make_tensor(w_e, device, g, w_e.get("grad", True))
    V = w_e["shape"][0]
    targets = torch.randint(0, V, t_e["shape"], generator=g).to(device)
    return [x, w, targets, ii["value"] if ii["kind"] == "scalar" else -100]


def _lce_perturb(argspec):
    spec = json.loads(json.dumps(argspec))
    spec[0]["shape"][0] += 8
    spec[2]["shape"][0] += 8
    return spec


_register(OpDef(
    name="linear_cross_entropy",
    eager=_lce_eager,
    signature="kernel(x: Tensor[N, D], weight: Tensor[V, D], targets: LongTensor[N], "
              "ignore_index: int) -> Tensor[] (scalar)  "
              "# mean cross_entropy(x @ weight.T, targets), entries with "
              "targets==ignore_index excluded. nn.Linear weight layout, no bias. "
              "Must support autograd wrt x and weight.",
    differentiable=True,
    make_inputs=_lce_make_inputs,
    perturb=_lce_perturb,
))


# ---------------------------------------------------------------- adapter-facing API

def ema_update(target, online, momentum):
    return _dispatch("ema_update", target, online, momentum)


def masked_gather_add(x, idx, pos):
    return _dispatch("masked_gather_add", x, idx, pos)


def layer_norm(x, weight, bias, eps=1e-5):
    return _dispatch("layer_norm", x, weight, bias, eps)


def gelu_mlp(x, w, b, approximate="tanh"):
    return _dispatch("gelu_mlp", x, w, b, approximate)


def rms_norm(x, weight=None, eps=None):
    return _dispatch("rms_norm", x, weight, eps)


def relu2_mlp(x, w, b):
    return _dispatch("relu2_mlp", x, w, b)


def swiglu_mlp(x, w1, w2):
    return _dispatch("swiglu_mlp", x, w1, w2)


def geglu_mlp(x, w1, w2, approximate="none"):
    return _dispatch("geglu_mlp", x, w1, w2, approximate)


def cross_entropy(logits, targets, ignore_index=-100):
    return _dispatch("cross_entropy", logits, targets, ignore_index)


def linear_cross_entropy(x, weight, targets, ignore_index=-100):
    return _dispatch("linear_cross_entropy", x, weight, targets, ignore_index)
