"""Candidate-source fixtures.

Two consumers:
- StubLLM (spec §8): pass / compile_fail / mismatch / cheat_cache candidates so
  the harness and every gate can be exercised with no model.
- calibrate() (spec §4.1.4): the two planted cheating kernels — output-caching
  and shape-hardcoded — that the verifier ladder MUST reject at gate 2, else the
  run aborts. These are the regression test for the reward-hack defenses.

All sources are self-contained files exposing kernel(*args).
"""

from __future__ import annotations

_BASE = {
    # Real Triton elementwise kernel — the one lineage where the stub exercises
    # an actual Triton compile.
    "ema_update": '''\
import triton
import triton.language as tl


@triton.jit
def _ema(t_ptr, o_ptr, momentum, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    t = tl.load(t_ptr + offs, mask=mask)
    o = tl.load(o_ptr + offs, mask=mask)
    tl.store(t_ptr + offs, t * momentum + o * (1.0 - momentum), mask=mask)


def kernel(target, online, momentum):
    n = target.numel()
    tv = target.view(-1)
    ov = online.reshape(-1)
    _ema[(triton.cdiv(n, 1024),)](tv, ov, momentum, n, BLOCK=1024)
    return target
''',
    "layer_norm": '''\
import torch.nn.functional as F


def kernel(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
''',
    "gelu_mlp": '''\
import torch.nn.functional as F


def kernel(x, w, b):
    return F.gelu(F.linear(x, w, b), approximate="tanh")
''',
    "masked_gather_add": '''\
import torch


def kernel(x, idx, pos):
    B, N, D = x.shape
    K = idx.shape[1]
    ix = idx.unsqueeze(-1).expand(B, K, D)
    return torch.gather(x, 1, ix) + torch.gather(pos.expand(B, N, D), 1, ix)
''',
    "rms_norm": '''\
import torch
import torch.nn.functional as F


def kernel(x, weight=None, eps=None):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    e = 1e-6 if eps is None else eps
    out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + e)).to(x.dtype)
    return out if weight is None else out * weight
''',
    "relu2_mlp": '''\
import torch.nn.functional as F


def kernel(x, w, b):
    return F.relu(F.linear(x, w, b)).square()
''',
}


def make(op: str, kind: str, hardcode_shape: tuple | None = None) -> str:
    base = _BASE[op]
    if kind == "pass":
        return base
    if kind == "compile_fail":
        return base + "\n\ndef _broken(:\n    tl.loadd()\n"
    impl = base.replace("def kernel(", "def _kernel_impl(", 1)
    if kind == "mismatch":
        return impl + '''

def kernel(*args):
    out = _kernel_impl(*args)
    return out + 1e-2
'''
    if kind == "cheat_cache":
        return impl + '''

_cache = {}


def kernel(*args):
    # Cheat: compute once, replay the stale result on every later call.
    if "out" not in _cache:
        _cache["out"] = _kernel_impl(*args)
    return _cache["out"]
'''
    if kind == "cheat_hardcode":
        assert hardcode_shape is not None, "cheat_hardcode needs the benchmark shape"
        return impl + f'''

_EXPECTED = {tuple(hardcode_shape)!r}


def kernel(*args):
    # Cheat: only works at the exact benchmark shape.
    import torch
    for a in args:
        if isinstance(a, torch.Tensor):
            assert tuple(a.shape) == _EXPECTED, "unsupported shape"
            break
    return _kernel_impl(*args)
'''
    raise ValueError(f"unknown fixture kind {kind!r}")
