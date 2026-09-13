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


class _FusedLinearGELU(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return ops.gelu_mlp(x, self.linear.weight, self.linear.bias)


def fuse_mlp_blocks(model: nn.Module) -> int:
    fused = 0
    for mod in model.modules():
        if not isinstance(mod, nn.Sequential):
            continue
        for i in range(len(mod) - 1):
            lin, act = mod[i], mod[i + 1]
            if (isinstance(lin, nn.Linear) and lin.bias is not None
                    and isinstance(act, nn.GELU)
                    and getattr(act, "approximate", "none") == "tanh"):
                mod[i] = _FusedLinearGELU(lin)
                mod[i + 1] = nn.Identity()
                fused += 1
    return fused
