"""ingest() (spec §4.2): import adapter, build model, pull one batch, run loss_fn
once, assert scalar/finite/requires-grad. Records param count and batch identity."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import torch

from kernelevo import bench


def load_adapter(path_or_module: str):
    """Bundled adapters load as modules ('adapters.jepa' / 'adapters/jepa.py');
    generated adapters (repo pipeline) load from an arbitrary file path, which
    is passed through unchanged so worker subprocesses can reload it."""
    if path_or_module.endswith(".py") and os.path.exists(path_or_module):
        p = os.path.abspath(path_or_module)
        spec = importlib.util.spec_from_file_location("kevo_adapter", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["kevo_adapter"] = mod
        spec.loader.exec_module(mod)
        name = p
    else:
        name = path_or_module
        if name.endswith(".py"):
            name = name[:-3]
        name = name.replace("/", ".").replace("\\", ".").strip(".")
        mod = importlib.import_module(name)
    for fn in ("build_model", "get_dataloader", "loss_fn"):
        if not hasattr(mod, fn):
            raise SystemExit(f"adapter {path_or_module} missing contract function {fn}()")
    return mod, name


def ingest(adapter, cfg) -> dict:
    from kernelevo import patch
    torch.manual_seed(cfg["seed"])
    model = adapter.build_model().to(cfg["device"])
    routed = patch.auto_route(model)
    if routed:
        print(f"[ingest] auto-routed onto registry: {routed}")
    batch = next(iter(adapter.get_dataloader("train")))
    loss = adapter.loss_fn(model, batch)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "loss_fn must return a scalar tensor"
    assert torch.isfinite(loss).item(), "loss is not finite on the first batch"
    assert loss.requires_grad, "loss does not require grad"
    n_params = sum(p.numel() for p in model.parameters())
    info = dict(
        n_params=n_params,
        samples_per_batch=bench.samples_per_batch(batch),
        dtype=str(next(model.parameters()).dtype),
        loss0=float(loss.detach()),
    )
    del model, batch, loss
    torch.cuda.empty_cache()
    return info
