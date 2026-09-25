"""ingest() (spec §4.2): import adapter, build model, pull one batch, run loss_fn
once, assert scalar/finite/requires-grad. Records param count and batch identity."""

from __future__ import annotations

import importlib
import importlib.util
from functools import wraps
from itertools import islice
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
    if os.path.basename(path_or_module) == "adapter.py":
        # Generated repo adapters frequently yield CPU batches while the
        # harness has already moved their model to CUDA. Apply the same device
        # transfer on every use of the adapter, not just the ingest check.
        from torch.utils._pytree import tree_map
        original_loss = mod.loss_fn

        @wraps(original_loss)
        def loss_with_device(model, batch):
            device = next(model.parameters()).device
            moved = tree_map(lambda value: value.to(device) if isinstance(value, torch.Tensor)
                             else value, batch)
            return original_loss(model, moved)

        mod.loss_fn = loss_with_device
    return mod, name


def check_training_signal(loss: torch.Tensor, model: torch.nn.Module) -> None:
    """Reject losses that do not train any parameter of the claimed model."""
    if loss.grad_fn is None:
        raise SystemExit("loss is detached from the model; no parameter gradient")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise SystemExit("model has no trainable parameters")
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    connected = [grad for grad in gradients if grad is not None]
    if not connected:
        raise SystemExit("loss is not connected to a trainable model parameter")
    if not all(bool(torch.isfinite(grad).all()) for grad in connected):
        raise SystemExit("loss produced a non-finite model gradient")
    if not any(bool(torch.count_nonzero(grad)) for grad in connected):
        raise SystemExit("loss produced only zero model gradients")


def ingest(adapter, cfg) -> dict:
    from kernelevo import patch
    torch.manual_seed(cfg["seed"])
    model = adapter.build_model().to(cfg["device"])
    batch = next(iter(adapter.get_dataloader("train")))
    # Routing-fidelity reference: loss on the UNROUTED model, same weights and
    # batch. auto_route replaces module forwards one-way; without this check a
    # misroute silently changes the model's math and no gate can see it
    # (incumbent and candidate both run on the changed model). Grad stays
    # ENABLED (adapters legitimately assert loss.requires_grad in loss_fn);
    # detach immediately, never backward.
    torch.manual_seed(cfg["seed"])  # identical dropout masks for both calls
    loss_unrouted = float(adapter.loss_fn(model, batch).detach())
    routed = patch.auto_route(model) if cfg.get("mode") != "recipe" else {}
    if routed:
        print(f"[ingest] auto-routed onto registry: {routed}")
        torch.manual_seed(cfg["seed"])
        loss_routed = float(adapter.loss_fn(model, batch).detach())
        drift = abs(loss_routed - loss_unrouted)
        tol = max(2e-2 * abs(loss_unrouted), 1e-3)
        if not (drift <= tol):
            raise SystemExit(
                f"[ingest] ROUTING FIDELITY FAILURE: loss {loss_unrouted:.6f} "
                f"(unrouted) vs {loss_routed:.6f} (routed), |Δ|={drift:.3g} > "
                f"tol {tol:.3g} — auto_route changed the model's math; aborting")
        print(f"[ingest] routing fidelity: |Δloss|={drift:.3g} (tol {tol:.3g}) ok")
    loss = adapter.loss_fn(model, batch)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "loss_fn must return a scalar tensor"
    assert torch.isfinite(loss).item(), "loss is not finite on the first batch"
    assert loss.requires_grad, "loss does not require grad"
    check_training_signal(loss, model)
    if cfg.get("mode") == "recipe":
        # Recipe training is scored on held-out loss. Check that path before
        # accepting the adapter, so a broken val loader gets repair feedback.
        was_training = model.training
        model.eval()
        val_losses = []
        for val_batch in islice(adapter.get_dataloader("val"),
                                cfg.get("recipe", {}).get("eval_batches", 8)):
            val_loss = adapter.loss_fn(model, val_batch)
            if not isinstance(val_loss, torch.Tensor) or val_loss.dim() != 0:
                raise SystemExit("recipe validation loss must be a scalar tensor")
            if not bool(torch.isfinite(val_loss).item()):
                raise SystemExit("recipe validation loss is not finite")
            val_losses.append(float(val_loss.detach()))
        if not val_losses:
            raise SystemExit("recipe validation loader yielded no batches")
        model.train(was_training)
        info_val_loss = sum(val_losses) / len(val_losses)
    else:
        info_val_loss = None
    n_params = sum(p.numel() for p in model.parameters())
    info = dict(
        n_params=n_params,
        samples_per_batch=bench.samples_per_batch(batch),
        dtype=str(next(model.parameters()).dtype),
        loss0=float(loss.detach()),
    )
    if info_val_loss is not None:
        info["val_loss0"] = info_val_loss
    del model, batch, loss
    torch.cuda.empty_cache()
    return info
