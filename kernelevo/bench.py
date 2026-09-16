"""CUDA-event timing helpers shared by calibrate() and the verify worker.
All comparisons re-measure the incumbent in-session; nothing here compares
against stored numbers."""

from __future__ import annotations

import statistics

import torch


def _median_event_time(run, warmup: int, iters: int, between=None) -> float:
    """Median wall time of run() in ms via CUDA events. `between` (if given)
    runs after each timed iteration, outside the event window — its cost never
    enters the measurement."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        if between is not None:
            between()
            torch.cuda.synchronize()
    return statistics.median(times)


def time_op(fn, args, grad_inputs, warmup: int, iters: int) -> float:
    """Isolation latency in µs. For differentiable ops this times fwd+bwd —
    LayerNorm-backward-style lineages are meaningless to time forward-only."""
    go = None
    floats = [a for a in args
              if isinstance(a, torch.Tensor) and a.is_floating_point()]

    def run():
        nonlocal go
        out = fn(*args)
        if grad_inputs:
            if go is None:
                go = torch.randn_like(out)
            torch.autograd.grad(out, grad_inputs, grad_outputs=go, retain_graph=False,
                                allow_unused=True)

    def perturb():
        # Anti-memoization (gate 3): rotate input content between timed
        # iterations so a cache keyed on input identity OR content cannot
        # serve replays — gate 2's fresh-seeds defense only covers the naive
        # replay-always cheat. Runs outside the event window (zero timing
        # cost) and identically for incumbent and candidate, so the A/B
        # comparison stays fair.
        with torch.no_grad():
            for t in floats:
                t.add_(torch.randn_like(t), alpha=1e-6)
            if go is not None:
                go.add_(torch.randn_like(go), alpha=1e-6)

    return _median_event_time(run, warmup, iters, between=perturb) * 1000.0


def time_steps(step, warmup: int, iters: int) -> float:
    """Median full-training-step time in ms."""
    return _median_event_time(step, warmup, iters)


def make_step(adapter, model, opt, data_iter, objective="train"):
    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            # Finite loaders (common in generated adapters) cycle: a fresh
            # loader yields the same batches in the same order, so timing
            # stays deterministic across measurements.
            data_iter = iter(adapter.get_dataloader("train"))
            return next(data_iter)

    if objective == "inference":
        def step():
            with torch.no_grad():
                return adapter.loss_fn(model, next_batch())
        return step

    def step():
        batch = next_batch()
        opt.zero_grad(set_to_none=True)
        loss = adapter.loss_fn(model, batch)
        loss.backward()
        opt.step()
        if hasattr(model, "post_optimizer_step"):
            model.post_optimizer_step()
        return loss

    return step


def make_optimizer(model):
    # Harness-owned, fixed hyperparameters (spec §3.1).
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)


def samples_per_batch(batch) -> int:
    if isinstance(batch, (tuple, list)):
        for t in batch:
            if isinstance(t, torch.Tensor):
                return t.shape[0]
    if isinstance(batch, dict):
        for t in batch.values():
            if isinstance(t, torch.Tensor):
                return t.shape[0]
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    raise ValueError("cannot infer samples per batch")
