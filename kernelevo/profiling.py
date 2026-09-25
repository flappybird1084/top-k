"""profile() (spec §4.3): warmup, torch.profiler over N steps, aggregate CUDA time
per kernelevo op region, keep shape histograms, extract inductor Triton seeds,
rank and threshold into targets.json."""

from __future__ import annotations

import inspect
import json
import os

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile

from kernelevo import bench, ops


def _op_time_us(prof) -> dict[str, float]:
    """CUDA time per kernelevo:: record_function region. Falls back to CPU time
    if this torch build reports zero device time for user annotations."""
    dev, cpu = {}, {}
    for evt in prof.key_averages():
        if not evt.key.startswith("kernelevo::"):
            continue
        name = evt.key.split("::", 1)[1]
        d = float(getattr(evt, "device_time_total", getattr(evt, "cuda_time_total", 0.0)) or 0.0)
        c = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
        dev[name] = dev.get(name, 0.0) + d
        cpu[name] = cpu.get(name, 0.0) + c
    if any(v > 0 for v in dev.values()):
        return dev
    print("[profile] WARNING: profiler reported no device time for op regions; "
          "ranking by CPU time instead")
    return cpu


def _count_flops_per_sample(adapter, model, batch, samples_per_batch: int):
    """Best-effort utilization metadata; gate timing must not depend on it."""
    from torch.utils.flop_counter import FlopCounterMode

    try:
        with FlopCounterMode(display=False) as fc:
            loss = adapter.loss_fn(model, batch)
            loss.backward()
        return fc.get_total_flops() / samples_per_batch
    except AssertionError as exc:
        # Some valid SDPA layouts execute and time correctly but PyTorch's FLOP
        # counter cannot classify their query/key/value shapes. Keep MFU unknown.
        if "sdpa_flop_count: query/key/value shapes are incompatible" not in str(exc):
            raise
        print("[profile] WARNING: SDPA FLOP count unavailable; MFU will be omitted")
        return None


def _extract_inductor_seed(op: ops.OpDef, argspec, device, out_path: str) -> str:
    """Compile the op's eager fn with inductor at the real shape and harvest the
    generated Triton source from PyCodeCache. Best-effort: falls back to the
    eager source so the seed prompt is never empty."""
    try:
        import torch._dynamo as dynamo
        from torch._inductor import codecache

        def known_files():
            mods = getattr(codecache.PyCodeCache, "modules", None)
            if mods is None:
                cache = getattr(codecache.PyCodeCache, "cache", {})
                mods = list(cache.values())
            return {getattr(m, "__file__", None) for m in mods} - {None}

        before = known_files()
        dynamo.reset()
        compiled = torch.compile(op.eager, dynamic=False)
        args = ops.make_inputs(op.name, argspec, device, seed=0)
        out = compiled(*args)
        if op.differentiable and isinstance(out, torch.Tensor) and out.requires_grad:
            out.backward(torch.randn_like(out))
        torch.cuda.synchronize()
        chunks = []
        for path in sorted(known_files() - before):
            try:
                text = open(path).read()
            except OSError:
                continue
            if "@triton" in text or "triton_heuristics" in text:
                chunks.append(f"# ---- inductor output: {os.path.basename(path)}\n{text}")
        if not chunks:
            raise RuntimeError("no triton code found in inductor output")
        src = "\n\n".join(chunks)
        kind = "inductor"
    except Exception as e:  # noqa: BLE001 — extraction is best-effort by design
        src = (f"# inductor seed extraction failed ({type(e).__name__}: {e}).\n"
               f"# Eager reference implementation for {op.name}:\n"
               + inspect.getsource(op.eager))
        kind = "eager_fallback"
    with open(out_path, "w") as f:
        f.write(src)
    return kind


def run_profile(adapter, cfg, info: dict, out_dir: str) -> dict:
    from kernelevo import patch
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    model = adapter.build_model().to(device)
    patch.auto_route(model)
    opt = bench.make_optimizer(model)
    step = bench.make_step(adapter, model, opt, iter(adapter.get_dataloader("train")))

    ops.reset_histograms()
    ops.set_profiling(True)
    try:
        for _ in range(cfg["profile_warmup"]):
            step()
        torch.cuda.synchronize()
        with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=True) as prof:
            for _ in range(cfg["profile_steps"]):
                step()
            torch.cuda.synchronize()
    finally:
        ops.set_profiling(False)

    op_times = _op_time_us(prof)
    step_ms = bench.time_steps(step, warmup=3, iters=10)
    step_us_total = step_ms * 1000.0 * 1  # per step; op_times cover profile_steps steps
    per_step = {k: v / cfg["profile_steps"] for k, v in op_times.items()}

    # flops_per_sample (spec §4.1.5) — counted here since the model is already up.
    torch.manual_seed(cfg["seed"])
    batch = next(iter(adapter.get_dataloader("train")))
    try:
        flops_per_sample = _count_flops_per_sample(
            adapter, model, batch, info["samples_per_batch"])
    finally:
        opt.zero_grad(set_to_none=True)

    seeds_dir = os.path.join(out_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)
    lineages = []
    ranked = sorted(per_step.items(), key=lambda kv: -kv[1])
    for name, t_us in ranked:
        pct = 100.0 * t_us / step_us_total
        if name not in cfg["allowed_ops"]:
            continue
        if pct < cfg["min_pct_step_time"]:
            continue
        shapes = ops.top_shapes(name, cfg["gate2_shapes"])
        if not shapes:
            continue
        seed_path = os.path.join(seeds_dir, f"{name}.py")
        seed_kind = _extract_inductor_seed(ops.REGISTRY[name], shapes[0], device, seed_path)
        lineages.append(dict(
            op=name, pct_step_time=round(pct, 2), per_step_us=round(t_us, 1),
            shapes=shapes,
            shape_hist={k: v for k, v in ops.REGISTRY[name].shape_hist.most_common(10)},
            seed_path=seed_path, seed_kind=seed_kind,
            signature=ops.REGISTRY[name].signature,
            differentiable=ops.REGISTRY[name].differentiable,
        ))

    targets = dict(
        step_time_ms=round(step_ms, 3),
        flops_per_sample=flops_per_sample,
        samples_per_batch=info["samples_per_batch"],
        lineages=lineages,
        all_op_pct={k: round(100.0 * v / step_us_total, 2) for k, v in ranked},
    )
    with open(os.path.join(out_dir, "targets.json"), "w") as f:
        json.dump(targets, f, indent=2)

    import gc
    del model, opt, prof, step
    gc.collect()
    torch.cuda.empty_cache()
    return targets
