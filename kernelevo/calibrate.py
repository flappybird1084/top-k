"""calibrate() (spec §4.1): once at startup, after profiling (it needs the real
shapes), before generation 1.

1. GPU identity → peak FLOPs (lookup, config override).
2. Noise floor: re-benchmark one inductor incumbent N times across a span;
   spread raises the gate-3/gate-4 acceptance margins (never lowers).
3. Numerical floor: eager vs inductor per target op; tolerance raised to 2x the
   observed floor if larger than the defaults.
4. Verifier self-test: the two planted cheats must be rejected at gate 2 or the
   run aborts — the regression test for the reward-hack defenses.
5. flops_per_sample comes from profiling (FlopCounterMode) and is echoed here.
"""

from __future__ import annotations

import gc
import os
import statistics
import time

import torch


def _release_gpu():
    """Big models leave tens of GB reserved between phases; worker subprocesses
    share the GPU and starve unless the orchestrator returns memory eagerly."""
    gc.collect()
    torch.cuda.empty_cache()

import config as cfgmod
from kernelevo import bench, fixtures, ops
from kernelevo.obs import weave_attributes


def _spread(vals: list[float]) -> float:
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else 0.0


def _dtype_key(dtype: torch.dtype) -> str:
    return {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "fp32")


def calibrate(cfg: dict, runner, targets: dict, out_dir: str) -> dict:
    device = cfg["device"]
    top = targets["lineages"][0]
    op = ops.REGISTRY[top["op"]]

    # 1. GPU identity
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    peak = cfgmod.peak_flops_for(gpu_name, cfg)
    cfg["peak_flops"] = peak
    if peak is None:
        print(f"[calibrate] no peak-FLOPs entry for {gpu_name!r}; MFU will be null "
              f"(set peak_flops_override in config)")

    # 2. Noise floor — inductor incumbent of the top lineage, gate-3 style,
    # repeated across a span; plus a short in-model spread for the gate-4 margin.
    compiled = torch.compile(op.eager, dynamic=False)
    argspec = top["shapes"][0]
    reps, span = cfg["calib_reps"], cfg["calib_span_s"]
    lat_reps = []
    for i in range(reps):
        args = ops.make_inputs(op.name, argspec, device, cfg["seed"])
        grad_inputs = ([a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
                       if op.differentiable else [])
        lat_reps.append(bench.time_op(compiled, args, grad_inputs,
                                      cfg["gate3_warmup"], cfg["gate3_iters"]))
        if i < reps - 1:
            time.sleep(span / max(1, reps - 1))
    del compiled
    _release_gpu()
    lat_spread = _spread(lat_reps)
    gate3_margin = max(cfg["gate3_margin"], lat_spread)
    cfg["gate3_margin"] = gate3_margin
    # gate-4 spread: reuse the profiled step time by re-measuring is expensive;
    # scale conservatively from the isolation spread instead, floor at default.
    gate4_margin = max(cfg["gate4_margin"], lat_spread / 2)
    cfg["gate4_margin"] = gate4_margin

    # 3. Numerical floor — eager vs inductor per lineage, observed shapes.
    floors = {}
    tol = {k: list(v) for k, v in cfg["tol"].items()}
    for lin in targets["lineages"]:
        o = ops.REGISTRY[lin["op"]]
        comp = torch.compile(o.eager, dynamic=False)
        for argspec in lin["shapes"]:
            with torch.no_grad():  # floors need outputs only; grads would hold
                a1 = ops.make_inputs(o.name, argspec, device, cfg["seed"])
                a2 = ops.make_inputs(o.name, argspec, device, cfg["seed"])
                out_e, out_c = o.eager(*a1), comp(*a2)
                diff = (out_c.float() - out_e.float()).abs()
                max_abs = float(diff.max())
                max_rel = float((diff / (out_e.float().abs() + 1e-12)).max())
            key = _dtype_key(out_e.dtype)
            floors[f"{lin['op']}/{key}"] = dict(max_abs=max_abs, max_rel=max_rel)
            tol[key][0] = max(tol[key][0], 2 * max_rel) if max_rel < 1 else tol[key][0]
            tol[key][1] = max(tol[key][1], 2 * max_abs)
            del a1, a2, out_e, out_c, diff
        del comp
    cfg["tol"] = tol
    _release_gpu()
    print(f"[calibrate] orchestrator GPU memory before self-test: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f}GB allocated, "
          f"{torch.cuda.memory_reserved() / 2**30:.2f}GB reserved")

    # 4. Verifier self-test — both planted cheats must die at gate 2.
    cheat_dir = os.path.join(out_dir, "candidates")
    os.makedirs(cheat_dir, exist_ok=True)
    first_tensor_shape = next(e["shape"] for e in top["shapes"][0] if e["kind"] == "tensor")
    cheat_results = {}
    with weave_attributes({"planted_cheat": True}):
        for kind in ("cheat_cache", "cheat_hardcode"):
            src = fixtures.make(top["op"], kind,
                                hardcode_shape=first_tensor_shape if "hardcode" in kind else None)
            path = os.path.join(cheat_dir, f"calib_{kind}.py")
            with open(path, "w") as f:
                f.write(src)
            ok, msg = runner.compile(path, top["op"])
            if not ok:
                raise SystemExit(f"[calibrate] planted cheat {kind} failed to even compile "
                                 f"({msg[:200]}); self-test is broken — aborting")
            v = runner.verify(path, top["op"], upto=2, incumbents={})
            cheat_results[kind] = dict(path=path, correct_ok=v.get("correct_ok"),
                                       failure_note=v.get("failure_note"))
            if v.get("correct_ok"):
                raise SystemExit(f"[calibrate] VERIFIER SELF-TEST FAILED: planted cheat "
                                 f"{kind} passed gate 2. Aborting run (spec §4.1.4).")
    print(f"[calibrate] both planted cheats rejected at gate 2 ✓")

    return dict(
        gpu_name=gpu_name,
        gpu_memory_gb=round(props.total_memory / 2**30, 1),
        peak_flops=peak,
        noise_reps_us=[round(v, 1) for v in lat_reps],
        noise_spread=round(lat_spread, 4),
        gate3_margin=gate3_margin,
        gate4_margin=gate4_margin,
        tol=tol,
        numerical_floors=floors,
        cheats_rejected=True,
        cheat_results=cheat_results,
        flops_per_sample=targets["flops_per_sample"],
    )
