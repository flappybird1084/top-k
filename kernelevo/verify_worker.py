"""Verifier ladder worker (spec §5): gates 2-4 for one candidate, in its own
process so a hung kernel is killed by the orchestrator's timeout and the CUDA
context dies with the process.

Assumes gate 1 (compile) already passed. Prints one line 'KEVO_RESULT {json}'
on stdout. gate_reached = highest gate PASSED (1 means compiled but wrong).

Usage: python -m kernelevo.verify_worker <job.json>
"""

from __future__ import annotations

import json
import statistics
import sys
import traceback

import torch

from kernelevo import bench, ops
from kernelevo.compile_worker import load_kernel
from kernelevo.ingest import load_adapter


def _tol_for(dtype: torch.dtype, tol: dict) -> tuple[float, float]:
    key = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "fp32")
    return tuple(tol[key])


def _compare(out_c, out_e, tol, label):
    if not isinstance(out_c, torch.Tensor):
        return f"{label}: expected a Tensor, got {type(out_c).__name__}"
    if out_c.shape != out_e.shape:
        return f"{label}: shape {tuple(out_c.shape)} != expected {tuple(out_e.shape)}"
    if out_c.dtype != out_e.dtype:
        return f"{label}: dtype {out_c.dtype} != expected {out_e.dtype}"
    rtol, atol = _tol_for(out_e.dtype, tol)
    c, e = out_c.detach().float(), out_e.detach().float()
    diff = (c - e).abs()
    thresh = atol + rtol * e.abs()
    bad = diff > thresh
    if bad.any():
        i = int(diff.argmax())
        idx = (() if e.dim() == 0 else
               tuple(int(x) for x in torch.unravel_index(torch.tensor(i), e.shape)))
        max_rel = float((diff / (e.abs() + 1e-12)).max())
        return (f"{label}: mismatch — max_abs_err={float(diff.max()):.3e} "
                f"max_rel_err={max_rel:.3e} at index {idx} "
                f"(candidate={float(c.flatten()[i]):.6e} expected={float(e.flatten()[i]):.6e}), "
                f"{int(bad.sum())}/{e.numel()} elements out of tolerance "
                f"(rtol={rtol}, atol={atol})")
    return None


def gate2(op: ops.OpDef, kernel, shapes, tol, trials, device, base_seed):
    """Outputs and gradients vs eager on top shapes + one unseen shape, fresh
    seeded inputs every trial (defeats output-caching cheats); the unseen shape
    defeats hardcoded sizes."""
    cases = [(s, False) for s in shapes] + [(op.perturb(shapes[0]), True)]
    for si, (argspec, unseen) in enumerate(cases):
        for trial in range(trials):
            seed = base_seed + si * 1000 + trial
            args_e = ops.make_inputs(op.name, argspec, device, seed)
            args_c = ops.make_inputs(op.name, argspec, device, seed)
            snap = [a.detach().clone() if isinstance(a, torch.Tensor) else None for a in args_c]
            label = f"shape#{si}{' (unseen)' if unseen else ''} trial {trial}"
            out_e = op.eager(*args_e)
            try:
                out_c = kernel(*args_c)
            except Exception as e:  # noqa: BLE001 — candidate code, anything can happen
                tb = traceback.format_exc().strip().splitlines()
                return f"{label}: candidate raised {type(e).__name__}: {e}\n" + "\n".join(tb[-4:])
            err = _compare(out_c, out_e, tol, f"{label} output")
            if err:
                return err
            for i, s in enumerate(snap):
                if s is not None and i not in op.mutates and not torch.equal(args_c[i], s):
                    return f"{label}: candidate mutated input arg {i}, which the op must not modify"
            if op.differentiable:
                grads_of_e = [a for a in args_e if isinstance(a, torch.Tensor) and a.requires_grad]
                grads_of_c = [a for a in args_c if isinstance(a, torch.Tensor) and a.requires_grad]
                if grads_of_e:
                    if not (isinstance(out_c, torch.Tensor) and out_c.requires_grad):
                        return (f"{label}: candidate output does not require grad — there is no "
                                f"autograd path. Wrap the Triton kernel in a "
                                f"torch.autograd.Function with a backward (which may itself be "
                                f"Triton), or compose with differentiable torch ops.")
                    torch.manual_seed(seed)
                    go = torch.randn_like(out_e.detach())
                    ge = torch.autograd.grad(out_e, grads_of_e, grad_outputs=go, allow_unused=True)
                    try:
                        gc = torch.autograd.grad(out_c, grads_of_c, grad_outputs=go.clone(),
                                                 allow_unused=True)
                    except Exception as e:  # noqa: BLE001
                        return f"{label}: backward raised {type(e).__name__}: {e}"
                    for gi, (a, b) in enumerate(zip(gc, ge)):
                        if b is None:
                            continue
                        if a is None:
                            return f"{label}: grad #{gi} is None (input not connected to output)"
                        err = _compare(a, b, tol, f"{label} grad #{gi}")
                        if err:
                            return err
    return None


def build_incumbent_fn(op_name: str, entry: dict):
    op = ops.REGISTRY[op_name]
    if entry["kind"] == "candidate":
        return load_kernel(entry["path"])
    if entry["kind"] == "inductor":
        return torch.compile(op.eager, dynamic=False)
    return op.eager


def gate3(op, kernel, incumbent_fn, argspec, g3, device, seed):
    def timed(fn):
        args = ops.make_inputs(op.name, argspec, device, seed)
        grad_inputs = ([a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
                       if op.differentiable else [])
        return bench.time_op(fn, args, grad_inputs, g3["warmup"], g3["iters"])

    inc_us = timed(incumbent_fn)   # incumbent re-measured in-session, right before
    cand_us = timed(kernel)
    ok = cand_us < inc_us * (1.0 - g3["margin"])
    return ok, cand_us, inc_us


def measure_in_model(adapter, impl_map, g4, device, seed):
    from kernelevo import patch
    torch.manual_seed(seed)
    model = adapter.build_model().to(device)
    patch.auto_route(model)
    opt = bench.make_optimizer(model)
    step = bench.make_step(adapter, model, opt, iter(adapter.get_dataloader("train")))
    with ops.swapped(impl_map):
        ms = bench.time_steps(step, g4["warmup"], g4["steps"])
    del model, opt
    torch.cuda.empty_cache()
    return ms


def main():
    job = json.load(open(sys.argv[1]))
    from kernelevo import patch
    patch.install()
    device, seed = job["device"], job["seed"]
    op = ops.REGISTRY[job["op"]]
    res = dict(gate_reached=1, correct_ok=False, accepted=False, failure_note=None,
               latency_us=None, incumbent_latency_us=None,
               step_time_ms=None, incumbent_step_time_ms=None,
               samples_per_s=None, mfu=None)

    kernel = load_kernel(job["candidate_path"])

    err = gate2(op, kernel, job["shapes"], job["tol"], job["gate2_trials"], device, seed)
    if err:
        res["failure_note"] = err
        print("KEVO_RESULT " + json.dumps(res))
        return
    res.update(gate_reached=2, correct_ok=True)
    if job["upto"] <= 2:
        print("KEVO_RESULT " + json.dumps(res))
        return

    incumbent_fn = build_incumbent_fn(job["op"], job["incumbents"][job["op"]])
    ok, cand_us, inc_us = gate3(op, kernel, incumbent_fn, job["shapes"][0],
                                job["gate3"], device, seed)
    res.update(latency_us=cand_us, incumbent_latency_us=inc_us)
    if not ok:
        res["failure_note"] = (f"gate 3: not faster in isolation — candidate {cand_us:.1f}us "
                               f"vs incumbent {inc_us:.1f}us (margin {job['gate3']['margin']:.0%})")
        print("KEVO_RESULT " + json.dumps(res))
        return
    res["gate_reached"] = 3
    if job["upto"] <= 3:
        print("KEVO_RESULT " + json.dumps(res))
        return

    adapter, _ = load_adapter(job["adapter"])
    incumbent_map = {name: build_incumbent_fn(name, entry)
                     for name, entry in job["incumbents"].items()}
    candidate_map = {**incumbent_map, job["op"]: kernel}
    # Interleaved A/B: alternate incumbent/candidate measurements so thermal
    # drift or background load hits both sides, and take medians — a single
    # one-shot pair made the objective-producing gate the least rigorous one.
    reps = max(1, int(job["gate4"].get("reps", 3)))
    inc_runs, cand_runs = [], []
    for _ in range(reps):
        inc_runs.append(measure_in_model(adapter, incumbent_map, job["gate4"], device, seed))
        cand_runs.append(measure_in_model(adapter, candidate_map, job["gate4"], device, seed))
    inc_ms = statistics.median(inc_runs)
    cand_ms = statistics.median(cand_runs)
    res.update(step_time_ms=cand_ms, incumbent_step_time_ms=inc_ms)
    sps = job["samples_per_batch"] / (cand_ms / 1000.0)
    res["samples_per_s"] = sps
    if job.get("peak_flops"):
        res["mfu"] = job["flops_per_sample"] * sps / job["peak_flops"]
    if cand_ms < inc_ms * (1.0 - job["gate4"]["margin"]):
        res.update(gate_reached=4, accepted=True)
    else:
        res["failure_note"] = (f"gate 4: isolation win did not translate in-model — "
                               f"step {cand_ms:.2f}ms vs incumbent {inc_ms:.2f}ms "
                               f"(margin {job['gate4']['margin']:.0%})")
    print("KEVO_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
