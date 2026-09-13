"""Gate 1 worker: import the candidate file and run its kernel once on the real
top shape (triggers Triton JIT). Exit 0 on success; compiler/runtime message on
stderr is fed back to the subagent verbatim (spec §4.5 repair table).

Usage: python -m kernelevo.compile_worker <job.json>
job = {candidate_path, op, argspec, device, seed}
"""

import importlib.util
import json
import sys

import torch


def load_kernel(path: str):
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel"):
        raise AttributeError("candidate file must expose kernel(*args)")
    return mod.kernel


def main():
    job = json.load(open(sys.argv[1]))
    from kernelevo import ops  # after torch, needs repo root on sys.path

    torch.manual_seed(job.get("seed", 0))
    kernel = load_kernel(job["candidate_path"])
    args = ops.make_inputs(job["op"], job["argspec"], job["device"], seed=job.get("seed", 0))
    kernel(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
