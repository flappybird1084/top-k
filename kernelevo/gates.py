"""Orchestrator side of the verifier ladder: compile checks in parallel worker
subprocesses (cached by source hash), gates 2-4 in a verify subprocess serialized
by a GPU lock, hang-kill via subprocess timeout, and the reward-hack source scan
(flags, never auto-rejects — spec §5)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCAN_PATTERNS = {
    "lru_cache": r"lru_cache",
    "global_state": r"^\s*global\s+\w",
    "module_cache": r"^_?\w*cache\w*\s*=|^\w+\s*=\s*\{\}\s*$",
    "torch_compile": r"torch\.compile",
    "file_io": r"\bopen\(|torch\.load|np\.load|pickle\.",
    "dynamic_import": r"__import__|importlib",
}


def source_hash(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def scan_flags(src: str) -> list[str]:
    return [name for name, pat in SCAN_PATTERNS.items()
            if re.search(pat, src, re.MULTILINE)]


class GateRunner:
    def __init__(self, cfg: dict, out_dir: str, adapter_name: str, targets: dict):
        self.cfg = cfg
        self.out_dir = out_dir
        self.adapter_name = adapter_name
        self.targets = targets
        self.gpu_lock = threading.Lock()
        self._cache_path = os.path.join(out_dir, "compile_cache.json")
        try:
            # only successes are trusted across runs — a cached failure may be
            # transient (OOM from a since-fixed memory hog, killed worker) and
            # would poison every later run of identical source (the planted
            # cheats!). Failures are cached in-memory for this run only.
            self._cache = {k: v for k, v in json.load(open(self._cache_path)).items()
                           if v and v[0]}
        except (OSError, ValueError):
            self._cache = {}
        self._env = dict(os.environ)
        # persist inductor artifacts across worker processes so torch.compile'd
        # incumbents are cheap after the first verify
        self._env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(out_dir, "inductor-cache"))

    def _lineage(self, op: str) -> dict:
        for lin in self.targets["lineages"]:
            if lin["op"] == op:
                return lin
        raise KeyError(op)

    def _run_worker(self, module: str, job: dict, timeout: int, name: str):
        job_path = job["candidate_path"] + f".{name}.json"
        with open(job_path, "w") as f:
            json.dump(job, f)
        try:
            return subprocess.run(
                [sys.executable, "-m", module, job_path],
                cwd=REPO_ROOT, env=self._env, timeout=timeout,
                capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None

    # ---------------------------------------------------------------- gate 1
    def compile(self, cand_path: str, op: str) -> tuple[bool, str]:
        src = open(cand_path).read()
        key = source_hash(src)
        if key in self._cache:
            return tuple(self._cache[key])
        job = dict(candidate_path=cand_path, op=op,
                   argspec=self._lineage(op)["shapes"][0],
                   device=self.cfg["device"], seed=self.cfg["seed"])
        proc = self._run_worker("kernelevo.compile_worker", job,
                                self.cfg["compile_timeout_s"], "compile")
        if proc is None:
            result = (False, f"compile/first-run exceeded {self.cfg['compile_timeout_s']}s; killed")
        elif proc.returncode == 0:
            result = (True, "")
        else:
            tail = (proc.stderr or proc.stdout or "no output").strip()
            result = (False, tail[-4000:])
        self._cache[key] = result
        with open(self._cache_path, "w") as f:
            json.dump({k: v for k, v in self._cache.items() if v[0]}, f)
        return result

    # ------------------------------------------------------------- gates 2-4
    def verify(self, cand_path: str, op: str, upto: int,
               incumbents: dict, timeout: int | None = None) -> dict:
        lin = self._lineage(op)
        job = dict(
            adapter=self.adapter_name, device=self.cfg["device"], seed=self.cfg["seed"],
            op=op, candidate_path=cand_path, upto=upto,
            shapes=lin["shapes"], tol=self.cfg["tol"],
            gate2_trials=self.cfg["gate2_trials"],
            gate3=dict(warmup=self.cfg["gate3_warmup"], iters=self.cfg["gate3_iters"],
                       margin=self.cfg["gate3_margin"]),
            gate4=dict(warmup=self.cfg["gate4_warmup"], steps=self.cfg["gate4_steps"],
                       margin=self.cfg["gate4_margin"]),
            incumbents=incumbents,
            flops_per_sample=self.targets["flops_per_sample"],
            peak_flops=self.cfg.get("peak_flops"),
            samples_per_batch=self.targets["samples_per_batch"],
        )
        timeout = timeout or self.cfg["verify_timeout_s"]
        with self.gpu_lock:
            proc = self._run_worker("kernelevo.verify_worker", job, timeout, f"verify{upto}")
        infra = dict(gate_reached=1, correct_ok=False, accepted=False,
                     latency_us=None, incumbent_latency_us=None, step_time_ms=None,
                     incumbent_step_time_ms=None, samples_per_s=None, mfu=None)
        if proc is None:
            infra["failure_note"] = (f"kernel hung or verify exceeded {timeout}s; "
                                     f"subprocess killed, CUDA context discarded")
            return infra
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("KEVO_RESULT "):
                return json.loads(line[len("KEVO_RESULT "):])
        tail = (proc.stderr or "no stderr").strip()[-4000:]
        infra["failure_note"] = f"verify worker crashed (exit {proc.returncode}):\n{tail}"
        return infra
