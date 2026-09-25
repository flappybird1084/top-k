"""Run the pinned repository benchmark through the website's Molab dispatcher.

This process runs on the EC2 backend. It dispatches one repository at a time
to the connected GPU notebook and records each attempt beside its artifacts.
Secrets are loaded from private files and never written to job.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernelevo.molab import MolabClient, MolabTarget  # noqa: E402
from scripts.run_repo_benchmark import archive_result, manifest_rows  # noqa: E402

CREDIT_ERROR = re.compile(r"(?:insufficient|exhausted|out of).*credits|"
                          r"credit balance|payment required|HTTP 402|Error code: 402|"
                          r"insufficient[_ ]quota|(?:quota|billing).*(?:exceeded|limit|disabled)|"
                          r"exceeded your current quota",
                          re.IGNORECASE)
RELAY_BUDGET_ERROR = re.compile(r"Relay closed for this run: this account reached "
                                r"its relay (?:requests|tokens|searches) budget", re.IGNORECASE)
WORKLOAD_GUIDANCE = {
    "huggingface/pytorch-image-models":
        "Choose a small timm Vision Transformer for classification so its "
        "LayerNorm and MLP blocks are part of the measured training step.",
}
RECIPE_BENCHMARK = {
    "phases": [
        {"kind": "architecture", "generations": 1, "candidates": 2,
         "train_seconds": 60},
        {"kind": "hyperparam", "generations": 1, "candidates": 2,
         "train_seconds": 60},
    ],
    "finals_top_k": 1,
    "finals_train_seconds": 120,
    "subagent_parallelism": 1,
}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def previous_job_finished(client: MolabClient, job_id: str) -> bool:
    """Fail closed if a disconnected dispatcher left its GPU job running."""
    work = f"/tmp/kevo_{job_id[:8]}"
    ok, out, err = client.run(
        "import os\n"
        f"_w={work!r}\n"
        "print('NO_WORK' if not os.path.isdir(_w) else "
        "('EXITED' if os.path.isfile(os.path.join(_w,'job.exit')) else 'RUNNING'))\n")
    if not ok:
        raise RuntimeError(f"Could not inspect prior notebook job: {err[:200]}")
    return out.strip().splitlines()[-1] in ("NO_WORK", "EXITED")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/repos.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--secrets", type=Path, required=True)
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--notebook-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--mode", choices=("kernel", "recipe"), default="kernel")
    ap.add_argument("--profile", choices=("DEV", "RUN"), default="DEV")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--spend-cap", type=float,
                    help="inference USD estimate per repository (kernel: 3, recipe: 0.75)")
    ap.add_argument("--start", type=int, default=1, help="One-based manifest row")
    ap.add_argument("--end", type=int, default=25, help="Inclusive manifest row")
    ap.add_argument("--wait-for", type=Path, help="Wait for a previous run's result.json")
    ap.add_argument("--plan", action="store_true", help="Print selected repositories only")
    args = ap.parse_args()
    if args.spend_cap is None:
        args.spend_cap = 0.75 if args.mode == "recipe" else 3.0
    if args.mode == "recipe" and args.generations != 2:
        ap.error("--generations applies to kernel mode; recipe phases use their recorded schedule")
    rows = manifest_rows(args.manifest)
    if not 1 <= args.start <= args.end <= len(rows):
        ap.error("--start and --end must select manifest rows in order")
    selected = rows[args.start - 1:args.end]
    if args.plan:
        for index, row in enumerate(selected, args.start):
            print(f"{index:02d} {row['repo']} {row['sha']} {args.mode} wandb:{args.model}")
        return 0

    if args.wait_for:
        print(f"[benchmark] waiting for {args.wait_for}", flush=True)
        while not args.wait_for.exists():
            time.sleep(30)
        previous = json.loads(args.wait_for.read_text())
        if previous.get("exit_code") != 0:
            print("[benchmark] previous run failed; inspect it before continuing", flush=True)
            return 1

    settings = dotenv_values(args.secrets)
    if not (settings.get("WANDB_INFERENCE_API_KEY") or settings.get("WANDB_API_KEY")):
        raise RuntimeError("W&B Inference credential missing from private secrets file")
    for key in ("WANDB_INFERENCE_API_KEY", "WANDB_API_KEY",
                "WANDB_INFERENCE_BASE_URL", "WANDB_INFERENCE_PROJECT",
                "WANDB_ENTITY", "WANDB_PROJECT"):
        if settings.get(key):
            os.environ[key] = settings[key]
    # Public-repo benchmarks need the model and GPU, not repeated web research.
    # Keeping research off saves relay quota and shortens sequential runs.
    remote_env = {"KEVO_WANDB_INFERENCE_RELAY": "1",
                  "KEVO_DISABLE_WEB_RESEARCH": "1"}
    token = args.token_file.read_text().strip()
    connection = {"notebook_url": args.notebook_url, "connection": "--token " + token}
    client = MolabClient(args.notebook_url, token)
    had_failures = False

    for index, row in enumerate(selected, args.start):
        run_dir = args.output / row["repo"].replace("/", "__")
        state_path = run_dir / "state.json"
        expected = {"index": index, "repo": row["repo"], "source_sha": row["sha"],
                    "workload": row["model"], "llm": f"wandb:{args.model}",
                    "profile": args.profile,
                    "generations": args.generations if args.mode == "kernel" else None,
                    "spend_cap_usd": args.spend_cap, "execution_target": "molab",
                    "mode": args.mode,
                    "precision_policy": ("fp32-params-cuda-bf16-autocast-v1"
                                         if args.mode == "recipe" else None),
                    "recipe": RECIPE_BENCHMARK if args.mode == "recipe" else None}
        if state_path.exists():
            old = json.loads(state_path.read_text())
            if any(old.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"{state_path} has a different benchmark configuration")
            if old.get("status") == "done" and old.get("result", {}).get("measured"):
                print(f"[benchmark] {index:02d} {row['repo']} already done", flush=True)
                continue
            if old.get("job_id") and not previous_job_finished(client, old["job_id"]):
                raise RuntimeError(f"Prior GPU job for {row['repo']} may still be running; "
                                   "inspect the notebook before retrying")
            attempt = old.get("attempt", 0) + 1
        else:
            attempt = 1
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = run_dir / f"attempt-{attempt}"
        while attempt_dir.exists():
            attempt += 1
            attempt_dir = run_dir / f"attempt-{attempt}"
        attempt_dir.mkdir()
        job = {"id": uuid.uuid4().hex,
               "repo": f"https://github.com/{row['repo']}/commit/{row['sha']}",
               "comments": f"Benchmark {row['model']}. Use a reproducible small training "
                           "configuration and the repository's real model implementation. "
                           "Generate a tiny deterministic synthetic training batch in memory, "
                           "with valid labels for this model. Do not download datasets or "
                           "depend on external data paths. "
                           f"{WORKLOAD_GUIDANCE.get(row['repo'], '')}",
               "max_debug_turns": 8, "profile": args.profile,
               "llm": f"wandb:{args.model}",
               "spend_cap": args.spend_cap, "mode": args.mode,
               "execution_target": "molab"}
        if args.mode == "recipe":
            job["recipe"] = RECIPE_BENCHMARK
        else:
            job["max_generations"] = args.generations
        save(attempt_dir / "job.json", job)
        state = {**expected, "status": "running", "attempt": attempt,
                 "started_at": time.time(), "job_id": job["id"],
                 "data_source": "deterministic synthetic batches",
                 "timing_scope": ("fixed wall-clock training budget; held-out validation loss"
                                  if args.mode == "recipe" else
                                  "full training step including host-to-GPU transfer")}
        save(state_path, state)
        log_path = attempt_dir / "dispatch.log"
        with log_path.open("w", buffering=1) as log:
            def write_line(line: str) -> None:
                log.write(line.rstrip("\n") + "\n")
                print(line, flush=True)

            write_line(f"[benchmark] {index:02d}/{len(rows)} repo={row['repo']} "
                       f"source_sha={row['sha']} llm={job['llm']} "
                       f"mode={args.mode} "
                       f"schedule={RECIPE_BENCHMARK if args.mode == 'recipe' else args.generations} "
                       f"spend_cap_usd={args.spend_cap}")
            try:
                rc = MolabTarget(connection).dispatch(
                    job, str(ROOT), remote_env, write_line,
                    artifacts_dir=str(attempt_dir / "artifacts"))
            except Exception as exc:
                write_line(f"[benchmark] dispatcher error: {type(exc).__name__}: {exc}")
                rc = 1
        archive = attempt_dir / "artifacts/archive.sqlite"
        try:
            state["result"] = archive_result(archive, args.mode)
        except Exception as exc:
            state["result"] = {"measured": False, "reason": f"archive unreadable: {exc}"}
        state.update(exit_code=rc,
                     status="done" if rc == 0 and state["result"].get("measured") else "failed",
                     finished_at=time.time())
        log_text = log_path.read_text(errors="replace")
        state["credits_exhausted"] = bool(CREDIT_ERROR.search(log_text))
        state["relay_budget_exhausted"] = bool(RELAY_BUDGET_ERROR.search(log_text))
        infrastructure_failure = (rc != 0 and
                                  "[molab] remote run finished with exit" not in log_text)
        if state["status"] == "failed":
            state["reason"] = ("W&B Inference credits exhausted" if state["credits_exhausted"]
                               else "relay usage budget exhausted" if state["relay_budget_exhausted"]
                               else "notebook dispatch or infrastructure failed" if infrastructure_failure
                               else state["result"].get("reason") or
                               f"dispatcher exited {rc}; inspect attempt log")
            had_failures = True
        save(state_path, state)
        print(f"[benchmark] {index:02d} exit={rc} result={state['result']}", flush=True)
        if state["credits_exhausted"] or state["relay_budget_exhausted"] or infrastructure_failure:
            return 1
        if state["status"] == "failed" and not previous_job_finished(client, job["id"]):
            print(f"[benchmark] {index:02d} GPU job may still be running; stopping", flush=True)
            return 1
        if state["status"] == "failed":
            print(f"[benchmark] {index:02d} failed; continuing to next repository", flush=True)
    return int(had_failures)


if __name__ == "__main__":
    raise SystemExit(main())
