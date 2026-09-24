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
                          r"credit balance|payment required|HTTP 402|Error code: 402",
                          re.IGNORECASE)
WORKLOAD_GUIDANCE = {
    "huggingface/pytorch-image-models":
        "Choose a small timm Vision Transformer for classification so its "
        "LayerNorm and MLP blocks are part of the measured training step.",
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
    ap.add_argument("--profile", choices=("DEV", "RUN"), default="DEV")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--spend-cap", type=float, default=3.0)
    ap.add_argument("--start", type=int, default=1, help="One-based manifest row")
    ap.add_argument("--end", type=int, default=25, help="Inclusive manifest row")
    ap.add_argument("--wait-for", type=Path, help="Wait for a previous run's result.json")
    ap.add_argument("--plan", action="store_true", help="Print selected repositories only")
    args = ap.parse_args()
    rows = manifest_rows(args.manifest)
    if not 1 <= args.start <= args.end <= len(rows):
        ap.error("--start and --end must select manifest rows in order")
    selected = rows[args.start - 1:args.end]
    if args.plan:
        for index, row in enumerate(selected, args.start):
            print(f"{index:02d} {row['repo']} {row['sha']} wandb:{args.model}")
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
    remote_env = {"KEVO_WANDB_INFERENCE_RELAY": "1"}
    token = args.token_file.read_text().strip()
    connection = {"notebook_url": args.notebook_url, "connection": "--token " + token}
    client = MolabClient(args.notebook_url, token)

    for index, row in enumerate(selected, args.start):
        run_dir = args.output / row["repo"].replace("/", "__")
        state_path = run_dir / "state.json"
        expected = {"index": index, "repo": row["repo"], "source_sha": row["sha"],
                    "workload": row["model"], "llm": f"wandb:{args.model}",
                    "profile": args.profile, "generations": args.generations,
                    "spend_cap_usd": args.spend_cap, "execution_target": "molab"}
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
               "llm": f"wandb:{args.model}", "max_generations": args.generations,
               "spend_cap": args.spend_cap, "mode": "kernel",
               "execution_target": "molab"}
        save(attempt_dir / "job.json", job)
        state = {**expected, "status": "running", "attempt": attempt,
                 "started_at": time.time(), "job_id": job["id"],
                 "data_source": "deterministic synthetic batches",
                 "timing_scope": "full training step including host-to-GPU transfer"}
        save(state_path, state)
        log_path = attempt_dir / "dispatch.log"
        with log_path.open("w", buffering=1) as log:
            def write_line(line: str) -> None:
                log.write(line.rstrip("\n") + "\n")
                print(line, flush=True)

            write_line(f"[benchmark] {index:02d}/{len(rows)} repo={row['repo']} "
                       f"source_sha={row['sha']} llm={job['llm']} "
                       f"generations={args.generations} spend_cap_usd={args.spend_cap}")
            try:
                rc = MolabTarget(connection).dispatch(
                    job, str(ROOT), remote_env, write_line,
                    artifacts_dir=str(attempt_dir / "artifacts"))
            except Exception as exc:
                write_line(f"[benchmark] dispatcher error: {type(exc).__name__}: {exc}")
                rc = 1
        archive = attempt_dir / "artifacts/archive.sqlite"
        try:
            state["result"] = archive_result(archive, "kernel")
        except Exception as exc:
            state["result"] = {"measured": False, "reason": f"archive unreadable: {exc}"}
        state.update(exit_code=rc,
                     status="done" if rc == 0 and state["result"].get("measured") else "failed",
                     finished_at=time.time())
        state["credits_exhausted"] = bool(CREDIT_ERROR.search(log_path.read_text(errors="replace")))
        if state["status"] == "failed":
            state["reason"] = ("W&B Inference credits exhausted" if state["credits_exhausted"]
                               else state["result"].get("reason") or
                               f"dispatcher exited {rc}; inspect attempt log")
        save(state_path, state)
        print(f"[benchmark] {index:02d} exit={rc} result={state['result']}", flush=True)
        if state["credits_exhausted"]:
            return 1
        if state["status"] == "failed" and not previous_job_finished(client, job["id"]):
            print(f"[benchmark] {index:02d} GPU job may still be running; stopping", flush=True)
            return 1
        if state["status"] == "failed":
            print(f"[benchmark] {index:02d} failed; continuing to next repository", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
