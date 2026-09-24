"""Run the pinned repository benchmark through the website's Molab dispatcher.

This process runs on the EC2 backend. It dispatches one repository at a time
to the connected GPU notebook and records each attempt beside its artifacts.
Secrets are loaded from private files and never written to job.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernelevo.molab import MolabTarget  # noqa: E402
from scripts.run_repo_benchmark import archive_result, manifest_rows  # noqa: E402

PASS_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "WANDB_API_KEY", "WANDB_ENTITY",
            "WANDB_PROJECT", "WEAVE_PROJECT", "WANDB_INFERENCE_API_KEY",
            "WANDB_INFERENCE_BASE_URL", "WANDB_INFERENCE_PROJECT", "SEARXNG_URL")
CREDIT_ERROR = re.compile(r"(?:insufficient|exhausted|out of).*credits|"
                          r"credit balance|payment required|HTTP 402|Error code: 402",
                          re.IGNORECASE)


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/repos.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--secrets", type=Path, required=True)
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--notebook-url", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
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
    remote_env = {key: settings[key] for key in PASS_ENV if settings.get(key)}
    if not (remote_env.get("WANDB_INFERENCE_API_KEY") or remote_env.get("WANDB_API_KEY")):
        raise RuntimeError("W&B Inference credential missing from private secrets file")
    token = args.token_file.read_text().strip()
    connection = {"notebook_url": args.notebook_url, "connection": "--token " + token}

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
            if old.get("status") == "done":
                print(f"[benchmark] {index:02d} {row['repo']} already done", flush=True)
                continue
            attempt = old.get("attempt", 0) + 1
        else:
            attempt = 1
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = run_dir / f"attempt-{attempt}"
        attempt_dir.mkdir()
        job = {"id": uuid.uuid4().hex,
               "repo": f"https://github.com/{row['repo']}/commit/{row['sha']}",
               "comments": f"Benchmark {row['model']}. Use a reproducible small training "
                           "configuration and the repository's real model implementation.",
               "max_debug_turns": 5, "profile": args.profile,
               "llm": f"wandb:{args.model}", "max_generations": args.generations,
               "spend_cap": args.spend_cap, "mode": "kernel",
               "execution_target": "molab"}
        save(attempt_dir / "job.json", job)
        state = {**expected, "status": "running", "attempt": attempt,
                 "started_at": time.time(), "job_id": job["id"]}
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
        state.update(exit_code=rc, status="done" if rc == 0 else "failed",
                     finished_at=time.time())
        archive = attempt_dir / "artifacts/archive.sqlite"
        state["result"] = archive_result(archive, "kernel")
        state["credits_exhausted"] = bool(CREDIT_ERROR.search(log_path.read_text(errors="replace")))
        save(state_path, state)
        print(f"[benchmark] {index:02d} exit={rc} result={state['result']}", flush=True)
        if rc or state["credits_exhausted"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
