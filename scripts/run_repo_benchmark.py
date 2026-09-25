"""Resumable, one-GPU benchmark over pinned public PyTorch repositories.

Plan is read-only. --run clones one shallow checkout per repository and executes
search.py serially so multiple training jobs never contend for the same GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "repos.json"
sys.path.insert(0, str(ROOT))
from scripts.inference_proxy import InferenceRelay


def manifest_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    names = [row["repo"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("manifest contains duplicate repositories")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid GitHub repository: {name!r}")
    for row in rows:
        if not re.fullmatch(r"[0-9a-f]{40}", row.get("sha", "")):
            raise ValueError(f"missing pinned commit for {row['repo']}")
    return rows


def preflight(model: str) -> None:
    if not os.getenv("WANDB_API_KEY") or not os.getenv("WANDB_ENTITY") or not os.getenv("WANDB_PROJECT"):
        raise SystemExit("Set WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT before --run")
    try:
        import torch
        from kernelevo.llm import make_wandb_inference_llm
    except ImportError as exc:
        raise SystemExit(f"Install requirements.txt in the GPU environment: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("--run requires a CUDA GPU")
    llm = make_wandb_inference_llm(model, max_tokens=256)
    ids = {item.id for item in llm.client.models.list().data}
    if model not in ids:
        raise SystemExit(f"W&B Inference model {model!r} is unavailable to this account")
    # One small completion tests authentication and the actual model endpoint,
    # rather than discovering a billing/auth error after cloning 30 repositories.
    answer = llm.complete([{"role": "user", "content":
                           "Return only a JSON object with an ok field set to true."}],
                          json_mode=True)
    try:
        valid = json.loads(answer.text).get("ok") is True
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise SystemExit("W&B Inference JSON-mode preflight did not return {'ok': true}")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def checkout(row: dict, cache: Path) -> tuple[Path, str]:
    name = row["repo"]
    dest = cache / name.replace("/", "__")
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone_env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--depth", "1",
                        "--filter=blob:none",
                        "--", f"https://github.com/{name}.git", str(dest)],
                       check=True, timeout=900, env=clone_env)
    sha = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                  text=True, timeout=30).strip()
    if sha != row["sha"]:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin",
                        row["sha"]], check=True, timeout=900)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", row["sha"]],
                       check=True, timeout=60)
        sha = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                      text=True, timeout=30).strip()
    if sha != row["sha"]:
        raise ValueError(f"checkout SHA mismatch for {name}")
    changes = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"],
                                      text=True, timeout=30).strip()
    if changes:
        raise ValueError(f"cached checkout for {name} has local changes")
    return dest, sha


def stop_process_tree(proc: subprocess.Popen, grace_seconds: float = 15) -> None:
    """Do not start the next GPU job while a timed-out child can still run."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=30)
        proc.wait(timeout=30)
        return

    def group_exists() -> bool:
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def signal_group(sig: signal.Signals) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    if group_exists():
        signal_group(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while group_exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    # The group leader can exit before a child does; always check the whole
    # group rather than assuming proc.wait() means GPU work has stopped.
    if group_exists():
        signal_group(signal.SIGKILL)
    proc.wait(timeout=30)


def job_env(proxy_url: str | None = None, home: Path | None = None) -> dict[str, str]:
    """Keep EC2/AWS/W&B operator credentials out of repository subprocesses."""
    names = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TEMP", "TMP",
             "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "VIRTUAL_ENV",
             "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
             "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "UV_CACHE_DIR",
             "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
             "HF_DATASETS_CACHE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
             "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_INFERENCE_PROJECT")
    environment = {key: os.environ[key] for key in names if os.environ.get(key)}
    if home is not None:
        environment["HOME"] = str(home)
    if proxy_url:
        environment["WANDB_INFERENCE_API_KEY"] = "benchmark-local-relay"
        environment["WANDB_INFERENCE_BASE_URL"] = proxy_url
    return environment


def archive_result(path: Path, mode: str) -> dict:
    if not path.is_file():
        return {"measured": False, "reason": "no archive"}
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        if mode == "kernel":
            rows = [dict(row) for row in db.execute(
                "SELECT step_time_ms, incumbent_step_time_ms, accepted, gate_reached "
                "FROM candidates WHERE generation > 0 ORDER BY rowid")]
            accepted = [r for r in rows if r["accepted"] and r["step_time_ms"]
                        and r["incumbent_step_time_ms"]]
            if not accepted:
                measured = any(r["incumbent_step_time_ms"] for r in rows)
                return {"measured": measured, "accepted": 0,
                        "reason": "no accepted kernel" if measured else
                        "no in-model candidate measurement"}
            # Match the search loop's final result: the first incumbent is the
            # reference and the fastest accepted step is the final candidate.
            best = min(accepted, key=lambda r: r["step_time_ms"])
            baseline = next(r["incumbent_step_time_ms"] for r in rows
                            if r["incumbent_step_time_ms"])
            return {"measured": True, "accepted": len(accepted),
                    "baseline_ms": baseline,
                    "candidate_ms": best["step_time_ms"],
                    "improvement_pct": round(100 * (baseline -
                                              best["step_time_ms"]) / baseline, 3)}
        architecture = db.execute(
            "SELECT 1 FROM candidates WHERE phase='architecture' "
            "AND val_loss IS NOT NULL LIMIT 1").fetchone()
        finalist = db.execute(
            "SELECT val_loss, train_secs, accepted, arch_fp FROM candidates WHERE phase='finals' "
            "AND val_loss IS NOT NULL ORDER BY val_loss LIMIT 1").fetchone()
        baseline = db.execute(
            "SELECT val_loss, train_secs, arch_fp FROM candidates WHERE phase='baseline' "
            "AND train_secs=? LIMIT 1", (finalist["train_secs"],)).fetchone() if finalist else \
            db.execute("SELECT val_loss, train_secs, arch_fp FROM candidates "
                       "WHERE phase='baseline' ORDER BY train_secs DESC LIMIT 1").fetchone()
        if not baseline or baseline["val_loss"] is None:
            return {"measured": False, "reason": "no baseline at final candidate budget"}
        if not architecture:
            return {"measured": False, "baseline_val_loss": baseline["val_loss"],
                    "reason": "no architecture candidate measurement"}
        if not finalist:
            return {"measured": False, "baseline_val_loss": baseline["val_loss"],
                    "reason": "no final candidate measurement"}
        structural_change = bool(finalist["arch_fp"] and baseline["arch_fp"] and
                                 finalist["arch_fp"] != baseline["arch_fp"])
        result = {"measured": True, "baseline_val_loss": baseline["val_loss"],
                  "final_budget_s": baseline["train_secs"],
                  "accepted": int(bool(finalist["accepted"] and structural_change)),
                  "architecture_changed": structural_change,
                  "candidate_val_loss": finalist["val_loss"]}
        if baseline["val_loss"] > 0:
            result["improvement_pct"] = round(
                100 * (baseline["val_loss"] - finalist["val_loss"]) /
                baseline["val_loss"], 3)
        else:
            result["reason"] = "baseline validation loss is not positive"
        if not structural_change:
            result["reason"] = "final model structure unchanged"
        if result["accepted"]:
            result["winner_val_loss"] = finalist["val_loss"]
        return result


def run_one(row: dict, args, root: Path) -> dict:
    name = row["repo"]
    slug = name.replace("/", "__")
    run_dir = root / slug
    state_path = run_dir / "benchmark.json"
    expected = {"repo": name, "model": row["model"], "mode": args.mode,
                "profile": args.profile, "llm": f"wandb:{args.model}",
                "generations": args.generations, "spend_cap": args.spend_cap,
                "source_sha": row["sha"]}
    attempt = 1
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if any(old.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{state_path} has a different benchmark configuration; "
                             "use a new --output directory")
        if not args.resume:
            raise ValueError(f"{state_path} already exists; use --resume")
        if old.get("status") == "done":
            return old
        attempt = old.get("attempt", 1) + 1
    state = {**expected, "attempt": attempt, "status": "preparing",
             "started_at": time.time()}
    attempt_dir = run_dir / ("run" if attempt == 1 else f"run-{attempt}")

    def record() -> None:
        save_json(state_path, state)
        save_json(run_dir / f"attempt-{attempt}.json", state)

    record()
    try:
        source, sha = checkout(row, root / "checkouts")
        state.update(commit=sha, status="running")
        record()
        command = [sys.executable, "-u", str(ROOT / "search.py"), "--repo", str(source),
                   "--comments", f"Benchmark {row['model']}. Use a reproducible small "
                   "training configuration and the repository's real model implementation.",
                   "--llm", f"wandb:{args.model}", "--mode", args.mode,
                   "--profile", args.profile, "--max-generations", str(args.generations),
                   "--spend-cap", str(args.spend_cap), "--out", str(attempt_dir)]
        log_path = run_dir / f"attempt-{attempt}.log"
        relay = InferenceRelay(args.model, args.spend_cap)
        job_home = attempt_dir / "home"
        job_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        with relay.serving() as proxy_url:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        env=job_env(proxy_url, job_home),
                                        start_new_session=(os.name != "nt"),
                                        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                                                       if os.name == "nt" else 0))
                try:
                    rc = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    stop_process_tree(proc)
                    rc = 124
                else:
                    # A successful parent may still leave a training worker
                    # behind; clear the group before the next repository.
                    if os.name != "nt":
                        stop_process_tree(proc, grace_seconds=2)
        state.update(inference_calls=relay.calls, inference_spend_usd=relay.spent_usd)
        if relay.credits_exhausted:
            state["credits_exhausted"] = True
        state.update(exit_code=rc, status="done" if rc == 0 else "failed",
                     finished_at=time.time())
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                url = re.search(r"https://wandb\.ai/[\w.-]+/[\w.-]+/runs/[\w]+", line)
                if url:
                    state["wandb_url"] = url.group(0)
                if re.search(r"(?:insufficient|exhausted|out of).*credits|"
                             r"credit balance|payment required|HTTP 402|Error code: 402",
                             line, re.IGNORECASE):
                    state["credits_exhausted"] = True
        state["result"] = archive_result(attempt_dir / "archive.sqlite", args.mode)
        if rc == 0 and not state["result"].get("measured"):
            state["status"] = "failed"
            state["reason"] = "search exited successfully without a measured result"
        if rc != 0:
            state["reason"] = "timeout" if rc == 124 else f"search failed; see {log_path.name}"
            if state.get("credits_exhausted"):
                state["reason"] = "W&B Inference credits exhausted"
    except Exception as exc:  # one broken repository must not erase other results
        state.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                     finished_at=time.time())
    record()
    return state


def publish_summary(results: list[dict], args) -> str:
    """Mirror the complete denominator, including failures, into one W&B run."""
    import wandb

    run = wandb.init(entity=os.environ["WANDB_ENTITY"],
                     project=os.environ["WANDB_PROJECT"],
                     name=f"{len(results)}-repo-{args.mode}-{args.profile.lower()}",
                     job_type="repo-benchmark-summary",
                     tags=["repo-benchmark", args.mode, args.profile.lower()],
                     config={"manifest": str(args.manifest), "model": args.model,
                             "generations": args.generations,
                             "spend_cap_per_repo": args.spend_cap,
                             "repository_count": len(results)})
    try:
        columns = ["repo", "commit", "workload", "optimizer_model", "status", "measured", "accepted",
                   "baseline", "candidate", "improvement_pct", "reason", "run_url"]
        data = []
        for result in results:
            metric = result.get("result") or {}
            data.append([result.get("repo"), result.get("commit"), result.get("model"),
                         result.get("llm"), result.get("status"),
                         metric.get("measured", False), metric.get("accepted"),
                         metric.get("baseline_ms", metric.get("baseline_val_loss")),
                         metric.get("candidate_ms", metric.get("candidate_val_loss")),
                         metric.get("improvement_pct"),
                         result.get("reason", metric.get("reason")),
                         result.get("wandb_url")])
        run.log({"repositories": wandb.Table(columns=columns, data=data)})
        run.summary["repositories_total"] = len(results)
        run.summary["runs_done"] = sum(r.get("status") == "done" for r in results)
        run.summary["runs_failed"] = sum(r.get("status") == "failed" for r in results)
        run.summary["measured"] = sum(bool((r.get("result") or {}).get("measured"))
                                      for r in results)
        run.summary["improved"] = sum(r.get("status") == "done" and
                                      (r.get("result") or {}).get("accepted", 0) > 0 and
                                      (r.get("result") or {}).get("improvement_pct", 0) > 0
                                      for r in results)
        return run.url
    finally:
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark-runs")
    ap.add_argument("--run", action="store_true", help="start paid inference and GPU jobs")
    ap.add_argument("--resume", action="store_true", help="skip completed repositories")
    ap.add_argument("--limit", type=int, default=None, help="pilot on the first N repositories")
    ap.add_argument("--mode", choices=("kernel", "recipe"), default="kernel")
    ap.add_argument("--profile", choices=("DEV", "RUN"), default="DEV")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--spend-cap", type=float, default=3.0,
                    help="USD estimate cap per repository")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per repository")
    args = ap.parse_args()
    rows = manifest_rows(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit must be positive")
        rows = rows[:args.limit]
    print(f"{len(rows)} repositories; {args.mode} / {args.profile}; "
          f"up to ${len(rows) * args.spend_cap:g} in estimated LLM spend caps")
    for row in rows:
        print(f"  {row['repo']}: {row['model']}")
    if not args.run:
        return
    if args.generations < 1 or args.spend_cap <= 0 or args.timeout < 1:
        ap.error("generations, spend cap and timeout must be positive")
    preflight(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['repo']}", flush=True)
        result = run_one(row, args, args.output)
        results.append(result)
        save_json(args.output / "summary.json", {"results": results})
        print(f"  {result['status']}: {result.get('result', {}).get('improvement_pct', 'n/a')}%",
              flush=True)
        if result.get("credits_exhausted"):
            print("W&B Inference credits exhausted; stopping before the next repository.",
                  file=sys.stderr, flush=True)
            break
    if results:
        try:
            url = publish_summary(results, args)
            save_json(args.output / "summary.json", {"results": results,
                                                      "wandb_summary_url": url})
            print(f"W&B benchmark summary: {url}")
        except Exception as exc:  # local results remain authoritative
            print(f"W&B benchmark summary upload failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
