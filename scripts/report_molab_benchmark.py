"""Summarize the pinned Molab benchmark, including every failed repository.

Run on the EC2 host after sourcing its private W&B environment. The report
contains no credentials and can be published to W&B with --publish.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_repo_benchmark import manifest_rows  # noqa: E402


def collect(manifest: Path, output: Path, mode: str = "kernel") -> dict:
    rows = manifest_rows(manifest)
    results = []
    for index, row in enumerate(rows, 1):
        state_path = output / row["repo"].replace("/", "__") / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (state.get("index") != index or state.get("repo") != row["repo"]
                    or state.get("source_sha") != row["sha"]
                    or state.get("mode", "kernel") != mode):
                raise ValueError(f"Benchmark state does not match manifest: {state_path}")
            result = {**state, "commit": row["sha"], "model": row["model"],
                      "mode": mode}
            if result["status"] != "done":
                # An interrupted search may have a promising intermediate candidate.
                # It is not a completed benchmark result.
                result["result"] = {"measured": False}
                if result["status"] == "failed":
                    result["reason"] = result.get("reason") or (
                        f"dispatch exited {result.get('exit_code', 'unknown')}; "
                        "inspect the private attempt log")
        else:
            result = {"index": index, "repo": row["repo"], "commit": row["sha"],
                      "model": row["model"], "mode": mode,
                      "status": "pending", "result": {}}
        results.append(result)
    return {"mode": mode, "repositories_total": len(results),
            "done": sum(r["status"] == "done" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "running": sum(r["status"] == "running" for r in results),
            "pending": sum(r["status"] == "pending" for r in results),
            "measured": sum(bool((r.get("result") or {}).get("measured")) for r in results),
            "improved": sum(r["status"] == "done" and
                            (r.get("result") or {}).get("improvement_pct", 0) > 0
                            for r in results),
            "results": results}


def evidence(report: dict, output: Path) -> dict:
    """Export numeric gate evidence only; never upload source, logs or secrets."""
    records = []
    for row in report["results"]:
        record = {key: row.get(key) for key in
                  ("index", "repo", "commit", "model", "mode", "llm", "status", "attempt",
                   "exit_code", "started_at", "finished_at", "data_source", "timing_scope")}
        record["result"] = row.get("result") or {}
        archive = (output / row["repo"].replace("/", "__") /
                   f"attempt-{row.get('attempt', 0)}" / "artifacts" / "archive.sqlite")
        if archive.is_file():
            try:
                with sqlite3.connect(archive) as db:
                    db.row_factory = sqlite3.Row
                    fields = ("generation, gate_reached, compile_ok, correct_ok, accepted, "
                              "val_loss, train_secs, phase, model_params" if
                              row.get("mode") == "recipe" else
                              "generation, gate_reached, compile_ok, correct_ok, accepted, "
                              "step_time_ms, incumbent_step_time_ms")
                    record["candidate_gates"] = [dict(candidate) for candidate in db.execute(
                        f"SELECT {fields} FROM candidates WHERE generation > 0 ORDER BY id")]
            except sqlite3.Error:
                record["archive_evidence"] = "unavailable or incompatible"
        records.append(record)
    return {"schema": "top-k-molab-benchmark-evidence-v1", "repositories": records}


def publish(report: dict, manifest: Path, evidence_path: Path) -> str:
    import wandb

    snapshot = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    phase = "partial" if report["running"] or report["pending"] else "complete"
    run = wandb.init(entity=os.environ["WANDB_ENTITY"],
                     project=os.environ["WANDB_PROJECT"],
                     name=f"{report['repositories_total']}-repo-molab-{report['mode']}-{phase}-{snapshot}",
                     job_type="repo-benchmark-summary",
                     tags=["repo-benchmark", "molab", report["mode"], phase],
                     config={"manifest": str(manifest),
                             "repository_count": report["repositories_total"],
                             "execution_target": "molab", "mode": report["mode"],
                             "snapshot_utc": snapshot,
                             "report_phase": phase})
    try:
        columns = ["repo", "commit", "workload", "mode", "optimizer_model", "status",
                   "measured", "accepted", "baseline", "candidate",
                   "improvement_pct", "reason", "data_source", "timing_scope"]
        data = []
        for row in report["results"]:
            metric = row.get("result") or {}
            data.append([row["repo"], row["commit"], row["model"], row["mode"],
                         row.get("llm"),
                         row["status"], bool(metric.get("measured")),
                         metric.get("accepted"),
                         metric.get("baseline_ms", metric.get("baseline_val_loss")),
                         metric.get("candidate_ms", metric.get("winner_val_loss")),
                         metric.get("improvement_pct"),
                         row.get("reason", metric.get("reason")),
                         row.get("data_source"), row.get("timing_scope")])
        run.log({"repositories": wandb.Table(columns=columns, data=data)})
        artifact = wandb.Artifact(f"molab-benchmark-evidence-{snapshot}", type="benchmark-evidence")
        artifact.add_file(str(evidence_path), name="evidence.json")
        run.log_artifact(artifact)
        for key in ("repositories_total", "done", "failed", "running", "pending",
                    "measured", "improved"):
            run.summary[key] = report[key]
        return run.url
    finally:
        run.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/repos.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("kernel", "recipe"), default="kernel")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Publish while some repositories are pending or running")
    args = ap.parse_args()
    report = collect(args.manifest, args.output, args.mode)
    report_path = args.output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    evidence_path = args.output / "evidence.json"
    evidence_path.write_text(json.dumps(evidence(report, args.output), indent=2,
                                        sort_keys=True) + "\n")
    print(f"Wrote {report_path}: {report['done']} done, {report['failed']} failed, "
          f"{report['running']} running, {report['pending']} pending")
    if args.publish:
        if (report["running"] or report["pending"]) and not args.allow_partial:
            ap.error("benchmark incomplete; use --allow-partial to publish now")
        url = publish(report, args.manifest, evidence_path)
        report["wandb_summary_url"] = url
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
