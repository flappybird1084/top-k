"""Summarize the pinned Molab benchmark, including every failed repository.

Run on the EC2 host after sourcing its private W&B environment. The report
contains no credentials and can be published to W&B with --publish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_repo_benchmark import manifest_rows  # noqa: E402


def collect(manifest: Path, output: Path) -> dict:
    rows = manifest_rows(manifest)
    results = []
    for index, row in enumerate(rows, 1):
        state_path = output / row["repo"].replace("/", "__") / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (state.get("index") != index or state.get("repo") != row["repo"]
                    or state.get("source_sha") != row["sha"]):
                raise ValueError(f"Benchmark state does not match manifest: {state_path}")
            result = {**state, "commit": row["sha"], "model": row["model"]}
        else:
            result = {"index": index, "repo": row["repo"], "commit": row["sha"],
                      "model": row["model"], "status": "pending", "result": {}}
        results.append(result)
    return {"repositories_total": len(results),
            "done": sum(r["status"] == "done" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "running": sum(r["status"] == "running" for r in results),
            "pending": sum(r["status"] == "pending" for r in results),
            "measured": sum(bool((r.get("result") or {}).get("measured")) for r in results),
            "improved": sum(r["status"] == "done" and
                            (r.get("result") or {}).get("improvement_pct", 0) > 0
                            for r in results),
            "results": results}


def publish(report: dict, manifest: Path) -> str:
    import wandb

    run = wandb.init(entity=os.environ["WANDB_ENTITY"],
                     project=os.environ["WANDB_PROJECT"],
                     name=f"{report['repositories_total']}-repo-molab-kernel-benchmark",
                     job_type="repo-benchmark-summary",
                     tags=["repo-benchmark", "molab", "kernel"],
                     config={"manifest": str(manifest),
                             "repository_count": report["repositories_total"],
                             "execution_target": "molab"})
    try:
        columns = ["repo", "commit", "workload", "optimizer_model", "status",
                   "measured", "accepted", "baseline_ms", "candidate_ms",
                   "improvement_pct", "reason", "data_source", "timing_scope"]
        data = []
        for row in report["results"]:
            metric = row.get("result") or {}
            data.append([row["repo"], row["commit"], row["model"], row.get("llm"),
                         row["status"], bool(metric.get("measured")),
                         metric.get("accepted"), metric.get("baseline_ms"),
                         metric.get("candidate_ms"), metric.get("improvement_pct"),
                         row.get("reason", metric.get("reason")),
                         row.get("data_source"), row.get("timing_scope")])
        run.log({"repositories": wandb.Table(columns=columns, data=data)})
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
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Publish while some repositories are pending or running")
    args = ap.parse_args()
    report = collect(args.manifest, args.output)
    report_path = args.output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {report_path}: {report['done']} done, {report['failed']} failed, "
          f"{report['running']} running, {report['pending']} pending")
    if args.publish:
        if (report["running"] or report["pending"]) and not args.allow_partial:
            ap.error("benchmark incomplete; use --allow-partial to publish now")
        url = publish(report, args.manifest)
        report["wandb_summary_url"] = url
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
