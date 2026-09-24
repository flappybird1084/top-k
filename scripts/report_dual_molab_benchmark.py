"""Publish one honest 25-row view of kernel and architecture Molab results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_molab_benchmark import collect, evidence  # noqa: E402


def combined(manifest: Path, kernel_output: Path, recipe_output: Path) -> dict:
    kernel = collect(manifest, kernel_output, "kernel")
    recipe = collect(manifest, recipe_output, "recipe")
    rows = []
    for k, r in zip(kernel["results"], recipe["results"], strict=True):
        if (k["index"], k["repo"], k["commit"]) != (r["index"], r["repo"], r["commit"]):
            raise ValueError("kernel and architecture rows do not match")
        rows.append({"index": k["index"], "repo": k["repo"], "commit": k["commit"],
                     "workload": k["model"], "kernel": k, "architecture": r})
    return {"schema": "top-k-dual-molab-benchmark-v1",
            "repositories_total": len(rows),
            "kernel": {key: kernel[key] for key in
                       ("done", "failed", "running", "pending", "measured", "improved")},
            "architecture": {key: recipe[key] for key in
                             ("done", "failed", "running", "pending", "measured", "improved")},
            "results": rows}


def numeric_evidence(report: dict, kernel_output: Path, recipe_output: Path) -> dict:
    kernel = evidence({"results": [row["kernel"] for row in report["results"]]},
                      kernel_output)["repositories"]
    recipe = evidence({"results": [row["architecture"] for row in report["results"]]},
                      recipe_output)["repositories"]
    return {"schema": "top-k-dual-molab-evidence-v1",
            "repositories": [{"index": row["index"], "repo": row["repo"],
                              "commit": row["commit"], "kernel": k, "architecture": r}
                             for row, k, r in zip(report["results"], kernel, recipe,
                                                  strict=True)]}


def publish(report: dict, evidence_path: Path) -> str:
    import wandb

    snapshot = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    phase = ("complete" if all(report[mode]["pending"] == 0 and
                               report[mode]["running"] == 0
                               for mode in ("kernel", "architecture")) else "partial")
    run = wandb.init(entity=os.environ["WANDB_ENTITY"],
                     project=os.environ["WANDB_PROJECT"],
                     name=f"{report['repositories_total']}-repo-kernel-and-architecture-{phase}-{snapshot}",
                     job_type="repo-benchmark-summary",
                     tags=["repo-benchmark", "molab", "kernel", "architecture", phase],
                     config={"repository_count": report["repositories_total"],
                             "kernel_metric": "training step milliseconds; lower is better",
                             "architecture_metric": "held-out validation loss after fixed training seconds; lower is better",
                             "snapshot_utc": snapshot})
    try:
        columns = ["repo", "commit", "workload", "kernel_model", "kernel_status",
                   "kernel_accepted", "baseline_ms", "candidate_ms",
                   "kernel_step_time_reduction_pct",
                   "kernel_reason", "architecture_model", "architecture_status",
                   "architecture_accepted", "baseline_val_loss", "candidate_val_loss",
                   "architecture_loss_improvement_pct", "architecture_reason"]
        data = []
        for row in report["results"]:
            k, a = row["kernel"], row["architecture"]
            km, am = k.get("result") or {}, a.get("result") or {}
            data.append([row["repo"], row["commit"], row["workload"],
                         k.get("llm"), k["status"], km.get("accepted"),
                         km.get("baseline_ms"), km.get("candidate_ms"),
                         km.get("improvement_pct"), k.get("reason", km.get("reason")),
                         a.get("llm"), a["status"], am.get("accepted"),
                         am.get("baseline_val_loss"), am.get("candidate_val_loss"),
                         am.get("improvement_pct"), a.get("reason", am.get("reason"))])
        run.log({"repositories": wandb.Table(columns=columns, data=data)})
        artifact = wandb.Artifact(f"dual-molab-evidence-{snapshot}",
                                  type="benchmark-evidence")
        artifact.add_file(str(evidence_path), name="evidence.json")
        run.log_artifact(artifact)
        for mode in ("kernel", "architecture"):
            for key, value in report[mode].items():
                run.summary[f"{mode}_{key}"] = value
        run.summary["repositories_total"] = report["repositories_total"]
        return run.url
    finally:
        run.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/repos.json")
    ap.add_argument("--kernel-output", type=Path, required=True)
    ap.add_argument("--recipe-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--secrets", type=Path, help="private W&B environment file")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    report = combined(args.manifest, args.kernel_output, args.recipe_output)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    evidence_path = args.output / "evidence.json"
    evidence_path.write_text(json.dumps(numeric_evidence(
        report, args.kernel_output, args.recipe_output), indent=2, sort_keys=True) + "\n")
    if args.publish:
        if (not args.allow_partial and any(report[mode]["pending"] or report[mode]["running"]
                                           for mode in ("kernel", "architecture"))):
            ap.error("benchmark incomplete; use --allow-partial to publish now")
        if args.secrets:
            settings = dotenv_values(args.secrets)
            for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
                if settings.get(key):
                    os.environ[key] = settings[key]
            if not os.getenv("WANDB_API_KEY") and settings.get("WANDB_INFERENCE_API_KEY"):
                os.environ["WANDB_API_KEY"] = settings["WANDB_INFERENCE_API_KEY"]
        report["wandb_summary_url"] = publish(report, evidence_path)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path}: kernel {report['kernel']}; architecture {report['architecture']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
