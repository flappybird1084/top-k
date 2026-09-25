"""Build the landing graph's compact, auditable benchmark data."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/results/top10-2026-09-25/summary.json"
DESTINATION = ROOT / "ui/landing-benchmarks.js"
SHORT_NAMES = {
    "huggingface/pytorch-image-models": "timm",
    "karpathy/nanochat": "nanochat",
    "huggingface/transformers": "Transformers",
    "huggingface/diffusers": "Diffusers",
    "Lightning-AI/litgpt": "LitGPT",
}


def accepted(result):
    return bool(result and result.get("measured") and result.get("accepted"))


def build():
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    results = []
    for row in summary["results"]:
        kernel = row["kernel"]
        architecture = row["architecture"]
        kernel_result = kernel.get("result", {})
        architecture_result = architecture.get("result", {})
        kernel_win = accepted(kernel_result)
        architecture_win = (
            accepted(architecture_result)
            and architecture_result.get("architecture_changed") is True
            and architecture_result.get("baseline_val_loss", 0) > 0
            and architecture_result.get("candidate_val_loss", float("inf"))
            < architecture_result["baseline_val_loss"]
        )
        if not (kernel_win or architecture_win):
            continue
        repo = row["repo"]
        entry = {"name": repo, "short": SHORT_NAMES.get(repo, repo.split("/")[-1])}
        if kernel_win:
            entry["kernel"] = {
                "baseline_ms": kernel_result["baseline_ms"],
                "candidate_ms": kernel_result["candidate_ms"],
                "improvement_pct": kernel_result["improvement_pct"],
                "generations": kernel["generations"],
            }
        if architecture_win:
            entry["architecture"] = {
                "baseline_val_loss": architecture_result["baseline_val_loss"],
                "candidate_val_loss": architecture_result["candidate_val_loss"],
                "improvement_pct": architecture_result["improvement_pct"],
                "budget_s": architecture_result["final_budget_s"],
                "caveat": (
                    "loss-floor" if architecture_result["baseline_val_loss"] < 1e-7
                    else "near-zero" if architecture_result["baseline_val_loss"] < 1e-3
                    else None
                ),
            }
        results.append(entry)
    payload = {
        "evidence_url": "https://github.com/flappybird1084/top-k/tree/main/benchmarks/results/top10-2026-09-25",
        "results": results,
    }
    safe_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return "// Generated from benchmarks/results/top10-2026-09-25/summary.json.\nwindow.TOPK_BENCHMARKS=" + safe_json + ";\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed data is stale")
    args = parser.parse_args()
    output = build()
    if args.check:
        if not DESTINATION.exists() or DESTINATION.read_text(encoding="utf-8") != output:
            parser.error("ui/landing-benchmarks.js is stale; regenerate it")
    else:
        DESTINATION.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
