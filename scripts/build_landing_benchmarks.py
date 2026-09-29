"""Build the landing graph's compact, auditable benchmark data."""

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
DESTINATION = ROOT / "ui/landing-benchmarks.js"
SHORT_NAMES = {
    "huggingface/pytorch-image-models": "timm",
    "karpathy/nanochat": "nanochat",
    "huggingface/transformers": "Transformers",
    "huggingface/diffusers": "Diffusers",
    "Lightning-AI/litgpt": "LitGPT",
}


def accepted(result):
    count = (result or {}).get("accepted")
    return bool(result and result.get("measured") is True
                and isinstance(count, int) and count > 0)

def lower(baseline, candidate):
    return (isinstance(baseline, (int, float)) and isinstance(candidate, (int, float))
            and math.isfinite(baseline) and math.isfinite(candidate)
            and baseline > 0 and 0 <= candidate < baseline)


def entries(source):
    summary = json.loads(source.read_text(encoding="utf-8"))
    if summary.get("schema") != "top-k-dual-molab-benchmark-v1":
        return []
    results = []
    for row in summary["results"]:
        kernel = row["kernel"]
        architecture = row["architecture"]
        kernel_result = kernel.get("result") or {}
        architecture_result = architecture.get("result") or {}
        kernel_win = (kernel.get("status") == "done" and accepted(kernel_result)
                      and lower(kernel_result.get("baseline_ms"), kernel_result.get("candidate_ms")))
        architecture_win = (
            architecture.get("status") == "done" and accepted(architecture_result)
            and architecture_result.get("architecture_changed") is True
            and lower(architecture_result.get("baseline_val_loss"),
                      architecture_result.get("candidate_val_loss"))
        )
        if not (kernel_win or architecture_win):
            continue
        repo = row["repo"]
        entry = {"name": repo, "short": SHORT_NAMES.get(repo, repo.split("/")[-1])}
        if kernel_win:
            entry["kernel"] = {
                "baseline_ms": kernel_result["baseline_ms"],
                "candidate_ms": kernel_result["candidate_ms"],
                "improvement_pct": round(100 * (1 - kernel_result["candidate_ms"] / kernel_result["baseline_ms"]), 3),
                "generations": kernel["generations"],
            }
        if architecture_win:
            entry["architecture"] = {
                "baseline_val_loss": architecture_result["baseline_val_loss"],
                "candidate_val_loss": architecture_result["candidate_val_loss"],
                "improvement_pct": round(100 * (1 - architecture_result["candidate_val_loss"] / architecture_result["baseline_val_loss"]), 3),
                "budget_s": architecture_result["final_budget_s"],
                "caveat": (
                    "loss-floor" if architecture_result["baseline_val_loss"] < 1e-7
                    else "near-zero" if architecture_result["baseline_val_loss"] < 1e-3
                    else None
                ),
            }
        entry["evidence_url"] = "https://github.com/flappybird1084/top-k/tree/main/" + source.parent.relative_to(ROOT).as_posix()
        entry["replay_available"] = (source.parent / "evidence.json").is_file()
        results.append(entry)
    return results


def payload(sources=None):
    # ISO-date report folders sort chronologically. Keep a complete comparison
    # from one report per project; never combine incompatible protocols.
    by_repo = {}
    def report_order(path):
        date = re.search(r"\d{4}-\d{2}-\d{2}", path.parent.name)
        return (date.group() if date else "", path.as_posix())
    for source in sorted(sources if sources is not None else RESULTS.glob("*/summary.json"), key=report_order):
        for entry in entries(source):
            by_repo[entry["name"]] = entry
    return {
        "evidence_url": "https://github.com/flappybird1084/top-k/tree/main/benchmarks/results",
        "results": list(by_repo.values()),
    }


def build(sources=None):
    safe_json = json.dumps(payload(sources), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return "// Generated from published benchmark reports under benchmarks/results/.\nwindow.TOPK_BENCHMARKS=" + safe_json + ";\n"


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
