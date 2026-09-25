import json

from scripts.report_dual_molab_benchmark import combined, numeric_evidence


def test_dual_report_keeps_modes_and_full_denominator(tmp_path):
    manifest = tmp_path / "repos.json"
    manifest.write_text(json.dumps([
        {"repo": "one/train", "model": "vision", "sha": "a" * 40},
        {"repo": "two/train", "model": "language", "sha": "b" * 40},
    ]))
    kernel_output = tmp_path / "kernel"
    recipe_output = tmp_path / "recipe"
    for directory, mode, metric in (
        (kernel_output, "kernel", {"measured": True, "accepted": 1,
                                   "baseline_ms": 10, "candidate_ms": 8,
                                   "improvement_pct": 20}),
        (recipe_output, "recipe", {"measured": True, "accepted": 1,
                                   "baseline_val_loss": 1.0, "candidate_val_loss": 0.9,
                                   "improvement_pct": 10}),
    ):
        run = directory / "one__train"
        run.mkdir(parents=True)
        (run / "state.json").write_text(json.dumps({
            "index": 1, "repo": "one/train", "source_sha": "a" * 40,
            "mode": mode, "status": "done", "llm": "wandb:Qwen/test",
            "result": metric,
        }))

    report = combined(manifest, kernel_output, recipe_output)

    assert report["repositories_total"] == 2
    assert report["kernel"]["improved"] == 1
    assert report["architecture"]["improved"] == 1
    assert report["kernel"]["pending"] == 1
    assert report["architecture"]["pending"] == 1
    assert report["results"][0]["kernel"]["result"]["baseline_ms"] == 10
    assert report["results"][0]["architecture"]["result"]["baseline_val_loss"] == 1.0
    assert report["results"][1]["kernel"]["status"] == "pending"
    numeric = numeric_evidence(report, kernel_output, recipe_output)
    assert len(numeric["repositories"]) == 2
    assert numeric["repositories"][0]["kernel"]["mode"] == "kernel"
    assert numeric["repositories"][0]["architecture"]["mode"] == "recipe"
