import json

import pytest

from scripts.report_molab_benchmark import collect


def test_report_preserves_full_denominator_and_measured_outcomes(tmp_path):
    manifest = tmp_path / "repos.json"
    manifest.write_text(json.dumps([
        {"repo": "one/train", "model": "vision", "sha": "a" * 40},
        {"repo": "two/train", "model": "language", "sha": "b" * 40},
        {"repo": "three/train", "model": "speech", "sha": "c" * 40},
    ]))
    output = tmp_path / "runs"
    first = output / "one__train"
    first.mkdir(parents=True)
    (first / "state.json").write_text(json.dumps({
        "index": 1, "repo": "one/train", "source_sha": "a" * 40,
        "status": "done", "llm": "wandb:Qwen/test",
        "result": {"measured": True, "improvement_pct": 7.5},
    }))
    second = output / "two__train"
    second.mkdir()
    (second / "state.json").write_text(json.dumps({
        "index": 2, "repo": "two/train", "source_sha": "b" * 40,
        "status": "failed", "result": {"measured": False},
    }))

    report = collect(manifest, output)

    assert (report["repositories_total"], report["done"], report["failed"],
            report["pending"], report["measured"], report["improved"]) == (3, 1, 1, 1, 1, 1)
    assert report["results"][0]["llm"] == "wandb:Qwen/test"
    assert report["results"][2]["status"] == "pending"


def test_report_rejects_a_stale_commit(tmp_path):
    manifest = tmp_path / "repos.json"
    manifest.write_text(json.dumps([
        {"repo": "one/train", "model": "vision", "sha": "a" * 40},
    ]))
    run = tmp_path / "runs" / "one__train"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({
        "index": 1, "repo": "one/train", "source_sha": "b" * 40,
        "status": "done", "result": {"measured": True},
    }))

    with pytest.raises(ValueError, match="does not match manifest"):
        collect(manifest, tmp_path / "runs")
