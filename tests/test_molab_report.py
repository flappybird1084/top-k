import json
import sqlite3

import pytest

from scripts.report_molab_benchmark import collect, evidence


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
        "result": {"measured": True, "accepted": 1, "improvement_pct": 7.5},
    }))
    second = output / "two__train"
    second.mkdir()
    (second / "state.json").write_text(json.dumps({
        "index": 2, "repo": "two/train", "source_sha": "b" * 40,
        "status": "failed", "exit_code": 1,
        "result": {"measured": True, "improvement_pct": 20.0},
    }))

    report = collect(manifest, output)

    assert (report["repositories_total"], report["done"], report["failed"],
            report["pending"], report["measured"], report["improved"]) == (3, 1, 1, 1, 1, 1)
    assert report["results"][0]["llm"] == "wandb:Qwen/test"
    assert report["results"][1]["result"] == {"measured": False}
    assert report["results"][1]["reason"].startswith("dispatch exited 1")
    assert report["results"][2]["status"] == "pending"

    archive = first / "attempt-0" / "artifacts" / "archive.sqlite"
    archive.parent.mkdir(parents=True)
    with sqlite3.connect(archive) as db:
        db.execute("CREATE TABLE candidates (id INTEGER, generation INTEGER, "
                   "gate_reached INTEGER, compile_ok INTEGER, correct_ok INTEGER, "
                   "accepted INTEGER, step_time_ms REAL, incumbent_step_time_ms REAL, "
                   "code_path TEXT)")
        db.execute("INSERT INTO candidates VALUES (1, 1, 4, 1, 1, 1, 9, 10, 'private.py')")
    gates = evidence(report, output)["repositories"][0]["candidate_gates"]
    assert gates[0]["gate_reached"] == 4
    assert "code_path" not in gates[0]
    broken = second / "attempt-0" / "artifacts" / "archive.sqlite"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a database")
    records = evidence(report, output)["repositories"]
    assert records[1]["archive_evidence"] == "unavailable or incompatible"
    assert len(records) == 3


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


def test_recipe_report_preserves_loss_evidence_and_mode(tmp_path):
    manifest = tmp_path / "repos.json"
    manifest.write_text(json.dumps([{
        "repo": "one/train", "model": "vision", "sha": "a" * 40,
    }]))
    output = tmp_path / "runs"
    run = output / "one__train"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({
        "index": 1, "repo": "one/train", "source_sha": "a" * 40,
        "mode": "recipe", "status": "done", "attempt": 1,
        "llm": "wandb:Qwen/test",
        "result": {"measured": True, "accepted": 1, "baseline_val_loss": 1.2,
                   "winner_val_loss": 1.0, "improvement_pct": 16.667},
    }))
    archive = run / "attempt-1" / "artifacts" / "archive.sqlite"
    archive.parent.mkdir(parents=True)
    with sqlite3.connect(archive) as db:
        db.execute("CREATE TABLE candidates (id INTEGER, generation INTEGER, "
                   "gate_reached INTEGER, compile_ok INTEGER, correct_ok INTEGER, "
                   "accepted INTEGER, val_loss REAL, train_secs REAL, phase TEXT, "
                   "model_params INTEGER, code_path TEXT)")
        db.execute("INSERT INTO candidates VALUES "
                   "(1, 1, 4, 1, 1, 1, 1.0, 120, 'finals', 1000, 'private.py')")

    report = collect(manifest, output, "recipe")
    assert report["mode"] == "recipe"
    assert report["improved"] == 1
    gates = evidence(report, output)["repositories"][0]["candidate_gates"]
    assert gates[0]["val_loss"] == 1.0
    assert "code_path" not in gates[0]
    with pytest.raises(ValueError, match="does not match manifest"):
        collect(manifest, output, "kernel")
