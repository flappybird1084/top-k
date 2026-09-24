"""Benchmark reporting must never turn an attempted run into a claimed win."""

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_repo_benchmark.py"
spec = importlib.util.spec_from_file_location("run_repo_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_manifest_is_25_distinct_repositories():
    rows = benchmark.manifest_rows(benchmark.DEFAULT_MANIFEST)
    assert len(rows) == 25
    assert len({row["repo"] for row in rows}) == 25


def test_result_uses_paired_baseline_and_rejects_unmeasured(tmp_path):
    path = tmp_path / "archive.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE candidates (generation INT, step_time_ms REAL, "
                   "incumbent_step_time_ms REAL, accepted INT, gate_reached INT, "
                   "phase TEXT, val_loss REAL, train_secs REAL)")
        db.execute("INSERT INTO candidates VALUES (1, 8, 10, 1, 4, NULL, NULL, NULL)")
        db.execute("INSERT INTO candidates VALUES (1, 7, 8, 1, 4, NULL, NULL, NULL)")
        db.execute("INSERT INTO candidates VALUES (1, 1, 10, 0, 2, NULL, NULL, NULL)")
    result = benchmark.archive_result(path, "kernel")
    assert result["improvement_pct"] == 30
    assert result["baseline_ms"] == 10
    assert result["candidate_ms"] == 7
    assert result["accepted"] == 2
    assert not benchmark.archive_result(tmp_path / "missing.sqlite", "kernel")["measured"]


def test_recipe_result_handles_zero_baseline_loss(tmp_path):
    path = tmp_path / "recipe.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE candidates (phase TEXT, val_loss REAL, "
                   "train_secs REAL, accepted INT)")
        db.execute("INSERT INTO candidates VALUES ('baseline', 0, 10, 1)")
        db.execute("INSERT INTO candidates VALUES ('finals', 0, 10, 1)")
    result = benchmark.archive_result(path, "recipe")
    assert result["measured"]
    assert "improvement_pct" not in result


def test_manifest_rejects_duplicate_and_non_github_identifier(tmp_path):
    path = tmp_path / "repos.json"
    path.write_text(json.dumps([{"repo": "one/two"}, {"repo": "one/two"}]))
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.manifest_rows(path)
    path.write_text(json.dumps([{"repo": "https://example.com/repo"}]))
    with pytest.raises(ValueError, match="invalid"):
        benchmark.manifest_rows(path)


def test_repository_job_does_not_inherit_aws_operator_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "operator-secret")
    monkeypatch.setenv("WANDB_API_KEY", "inference-key")
    env = benchmark.job_env()
    assert "AWS_ACCESS_KEY_ID" not in env
    assert env["WANDB_API_KEY"] == "inference-key"


def test_online_summary_keeps_failures_in_denominator(monkeypatch):
    captured = {}

    class FakeRun:
        url = "https://wandb.ai/team/project/runs/summary"
        summary = {}

        def log(self, data):
            captured.update(data)

        def finish(self):
            pass

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kw: FakeRun(),
        Table=lambda **kw: kw))
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "project")
    args = types.SimpleNamespace(mode="kernel", profile="DEV", model="example",
                                 manifest="manifest", generations=2, spend_cap=10)
    results = [{"repo": "a/b", "status": "done",
                "result": {"measured": True, "improvement_pct": 4.0}},
               {"repo": "c/d", "status": "failed",
                "result": {"measured": False}, "reason": "ingest failed"}]
    url = benchmark.publish_summary(results, args)
    assert url.endswith("/summary")
    assert len(captured["repositories"]["data"]) == 2
    assert captured["repositories"]["data"][1][10] == "ingest failed"
