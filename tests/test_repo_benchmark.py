"""Benchmark reporting must never turn an attempted run into a claimed win."""

import importlib.util
import http.client
import json
import os
import sqlite3
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

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
    env = benchmark.job_env("http://127.0.0.1:1234/v1")
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "WANDB_API_KEY" not in env
    assert env["WANDB_INFERENCE_API_KEY"] != "inference-key"


def test_inference_relay_limits_model_and_keeps_upstream_key(monkeypatch):
    from scripts.inference_proxy import InferenceRelay

    monkeypatch.setenv("WANDB_API_KEY", "private-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "benchmark")
    seen = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "usage": {"prompt_tokens": 100,
                                         "completion_tokens": 50}}).encode()

    def fake_urlopen(request, timeout):
        seen.append(request.get_header("Authorization"))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    relay = InferenceRelay("deepseek-ai/DeepSeek-V4-Flash-0731", 3.0)
    with relay.serving() as url:
        target = urlsplit(url)
        client = http.client.HTTPConnection(target.hostname, target.port)
        client.request("POST", "/v1/chat/completions", body=json.dumps({
            "model": "wrong", "max_tokens": 100, "messages": []}),
            headers={"Content-Type": "application/json"})
        first = client.getresponse()
        assert first.status == 400
        first.read()
        client.request("POST", "/v1/chat/completions", body=json.dumps({
            "model": relay.model, "max_tokens": 100, "messages": []}),
            headers={"Content-Type": "application/json"})
        second = client.getresponse()
        assert second.status == 200
        second.read()
        client.close()
    assert seen == ["Bearer private-key"]
    assert relay.calls == 1
    assert relay.spent_usd > 0


@pytest.mark.skipif(os.name == "nt" or not Path("/proc").exists(),
                    reason="requires Linux process groups")
def test_timeout_stops_gpu_child_after_parent_exits(tmp_path):
    pidfile = tmp_path / "child.pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent_code = ("import subprocess,time; "
                   f"p=subprocess.Popen([{sys.executable!r},'-c',{child_code!r}]); "
                   f"open({str(pidfile)!r},'w').write(str(p.pid)); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    try:
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(0.01)
        assert pidfile.exists()
        child_pid = int(pidfile.read_text())
        benchmark.stop_process_tree(proc, grace_seconds=0.3)
        stat = Path(f"/proc/{child_pid}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait()


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
