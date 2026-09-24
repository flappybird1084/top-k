"""Benchmark reporting must never turn an attempted run into a claimed win."""

import importlib.util
import http.client
import io
import json
import os
import sqlite3
import subprocess
import sys
import time
import types
import urllib.request
import urllib.error
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


def test_github_commit_url_checks_out_pinned_sha(monkeypatch, tmp_path):
    from kernelevo.adapter_writer import fetch_repo

    sha = "a" * 40
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return types.SimpleNamespace(stdout=sha + "\n")
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    messages = []
    fetch_repo(f"https://github.com/example/project/commit/{sha}",
               str(tmp_path), log=messages.append)
    assert commands[0][-3:] == ["--", "https://github.com/example/project", str(tmp_path / "repo")]
    assert commands[1][-4:] == ["--depth", "1", "origin", sha]
    assert commands[2][-2:] == ["--detach", sha]
    assert messages[-1] == f"[adapter] source commit {sha}"

    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="differs from requested"):
        fetch_repo(f"https://github.com/example/project/commit/{'b' * 40}",
                   str(tmp_path))

    with pytest.raises(ValueError, match="40-character SHA"):
        fetch_repo("https://github.com/example/project/commit/abc", str(tmp_path))


def test_molab_retry_refuses_a_still_running_gpu_job():
    from scripts.run_molab_benchmark import previous_job_finished

    class Client:
        result = "RUNNING\n"
        def run(self, code):
            assert "/tmp/kevo_12345678" in code
            return True, self.result, ""

    client = Client()
    assert not previous_job_finished(client, "12345678" + "a" * 24)
    client.result = "EXITED\n"
    assert previous_job_finished(client, "12345678" + "a" * 24)


def test_generated_adapter_receives_nested_batches_on_model_device(tmp_path):
    import torch
    from kernelevo.ingest import load_adapter

    path = tmp_path / "adapter.py"
    path.write_text("""import torch
def build_model(): return torch.nn.Linear(1, 1)
def get_dataloader(split): return []
def loss_fn(model, batch):
    return batch['x'][0].device.type, batch['x'][1][0].device.type
""")
    adapter, _ = load_adapter(str(path))
    model = torch.nn.Linear(1, 1, device="meta")
    batch = {"x": [torch.zeros(1), (torch.ones(1),)]}
    assert adapter.loss_fn(model, batch) == ("meta", "meta")
    from kernelevo.bench import samples_per_batch
    assert samples_per_batch(batch) == 1


def test_ingest_rejects_loss_without_model_gradient():
    import torch
    from kernelevo.ingest import check_training_signal

    model = torch.nn.Linear(1, 1)
    with pytest.raises(SystemExit, match="detached"):
        check_training_signal(torch.tensor(0.0, requires_grad=True), model)
    with pytest.raises(SystemExit, match="only zero"):
        check_training_signal((model.weight * 0).sum(), model)
    check_training_signal((model.weight ** 2).sum(), model)


def test_repository_job_does_not_inherit_operator_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "operator-secret")
    monkeypatch.setenv("WANDB_API_KEY", "inference-key")
    env = benchmark.job_env("http://127.0.0.1:1234/v1", tmp_path)
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "WANDB_API_KEY" not in env
    assert env["WANDB_INFERENCE_API_KEY"] != "inference-key"
    assert env["HOME"] == str(tmp_path)


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


def test_inference_relay_marks_provider_credit_error(monkeypatch):
    from scripts.inference_proxy import InferenceRelay

    monkeypatch.setenv("WANDB_API_KEY", "private-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "benchmark")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {},
                                     io.BytesIO(b'{"error":{"code":"insufficient_quota"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    relay = InferenceRelay("deepseek-ai/DeepSeek-V4-Flash-0731", 3.0)
    with relay.serving() as url:
        target = urlsplit(url)
        client = http.client.HTTPConnection(target.hostname, target.port)
        client.request("POST", "/v1/chat/completions", body=json.dumps({
            "model": relay.model, "max_tokens": 100, "messages": []}),
            headers={"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 402
        response.read()
        client.close()
    assert relay.credits_exhausted


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
