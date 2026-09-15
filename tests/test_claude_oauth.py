"""claude_oauth provider: local CLI parsing, relay round-trip, pool wiring."""
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _fake_claude(tmp_path, payload, monkeypatch):
    """Put a fake `claude` executable on PATH that prints one JSON result."""
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\ncat > /dev/null\necho '" + json.dumps(payload) + "'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


def test_complete_local_parses_result(tmp_path, monkeypatch):
    from kernelevo import claude_oauth
    _fake_claude(tmp_path, {"type": "result", "subtype": "success",
                            "result": "hello from claude",
                            "usage": {"input_tokens": 12, "output_tokens": 5}}, monkeypatch)
    out = claude_oauth.complete_local({"messages": [{"role": "user", "content": "hi"}]})
    assert out == {"text": "hello from claude", "input_tokens": 12, "output_tokens": 5}


def test_complete_local_rejects_empty(tmp_path, monkeypatch):
    from kernelevo import claude_oauth
    _fake_claude(tmp_path, {"type": "result", "subtype": "success", "result": ""}, monkeypatch)
    try:
        claude_oauth.complete_local({"messages": []})
        assert False, "empty response should raise"
    except RuntimeError:
        pass


def test_relay_request_kind(tmp_path, monkeypatch):
    """With KEVO_RELAY_DIR set, complete() writes a claude_oauth-kind request
    and returns the response file's payload."""
    from kernelevo import claude_oauth
    monkeypatch.setenv("KEVO_RELAY_DIR", str(tmp_path))
    import threading

    def answer():
        import time
        for _ in range(100):
            reqs = list(tmp_path.glob("*.req.json"))
            if reqs:
                body = json.loads(reqs[0].read_text())
                assert body["kind"] == "claude_oauth"
                rid = reqs[0].name[:-len(".req.json")]
                (tmp_path / (rid + ".res.json")).write_text(
                    json.dumps({"text": "relayed", "input_tokens": 1, "output_tokens": 2}))
                return
            time.sleep(0.05)

    t = threading.Thread(target=answer)
    t.start()
    out = claude_oauth.complete({"messages": [{"role": "user", "content": "q"}]})
    t.join()
    assert out["text"] == "relayed"


def test_llm_pool_constructs_provider():
    import config
    from kernelevo.llm import ClaudeOAuthLLM, LLMPool
    cfg = config.load("DEV")
    cfg["llm"] = "claude_oauth:sonnet"
    cfg["planner_llm"] = cfg["subagent_llm"] = cfg["curator_llm"] = None
    pool = LLMPool(cfg)
    assert isinstance(pool.planner, ClaudeOAuthLLM)
    assert pool.planner.model == "sonnet"
    assert pool.planner.usage_usd() == 0.0


def test_relay_class_uses_claude_worker():
    from kernelevo import claude_oauth, codex_oauth
    assert issubclass(claude_oauth.Relay, codex_oauth.Relay)
    assert claude_oauth.Relay.label == "Claude"
    assert claude_oauth.Relay.worker is claude_oauth.complete_local
    assert codex_oauth.Relay.worker is codex_oauth.complete_local


def test_unfence():
    from kernelevo.claude_oauth import _unfence
    assert _unfence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _unfence('{"a": 1}') == '{"a": 1}'
    assert _unfence('```\n{"a": 1}\n```') == '{"a": 1}'
