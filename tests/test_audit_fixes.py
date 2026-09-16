"""Regression tests for findings/06-audit.md priority fixes."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---- finding 27: extract_code picks the answer, not the biggest block ----

def test_extract_code_prefers_python_tag_over_longer_reference():
    from kernelevo.llm import extract_code
    text = ("For reference, the original was:\n```\n" + "old = 1\n" * 50 +
            "```\nHere is the corrected file:\n```python\nnew = 2\n```")
    assert extract_code(text) == "new = 2\n"


def test_extract_code_takes_last_block():
    from kernelevo.llm import extract_code
    text = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
    assert extract_code(text) == "second = 2\n"


def test_extract_code_single_line_fence():
    from kernelevo.llm import extract_code
    assert extract_code("```x = 1```") == "x = 1\n"


def test_extract_code_bare_text_strips_stray_fences():
    from kernelevo.llm import extract_code
    assert extract_code("``x = 1``") == "x = 1\n"


# ---- finding 24: the relay releases its slot on every path ----

class _FailingClient:
    def __init__(self, fail=True):
        self.fail = fail
        self.calls = 0

    def run(self, code):
        self.calls += 1
        if self.fail:
            return False, "", "delivery broke"
        return True, "OAUTH-OK", ""


def _make_relay(answers):
    from kernelevo.codex_oauth import Relay

    class TestRelay(Relay):
        label = "Test"
        worker = staticmethod(lambda request: answers(request))
    return TestRelay()


def test_relay_releases_slot_after_delivery_failures():
    """The wedge held the slot forever: no drop, relay dead at 8 pending. The
    slot is dropped after MAX_DELIVERY_FAILS — and the dropped request is NOT
    picked up again, because re-serving it would pay the provider a second
    time for an answer nobody collected (finding 8)."""
    relay = _make_relay(lambda req: dict(text="hi", input_tokens=1, output_tokens=1))
    client = _FailingClient(fail=True)
    lines = []
    req = [dict(id="a" * 32, kind="test", messages=[])]
    for _ in range(relay.MAX_DELIVERY_FAILS + 3):
        relay.service(client, "/tmp/w", req, lines.append)
        time.sleep(0.05)  # let the worker future resolve
    assert any("undeliverable" in l for l in lines), "slot never dropped"
    assert relay.pending == {}, "slot never released"
    assert [l for l in lines if "Request sent" in l] == [
        "[agent] Request sent through local Test OAuth session."], "request was re-billed"
    # a different request id still gets served: the relay is not wedged
    relay.service(client, "/tmp/w", [dict(id="f" * 32, kind="test", messages=[])],
                  lines.append)
    assert len([l for l in lines if "Request sent" in l]) == 2


def test_relay_does_not_reserve_the_same_request_twice(tmp_path):
    """The sandbox's request file stays on the notebook until its own wait
    ends. Answering it once and then seeing it again must not re-run it."""
    calls = []

    def answer(request):
        calls.append(request)
        return dict(text="hi", input_tokens=1, output_tokens=1)

    relay = _make_relay(answer)
    client = _FailingClient(fail=False)
    req = [dict(id="e" * 32, kind="test", messages=[])]
    lines = []
    deadline = time.time() + 2
    while not any("delivered" in l for l in lines) and time.time() < deadline:
        relay.service(client, "/tmp/w", req, lines.append)
        time.sleep(0.02)
    assert any("delivered" in l for l in lines)
    for _ in range(3):
        relay.service(client, "/tmp/w", req, lines.append)
    assert len(calls) == 1


def test_relay_releases_slot_on_worker_exception_then_delivers_error():
    """A worker exception must be logged, delivered to the sandbox as an
    error answer, and release the slot (re-submission afterwards is fine —
    the remote request file persists until answered)."""
    def boom(req):
        raise RuntimeError("provider down")
    relay = _make_relay(boom)
    client = _FailingClient(fail=False)
    lines = []
    req = [dict(id="b" * 32, kind="test", messages=[])]
    deadline = time.time() + 2
    while not any("delivered" in l for l in lines) and time.time() < deadline:
        relay.service(client, "/tmp/w", req, lines.append)
        time.sleep(0.02)
    assert any("failed" in l for l in lines), "worker exception never logged"
    assert any("delivered" in l for l in lines), "error answer never delivered"


def test_relay_expires_stale_requests():
    relay = _make_relay(lambda req: time.sleep(60))
    relay.EXPIRY_S = 0
    client = _FailingClient()
    req = [dict(id="c" * 32, kind="test", messages=[])]
    relay.service(client, "/tmp/w", req, lambda l: None)   # submit
    time.sleep(0.01)
    relay.service(client, "/tmp/w", [], lambda l: None)    # expiry sweep
    assert relay.pending == {}


# ---- finding 25: judges_pool worker survives a corrupt job ----

def test_judges_pool_worker_survives_bad_job(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    conn = tmp_path / "conn.json"
    conn.write_text(json.dumps([{"url": "https://x", "token": "t"}]))
    ran = []

    def load_job(jid):
        if jid == "badjob":
            raise ValueError("corrupt job.json")
        return dict(id=jid, status="queued", mode="recipe")

    from kernelevo.judges_pool import NotebookPool
    pool = NotebookPool(str(conn), run_job=ran.append, load_job=load_job,
                        save_job=lambda j: None, expires_at=time.time() + 3600)
    pool.queues[0].put("badjob")
    pool.queues[0].put("goodjob")
    pool.start()
    pool.queues[0].join()          # would hang forever if the worker thread died
    assert ran == ["goodjob"]
    assert pool.status()[0]["busy"] is False


# ---- finding 17: recipe archives get a stop_reason ----

def test_set_stop_reason_roundtrip(tmp_path):
    from kernelevo.archive import Archive
    a = Archive(str(tmp_path / "a.sqlite"))
    a.db.execute("INSERT INTO models (name) VALUES ('m')")
    gid = a.start_generation(1)
    a.set_stop_reason(gid, "recipe_complete")
    row = a.db.execute("SELECT stop_reason FROM generations WHERE id=?", (gid,)).fetchone()
    assert row["stop_reason"] == "recipe_complete"


# ---- review fix: the search relay stamps the per-run token the policy needs ----

def test_search_relay_request_carries_the_run_token(tmp_path, monkeypatch):
    """The dispatcher's RelayPolicy refuses any search that does not carry the
    active run's relay token; the real writer must stamp it. Regression: the
    writer emitted {query, n} only, so once the token gate landed every relayed
    search — operator and opted-in alike — was refused. Exercises the actual
    writer, not a hand-stamped synthetic request."""
    import threading
    from kernelevo import websearch
    from kernelevo.relay_policy import OwnerLedger, RelayPolicy

    monkeypatch.setenv("KEVO_RELAY_DIR", str(tmp_path))
    monkeypatch.setenv("KEVO_RELAY_TOKEN", "run-secret")
    monkeypatch.setattr(websearch, "RELAY_TIMEOUT_S", 5)
    websearch._cache.clear()

    worker = threading.Thread(target=websearch._relay_search, args=("triton softmax", 3))
    worker.start()
    try:
        deadline = time.time() + 3
        req_file = None
        while time.time() < deadline and req_file is None:
            hits = list(tmp_path.glob("*.req.json"))
            req_file = hits[0] if hits else None
            if req_file is None:
                time.sleep(0.02)
        assert req_file is not None, "the writer never dropped a request file"
        request = json.loads(req_file.read_text())
        request["id"] = req_file.name[: -len(".req.json")]
        assert request.get("relay_token") == "run-secret"

        # the dispatcher's policy must now ADMIT the stamped request
        monkeypatch.setenv("KEVO_ALLOW_OPERATOR_SEARCH_RELAY", "1")
        p = RelayPolicy({"id": "j" * 32, "visitor": "github:7"}, "run-secret",
                        OwnerLedger(tmp_path / "ledger.json"))
        assert p.check_search(request) is None, "a correctly-stamped search was refused"

        # unblock the writer so its thread exits cleanly
        (tmp_path / (request["id"] + ".res.json")).write_text(json.dumps([]))
    finally:
        worker.join(timeout=5)
    assert not worker.is_alive()


# ---- review fix: a visitor's job never runs on the operator's own box ----

def test_visitor_job_refuses_local_execution(tmp_path, monkeypatch):
    """The local execution branch runs search.py under the server's full
    environment (operator ANTHROPIC_API_KEY / WANDB_API_KEY included). A public
    (visitor-owned) job must be refused there rather than handed the operator's
    credentials. Regression: the branch ran it without scoping the environment."""
    from unittest.mock import patch
    import web

    monkeypatch.setattr(web, "JOBS_DIR", str(tmp_path))
    jid = "vjob"
    (tmp_path / jid).mkdir()
    web.save_job(dict(id=jid, execution_target="local", visitor="github:7",
                      profile="DEV", adapter="adapters/lm.py",
                      stage="starting", status="queued"))

    with patch.object(web.subprocess, "Popen") as popen:
        try:
            web._run_job(jid)
            refusal = None
        except RuntimeError as e:
            refusal = str(e)
    assert refusal and "local execution is not available" in refusal
    popen.assert_not_called()
    assert web.load_job(jid)["status"] != "running"


# ============ PR #5 Copilot review findings ============

def test_relay_bills_usage_once_across_delivery_retries(tmp_path, monkeypatch):
    """codex_oauth finding: a failed delivery keeps the request pending, so the
    next poll re-enters the completed branch with the same future. Usage must be
    billed once, not once per delivery attempt."""
    monkeypatch.setenv("KEVO_ALLOW_OPERATOR_LLM_RELAY", "1")
    from kernelevo.relay_policy import RelayPolicy, OwnerLedger

    class _Flaky:
        def __init__(self, fail):
            self.fail, self.calls = fail, 0

        def run(self, code):
            self.calls += 1
            return (False, "", "boom") if self.calls <= self.fail else (True, "OAUTH-OK", "")

    relay = _make_relay(lambda req: dict(text="hi", input_tokens=3, output_tokens=4))
    client = _Flaky(fail=2)   # two failed deliveries, then success (< MAX_DELIVERY_FAILS)
    policy = RelayPolicy({"id": "c" * 32, "visitor": "github:7"}, "relay-secret",
                         OwnerLedger(tmp_path / "ledger.json"))
    req = [dict(id="c" * 32, kind="test", relay_token="relay-secret",
                messages=[{"role": "user", "content": "hi"}])]
    lines = []
    deadline = time.time() + 2
    while client.calls < 3 and time.time() < deadline:
        relay.service(client, "/tmp/w", req, lines.append, policy=policy)
        time.sleep(0.02)
    assert client.calls >= 3, "delivery never retried through to success"
    assert policy.tokens == 7, f"usage billed {policy.tokens}, expected 7 (billed once)"


def test_relay_fails_closed_when_the_owner_ledger_cannot_persist(tmp_path, monkeypatch):
    """relay_policy finding: a filesystem failure writing the per-owner ledger
    must stop the relay, not be read as 'no cap crossed' — which would silently
    disable the rolling per-account budget (a fail-open)."""
    monkeypatch.setenv("KEVO_ALLOW_OPERATOR_LLM_RELAY", "1")
    monkeypatch.setenv("KEVO_ALLOW_OPERATOR_SEARCH_RELAY", "1")
    from kernelevo.relay_policy import RelayPolicy, OwnerLedger, LEDGER_UNAVAILABLE
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")     # its child path can never be created
    ledger = OwnerLedger(blocker / "ledger.json")
    assert ledger.charge("github:7", requests=1) == LEDGER_UNAVAILABLE
    ask = dict(relay_token="relay-secret", messages=[{"role": "user", "content": "x"}])
    p = RelayPolicy({"id": "a" * 32, "visitor": "github:7"}, "relay-secret", ledger)
    assert "accounting is unavailable" in (p.check_llm(ask) or "")
    p2 = RelayPolicy({"id": "b" * 32, "visitor": "github:7"}, "relay-secret", ledger)
    assert "accounting is unavailable" in (
        p2.check_search(dict(relay_token="relay-secret", query="triton")) or "")


def test_public_discovery_skips_the_operator_oauth_session(tmp_path, monkeypatch):
    """ui_server finding: a visitor's repo must never trigger the operator's
    Codex OAuth + live web search discovery. That runs before the job reaches
    the visitor's notebook and bypasses every relay cap; a public job should go
    straight to awaiting_data instead."""
    import web
    import ui_server
    from kernelevo import repo_discovery
    monkeypatch.setattr(web, "JOBS_DIR", str(tmp_path))
    jid = "d" * 32
    (tmp_path / jid).mkdir()
    (tmp_path / jid / "job.json").write_text(json.dumps(
        {"id": jid, "status": "exploring", "repo": "https://github.com/x/y",
         "visitor": "github:7"}))

    def _boom(repo):
        raise AssertionError("operator discovery ran for a visitor job")
    monkeypatch.setattr(repo_discovery, "discover", _boom)
    ui_server.discover_repository(jid)
    job = json.loads((tmp_path / jid / "job.json").read_text())
    assert job["status"] == "awaiting_data"


def test_public_pool_forces_molab_execution_target(monkeypatch):
    """judges_pool finding: a queued public run always targets the visitor's own
    notebook, even if a stray KEVO_UI_TARGET=local leaked into the job."""
    import web
    from kernelevo.judges_pool import NotebookPool
    monkeypatch.setattr(web, "list_jobs", lambda: [])
    store = {"j" * 32: {"id": "j" * 32, "mode": "recipe", "execution_target": "local",
                        "molab": {"connection": "--token t",
                                  "notebook_url": "https://n.sb.molab.run"}}}
    pool = NotebookPool(None, run_job=lambda jid: None,
                        load_job=lambda jid: dict(store[jid]),
                        save_job=lambda job: store.__setitem__(job["id"], job),
                        expires_at=time.time() + 3600, concurrency=1)
    pool.put("j" * 32)
    assert store["j" * 32]["execution_target"] == "molab"
