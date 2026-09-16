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
    """The wedge held the slot forever: no drop, no re-submission, relay dead
    at 8 pending. Fixed behavior: the slot is dropped after MAX_DELIVERY_FAILS
    and a still-pending remote request can be re-submitted afterwards."""
    relay = _make_relay(lambda req: dict(text="hi", input_tokens=1, output_tokens=1))
    client = _FailingClient(fail=True)
    lines = []
    req = [dict(id="a" * 32, kind="test", messages=[])]
    for _ in range(relay.MAX_DELIVERY_FAILS + 3):
        relay.service(client, "/tmp/w", req, lines.append)
        time.sleep(0.05)  # let the worker future resolve
    assert any("undeliverable" in l for l in lines), "slot never dropped"
    submits = [l for l in lines if "Request sent" in l]
    assert len(submits) >= 2, "dropped request could not be re-submitted"


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
