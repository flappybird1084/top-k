"""Front-page Settings dialog: per-job overrides with env fallback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import ui_server


def test_sanitize_accepts_valid_settings():
    out = ui_server.sanitize_settings({"settings": dict(
        llm="claude_oauth:sonnet", profile="DEV", spend_cap=25,
        max_debug_turns=4, molab_connection="Connect at https://sb-x.sb.molab.run --token abc123",
        arch_gens=1, arch_cands=4, arch_secs=30, hp_gens=1, finals_k=1,
        finals_secs=120, parallelism=4, loss_margin=0.005)})
    assert out["llm"] == "claude_oauth:sonnet"
    assert out["profile"] == "DEV"
    assert out["spend_cap"] == 25.0
    assert out["max_debug_turns"] == 4
    assert "molab.run" in out["molab_connection"]
    r = out["recipe"]
    assert r["phases"][0] == dict(kind="architecture", generations=1,
                                  candidates=4, train_seconds=30)
    assert r["phases"][2]["generations"] == 1
    assert r["finals_top_k"] == 1 and r["finals_train_seconds"] == 120
    assert r["subagent_parallelism"] == 4 and r["loss_margin_rel"] == 0.005


def test_sanitize_rejects_bad_values():
    out = ui_server.sanitize_settings({"settings": dict(
        llm="rm -rf /", profile="EVIL", spend_cap=99999,
        max_debug_turns="lots", arch_gens=-3, molab_connection="x" * 5000)})
    assert out == {}
    assert ui_server.sanitize_settings({}) == {}
    assert ui_server.sanitize_settings({"settings": "not-a-dict"}) == {}


def test_sanitize_ignored_in_judge_deployment(monkeypatch):
    monkeypatch.setenv("JUDGES_EXPIRES_AT", "9999999999")
    out = ui_server.sanitize_settings({"settings": dict(llm="stub", spend_cap=5)})
    assert out == {}


def test_create_applies_settings_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KEVO_UI_LLM", "wandb")
    monkeypatch.setenv("KEVO_UI_PROFILE", "RUN")
    monkeypatch.setattr(ui_server, "start_discovery", lambda jid: None)
    monkeypatch.setattr(ui_server.web, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(ui_server, "JOBS", Path(tmp_path), raising=False)
    with ui_server.app.test_client() as client:
        r = client.post("/api/runs", json=dict(
            repo="https://github.com/x/y", mode="recipe",
            settings=dict(llm="claude_oauth:sonnet", profile="DEV", spend_cap=20)),
            headers={"Idempotency-Key": "settings-test-1"})
        assert r.status_code == 202, r.get_json()
        jid = r.get_json()["id"]
    job = ui_server.web.load_job(jid)
    assert job["llm"] == "claude_oauth:sonnet"   # user beats KEVO_UI_LLM
    assert job["profile"] == "DEV"               # user beats KEVO_UI_PROFILE
    assert job["spend_cap"] == 20.0


def test_create_falls_back_to_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KEVO_UI_LLM", "wandb")
    monkeypatch.setattr(ui_server, "start_discovery", lambda jid: None)
    monkeypatch.setattr(ui_server.web, "JOBS_DIR", str(tmp_path))
    with ui_server.app.test_client() as client:
        r = client.post("/api/runs", json=dict(repo="https://github.com/x/y",
                                               mode="recipe", settings={}),
                        headers={"Idempotency-Key": "settings-test-2"})
        assert r.status_code == 202, r.get_json()
        jid = r.get_json()["id"]
    job = ui_server.web.load_job(jid)
    assert job["llm"] == "wandb"


def test_settings_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("KEVO_UI_LLM", "claude_oauth:opus")
    monkeypatch.setenv("KEVO_MOLAB_CONNECTION_FILE", str(tmp_path / "none.json"))
    with ui_server.app.test_client() as client:
        d = client.get("/api/settings").get_json()
    assert d["llm"] == "claude_oauth:opus"
    assert d["connection_file"] is False


def test_masked_hides_pasted_connection():
    import web
    j = web.masked(dict(id="x", molab_connection="--token SECRET", molab={"t": 1}))
    assert "molab_connection" not in j and "molab" not in j
