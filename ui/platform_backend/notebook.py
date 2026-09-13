import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import re
    from pathlib import Path
    refresh = mo.ui.refresh(options=[5, 10, 30], default_interval=5)
    return mo, json, os, re, Path, refresh


@app.cell
def _(mo, json, os, re, Path, refresh):
    refresh
    _request = mo.app_meta().request
    _run = _request.query_params.get("run", "") if _request else ""
    if isinstance(_run, (list, tuple)):
        _run = _run[0] if _run else ""
    _state = {}
    if re.fullmatch(r"[a-f0-9]{32}", _run):
        _root = Path(os.environ.get("TOPK_STATE_DIR", str(Path.home()/".local/state/topkernel-platform")))
        _file = _root / (_run + ".json")
        if _file.is_file():
            _state = json.loads(_file.read_text())
    run_snapshot = _state
    return (run_snapshot,)


@app.cell
def _(mo, run_snapshot):
    _rows = [{"kernel": r.get("lineage_id"), "candidate": r.get("id"),
              "generation": r.get("generation"), "verified": bool(r.get("accepted")),
              "step_ms": r.get("step_time_ms")} for r in run_snapshot.get("candidates", [])]
    mo.vstack([
        mo.md("### Run notebook"),
        mo.md("Status: **" + run_snapshot.get("status", "waiting") + "**"),
        mo.ui.table(_rows, selection=None) if _rows else mo.md("No kernel measurements yet."),
    ])
    return


if __name__ == "__main__":
    app.run()
