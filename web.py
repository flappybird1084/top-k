"""Barebones web frontend (placeholder until the real one lands).

  python web.py             # http://127.0.0.1:8420
  python web.py --host 0.0.0.0 --port 8420

Entry page: git repo (or demo adapter), max debug turns, comments, molab
connection details, W&B settings. Submitting queues a job; a worker thread runs
jobs one at a time (the GPU is owned by one loop) by shelling out to search.py
and streaming its output to the job log. Molab-target jobs are stored but
blocked until remote dispatch is wired (kernelevo/molab.py).

Run this on the GPU box (or inside molab) with the 'local' target.
W&B keys submitted in the form are kept in the job file on this machine only
and masked in the UI/API.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, redirect, render_template_string, request, url_for

load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(ROOT, "jobs")
app = Flask(__name__)
_queue: "queue.Queue[str]" = queue.Queue()

STAGES = [("[adapter]", "writing adapter"), ("[ingest]", "ingesting"),
          ("[profile]", "profiling"), ("[calibrate]", "calibrating"),
          ("=== generation", "optimizing"), ("[loop] stopped", "finishing")]


def _job_path(jid):
    return os.path.join(JOBS_DIR, jid, "job.json")


def load_job(jid):
    with open(_job_path(jid)) as f:
        return json.load(f)


def save_job(job):
    os.makedirs(os.path.join(JOBS_DIR, job["id"]), exist_ok=True)
    with open(_job_path(job["id"]), "w") as f:
        json.dump(job, f, indent=2)


def list_jobs():
    if not os.path.isdir(JOBS_DIR):
        return []
    jobs = []
    for jid in os.listdir(JOBS_DIR):
        try:
            jobs.append(load_job(jid))
        except (OSError, ValueError):
            continue
    return sorted(jobs, key=lambda j: j["created_at"], reverse=True)


def masked(job):
    j = dict(job)
    if j.get("wandb", {}).get("api_key"):
        j["wandb"] = {**j["wandb"], "api_key": "••••"}
    return j


def _worker():
    while True:
        jid = _queue.get()
        try:
            _run_job(jid)
        except Exception as e:  # noqa: BLE001 — a job failure must not kill the worker
            job = load_job(jid)
            job.update(status="failed", stage=f"web worker error: {e}")
            save_job(job)


PASS_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "WANDB_API_KEY", "WANDB_ENTITY",
            "WANDB_PROJECT", "WEAVE_PROJECT", "WANDB_INFERENCE_API_KEY",
            "WANDB_INFERENCE_BASE_URL")


def _job_env(job):
    env = {k: os.environ[k] for k in PASS_ENV if os.environ.get(k)}
    for key, envname in (("api_key", "WANDB_API_KEY"), ("entity", "WANDB_ENTITY"),
                         ("project", "WANDB_PROJECT")):
        if job.get("wandb", {}).get(key):
            env[envname] = job["wandb"][key]
    return env


def _run_job(jid):
    job = load_job(jid)
    log_path = os.path.join(JOBS_DIR, jid, "log.txt")
    logf = open(log_path, "a")

    def write_line(line):
        logf.write(line.rstrip("\n") + "\n")
        logf.flush()
        for marker, stage in STAGES:
            if marker in line and job["stage"] != stage:
                job["stage"] = stage
                save_job(job)

    try:
        if job["execution_target"] == "molab":
            from kernelevo.molab import MolabTarget
            job.update(status="running", stage="connecting to molab")
            save_job(job)
            try:
                rc = MolabTarget(job.get("molab")).dispatch(
                    job, ROOT, _job_env(job), write_line)
            except Exception as e:  # noqa: BLE001 — connection/protocol errors
                write_line(f"[molab] dispatch error: {type(e).__name__}: {e}")
                rc = 1
        else:
            run_dir = os.path.join(JOBS_DIR, jid, "run")
            cmd = [sys.executable, "-u", "search.py", "--profile", job["profile"],
                   "--out", run_dir]
            if job.get("repo"):
                cmd += ["--repo", job["repo"], "--comments", job.get("comments", ""),
                        "--max-debug-turns", str(job["max_debug_turns"])]
            else:
                cmd += ["--adapter", job["adapter"]]
            if job.get("llm"):
                cmd += ["--llm", job["llm"]]
            job.update(status="running", stage="starting", run_dir=run_dir)
            save_job(job)
            write_line(f"$ {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **_job_env(job)},
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            for line in proc.stdout:
                write_line(line)
            rc = proc.wait()
        job.update(status="done" if rc == 0 else "failed",
                   stage=job["stage"] if rc == 0 else f"{job['stage']} (exit {rc})")
        save_job(job)
    finally:
        logf.close()


PAGE = """<!doctype html><meta charset=utf-8>
<title>kernel evolution</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;
      padding:0 1rem;color:#1a202c;background:#f7f8fa}
 h1{font-size:1.3rem} h2{font-size:1.05rem;margin-top:2rem}
 fieldset{border:1px solid #d5dae1;border-radius:8px;margin:1rem 0;padding:.8rem 1rem;
          background:#fff}
 legend{font-weight:600;font-size:.85rem;padding:0 .4rem}
 label{display:block;margin:.5rem 0 .1rem;font-size:.85rem;color:#4a5568}
 input[type=text],input[type=password],input[type=number],textarea,select{
   width:100%;box-sizing:border-box;padding:.45rem;border:1px solid #cbd2db;
   border-radius:6px;font:inherit}
 textarea{min-height:4rem}
 button{margin-top:1rem;padding:.55rem 1.4rem;border:0;border-radius:6px;
        background:#2b6cb0;color:#fff;font:inherit;cursor:pointer}
 table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px}
 td,th{padding:.45rem .6rem;border-bottom:1px solid #e4e8ee;text-align:left;
       font-size:.85rem}
 .st{padding:.1rem .5rem;border-radius:999px;font-size:.75rem;white-space:nowrap}
 .queued{background:#e2e8f0}.running{background:#bee3f8}.done{background:#c6f6d5}
 .failed{background:#fed7d7}.blocked_molab{background:#feebc8}.interrupted{background:#e2e8f0}
 pre{background:#11151c;color:#d7dde6;padding:1rem;border-radius:8px;overflow-x:auto;
     font-size:.78rem;max-height:32rem;overflow-y:auto}
 a{color:#2b6cb0} .muted{color:#718096;font-size:.85rem}
 .warn{background:#fff5f5;border:1px solid #fc8181;border-radius:8px;color:#c53030;
       padding:.7rem 1rem;margin:1rem 0;font-size:.85rem}
 .warn ul{margin:.3rem 0 0;padding-left:1.2rem} .warn code{font-weight:600}
</style>
<h1>kernel evolution</h1>
{{ warn|safe }}
{{ body|safe }}"""

FORM = """
<form method=post action="{{ url_for('create_job') }}">
<fieldset><legend>target</legend>
 <label>git repo (URL or local path) — the system writes and verifies an adapter
        for the repo's stated model</label>
 <input type=text name=repo placeholder="https://github.com/user/repo">
 <label>…or a bundled demo adapter (used only if repo is empty)</label>
 <select name=adapter><option value="">(none)</option>
  <option>adapters/jepa.py</option><option>adapters/lm.py</option></select>
 <label>comments / guidance (which model, which data, anything the agent should know)</label>
 <textarea name=comments></textarea>
 <label>max debug turns (adapter repair attempts before giving up)</label>
 <input type=number name=max_debug_turns value=5 min=1 max=20>
</fieldset>
<fieldset><legend>run</legend>
 <label>profile</label>
 <select name=profile><option>DEV</option><option>RUN</option></select>
 <label>llm override (blank = from profile)</label>
 <select name=llm><option value="">(from profile)</option><option>stub</option>
  <option>anthropic</option><option>openai</option><option>wandb</option></select>
 <label>execution target</label>
 <select name=execution_target>
  <option value=molab>molab remote — runs on the notebook's GPU</option>
  <option value=local>local — this machine's GPU</option></select>
</fieldset>
<fieldset><legend>molab connection (per-notebook)</legend>
 <label>paste the whole "Pair with agent" prompt from molab here — it contains the
        notebook URL and the real auth token (the on-screen ****-masked token won't
        work, copy the prompt itself). Keep the notebook tab open while the job runs.</label>
 <textarea name=molab_connection placeholder="Use the /marimo-pair skill … Connect to the notebook at: https://sb-….sb.molab.run/ … --token '…'"></textarea>
 <label>notebook URL (optional if it's in the pasted prompt)</label>
 <input type=text name=molab_url>
</fieldset>
<fieldset><legend>weights &amp; biases</legend>
 <label>api key (blank = server .env)</label><input type=password name=wandb_key>
 <label>entity</label><input type=text name=wandb_entity>
 <label>project</label><input type=text name=wandb_project>
</fieldset>
<button>launch</button>
</form>
<h2>jobs</h2>
{% if jobs %}<table><tr><th>id</th><th>target</th><th>status</th><th>stage</th></tr>
{% for j in jobs %}<tr>
 <td><a href="{{ url_for('job_page', jid=j['id']) }}">{{ j['id'][:8] }}</a></td>
 <td>{{ j.get('repo') or j.get('adapter') }}</td>
 <td><span class="st {{ j['status'] }}">{{ j['status'] }}</span></td>
 <td class=muted>{{ j.get('stage','') }}</td></tr>{% endfor %}</table>
{% else %}<p class=muted>no jobs yet</p>{% endif %}"""

JOB = """
{% if job['status'] in ('queued','running') %}<meta http-equiv=refresh content=3>{% endif %}
<p><a href="{{ url_for('index') }}">&larr; jobs</a></p>
<h2>job {{ job['id'][:8] }}
 <span class="st {{ job['status'] }}">{{ job['status'] }}</span></h2>
<p class=muted>{{ job.get('stage','') }}</p>
<table>
 <tr><td>target</td><td>{{ job.get('repo') or job.get('adapter') }}</td></tr>
 <tr><td>profile / llm</td><td>{{ job['profile'] }} / {{ job.get('llm') or '(profile default)' }}</td></tr>
 <tr><td>max debug turns</td><td>{{ job['max_debug_turns'] }}</td></tr>
 <tr><td>execution</td><td>{{ job['execution_target'] }}</td></tr>
 <tr><td>comments</td><td>{{ job.get('comments') or '—' }}</td></tr>
 <tr><td>run dir</td><td>{{ job.get('run_dir') or '—' }}</td></tr>
</table>
<h2>log</h2>
<pre>{{ log }}</pre>"""


def _env_warnings() -> str:
    """Red banner for API keys missing from the server env / .env that would
    make runs fail. Static trusted text only — no user input."""
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("<code>ANTHROPIC_API_KEY</code> — jobs with the "
                       "<b>anthropic</b> LLM (the RUN profile's default) will fail")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("<code>OPENAI_API_KEY</code> — jobs with the <b>openai</b> "
                       "LLM will fail")
    if not (os.environ.get("WANDB_INFERENCE_API_KEY") or os.environ.get("WANDB_API_KEY")):
        missing.append("<code>WANDB_INFERENCE_API_KEY</code> (or <code>WANDB_API_KEY"
                       "</code>) — jobs with the <b>wandb</b> LLM will fail")
    if not os.environ.get("WANDB_API_KEY"):
        missing.append("<code>WANDB_API_KEY</code> — W&amp;B/Weave logging disabled "
                       "(can also be entered per job in the form)")
    if not missing:
        return ""
    return ("<div class=warn><b>⚠ missing API keys</b> — fill them in "
            "<code>.env</code> next to <code>web.py</code> and restart "
            "(stub-LLM jobs are unaffected):<ul>"
            + "".join(f"<li>{m}</li>" for m in missing) + "</ul></div>")


@app.get("/")
def index():
    body = render_template_string(FORM, jobs=list_jobs())
    return render_template_string(PAGE, body=body, warn=_env_warnings())


@app.post("/jobs")
def create_job():
    f = request.form
    job = dict(
        id=uuid.uuid4().hex, created_at=time.time(), status="queued", stage="queued",
        repo=f.get("repo", "").strip() or None,
        adapter=f.get("adapter", "").strip() or "adapters/jepa.py",
        comments=f.get("comments", "").strip(),
        max_debug_turns=int(f.get("max_debug_turns") or 5),
        profile=f.get("profile", "DEV"),
        llm=f.get("llm", "").strip() or None,
        execution_target=f.get("execution_target", "local"),
        molab=dict(notebook_url=f.get("molab_url", "").strip(),
                   connection=f.get("molab_connection", "").strip()),
        wandb=dict(api_key=f.get("wandb_key", "").strip(),
                   entity=f.get("wandb_entity", "").strip(),
                   project=f.get("wandb_project", "").strip()),
    )
    save_job(job)
    _queue.put(job["id"])
    return redirect(url_for("job_page", jid=job["id"]))


@app.get("/jobs/<jid>")
def job_page(jid):
    job = load_job(jid)
    log_path = os.path.join(JOBS_DIR, jid, "log.txt")
    try:
        lines = open(log_path, errors="replace").read().splitlines()
        log = "\n".join(lines[-200:])
    except OSError:
        log = "(no log yet)"
    body = render_template_string(JOB, job=masked(job), log=log)
    return render_template_string(PAGE, body=body, warn=_env_warnings())


@app.get("/api/jobs/<jid>")
def job_api(jid):
    return masked(load_job(jid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()
    os.makedirs(JOBS_DIR, exist_ok=True)
    for job in list_jobs():  # recover from a previous server session
        if job["status"] == "running":
            job.update(status="interrupted", stage="server restarted mid-run")
            save_job(job)
        elif job["status"] == "queued":
            _queue.put(job["id"])
    threading.Thread(target=_worker, daemon=True).start()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
