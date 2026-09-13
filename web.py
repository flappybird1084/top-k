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
          ("[baseline]", "measuring baseline"),
          ("=== generation", "optimizing"),
          ("=== recipe", "optimizing (recipe)"),
          ("=== finals", "finals"), ("[loop] stopped", "finishing")]


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
            "WANDB_INFERENCE_BASE_URL", "WANDB_INFERENCE_PROJECT", "SEARXNG_URL")


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
            job.update(status="running", stage="connecting to molab")
            save_job(job)
            # subprocess so each job runs the CURRENT dispatch code from disk,
            # even if this server process has been up for days
            env = dict(os.environ)
            env["KEVO_REMOTE_ENV"] = json.dumps(_job_env(job))
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "kernelevo.molab_dispatch",
                 _job_path(jid), ROOT, os.path.join(JOBS_DIR, jid, "run")],
                cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                write_line(line)
            rc = proc.wait()
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
            if job.get("spend_cap"):
                cmd += ["--spend-cap", str(job["spend_cap"])]
            if job.get("mode") == "recipe":
                cmd += ["--mode", "recipe"]
                if job.get("recipe"):
                    cmd += ["--recipe-json", json.dumps(job["recipe"])]
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
 .head{background:#fff;border:1px solid #d5dae1;border-radius:8px;padding:.8rem 1rem;
       margin:1rem 0;display:flex;gap:2rem;flex-wrap:wrap}
 .head b{font-size:1.25rem} .good{color:#276749} .bad{color:#c53030}
 details.gen{background:#fff;border:1px solid #d5dae1;border-radius:8px;
             margin:.5rem 0;padding:.15rem .8rem}
 details.gen>summary{font-weight:600;font-size:.9rem;padding:.4rem 0;cursor:pointer}
 details.cand{border-top:1px solid #edf0f4;margin:.2rem 0;padding:.1rem .4rem}
 details.cand>summary{font-size:.85rem;padding:.35rem 0;cursor:pointer}
 .pill{padding:.05rem .45rem;border-radius:999px;font-size:.72rem;margin-right:.4rem}
 .p-acc{background:#c6f6d5;color:#276749}.p-rej{background:#fed7d7;color:#c53030}
 .p-slow{background:#feebc8;color:#975a16}.p-infra{background:#e2e8f0;color:#4a5568}
 details.cand table{margin:.4rem 0}
 details.code>summary{font-size:.8rem;color:#2b6cb0;cursor:pointer;padding:.2rem 0}
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
 <label>mode</label>
 <select name=mode>
  <option value=kernel>kernel evolution — Triton kernels vs torch.compile</option>
  <option value=recipe>recipe golf — architecture + hyperparams vs held-out val loss</option>
 </select>
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
<fieldset><legend>recipe golf knobs (used only in recipe mode)</legend>
 <label>architecture generations · candidates · train seconds each</label>
 <div style="display:flex;gap:.5rem">
  <input type=number name=arch_gens value=2 min=0>
  <input type=number name=arch_cands value=8 min=1>
  <input type=number name=arch_secs value=60 min=10></div>
 <label>hyperparam generations · candidates · train seconds each</label>
 <div style="display:flex;gap:.5rem">
  <input type=number name=hp_gens value=1 min=0>
  <input type=number name=hp_cands value=8 min=1>
  <input type=number name=hp_secs value=120 min=10></div>
 <label>mixed generations · train seconds (0 = skipped)</label>
 <div style="display:flex;gap:.5rem">
  <input type=number name=mixed_gens value=0 min=0>
  <input type=number name=mixed_secs value=180 min=10></div>
 <label>finals: top-K · train seconds each</label>
 <div style="display:flex;gap:.5rem">
  <input type=number name=finals_k value=2 min=1>
  <input type=number name=finals_secs value=300 min=30></div>
 <label>param budget ratio · loss margin (rel) · eval batches · parallel agents</label>
 <div style="display:flex;gap:.5rem">
  <input type=number name=param_ratio value=1.10 step=0.01>
  <input type=number name=loss_margin value=0.003 step=0.001>
  <input type=number name=eval_batches value=8 min=1>
  <input type=number name=parallelism value=8 min=1></div>
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

HEADLINE_T = """{% if evo and evo['recipe'] and evo['base_val'] %}
<div class=head>
 <div>baseline val loss<br><b>{{ '%.4f'|format(evo['base_val']) }}</b></div>
 <div>best evolved val loss<br><b>{{ '%.4f'|format(evo['best_val'])
      if evo['best_val'] else '—' }}</b></div>
 <div>improvement<br><b class="{{ 'good' if evo['val_pct'] and evo['val_pct'] > 0 }}">
  {{ '−%.2f%%'|format(evo['val_pct']) if evo['val_pct'] and evo['val_pct'] > 0
     else 'none yet' }}</b></div>
 <div>accepted recipes<br><b>{{ evo['n_accepted'] }}</b></div>
</div>
{% elif evo and evo['baseline_ms'] %}
<div class=head>
 {% if evo['eager_ms'] %}<div>eager step (no compile)<br>
  <b>{{ '%.2f'|format(evo['eager_ms']) }}ms</b></div>{% endif %}
 <div>torch.compile step<br><b>{{ '%.2f'|format(evo['baseline_ms']) }}ms</b></div>
 <div>best evolved step<br><b>{{ '%.2f'|format(evo['best_ms']) if evo['best_ms']
      else '—' }}{{ 'ms' if evo['best_ms'] else '' }}</b></div>
 <div>vs torch.compile<br><b class="{{ 'good' if evo['improvement_pct'] else '' }}">
  {{ '−%.1f%%'|format(evo['improvement_pct']) if evo['improvement_pct']
     else 'none yet' }}</b></div>
 {% if evo['vs_eager_pct'] %}<div>vs eager<br>
  <b class=good>−{{ '%.1f%%'|format(evo['vs_eager_pct']) }}</b></div>{% endif %}
 <div>accepted kernels<br><b>{{ evo['n_accepted'] }}</b></div>
</div>
{% endif %}"""

FILES_T = """{% if files %}<table><tr><th>file</th><th>size</th></tr>
{% for f in files %}<tr>
 <td><a href="{{ url_for('job_file', jid=jid, path=f[0]) }}">{{ f[0] }}</a></td>
 <td class=muted>{{ f[1] }}</td></tr>{% endfor %}</table>
{% else %}<p class=muted>nothing synced yet — files appear as the run produces
 them</p>{% endif %}"""

GENS_T = """{% if evo and evo['generations'] %}
<h2>generations</h2>
{% for g in evo['generations'] %}
<details class=gen id="gen-{{ g['n'] }}" {{ 'open' if loop.last }}>
 <summary>generation {{ g['n'] }} — {{ g['cands']|length }} candidate(s),
  {{ g['n_acc'] }} accepted</summary>
 {% for c in g['cands'] %}
 <details class=cand id="cand-{{ c['id'] }}">
  <summary><span class="pill {{ c['pill'] }}">{{ c['op_name'] }}</span>
   {{ c['headline'] }}</summary>
  <table>
   <tr><td>strategy</td><td>{{ c['strategy'] }}</td></tr>
   <tr><td>gate reached</td><td>{{ c['gate_reached'] }} / 4
       (repairs used: {{ c['repairs_used'] }})</td></tr>
   {% if c['latency_us'] %}<tr><td>isolation latency</td>
    <td>{{ '%.1f'|format(c['latency_us']) }}µs vs incumbent
        {{ '%.1f'|format(c['incumbent_latency_us']) }}µs</td></tr>{% endif %}
   {% if c['step_time_ms'] %}<tr><td>in-model step</td>
    <td>{{ '%.2f'|format(c['step_time_ms']) }}ms vs incumbent
        {{ '%.2f'|format(c['incumbent_step_time_ms']) }}ms</td></tr>{% endif %}
   {% if c['samples_per_s'] %}<tr><td>samples/s · MFU</td>
    <td>{{ '%.1f'|format(c['samples_per_s']) }} ·
        {{ '%.3f'|format(c['mfu']) if c['mfu'] else '—' }}</td></tr>{% endif %}
   {% if c['failure_note'] %}<tr><td>failure</td>
    <td>{{ c['failure_note'][:400] }}</td></tr>{% endif %}
   {% if c['model_name'] %}<tr><td>written by</td><td>{{ c['model_name'] }}</td></tr>{% endif %}
  </table>
  {% if c['code'] %}
  <details class=code id="code-{{ c['id'] }}"><summary>kernel source
    {% if c['code_rel'] %}(<a href="{{ url_for('job_file', jid=jid,
      path=c['code_rel']) }}">raw</a>){% endif %}</summary>
   <pre>{{ c['code'] }}</pre>
  </details>
  {% endif %}
 </details>
 {% endfor %}
</details>
{% endfor %}
{% endif %}"""

JOB = """
<p><a href="{{ url_for('index') }}">&larr; jobs</a></p>
<h2>job {{ job['id'][:8] }}
 <span class="st {{ job['status'] }}" id=statuspill>{{ job['status'] }}</span></h2>
<p class=muted id=stageline>{{ job.get('stage','') }}</p>
<table>
 <tr><td>target</td><td>{{ job.get('repo') or job.get('adapter') }}</td></tr>
 <tr><td>profile / llm</td><td>{{ job['profile'] }} / {{ job.get('llm') or '(profile default)' }}</td></tr>
 <tr><td>max debug turns</td><td>{{ job['max_debug_turns'] }}</td></tr>
 <tr><td>execution</td><td>{{ job['execution_target'] }}</td></tr>
 <tr><td>comments</td><td>{{ job.get('comments') or '—' }}</td></tr>
 <tr><td>run dir</td><td>{{ job.get('run_dir') or '—' }}</td></tr>
</table>
<div id=headline>{{ headline_html|safe }}</div>
<h2>generated code &amp; artifacts</h2>
<div id=filesbox>{{ files_html|safe }}</div>
<h2>log</h2>
<pre id=log>{{ log }}</pre>
<div id=gensbox>{{ gens_html|safe }}</div>
<script>
const POLLING = {{ 'true' if job['status'] in ('queued', 'running') else 'false' }};
function morph(id, html) {
  const el = document.getElementById(id);
  const open = new Set(), closed = new Set();
  el.querySelectorAll('details[id]').forEach(x => (x.open ? open : closed).add(x.id));
  if (el.innerHTML === html) return;
  el.innerHTML = html;
  el.querySelectorAll('details[id]').forEach(x => {
    if (open.has(x.id)) x.open = true;
    else if (closed.has(x.id)) x.open = false;
  });
}
async function tick() {
  try {
    const r = await fetch(location.pathname + '/partial');
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    const pre = document.getElementById('log');
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 12;
    if (pre.textContent !== d.log) {
      pre.textContent = d.log;
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    }
    morph('headline', d.headline_html);
    morph('filesbox', d.files_html);
    morph('gensbox', d.gens_html);
    const sp = document.getElementById('statuspill');
    sp.textContent = d.status; sp.className = 'st ' + d.status;
    document.getElementById('stageline').textContent = d.stage || '';
    if (d.status === 'queued' || d.status === 'running') setTimeout(tick, 3000);
  } catch (e) { setTimeout(tick, 6000); }
}
if (POLLING) setTimeout(tick, 3000);
</script>"""


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
        mode=f.get("mode", "kernel"),
        recipe=dict(
            phases=[
                dict(kind="architecture", generations=int(f.get("arch_gens") or 2),
                     candidates=int(f.get("arch_cands") or 8),
                     train_seconds=int(f.get("arch_secs") or 60)),
                dict(kind="mixed", generations=int(f.get("mixed_gens") or 0),
                     candidates=int(f.get("arch_cands") or 8),
                     train_seconds=int(f.get("mixed_secs") or 180)),
                dict(kind="hyperparam", generations=int(f.get("hp_gens") or 1),
                     candidates=int(f.get("hp_cands") or 8),
                     train_seconds=int(f.get("hp_secs") or 120)),
            ],
            finals_top_k=int(f.get("finals_k") or 2),
            finals_train_seconds=int(f.get("finals_secs") or 300),
            param_budget_ratio=float(f.get("param_ratio") or 1.10),
            loss_margin_rel=float(f.get("loss_margin") or 0.003),
            eval_batches=int(f.get("eval_batches") or 8),
            subagent_parallelism=int(f.get("parallelism") or 8),
        ) if f.get("mode") == "recipe" else None,
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


def _pct(new, old):
    return None if not new or not old else 100.0 * (old - new) / old


def _job_evolution(jid):
    """Read the (synced) archive and shape it for the per-generation panel."""
    import sqlite3
    db_path = os.path.join(JOBS_DIR, jid, "run", "archive.sqlite")
    if not os.path.exists(db_path):
        return None
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        # the archive persists across relaunches of a job; show the latest run
        rows = [dict(r) for r in db.execute(
            "SELECT c.*, l.op_name FROM candidates c "
            "JOIN lineages l ON c.lineage_id = l.id "
            "WHERE l.model_id = (SELECT MAX(id) FROM models) "
            "ORDER BY c.generation, c.id")]
        db.close()
    except sqlite3.Error:
        return None
    recipe_mode = any(r.get("val_loss") is not None for r in rows)
    baselines = {r.get("train_secs"): r.get("val_loss") for r in rows
                 if r["generation"] == 0 and r.get("val_loss") is not None}
    baseline = next((r["incumbent_step_time_ms"] for r in rows
                     if r["incumbent_step_time_ms"]), None)
    accepted = [r for r in rows if r["accepted"] and r["step_time_ms"]]
    best = min((r["step_time_ms"] for r in accepted), default=None)
    eager_ms = None
    try:
        tj = json.load(open(os.path.join(JOBS_DIR, jid, "run", "targets.json")))
        eager_ms = tj.get("step_time_ms")
    except (OSError, ValueError):
        pass
    gens = {}
    for r in rows:
        if r["generation"] == 0:
            continue  # seed rows, not agent work
        c = dict(r)
        c["code_rel"] = (f"candidates/{os.path.basename(r['code_path'])}"
                         if r.get("code_path") else None)
        code_local = (os.path.join(JOBS_DIR, jid, "run", c["code_rel"])
                      if c["code_rel"] else None)
        c["code"] = None
        if code_local and os.path.exists(code_local):
            c["code"] = open(code_local, errors="replace").read()[:15000]
        if recipe_mode:
            v, secs = r.get("val_loss"), r.get("train_secs")
            base = baselines.get(secs)
            if v is not None and base:
                d = 100.0 * (base - v) / base
                c["pill"] = "p-acc" if r["accepted"] else "p-slow"
                c["headline"] = (f"val loss {v:.4f} ({d:+.2f}% vs baseline "
                                 f"@{int(secs)}s){' ACCEPTED' if r['accepted'] else ''}")
            elif (r.get("failure_note") or "").startswith("[infra]"):
                c["pill"], c["headline"] = "p-infra", "infrastructure failure"
            else:
                c["pill"] = "p-rej"
                c["headline"] = (r.get("failure_note") or "failed to load")[:90]
            gens.setdefault(r["generation"], []).append(c)
            continue
        st, inc = r["step_time_ms"], r["incumbent_step_time_ms"]
        lat, ilat = r["latency_us"], r["incumbent_latency_us"]
        if r["accepted"]:
            c["pill"], c["headline"] = ("p-acc",
                f"ACCEPTED  −{inc - st:.2f}ms step ({_pct(st, inc):+.1f}% faster)")
        elif st and inc:
            d = _pct(st, inc)
            c["pill"] = "p-slow"
            c["headline"] = (f"in-model {d:+.1f}% vs incumbent — under the "
                             f"acceptance margin" if d and d > 0 else
                             f"in-model {d:+.1f}% — not faster")
        elif lat and ilat:
            c["pill"], c["headline"] = "p-slow", \
                f"isolation {_pct(lat, ilat):+.1f}% vs incumbent — failed gate 3"
        elif (r.get("failure_note") or "").startswith("[infra]"):
            c["pill"], c["headline"] = "p-infra", "infrastructure failure (not the kernel's fault)"
        elif r["gate_reached"] == 1:
            c["pill"], c["headline"] = "p-rej", "failed correctness (gate 2)"
        elif r["gate_reached"] == 0:
            c["pill"], c["headline"] = "p-rej", "failed to compile (gate 1)"
        else:
            c["pill"], c["headline"] = "p-slow", "correct — benchmarks did not run"
        gens.setdefault(r["generation"], []).append(c)
    best_val = min((r["val_loss"] for r in rows
                    if r.get("val_loss") is not None and r["generation"] > 0),
                   default=None) if recipe_mode else None
    base_val = max(baselines.items())[1] if baselines else None  # longest budget
    return dict(
        recipe=recipe_mode, base_val=base_val, best_val=best_val,
        val_pct=_pct(best_val, base_val) if recipe_mode else None,
        baseline_ms=baseline, best_ms=best, eager_ms=eager_ms,
        improvement_pct=_pct(best, baseline),
        vs_eager_pct=_pct(best, eager_ms),
        n_accepted=len(accepted) if not recipe_mode else
            sum(1 for r in rows if r["accepted"] and r["generation"] > 0),
        generations=[{"n": g, "cands": cs,
                      "n_acc": sum(1 for c in cs if c["accepted"])}
                     for g, cs in sorted(gens.items())])


def _job_files(jid):
    base = os.path.join(JOBS_DIR, jid, "run")
    out = []
    for root, dirs, fs in os.walk(base):
        dirs[:] = [d for d in dirs
                   if d not in ("inductor-cache", "wandb", "repo", "__pycache__")]
        for f in fs:
            p = os.path.join(root, f)
            out.append((os.path.relpath(p, base), f"{os.path.getsize(p):,}B"))
    return sorted(out)


def _fragments(jid, job):
    try:
        lines = open(os.path.join(JOBS_DIR, jid, "log.txt"),
                     errors="replace").read().splitlines()
        log = "\n".join(lines[-300:])
    except OSError:
        log = "(no log yet)"
    evo = _job_evolution(jid)
    return dict(
        status=job["status"], stage=job.get("stage", ""), log=log,
        headline_html=render_template_string(HEADLINE_T, evo=evo),
        files_html=render_template_string(FILES_T, files=_job_files(jid), jid=jid),
        gens_html=render_template_string(GENS_T, evo=evo, jid=jid))


@app.get("/jobs/<jid>")
def job_page(jid):
    job = load_job(jid)
    body = render_template_string(JOB, job=masked(job), **_fragments(jid, job))
    return render_template_string(PAGE, body=body, warn=_env_warnings())


@app.get("/jobs/<jid>/partial")
def job_partial(jid):
    return _fragments(jid, load_job(jid))


@app.get("/jobs/<jid>/file")
def job_file(jid):
    from flask import Response, abort, send_file
    base = os.path.realpath(os.path.join(JOBS_DIR, jid, "run"))
    full = os.path.realpath(os.path.join(base, request.args.get("path", "")))
    if not full.startswith(base + os.sep) or not os.path.isfile(full):
        abort(404)
    if full.endswith((".py", ".json", ".txt")):
        return Response(open(full, errors="replace").read(), mimetype="text/plain")
    return send_file(full, as_attachment=True)


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
