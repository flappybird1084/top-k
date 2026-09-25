"""Remote dispatch to a molab (marimo cloud) notebook, where the GPU lives.

Protocol (from the marimo-pair skill's execute-code.sh, verified against its
source): GET {base}/api/sessions and POST {base}/api/kernel/execute with
'Authorization: Bearer <token>' and 'Marimo-Session-Id: <id>'; execution
streams back as SSE events stdout / stderr / done ({"success": bool}).
Each molab notebook has its own URL + auth token — the "Pair with agent"
prompt contains both; users paste that blob into the web form.

dispatch() runs a whole job remotely: preflight (session + GPU), upload the
project as a gzipped tarball via execute calls, install deps, launch search.py
detached (log + exit-code files), then poll the remote log back line by line.
The session id is re-resolved on every call because marimo renames sessions
when the browser reconnects; the notebook tab must stay open.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tarfile
import time
import urllib.error
import urllib.request

from kernelevo import relay_policy

REMOTE_DEPS = ["triton", "numpy", "pandas", "python-dotenv", "wandb", "weave",
               "anthropic", "openai", "datasets", "tiktoken"]


def repo_runtime_deps(repo_url: str) -> list[str]:
    """Dependencies needed by upstream source, isolated from the notebook Python."""
    from urllib.parse import urlparse

    parsed = urlparse(repo_url)
    if parsed.hostname != "github.com":
        return []
    owner_repo = tuple(parsed.path.strip("/").split("/")[:2])
    # Install only the packages each pinned repository imports. The dispatcher
    # uses --no-deps in a per-job target, so explicitly list small transitive
    # imports instead of pulling a second PyTorch into the notebook runtime.
    return {
        ("huggingface", "transformers"): ["tokenizers>=0.23.1,<0.24.0"],
        ("DLR-RM", "stable-baselines3"): [
            "gymnasium>=0.29.1,<2.0", "farama-notifications>=0.0.4", "cloudpickle"],
        ("Lightning-AI", "litgpt"): [
            "lightning>=2.6.1,<3", "lightning-utilities>=0.14,<1",
            "torchmetrics>=1.3,<2", "fsspec", "packaging", "PyYAML"],
        ("facebookresearch", "detectron2"): [
            "fvcore>=0.1.5,<0.1.6", "iopath>=0.1.7,<0.1.10",
            "omegaconf>=2.1,<2.4", "yacs>=0.1.8", "hydra-core>=1.1",
            "termcolor>=1.1", "portalocker", "antlr4-python3-runtime==4.9.3"],
    }.get(owner_repo, [])
UPLOAD_CHUNK = 400_000  # base64 chars per execute call
EXCLUDE_DIRS = {".git", "__pycache__", "runs", "jobs", ".venv", "venv", "wandb",
                "notebooks"}
EXCLUDE_FILES = {".env"}  # secrets travel via the launch env, never in the tar


def parse_connection(details: dict) -> tuple[str, str | None]:
    """Extract (base_url, token) from the form fields — either explicit values
    or the whole 'Pair with agent' prompt pasted into the connection box."""
    blob = " ".join(filter(None, [(details or {}).get("notebook_url", ""),
                                  (details or {}).get("connection", "")]))
    m = (re.search(r"https?://[^\s'\"`]*molab\.run[^\s'\"`]*", blob)
         or re.search(r"https?://[^\s'\"`]+", blob))
    if not m:
        raise ValueError("no notebook URL found in the molab connection details")
    url = m.group(0).rstrip("/.,")
    token = None
    # The pair prompt renders the token bare, quoted, or inside markdown
    # backticks (--token e27…c`.), so capture only token-legal characters.
    for pat in (r"--token\s+['\"`]*([A-Za-z0-9_\-*]+)", r"MARIMO_TOKEN=['\"`]*([A-Za-z0-9_\-*]+)",
                r"token[:=]\s*['\"`]*([A-Za-z0-9_\-*]{8,})"):
        tm = re.search(pat, blob)
        if tm:
            token = tm.group(1)
            break
    # The server validates credentials. A prefix heuristic rejects working
    # notebook credentials that happen to start with an asterisk.
    return url, token


class MolabClient:
    def __init__(self, url: str, token: str | None):
        self.base = url.rstrip("/")
        self.token = token

    def _headers(self, extra=None):
        # molab sits behind Cloudflare, which bans the default Python-urllib
        # user agent (error 1010) — any custom UA passes.
        h = {"User-Agent": "kernelevo/0.1", **(extra or {})}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def sessions(self) -> dict:
        req = urllib.request.Request(self.base + "/api/sessions",
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.load(resp)

    def session_id(self) -> str:
        s = self.sessions()
        if not s:
            raise RuntimeError("no active molab sessions — open the notebook in the "
                               "browser and keep the tab open")
        return next(iter(s))

    def run(self, code: str, timeout: int = 300) -> tuple[bool, str, str]:
        """Execute code in the notebook kernel; returns (success, stdout, stderr).
        Session id resolved fresh per call (ids go stale on browser reconnect)."""
        sid = self.session_id()
        req = urllib.request.Request(
            self.base + "/api/kernel/execute",
            data=json.dumps({"code": code}).encode(),
            headers=self._headers({"Content-Type": "application/json",
                                   "Marimo-Session-Id": sid}))
        out, err, event, done, success = [], [], "", False, True
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if event == "stdout":
                        out.append(payload.get("data", ""))
                    elif event == "stderr":
                        err.append(payload.get("data", ""))
                    elif event == "done":
                        success = payload.get("success") is not False
                        o = (payload.get("output") or {}).get("data")
                        if o:
                            out.append(str(o))
                        done = True
                        break
        if not done:
            raise RuntimeError("molab ended the stream without a result "
                               "(browser reconnect mid-call? retrying usually works)")
        return success, "".join(out), "".join(err)


def _project_tarball(root: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for f in files:
                if f in EXCLUDE_FILES or f.endswith((".pyc", ".sqlite")):
                    continue
                p = os.path.join(dirpath, f)
                if os.path.getsize(p) > 1_000_000:
                    continue
                tar.add(p, arcname=os.path.relpath(p, root))
    return buf.getvalue()


RELAY_TOMBSTONE_S = 1800

ARTIFACT_EXTS = (".py", ".json", ".sqlite", ".txt")
ARTIFACT_SKIP_DIRS = ("repo", "inductor-cache", "wandb", "__pycache__",
                      "compile_cache", "search_relay")


def _service_relay(client: MolabClient, work: str, pending: list, write_line,
                   relay_dir: str | None = None, policy=None, served=None):
    """Serve remote search requests locally: the notebook can't reach the
    (tailnet-private) SearXNG, but this dispatcher can.

    `policy` decides whether this run may use the operator's search service at
    all, and how much; `served` tombstones request ids so a query the sandbox
    has not yet collected is never run — or billed — twice."""
    from kernelevo import websearch
    relay = relay_dir or work + "/run/search_relay"
    served = {} if served is None else served
    now = time.monotonic()
    for rid, stamped in list(served.items()):
        if now - stamped > RELAY_TOMBSTONE_S:
            del served[rid]
    for req in pending[:4]:
        rid, query, n = req.get("id"), req.get("query", ""), req.get("n", 5)
        if rid in served:
            continue
        refusal = policy.check_search(req) if policy is not None else None
        if refusal:
            served[rid] = now
            write_line(f"[research-relay] request refused: {refusal}")
            results = [{"error": refusal}]
        else:
            try:
                results = websearch.direct_search(query, n)
            except Exception as e:  # noqa: BLE001 — report the failure to the requester
                results = [{"error": f"relay search failed: {e}"}]
            write_line(f"[research-relay] {query[:70]} -> {len(results)} result(s)")
        served[rid] = now
        payload = json.dumps(results)
        client.run(
            "import os\n"
            f"os.makedirs({relay!r}, exist_ok=True)\n"
            f"_p = {relay + '/' + str(rid) + '.res.json'!r}\n"
            f"open(_p + '.tmp', 'w').write({payload!r})\n"
            "os.replace(_p + '.tmp', _p)\n"
            "print('RELAYED')\n")


def _fetch_file(client: MolabClient, remote: str, local: str, size: int):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local + ".part", "wb") as f:
        off = 0
        while off < size:
            ok, out, _ = client.run(
                "import base64\n"
                f"_f = open({remote!r}, 'rb')\n"
                f"_f.seek({off})\n"
                "print(base64.b64encode(_f.read(200000)).decode())\n")
            if not ok:
                return False
            chunk = base64.b64decode(out.strip().splitlines()[-1])
            if not chunk:
                break
            f.write(chunk)
            off += len(chunk)
    os.replace(local + ".part", local)
    return True


def fetch_archive(client: MolabClient, work: str, dest_dir: str) -> bool:
    """Mid-run sync: archive.sqlite plus any NEW candidate/recipe source files,
    so a terminated sandbox can never take accepted code with it."""
    ok, out, _ = client.run(
        "import os, json, glob\n"
        f"_w = {work!r}\n"
        "_files = []\n"
        "for _p in ([os.path.join(_w, 'run', 'archive.sqlite')]\n"
        "           + glob.glob(os.path.join(_w, 'run', 'recipes', '*.py'))\n"
        "           + glob.glob(os.path.join(_w, 'run', 'candidates', '*.py'))\n"
        "           + glob.glob(os.path.join(_w, 'run', 'adapter.py'))\n"
        "           + glob.glob(os.path.join(_w, 'run', 'adapter_attempts.json'))):\n"
        "    if os.path.exists(_p):\n"
        "        _files.append([os.path.relpath(_p, os.path.join(_w, 'run')),\n"
        "                       os.path.getsize(_p)])\n"
        "print(json.dumps(_files))\n")
    if not ok:
        return False
    files = json.loads(out.strip().splitlines()[-1])
    for rel, size in files:
        local = os.path.join(dest_dir, rel)
        # source files are immutable once written — skip already-synced ones
        if rel.endswith(".py") and os.path.exists(local) \
                and os.path.getsize(local) == size:
            continue
        _fetch_file(client, work + "/run/" + rel, local, size)
    return True


def fetch_artifacts(client: MolabClient, work: str, dest_dir: str, write_line) -> int:
    """Copy the remote run's small artifacts (kernels, adapter, seeds, targets,
    archive) back next to the job so the web UI can show them."""
    ok, out, _ = client.run(
        "import os, json\n"
        f"_r = os.path.join({work!r}, 'run')\n"
        "_files = []\n"
        "for _root, _dirs, _fs in os.walk(_r):\n"
        f"    _dirs[:] = [d for d in _dirs if d not in {ARTIFACT_SKIP_DIRS!r}]\n"
        "    for _f in _fs:\n"
        f"        if _f.endswith({ARTIFACT_EXTS!r}) and not _f.endswith('.ingest.json'):\n"
        "            _p = os.path.join(_root, _f)\n"
        "            if os.path.getsize(_p) <= 2_000_000:\n"
        "                _files.append([os.path.relpath(_p, _r), os.path.getsize(_p)])\n"
        "print(json.dumps(_files))\n")
    if not ok:
        return 0
    files = json.loads(out.strip().splitlines()[-1])
    for rel, size in files:
        _fetch_file(client, work + "/run/" + rel, os.path.join(dest_dir, rel), size)
    write_line(f"[molab] synced {len(files)} artifact file(s) into the job directory")
    return len(files)


class MolabTarget:
    def __init__(self, details: dict):
        self.details = details or {}

    def dispatch(self, job: dict, project_root: str, env_updates: dict,
                 write_line, require_gpu: bool = True,
                 poll_interval: float = 5.0, artifacts_dir: str | None = None) -> int:
        """Run the job on the remote notebook; streams its log through
        write_line(line). Returns the remote search.py exit code."""
        url, token = parse_connection(self.details)
        client = MolabClient(url, token)
        write_line(f"[molab] connecting to {url}")
        client.session_id()  # fail fast on auth/reachability

        ok, out, err = client.run(
            "import shutil, subprocess\n"
            "smi = shutil.which('nvidia-smi')\n"
            "print(subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],"
            " capture_output=True, text=True).stdout.strip() if smi else 'NO-GPU')\n")
        gpu = (out.strip().splitlines() or ["NO-GPU"])[-1]
        write_line(f"[molab] GPU: {gpu}")
        if require_gpu and (not ok or "NO-GPU" in gpu):
            write_line("[molab] no GPU visible in the notebook — aborting")
            return 1

        # Marimo cell semantics (underscore names are cell-private, public names
        # single-owner) make kernel variables unreliable across execute calls,
        # so the upload accumulates in a remote file instead.
        work = f"/tmp/kevo_{job['id'][:8]}"
        b64_path = work + ".b64"
        tar_b64 = base64.b64encode(_project_tarball(project_root)).decode()
        write_line(f"[molab] uploading project ({len(tar_b64) // 1024}KB base64) "
                   f"to {work}")
        client.run(f"open({b64_path!r}, 'w').close()\nprint('INIT')\n")
        for i in range(0, len(tar_b64), UPLOAD_CHUNK):
            ok, out, err = client.run(
                f"with open({b64_path!r}, 'a') as _f:\n"
                f"    _f.write({tar_b64[i:i + UPLOAD_CHUNK]!r})\n"
                "print('CHUNK-OK')\n")
            if not ok or "CHUNK-OK" not in out:
                write_line(f"[molab] upload chunk failed: {err[-300:] or out[-300:]}")
                return 1
        ok, out, err = client.run(
            "import base64, tarfile, io, os\n"
            f"_w = {work!r}\n"
            "os.makedirs(_w, exist_ok=True)\n"
            f"_raw = base64.b64decode(open({b64_path!r}).read())\n"
            "_t = tarfile.open(fileobj=io.BytesIO(_raw), mode='r:gz')\n"
            "try:\n"
            "    _t.extractall(_w, filter='data')\n"
            "except TypeError:\n"
            "    _t.extractall(_w)\n"
            f"os.remove({b64_path!r})\n"
            "print('EXTRACTED', len(_raw))\n")
        if not ok or "EXTRACTED" not in out:
            write_line(f"[molab] extract failed: {err[-300:] or out[-300:]}")
            return 1

        write_line("[molab] installing dependencies (this can take a few minutes)")
        ok, out, err = client.run(
            "import subprocess, sys, shutil\n"
            f"deps = {REMOTE_DEPS!r}\n"
            "r = None\n"
            "if shutil.which('uv'):\n"
            "    r = subprocess.run(['uv', 'pip', 'install', '-q', '--python',"
            " sys.executable, *deps], capture_output=True, text=True)\n"
            "if r is None or r.returncode != 0:\n"
            "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',"
            " *deps], capture_output=True, text=True)\n"
            "print('PIP', r.returncode)\n"
            "print((r.stderr or '')[-1200:])\n", timeout=900)
        if not ok or "PIP 0" not in out:
            write_line(f"[molab] dependency install failed:\n{out[-500:]}\n{err[-500:]}")
            return 1

        job_deps = repo_runtime_deps(job.get("repo", ""))
        job_deps_dir = os.path.join(work, "runtime-deps")
        if job_deps:
            write_line(f"[molab] installing isolated repository dependencies: {job_deps}")
            ok, out, err = client.run(
                "import subprocess, sys, os, shutil\n"
                f"_target = {job_deps_dir!r}\n"
                f"_deps = {job_deps!r}\n"
                "os.makedirs(_target, exist_ok=True)\n"
                "_r = None\n"
                "if shutil.which('uv'):\n"
                "    _r = subprocess.run(['uv', 'pip', 'install', '-q', '--python', "
                "sys.executable, '--no-deps', '--target', _target, *_deps], "
                "capture_output=True, text=True)\n"
                "if _r is None or _r.returncode != 0:\n"
                "    _r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
                "'--no-deps', '--target', _target, *_deps], capture_output=True, text=True)\n"
                "print('JOB_DEPS', _r.returncode)\n"
                "print((_r.stderr or '')[-1200:])\n", timeout=900)
            if not ok or "JOB_DEPS 0" not in out:
                write_line(f"[molab] isolated dependency install failed:\n"
                           f"{out[-500:]}\n{err[-500:]}")
                return 1

        args = ["search.py", "--profile", job["profile"], "--out",
                os.path.join(work, "run")]
        if job.get("repo"):
            args += ["--repo", job["repo"], "--comments", job.get("comments", ""),
                     "--max-debug-turns", str(job["max_debug_turns"])]
        else:
            args += ["--adapter", job["adapter"]]
        if job.get("llm"):
            args += ["--llm", job["llm"]]
        if job.get("max_generations"):
            args += ["--max-generations", str(job["max_generations"])]
        if job.get("spend_cap"):
            args += ["--spend-cap", str(job["spend_cap"])]
        if job.get("mode") == "recipe":
            args += ["--mode", "recipe"]
            if job.get("recipe"):
                args += ["--recipe-json", json.dumps(job["recipe"])]

        # Judge runs get a relay dir OUTSIDE the runner-owned work tree (the
        # sandbox chowns all of `work` to the untrusted uid — audit finding 19)
        relay_dir = (work + "_relay") if job.get('judge_expires_at') \
            else work + "/run/search_relay"
        # Secret stamped into this run's launch environment. Every relay
        # request must carry it back, so work the dispatcher answers is
        # attributable to the job it is currently dispatching and to no other.
        relay_token = relay_policy.new_token()
        policy = relay_policy.RelayPolicy(job, relay_token)
        for note in (policy.operator_llm_allowed(), policy.operator_search_allowed()):
            if note:
                write_line("[relay] " + note)
        env_updates = dict(env_updates, KEVO_RELAY_DIR=relay_dir,
                           KEVO_RELAY_TOKEN=relay_token)
        sandbox_setup = ''
        if job.get('judge_expires_at'):
            remaining = min(1800, int(float(job['judge_expires_at']) - time.time()))
            if remaining < 60:
                write_line('[molab] judging window ended before launch')
                return 1
            sandbox_setup = (
                "sys.path.insert(0, _w)\n"
                "_sandbox_ns = {}\n"
                "exec(compile(open(_w + '/kernelevo/judges_sandbox.py').read(), _w + '/kernelevo/judges_sandbox.py', 'exec'), _sandbox_ns)\n"
                "_sandbox_command = _sandbox_ns['user_command']\n"
                f"_cmd = _sandbox_command(_w, _cmd, {int(job.get('judge_uid',0))}, {relay_dir!r}, {relay_token!r})\n"
                f"_cmd = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10', {str(remaining)!r}] + _cmd\n"
            )
        ok, out, err = client.run(
            "import subprocess, os, sys, json, shlex, time\n"
            f"_w = {work!r}\n"
            "for _stale in ('job.exit', 'job.log'):\n"
            "    _p = os.path.join(_w, _stale)\n"
            "    if os.path.exists(_p):\n"
            "        os.remove(_p)  # a stale exit file makes the poller think the new run died\n"
            f"_env = dict(os.environ); _env.update(json.loads({json.dumps(env_updates)!r}))\n"
            + (f"_env['PYTHONPATH'] = {job_deps_dir!r} + os.pathsep + "
               "_env.get('PYTHONPATH', '')\n" if job_deps else "") +
            f"_cmd = [sys.executable, '-u'] + json.loads({json.dumps(args)!r})\n"
            + sandbox_setup +
            "_lease = '/tmp/kevo_gpu_lease'\n"
            "try:\n"
            "    os.mkdir(_lease)  # atomic across website and benchmark dispatchers\n"
            "except FileExistsError:\n"
            "    _old_owner = os.path.join(_lease, 'owner')\n"
            "    _old_id = open(_old_owner).read().strip() if os.path.isfile(_old_owner) else ''\n"
            "    _old_work = '/tmp/kevo_' + _old_id[:8]\n"
            "    _live = subprocess.run(['pgrep', '-af', _old_work], capture_output=True, text=True).stdout.strip() if len(_old_id) == 32 else 'unknown'\n"
            "    _gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()\n"
            "    if time.time() - os.stat(_lease).st_mtime < 60 or _live or _gpu:\n"
            "        raise RuntimeError('GPU is leased to another notebook job')\n"
            "    if os.path.isfile(_old_owner): os.unlink(_old_owner)\n"
            "    os.rmdir(_lease); os.mkdir(_lease)\n"
            "_owner = os.path.join(_lease, 'owner')\n"
            f"open(_owner, 'w').write({job['id']!r})\n"
            "_sh = ' '.join(shlex.quote(c) for c in _cmd) + "
            "' > job.log 2>&1; _rc=$?; echo $_rc > job.exit; "
            f"if [ \"$(cat /tmp/kevo_gpu_lease/owner 2>/dev/null)\" = \"{job['id']}\" ]; then "
            "rm -f /tmp/kevo_gpu_lease/owner; rmdir /tmp/kevo_gpu_lease; fi; exit $_rc'\n"
            "try:\n"
            "    _p = subprocess.Popen(['bash', '-c', _sh], cwd=_w, env=_env,"
            " start_new_session=True)\n"
            "except BaseException:\n"
            "    os.unlink(_owner); os.rmdir(_lease); raise\n"
            "print('LAUNCHED', _p.pid)\n"
            "try:\n"
            "    import marimo as mo\n"
            "    mo.status.toast('kernel-evolution: optimizing on this GPU')\n"
            "except Exception:\n"
            "    pass\n")
        if not ok or "LAUNCHED" not in out:
            write_line(f"[molab] launch failed: {err[-400:] or out[-400:]}")
            return 1
        write_line(f"[molab] {out.strip().splitlines()[0]} — streaming remote log")

        from kernelevo.claude_oauth import Relay as ClaudeRelay
        from kernelevo.codex_oauth import Relay, RELAY_STALE_S
        from kernelevo.wandb_relay import Relay as WandbRelay
        oauth_relay = Relay()
        claude_relay = ClaudeRelay()
        wandb_relay = WandbRelay()
        served_searches: dict = {}
        offset, misses = 0, 0
        last_archive_sync = 0.0
        while True:
            time.sleep(poll_interval)
            if artifacts_dir and time.time() - last_archive_sync > 10:
                try:
                    fetch_archive(client, work, artifacts_dir)
                except Exception:  # noqa: BLE001 — mid-run sync is best-effort
                    pass
                last_archive_sync = time.time()
            poll = (
                "import os, json, time\n"
                f"_w = {work!r}\n"
                f"_off = {offset}\n"
                "_p = os.path.join(_w, 'run', 'job.log')\n"
                "_p = _p if os.path.exists(_p) else os.path.join(_w, 'job.log')\n"
                "_data = b''\n"
                "if os.path.exists(_p):\n"
                "    _f = open(_p, 'rb'); _f.seek(_off); _data = _f.read(50000); _f.close()\n"
                "_ex = None\n"
                "_xp = os.path.join(_w, 'job.exit')\n"
                "if os.path.exists(_xp) and len(_data) == 0:\n"
                "    _ex = int(open(_xp).read().strip() or 1)\n"
                "_relay = []\n"
                f"_rd = {relay_dir!r}\n"
                f"_stale = {RELAY_STALE_S}\n"
                "if os.path.isdir(_rd):\n"
                "    for _f2 in os.listdir(_rd):\n"
                # A worker killed mid-request leaves its .req.json behind. Without
                # this sweep the dispatcher keeps re-listing it after the pending
                # slot expires and re-bills the local OAuth session forever.
                "        if _f2.endswith(('.req.json', '.res.json', '.tmp')):\n"
                "            try:\n"
                "                if time.time() - os.path.getmtime(os.path.join(_rd, _f2)) > _stale:\n"
                "                    os.unlink(os.path.join(_rd, _f2))\n"
                "                    continue\n"
                "            except OSError:\n"
                "                continue\n"
                "        if _f2.endswith('.req.json'):\n"
                "            _rid = _f2[:-len('.req.json')]\n"
                "            if not os.path.exists(os.path.join(_rd, _rid + '.res.json')):\n"
                "                try:\n"
                "                    _r = json.load(open(os.path.join(_rd, _f2)))\n"
                "                    _relay.append(dict(_r, id=_rid))\n"
                "                except ValueError:\n"
                "                    pass\n"
                "print(json.dumps({'off': _off + len(_data), 'exit': _ex, 'relay': _relay}))\n"
                "print(_data.decode('utf-8', 'replace'), end='')\n")
            try:
                ok, out, err = client.run(poll)
                misses = 0
            except (urllib.error.URLError, RuntimeError, OSError) as e:
                misses += 1
                if misses >= 5:
                    write_line(f"[molab] lost the notebook session ({e}) — is the "
                               "browser tab still open? Job keeps running remotely; "
                               f"log at {work}/job.log")
                    return 1
                continue
            head, _, chunk = out.partition("\n")
            try:
                status = json.loads(head)
            except ValueError:
                continue
            offset = status["off"]
            for line in chunk.splitlines():
                write_line(line)
            if status.get("relay"):
                try:
                    oauth_relay.service(client, work, [r for r in status["relay"] if r.get("kind")=="codex_oauth"], write_line, relay_dir=relay_dir, policy=policy)
                    claude_relay.service(client, work, [r for r in status["relay"] if r.get("kind")=="claude_oauth"], write_line, relay_dir=relay_dir, policy=policy)
                    wandb_relay.service(client, work, [r for r in status["relay"] if r.get("kind")=="wandb_inference"], write_line, relay_dir=relay_dir, policy=policy)
                    _service_relay(client, work, [r for r in status["relay"] if r.get("kind") not in ("codex_oauth", "claude_oauth", "wandb_inference")], write_line, relay_dir=relay_dir, policy=policy, served=served_searches)
                except Exception as e:  # noqa: BLE001 — relay is best-effort
                    write_line(f"[research-relay] servicing failed: {e}")
            if status["exit"] is not None:
                # Nothing left in the relay directory is attributable to a run
                # that has exited, so stop answering for it.
                policy.finished()
                write_line(f"[molab] remote run finished with exit {status['exit']}")
                if artifacts_dir:
                    try:
                        fetch_artifacts(client, work, artifacts_dir, write_line)
                    except Exception as e:  # noqa: BLE001 — sync is best-effort
                        write_line(f"[molab] artifact sync failed: {e}; files remain "
                                   f"at {work}/run on the notebook")
                return int(status["exit"])
