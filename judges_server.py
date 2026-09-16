"""Restricted public gateway for the time-limited judging deployment.

Every route is behind GitHub sign-in. A signed-in user brings their own GPU
notebook and their own Weights & Biases account; this process never hands out
the operator's notebooks, tokens, or API keys, and runs are owned by the
immutable GitHub numeric user id.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

from flask import Flask, jsonify, request, g, send_from_directory

import ui_server
import web
from github_signin import GitHubSignIn, RateLimiter
from kernelevo.judges_pool import NotebookPool

DEFAULT_ORIGIN = 'https://top-kernel-demo.andre520395.chatgpt.site'
# Written by the edge worker; a request that reaches the origin without them
# did not come through the edge and is refused.
EDGE_AUTH_HEADER = 'X-TopK-Edge-Auth'
CLIENT_IP_HEADER = 'X-TopK-Client-IP'
AUTH_PATHS = ('/auth/github/login', '/auth/github/callback')
PUBLIC_API = ('/api/auth/config',)


class IntegrationStore:
    """Per-user notebook and W&B settings, kept out of the job store and out of
    every API response. Files are private to the gateway user and named by a
    hash of the GitHub user id, so the directory listing names no accounts."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.lock = threading.Lock()

    def _path(self, user_id):
        return self.root / (hashlib.sha256(user_id.encode()).hexdigest() + '.json')

    def load(self, user_id):
        try:
            return json.loads(self._path(user_id).read_text())
        except (OSError, ValueError):
            return {}

    def save(self, user_id, record):
        path = self._path(user_id)
        with self.lock:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record))
            temporary.chmod(0o600)
            temporary.replace(path)

    @staticmethod
    def public(record):
        """What the browser is allowed to see: whether each integration is
        connected, plus the non-secret labels needed to confirm the right one.
        Notebook tokens and W&B API keys never appear here."""
        notebook = record.get('molab') or {}
        wandb = record.get('wandb') or {}
        return dict(
            notebook=dict(configured=bool(notebook.get('url') and notebook.get('token')),
                          url=notebook.get('url', '')),
            wandb=dict(configured=bool(wandb.get('api_key')),
                       entity=wandb.get('entity', ''), project=wandb.get('project', '')))


def parse_notebook(pair_prompt):
    from kernelevo.molab import parse_connection
    from urllib.parse import urlsplit
    url, token = parse_connection({'connection': pair_prompt})
    parts = urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        raise ValueError('The notebook URL must be an HTTPS address without credentials.')
    if not token:
        raise ValueError('That prompt has no access token. Copy the whole "Pair with agent" '
                         'prompt — the token on screen is masked, only the copied text has it.')
    return {'url': url, 'token': token}


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get('JUDGES_SESSION_SECRET') or secrets.token_hex(32)
    origin = os.environ.get('JUDGES_ORIGIN', DEFAULT_ORIGIN)
    github = GitHubSignIn(app, origin)
    # Shared secret installed on the edge worker. Missing means the deployment
    # is half-configured, and the gateway refuses everything rather than
    # accepting requests that bypassed the edge.
    edge_secret = os.environ.get('JUDGES_EDGE_SECRET', '')
    expires_at = float(os.environ['JUDGES_EXPIRES_AT'])
    app.config.update(MAX_CONTENT_LENGTH=16384, SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    admission_lock = threading.Lock()
    max_jobs = int(os.getenv('JUDGES_MAX_RUNS', '12'))
    integrations = IntegrationStore(
        os.getenv('JUDGES_INTEGRATION_DIR',
                  str(Path.home() / '.local/state/kernel-evolution/integrations')))
    # Public runs never touch an operator notebook: the queue uses whichever
    # notebook the signed-in user connected.
    pool = NotebookPool(None, web._run_job, web.load_job, web.save_job, expires_at)
    web._queue = pool
    app.extensions['notebook_pool'] = pool
    app.extensions['integrations'] = integrations
    app.extensions['github'] = github

    run_limit = RateLimiter(int(os.getenv('JUDGES_RUN_RATE', '6')), 3600)
    api_limit = RateLimiter(int(os.getenv('JUDGES_API_RATE', '240')), 60)
    # Session checks happen before there is an account to charge, so they are
    # bounded by client address instead.
    auth_limit = RateLimiter(int(os.getenv('JUDGES_AUTH_RATE', '120')), 60)

    def client_ip():
        """Only the edge may name the client, and only after proving it is the
        edge — otherwise any caller could spoof a rate-limit identity."""
        return request.headers.get(CLIENT_IP_HEADER) or request.remote_addr or 'unknown'

    github.client_key = client_ip

    def from_edge():
        presented = request.headers.get(EDGE_AUTH_HEADER, '')
        return bool(edge_secret) and hmac.compare_digest(presented, edge_secret)

    @app.before_request
    def authenticate_gateway():
        if not edge_secret:
            return jsonify(error='This service is not fully configured yet.'), 503
        if not from_edge():
            return jsonify(error='Not found'), 404
        if request.path in AUTH_PATHS and request.method == 'GET':
            return None
        if request.headers.get('Origin') != origin:
            return jsonify(error='Origin rejected'), 403
        if request.method == 'OPTIONS':
            return '', 204
        if request.method not in ('GET', 'POST', 'HEAD'):
            return jsonify(error='Method not allowed'), 405
        if request.path in PUBLIC_API:
            return None
        if not github.enabled:
            return jsonify(error='GitHub sign-in is not configured yet.'), 503
        if request.path in ('/api/auth/me', '/api/auth/logout'):
            if not auth_limit.allow(client_ip()):
                return jsonify(error='Too many requests. Slow down and try again shortly.'), 429
            return None
        user = github.identity()
        if not user:
            return jsonify(error='Sign in with GitHub to continue.'), 401
        g.visitor = user['id']
        if not api_limit.allow(g.visitor):
            return jsonify(error='Too many requests. Slow down and try again shortly.'), 429
        if request.method == 'POST' and time.time() >= expires_at:
            return jsonify(error='The live judging window has ended.'), 410

    @app.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, Idempotency-Key'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Vary'] = 'Origin'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers.setdefault('Referrer-Policy', 'same-origin')
        return response

    def owned(jid):
        if not re.fullmatch('[a-f0-9]{32}', jid):
            return False
        job = ui_server.read_json(ui_server.job_path(jid) / 'job.json', {})
        return bool(job.get('visitor')) and job.get('visitor') == g.visitor

    def delegate(path):
        # The development app only receives a narrowly allowed local request.
        headers = {'Content-Type': 'application/json'}
        if request.headers.get('Idempotency-Key'):
            headers['Idempotency-Key'] = hashlib.sha256(
                (g.visitor + request.headers['Idempotency-Key']).encode()).hexdigest()
        with ui_server.app.test_client() as client:
            r = client.open(path, method=request.method, data=request.get_data(), headers=headers)
            return app.response_class(r.data, status=r.status_code, content_type=r.content_type)

    @app.get('/api/health')
    def health():
        # Readiness only: queue depth, worker counts and the exact expiry are
        # operational detail, not something an anonymous prober should learn.
        return dict(ready=time.time() < expires_at)

    @app.route('/api/integrations', methods=['GET', 'POST'])
    def user_integrations():
        record = integrations.load(g.visitor)
        if request.method == 'GET':
            return jsonify(IntegrationStore.public(record))
        if not request.is_json:
            return jsonify(error='Send JSON'), 415
        payload = request.get_json(silent=True) or {}
        updated = dict(record)
        pair = str(payload.get('pair_prompt') or '').strip()
        if pair:
            if len(pair) > 4000:
                return jsonify(error='That pair prompt is too long.'), 400
            try:
                updated['molab'] = parse_notebook(pair)
            except ValueError as e:
                return jsonify(error=str(e)), 400
        key = str(payload.get('wandb_api_key') or '').strip()
        entity = str(payload.get('wandb_entity') or '').strip()
        project = str(payload.get('wandb_project') or '').strip()
        if key or entity or project:
            wandb = dict(updated.get('wandb') or {})
            for name, value, pattern in (('api_key', key, r'[A-Za-z0-9_\-]{20,120}'),
                                         ('entity', entity, r'[\w.-]{1,64}'),
                                         ('project', project, r'[\w.-]{1,64}')):
                if not value:
                    continue
                if not re.fullmatch(pattern, value):
                    return jsonify(error='Check the W&B ' + name.replace('_', ' ') + '.'), 400
                wandb[name] = value
            updated['wandb'] = wandb
        updated['updated_at'] = time.time()
        integrations.save(g.visitor, updated)
        return jsonify(IntegrationStore.public(updated))

    @app.post('/api/runs')
    def create_run():
        if not request.is_json:
            return jsonify(error='Send JSON'), 415
        if len(request.headers.get('Idempotency-Key', '')) > 200:
            return jsonify(error='Invalid request identifier'), 400
        record = integrations.load(g.visitor)
        notebook = record.get('molab') or {}
        if not (notebook.get('url') and notebook.get('token')):
            return jsonify(error='Connect your own marimo notebook first: start a notebook, '
                                 'choose "Pair with agent", and paste that prompt in setup.'), 428
        if not run_limit.allow(g.visitor):
            return jsonify(error='You have started several runs recently. Try again later.'), 429
        with admission_lock, ui_server.lock:
            jobs = web.list_jobs()
            pending = [j for j in jobs if j['status'] in ('exploring', 'awaiting_data', 'queued', 'running')]
            if len(pending) >= max_jobs:
                return jsonify(error='All GPU queue slots are reserved. Please try again later.'), 429
            own = [j for j in pending if j.get('visitor') == g.visitor]
            if own:
                return jsonify(id=own[0]['id']), 202
            response = delegate('/api/runs')
            if response.status_code == 202:
                jid = response.get_json()['id']
                job = web.load_job(jid)
                job['visitor'] = g.visitor
                job['molab'] = {'notebook_url': notebook['url'],
                                'connection': '--token ' + notebook['token']}
                job['wandb'] = dict(record.get('wandb') or {})
                web.save_job(job)
            return response

    @app.route('/api/runs/<jid>', methods=['GET'])
    @app.route('/api/runs/<jid>/<action>', methods=['GET', 'POST'])
    def run(jid, action=None):
        if not owned(jid):
            return jsonify(error='Run not found for this account'), 404
        allowed = {None: 'GET', 'runtime': 'GET', 'wandb': 'GET', 'data': 'POST'}
        if action not in allowed or request.method != allowed[action]:
            return jsonify(error='Not found'), 404
        return delegate('/api/runs/' + jid + ('/' + action if action else ''))

    @app.get('/api/recorded/runs')
    def recorded_runs():
        """The recorded demo evidence carries private Weave links and run logs
        from the operator's W&B account, so it is served here — to signed-in
        users — instead of sitting in the published static bundle."""
        source = ui_server.UI / 'assets' / 'recorded-data.js'
        try:
            text = source.read_text()
        except OSError:
            return jsonify(error='No recorded evidence is installed.'), 404
        _, _, body = text.partition('=')
        try:
            return app.response_class(json.dumps(json.loads(body.strip().rstrip(';'))),
                                      content_type='application/json')
        except ValueError:
            return jsonify(error='No recorded evidence is installed.'), 404

    @app.get('/')
    def index():
        return send_from_directory(ui_server.UI, 'index.html')

    @app.get('/assets/<path:name>')
    def assets(name):
        if Path(name).suffix.lower() not in {'.html', '.js', '.css', '.png', '.svg', '.woff2', '.json'}:
            return jsonify(error='Not found'), 404
        if name.startswith('recorded') and not getattr(g, 'visitor', None):
            return jsonify(error='Not found'), 404
        return send_from_directory(ui_server.UI / 'assets', name)

    @app.get('/<name>')
    def public_file(name):
        if name not in {'evolution.js', 'demo.js', 'index.html', 'front.js', 'front.css',
                        'live-api.js', 'github-auth.css', 'workspace.html', 'run.js',
                        'run.css', 'results.html', 'results.js'}:
            return jsonify(error='Not found'), 404
        return send_from_directory(ui_server.UI, name)

    return app


if __name__ == '__main__':
    import uvicorn
    from starlette.middleware.wsgi import WSGIMiddleware
    application = create_app()
    application.extensions['notebook_pool'].start()
    uvicorn.run(WSGIMiddleware(application), host='127.0.0.1', port=8768)
