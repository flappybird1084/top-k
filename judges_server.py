"""Restricted public gateway for the time-limited judging deployment.

Signed browser sessions own their runs. Only the published site's origin and
the run-scoped API routes are accepted; development/admin routes stay private.
"""
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, g, send_from_directory
from itsdangerous import URLSafeTimedSerializer, BadSignature

import ui_server
import web
from kernelevo.judges_pool import NotebookPool


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ['JUDGES_SESSION_SECRET']
    signer = URLSafeTimedSerializer(app.secret_key, salt='judges-visitor')
    origin = 'https://top-kernel-demo.andre520395.chatgpt.site'
    expires_at = float(os.environ['JUDGES_EXPIRES_AT'])
    app.config.update(MAX_CONTENT_LENGTH=16384, SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    admission_lock = threading.Lock()
    max_jobs = int(os.getenv('JUDGES_MAX_RUNS', '12'))
    pool = NotebookPool(os.environ['JUDGES_NOTEBOOKS'], web._run_job,
                        web.load_job, web.save_job, expires_at)
    web._queue = pool
    app.extensions['notebook_pool'] = pool

    @app.before_request
    def authenticate_gateway():
        if request.headers.get('Origin') != origin:
            return jsonify(error='Origin rejected'), 403
        if request.method == 'OPTIONS':
            return '', 204
        if request.method not in ('GET', 'POST', 'HEAD'):
            return jsonify(error='Method not allowed'), 405
        if request.method == 'POST' and time.time() >= expires_at:
            return jsonify(error='The live judging window has ended.'), 410
        if request.path == '/api/session':
            return None
        try:
            g.visitor = signer.loads(request.headers.get('Authorization', '').removeprefix('Bearer '), max_age=21600)
        except BadSignature:
            return jsonify(error='Start a new browser session'), 401

    @app.post('/api/session')
    def new_session():
        return jsonify(token=signer.dumps(secrets.token_hex(24)))

    @app.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, Idempotency-Key'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Vary'] = 'Origin'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    def owned(jid):
        if not re.fullmatch('[a-f0-9]{32}', jid):
            return False
        job = ui_server.read_json(ui_server.job_path(jid) / 'job.json', {})
        return job.get('visitor') == g.visitor

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
        return dict(accepting=time.time() < expires_at, expires_at=expires_at, workers=pool.status())

    @app.post('/api/runs')
    def create_run():
        if not request.is_json:
            return jsonify(error='Send JSON'), 415
        if len(request.headers.get('Idempotency-Key', '')) > 200:
            return jsonify(error='Invalid request identifier'), 400
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
                web.save_job(job)
            return response

    @app.route('/api/runs/<jid>', methods=['GET'])
    @app.route('/api/runs/<jid>/<action>', methods=['GET', 'POST'])
    def run(jid, action=None):
        if not owned(jid):
            return jsonify(error='Run not found in this browser session'), 404
        allowed = {None: 'GET', 'runtime': 'GET', 'wandb': 'GET', 'data': 'POST'}
        if action not in allowed or request.method != allowed[action]:
            return jsonify(error='Not found'), 404
        return delegate('/api/runs/' + jid + ('/' + action if action else ''))

    @app.get('/')
    def index():
        return send_from_directory(ui_server.UI, 'index.html')

    @app.get('/assets/<path:name>')
    def assets(name):
        if Path(name).suffix.lower() not in {'.html', '.js', '.css', '.png', '.svg', '.woff2', '.json'}:
            return jsonify(error='Not found'), 404
        return send_from_directory(ui_server.UI / 'assets', name)

    @app.get('/<name>')
    def public_file(name):
        if name not in {'evolution.js', 'demo.js', 'index.html', 'front.js', 'front.css',
                        'workspace.html', 'run.js', 'run.css', 'results.html', 'results.js'}:
            return jsonify(error='Not found'), 404
        return send_from_directory(ui_server.UI, name)

    return app


if __name__ == '__main__':
    import uvicorn
    from starlette.middleware.wsgi import WSGIMiddleware
    application = create_app()
    application.extensions['notebook_pool'].start()
    uvicorn.run(WSGIMiddleware(application), host='127.0.0.1', port=8768)
