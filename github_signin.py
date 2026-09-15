"""GitHub identity for the existing gateway; OAuth credentials never reach the GPU."""
import base64
from contextlib import closing, contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from flask import Blueprint, Response, jsonify, redirect, request


class GitHubSignIn:
    def __init__(self, app, site_origin):
        self.origin = site_origin
        self.client_id = os.getenv('GITHUB_CLIENT_ID', '')
        self.client_secret = os.getenv('GITHUB_CLIENT_SECRET', '')
        self.backend = os.getenv('GITHUB_AUTH_ORIGIN', '').rstrip('/')
        self.enabled = bool(self.client_id and self.client_secret and self.backend)
        if any((self.client_id, self.client_secret, self.backend)) and not self.enabled:
            raise ValueError('Configure all GitHub OAuth settings together')
        if self.enabled:
            u = urlsplit(self.backend)
            if u.scheme != 'https' or not u.hostname or u.path or u.query or u.fragment or u.username or u.password:
                raise ValueError('GITHUB_AUTH_ORIGIN must be a fixed HTTPS origin')
        self.callback = self.backend + '/auth/github/callback'
        self.path = Path(os.getenv('GITHUB_AUTH_DB', str(Path.home()/'.local/state/kernel-evolution/github-auth.sqlite')))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.chmod(0o600)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS states (id TEXT PRIMARY KEY, verifier TEXT, expires REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT, login TEXT, expires REAL)')
        bp = Blueprint('github_signin', __name__)
        bp.add_url_rule('/auth/github/login', view_func=self.login)
        bp.add_url_rule('/auth/github/callback', view_func=self.finish)
        bp.add_url_rule('/api/auth/config', view_func=lambda: jsonify(enabled=self.enabled))
        bp.add_url_rule('/api/auth/me', view_func=self.me)
        bp.add_url_rule('/api/auth/logout', view_func=self.logout, methods=['POST'])
        app.register_blueprint(bp)

    @contextmanager
    def db(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            yield db

    @staticmethod
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def github(url, data=None, token=None):
        headers = {'Accept': 'application/json', 'User-Agent': 'Top-Kernel'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        body = None
        if data is not None:
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
            body = urlencode(data).encode()
        with urlopen(Request(url, data=body, headers=headers), timeout=15) as response:
            return json.load(response)

    def login(self):
        if not self.enabled:
            return self.result({'error': 'GitHub sign-in is not configured yet.'}, 503)
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        with self.db() as db:
            db.execute('DELETE FROM states WHERE expires < ?', (time.time(),))
            db.execute('INSERT INTO states VALUES (?,?,?)', (self.digest(state), verifier, time.time()+600))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        params = dict(client_id=self.client_id, redirect_uri=self.callback, state=state,
                      scope='read:user', code_challenge=challenge, code_challenge_method='S256')
        response = redirect('https://github.com/login/oauth/authorize?' + urlencode(params))
        response.set_cookie('__Host-topk-oauth', state, secure=True, httponly=True, samesite='Lax', max_age=600, path='/')
        return response

    def finish(self):
        state = request.args.get('state', '')
        cookie = request.cookies.get('__Host-topk-oauth', '')
        if not self.enabled or not state or not hmac.compare_digest(state, cookie):
            return self.result({'error': 'Sign-in expired or could not be verified. Please try again.'}, 400)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT verifier,expires FROM states WHERE id=?', (self.digest(state),)).fetchone()
            db.execute('DELETE FROM states WHERE id=?', (self.digest(state),))
        if not row or row[1] < time.time() or request.args.get('error') or not request.args.get('code'):
            return self.result({'error': 'Sign-in was cancelled or expired. Please try again.'}, 400)
        try:
            exchange = self.github('https://github.com/login/oauth/access_token', dict(
                client_id=self.client_id, client_secret=self.client_secret,
                redirect_uri=self.callback, code=request.args['code'], code_verifier=row[0]))
            access = exchange.get('access_token')
            if not isinstance(access, str) or not access:
                raise ValueError('No access token')
            user = self.github('https://api.github.com/user', token=access)
            if not isinstance(user.get('id'), int) or not isinstance(user.get('login'), str):
                raise ValueError('Invalid identity')
        except Exception:
            return self.result({'error': 'GitHub could not complete sign-in. Please try again.'}, 502)
        token = secrets.token_urlsafe(48)
        identity = {'id': 'github:'+str(user['id']), 'login': user['login']}
        with self.db() as db:
            db.execute('DELETE FROM sessions WHERE expires < ?', (time.time(),))
            db.execute('INSERT INTO sessions VALUES (?,?,?,?)', (self.digest(token), identity['id'], identity['login'], time.time()+21600))
        return self.result({'token': token, 'user': identity})

    def result(self, payload, status=200):
        nonce = secrets.token_urlsafe(24)
        message = json.dumps(dict(type='topk-github-auth', **payload)).replace('<', '\\u003c')
        # Fixed target origin, no token in URLs, and no GitHub access token in the response.
        html = f'''<!doctype html><meta charset="utf-8"><title>Top-Kernel sign-in</title>
<p id="status"></p><script nonce="{nonce}">
const result={message};document.getElementById('status').textContent=result.error||'Signed in. You can close this window.';
if(window.opener){{window.opener.postMessage(result,{json.dumps(self.origin)});if(!result.error)window.close();}}
</script>'''
        response = Response(html, status=status, content_type='text/html; charset=utf-8')
        response.headers['Content-Security-Policy'] = f"default-src 'none'; script-src 'nonce-{nonce}'; frame-ancestors 'none'; base-uri 'none'"
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.delete_cookie('__Host-topk-oauth', secure=True, httponly=True, samesite='Lax', path='/')
        return response

    def identity(self):
        token = request.headers.get('Authorization', '').removeprefix('Bearer ')
        if not token or len(token)>256:
            return None
        with self.db() as db:
            row = db.execute('SELECT user_id,login FROM sessions WHERE id=? AND expires>?', (self.digest(token), time.time())).fetchone()
        return {'id': row[0], 'login': row[1]} if row else None

    def me(self):
        user = self.identity()
        return (jsonify(user=user), 200) if user else (jsonify(error='Sign in with GitHub to continue.'), 401)

    def logout(self):
        token = request.headers.get('Authorization', '').removeprefix('Bearer ')
        with self.db() as db:
            db.execute('DELETE FROM sessions WHERE id=?', (self.digest(token),))
        return '', 204
