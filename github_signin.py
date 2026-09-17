"""GitHub identity for the existing gateway; OAuth credentials never reach the GPU."""
import base64
from contextlib import closing, contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from flask import Blueprint, Response, jsonify, redirect, request

# Tokens are secrets.token_urlsafe(48); anything outside this shape is a probe.
TOKEN = re.compile(r'[A-Za-z0-9_-]{32,256}')

# Idle lifetime of a browser session, and the absolute ceiling no amount of
# activity extends past. Authenticated API activity slides the idle window
# forward; only a fresh GitHub sign-in mints a new token. Rotating on every
# identity check used to log people out mid-run — two tabs, or a status poll
# racing a run request, each revoked the token the other was still holding
# (audit finding 4).
SESSION_IDLE_S = 3600
SESSION_MAX_S = 6 * 3600


def bearer(header):
    """Strict `Authorization: Bearer <token>` parsing.

    `removeprefix` accepted 'Bearerx', 'bearer ...' with the space inside the
    token, and whitespace-padded values — each one a different string hashing
    to a different session row, which made the same credential behave
    differently depending on how it was spelled."""
    if not isinstance(header, str) or not header.startswith('Bearer '):
        return ''
    token = header[len('Bearer '):]
    return token if TOKEN.fullmatch(token) else ''


class RateLimiter:
    """Fixed-window counter with a hard cap on tracked keys.

    An unbounded dict keyed by client address is itself the attack: enough
    distinct sources and the gateway runs out of memory. At capacity this
    prunes expired windows and, failing that, refuses — the limiter never
    grows past `capacity` entries."""

    def __init__(self, limit, window, capacity=4096):
        self.limit, self.window, self.capacity = limit, window, capacity
        self.windows = {}

    def allow(self, key):
        now = time.time()
        start = now - (now % self.window)
        if key not in self.windows and len(self.windows) >= self.capacity:
            for stale, (began, _n) in list(self.windows.items()):
                if began < start:
                    del self.windows[stale]
            if len(self.windows) >= self.capacity:
                return False
        began, count = self.windows.get(key, (start, 0))
        if began < start:
            began, count = start, 0
        self.windows[key] = (began, count + 1)
        return count < self.limit


class GitHubSignIn:
    def __init__(self, app, site_origin, allowed_login=None):
        self.origin = site_origin
        self.allowed_login = allowed_login or (lambda login: True)
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
            db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT, login TEXT, expires REAL, issued REAL)')
            try:   # databases created before rotation existed
                db.execute('ALTER TABLE sessions ADD COLUMN issued REAL')
            except sqlite3.OperationalError:
                pass
        self.sign_in_limit = RateLimiter(int(os.getenv('GITHUB_SIGNIN_RATE', '12')), 600)
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

    def client_key(self):
        """Overridden by the gateway, which knows which forwarded address it
        is allowed to believe."""
        return request.remote_addr or 'unknown'

    def login(self):
        if not self.enabled:
            return self.result({'error': 'GitHub sign-in is not configured yet.'}, 503)
        if not self.sign_in_limit.allow(self.client_key()):
            return self.result({'error': 'Too many sign-in attempts. Wait a few minutes and try again.'}, 429)
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
        if not self.sign_in_limit.allow(self.client_key()):
            return self.result({'error': 'Too many sign-in attempts. Wait a few minutes and try again.'}, 429)
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
        identity = {'id': 'github:'+str(user['id']), 'login': user['login']}
        if not self.allowed(identity):
            return self.result({'error': 'This GitHub account is not approved for Top-Kernel yet.'}, 403)
        return self.result({'token': self.issue(identity), 'user': identity})

    def allowed(self, identity):
        return bool(identity and isinstance(identity.get('login'), str)
                    and self.allowed_login(identity['login']))

    def issue(self, identity, issued=None):
        """Mint a session. Called on a completed GitHub sign-in and nowhere
        else. `issued` carries the original sign-in time so a future rotation
        could not push a session past SESSION_MAX_S."""
        token, now = secrets.token_urlsafe(48), time.time()
        issued = issued or now
        expires = min(now + SESSION_IDLE_S, issued + SESSION_MAX_S)
        with self.db() as db:
            db.execute('DELETE FROM sessions WHERE expires < ?', (now,))
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',
                       (self.digest(token), identity['id'], identity['login'], expires, issued))
        return token

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
        """Look up the presented session and, if it is live, slide its idle
        window forward. One statement, so concurrent requests on the same
        session cannot race each other into a logout."""
        token = bearer(request.headers.get('Authorization', ''))
        if not token:
            return None
        now, digest = time.time(), self.digest(token)
        with self.db() as db:
            row = db.execute('SELECT user_id,login,issued FROM sessions WHERE id=? AND expires>?',
                             (digest, now)).fetchone()
            if not row:
                return None
            issued = row[2] or now
            if issued + SESSION_MAX_S <= now:
                db.execute('DELETE FROM sessions WHERE id=?', (digest,))
                return None
            db.execute('UPDATE sessions SET expires=?, issued=? WHERE id=?',
                       (min(now + SESSION_IDLE_S, issued + SESSION_MAX_S), issued, digest))
        return {'id': row[0], 'login': row[1], 'issued': issued}

    def me(self):
        """Who is signed in. Answering this is activity, so it extends the
        session (in identity()) — but it hands back no new token: a long run
        monitored from two tabs must not log itself out."""
        user = self.identity()
        if not user:
            return jsonify(error='Sign in with GitHub to continue.'), 401
        if not self.allowed(user):
            return jsonify(error='This GitHub account is not approved for Top-Kernel yet.'), 403
        return jsonify(user={'id': user['id'], 'login': user['login']}), 200

    def revoke(self):
        token = bearer(request.headers.get('Authorization', ''))
        if not token:
            return False
        with self.db() as db:
            deleted = db.execute('DELETE FROM sessions WHERE id=?', (self.digest(token),))
        return bool(deleted.rowcount)

    def logout(self):
        """Guarded: an unauthenticated or malformed call is refused rather than
        silently answering 204 for a session it never held."""
        if not self.revoke():
            return jsonify(error='Sign in with GitHub to continue.'), 401
        return '', 204
