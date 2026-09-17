import json
import os
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs
from flask import Flask
from github_signin import GitHubSignIn

ORIGIN='https://top-kernel-demo.andre520395.chatgpt.site'

class GitHubAuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        env=patch.dict(os.environ,{'GITHUB_CLIENT_ID':'test-id','GITHUB_CLIENT_SECRET':'test-secret',
            'GITHUB_AUTH_ORIGIN':'https://auth.example.com','GITHUB_AUTH_DB':str(Path(self.tmp.name)/'auth.sqlite')})
        env.start();self.addCleanup(env.stop)
        self.app=Flask(__name__);self.auth=GitHubSignIn(self.app,ORIGIN);self.client=self.app.test_client()
    def begin(self):
        r=self.client.get('/auth/github/login',base_url='https://auth.example.com')
        self.assertIn('Secure',r.headers['Set-Cookie']);self.assertIn('HttpOnly',r.headers['Set-Cookie'])
        q=parse_qs(urlsplit(r.location).query)
        self.assertEqual(q['scope'],['read:user']);self.assertEqual(q['code_challenge_method'],['S256'])
        return q['state'][0]
    def finish(self,state):
        with patch.object(self.auth,'github',side_effect=[{'access_token':'github-secret-token'},{'id':42,'login':'tester'}]) as api:
            r=self.client.get('/auth/github/callback',query_string={'state':state,'code':'code'},base_url='https://auth.example.com')
            return r,api
    def token(self,response):
        return json.loads(re.search(r'const result=(.*?);document',response.text).group(1))['token']
    def test_signin_pkce_identity_and_no_github_token_leak(self):
        state=self.begin();r,api=self.finish(state);self.assertEqual(r.status_code,200)
        self.assertNotIn('github-secret-token',r.text);self.assertNotIn('test-secret',r.text)
        self.assertIn('code_verifier',api.call_args_list[0].args[1])
        token=self.token(r);me=self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token})
        self.assertEqual(me.json['user'],{'id':'github:42','login':'tester'})
        with self.auth.db() as db:self.assertNotEqual(db.execute('SELECT id FROM sessions').fetchone()[0],token)
    def test_unapproved_callback_does_not_create_a_session(self):
        self.auth.allowed_login=lambda login: login == 'approved'
        state=self.begin();r,_=self.finish(state)
        self.assertEqual(r.status_code,403)
        self.assertNotIn('token',r.text)
        with self.auth.db() as db:self.assertIsNone(db.execute('SELECT id FROM sessions').fetchone())
    def test_wrong_state_rejected_without_exchange(self):
        self.begin();r,api=self.finish('wrong');self.assertEqual(r.status_code,400);api.assert_not_called()
    def test_callback_in_other_browser_rejected(self):
        state=self.begin();other=self.app.test_client();r=other.get('/auth/github/callback',query_string={'state':state,'code':'code'});self.assertEqual(r.status_code,400)
    def test_state_single_use(self):
        state=self.begin();r,_=self.finish(state);self.assertEqual(r.status_code,200)
        self.client.set_cookie('__Host-topk-oauth',state,domain='auth.example.com')
        r,api=self.finish(state);self.assertEqual(r.status_code,400);api.assert_not_called()
    def test_state_expired(self):
        state=self.begin()
        with self.auth.db() as db:db.execute('UPDATE states SET expires=0')
        r,api=self.finish(state);self.assertEqual(r.status_code,400);api.assert_not_called()
    def test_logout_revokes_session(self):
        r,_=self.finish(self.begin());h={'Authorization':'Bearer '+self.token(r)}
        self.assertEqual(self.client.post('/api/auth/logout',headers=h).status_code,204)
        self.assertEqual(self.client.get('/api/auth/me',headers=h).status_code,401)
    def test_session_expired(self):
        r,_=self.finish(self.begin());token=self.token(r)
        with self.auth.db() as db:db.execute('UPDATE sessions SET expires=0')
        self.assertEqual(self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token}).status_code,401)
    def test_provider_error_sanitized(self):
        state=self.begin()
        with patch.object(self.auth,'github',side_effect=RuntimeError('secret diagnostics')):
            r=self.client.get('/auth/github/callback',query_string={'state':state,'code':'code'},base_url='https://auth.example.com')
        self.assertEqual(r.status_code,502);self.assertNotIn('secret diagnostics',r.text)
    def test_session_survives_checks_and_rejects_malformed_bearers(self):
        """Finding 4: /api/auth/me used to mint a new token and revoke the one
        presented, so a second tab — or a status poll racing a run request —
        logged the user out mid-run."""
        r,_=self.finish(self.begin());token=self.token(r)
        first=self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token})
        self.assertEqual(first.status_code,200)
        self.assertNotIn('token',first.json)
        # the same credential keeps working, from as many tabs as you like
        for _ in range(3):
            self.assertEqual(self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token}).status_code,200)
        for header in ('Bearer','Bearer  '+token,'bearer '+token,'Bearer'+token,'Bearer '+token+' '):
            self.assertEqual(self.client.get('/api/auth/me',headers={'Authorization':header}).status_code,401,header)
    def test_activity_slides_the_idle_window_forward(self):
        """A run watched for an hour must not expire out from under its owner:
        authenticated activity extends the session."""
        import github_signin
        r,_=self.finish(self.begin());token=self.token(r)
        with self.auth.db() as db:
            db.execute('UPDATE sessions SET expires=?',(time.time()+30,))
        self.assertEqual(self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token}).status_code,200)
        with self.auth.db() as db:
            expires=db.execute('SELECT expires FROM sessions').fetchone()[0]
        self.assertGreater(expires,time.time()+github_signin.SESSION_IDLE_S-60)
    def test_activity_cannot_outlive_the_absolute_session_age(self):
        import github_signin
        r,_=self.finish(self.begin());token=self.token(r)
        with self.auth.db() as db:
            db.execute('UPDATE sessions SET issued=?',(time.time()-github_signin.SESSION_MAX_S-1,))
        self.assertEqual(self.client.get('/api/auth/me',headers={'Authorization':'Bearer '+token}).status_code,401)
    def test_logout_requires_a_real_session(self):
        self.assertEqual(self.client.post('/api/auth/logout').status_code,401)
        self.assertEqual(self.client.post('/api/auth/logout',headers={'Authorization':'Bearer not-a-session-token-value'}).status_code,401)
    def test_sign_in_attempts_are_rate_limited(self):
        self.auth.sign_in_limit=github_limiter=__import__('github_signin').RateLimiter(2,600)
        self.assertEqual(self.client.get('/auth/github/login',base_url='https://auth.example.com').status_code,302)
        self.assertEqual(self.client.get('/auth/github/login',base_url='https://auth.example.com').status_code,302)
        self.assertEqual(self.client.get('/auth/github/login',base_url='https://auth.example.com').status_code,429)
        self.assertTrue(len(github_limiter.windows)<=2)
    def test_rate_limiter_table_is_bounded(self):
        from github_signin import RateLimiter
        limiter=RateLimiter(5,60,capacity=10)
        for i in range(200):limiter.allow('client-%d'%i)
        self.assertLessEqual(len(limiter.windows),10)

if __name__=='__main__':unittest.main()
