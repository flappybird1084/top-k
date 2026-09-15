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
    def test_gateway_rejects_old_anonymous_token_and_foreign_origin(self):
        from itsdangerous import URLSafeTimedSerializer
        import judges_server
        with patch.dict(os.environ,{'JUDGES_SESSION_SECRET':'test-signing-secret','JUDGES_EXPIRES_AT':str(time.time()+3600),'JUDGES_NOTEBOOKS':'unused'}),patch.object(judges_server,'NotebookPool'),patch.object(judges_server.web,'_queue'):
            app=judges_server.create_app();c=app.test_client()
            token=URLSafeTimedSerializer('test-signing-secret',salt='judges-visitor').dumps('old-anonymous')
            r=c.post('/api/session',headers={'Origin':ORIGIN,'Authorization':'Bearer '+token})
            self.assertEqual(r.status_code,401)
            self.assertEqual(c.post('/api/auth/logout',headers={'Origin':'https://evil.example'}).status_code,403)
            self.assertEqual(c.get('/api/auth/config',headers={'Origin':ORIGIN}).json,{'enabled':True})

if __name__=='__main__':unittest.main()
