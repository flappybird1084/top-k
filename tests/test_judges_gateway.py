"""Public gateway: sign-in gating, per-user integrations, and fail-closed config."""
import json
import os
from pathlib import Path
import sys
import time
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

ORIGIN = 'https://top-kernel-demo.andre520395.chatgpt.site'
EDGE = 'edge-shared-secret-value'


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    import judges_server
    for key, value in (('GITHUB_CLIENT_ID', 'cid'), ('GITHUB_CLIENT_SECRET', 'csecret'),
                       ('GITHUB_AUTH_ORIGIN', 'https://auth.example.com'),
                       ('GITHUB_AUTH_DB', str(tmp_path / 'auth.sqlite')),
                       ('GITHUB_SIGNIN_RATE', '50'),
                       ('JUDGES_EXPIRES_AT', str(time.time() + 3600)),
                       ('JUDGES_EDGE_SECRET', EDGE),
                       ('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations')),
                       ('JUDGES_ORIGIN', ORIGIN)):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(judges_server, 'NotebookPool', lambda *a, **k: None)
    # create_app() installs its own queue; record the real one first so
    # monkeypatch puts it back for the rest of the suite.
    monkeypatch.setattr(judges_server.web, '_queue',
                        getattr(judges_server.web, '_queue', None), raising=False)
    monkeypatch.setattr(judges_server.web, 'JOBS_DIR', str(tmp_path / 'jobs'))
    (tmp_path / 'jobs').mkdir()
    app = judges_server.create_app()
    return app


def sign_in(app, user_id=7, login='tester'):
    github = app.extensions['github']
    identity = {'id': 'github:%d' % user_id, 'login': login}
    return github.issue(identity)


def headers(token=None, **extra):
    h = {'Origin': ORIGIN, 'X-TopK-Edge-Auth': EDGE, **extra}
    if token:
        h['Authorization'] = 'Bearer ' + token
    return h


def test_requests_that_skip_the_edge_are_refused(gateway):
    client = gateway.test_client()
    token = sign_in(gateway)
    assert client.get('/api/health', headers={'Origin': ORIGIN}).status_code == 404
    assert client.get('/api/health', headers={'Origin': ORIGIN,
                                              'X-TopK-Edge-Auth': 'wrong'}).status_code == 404
    assert client.get('/api/health', headers=headers(token)).status_code == 200


def test_missing_edge_secret_fails_closed(tmp_path, monkeypatch):
    import judges_server
    for key, value in (('GITHUB_CLIENT_ID', 'cid'), ('GITHUB_CLIENT_SECRET', 'csecret'),
                       ('GITHUB_AUTH_ORIGIN', 'https://auth.example.com'),
                       ('GITHUB_AUTH_DB', str(tmp_path / 'auth.sqlite')),
                       ('JUDGES_EXPIRES_AT', str(time.time() + 3600)),
                       ('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations'))):
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('JUDGES_EDGE_SECRET', raising=False)
    monkeypatch.setattr(judges_server, 'NotebookPool', lambda *a, **k: None)
    monkeypatch.setattr(judges_server.web, '_queue',
                        getattr(judges_server.web, '_queue', None), raising=False)
    client = judges_server.create_app().test_client()
    assert client.get('/api/auth/config', headers={'Origin': ORIGIN}).status_code == 503


def test_health_reports_readiness_only(gateway):
    body = gateway.test_client().get('/api/health', headers=headers(sign_in(gateway))).json
    assert body == {'ready': True}


def test_every_route_needs_a_signed_in_user(gateway):
    client = gateway.test_client()
    for path in ('/api/health', '/api/integrations', '/api/runs/' + 'a' * 32):
        assert client.get(path, headers=headers()).status_code == 401, path
    # the legacy anonymous-session endpoint is gone for signed-in users too
    assert client.post('/api/session', headers=headers(sign_in(gateway))).status_code == 404


def test_integration_secrets_never_come_back(gateway):
    client = gateway.test_client()
    token = sign_in(gateway)
    saved = client.post('/api/integrations', headers=headers(token), json={
        'pair_prompt': 'connect to https://notebook.molab.run/abc --token secret-token-value',
        'wandb_api_key': 'k' * 40, 'wandb_entity': 'someone', 'wandb_project': 'kernels'})
    assert saved.status_code == 200
    body = saved.get_data(as_text=True)
    assert 'secret-token-value' not in body and 'k' * 40 not in body
    assert saved.json['notebook'] == {'configured': True, 'url': 'https://notebook.molab.run/abc'}
    assert saved.json['wandb'] == {'configured': True, 'entity': 'someone', 'project': 'kernels'}
    # and a different GitHub account sees nothing of it
    other = client.get('/api/integrations', headers=headers(sign_in(gateway, 9, 'other'))).json
    assert other['notebook']['configured'] is False and other['wandb']['configured'] is False


def test_pair_prompt_without_a_token_is_rejected(gateway):
    r = gateway.test_client().post('/api/integrations', headers=headers(sign_in(gateway)),
                                   json={'pair_prompt': 'https://notebook.molab.run/abc'})
    assert r.status_code == 400 and 'token' in r.json['error']


def test_run_needs_the_users_own_notebook(gateway):
    r = gateway.test_client().post('/api/runs', headers=headers(sign_in(gateway)),
                                   json={'repo': 'https://github.com/a/b'})
    assert r.status_code == 428 and 'Pair with agent' in r.json['error']


def test_runs_are_scoped_to_the_github_user_id(gateway, tmp_path):
    import web
    jid = 'b' * 32
    (Path(web.JOBS_DIR) / jid).mkdir(parents=True)
    (Path(web.JOBS_DIR) / jid / 'job.json').write_text(json.dumps(
        {'id': jid, 'created_at': 0, 'status': 'running', 'visitor': 'github:7'}))
    client = gateway.test_client()
    assert client.get('/api/runs/' + jid, headers=headers(sign_in(gateway, 9))).status_code == 404


def test_recorded_evidence_is_not_served_anonymously(gateway):
    client = gateway.test_client()
    assert client.get('/assets/recorded-data.js', headers=headers()).status_code == 401
    assert client.get('/assets/recorded/kernel.json', headers=headers()).status_code == 401


def test_a_users_own_notebook_is_never_replaced_by_the_operators(tmp_path, monkeypatch):
    """The data step used to fill in the server's connection file for any job
    without one — which would have handed a visitor our GPU notebook."""
    import ui_server
    import web
    monkeypatch.setattr(web, 'JOBS_DIR', str(tmp_path / 'jobs'))
    operator = tmp_path / 'molab.json'
    operator.write_text(json.dumps({'url': 'https://operator.molab.run/x', 'token': 'operator-token'}))
    monkeypatch.setenv('KEVO_MOLAB_CONNECTION_FILE', str(operator))
    monkeypatch.setattr(ui_server, 'validate_provider', lambda job: None)
    jid = 'c' * 32
    (Path(web.JOBS_DIR) / jid).mkdir(parents=True)
    job = {'id': jid, 'created_at': 0, 'status': 'awaiting_data', 'mode': 'kernel',
           'execution_target': 'molab', 'profile': 'DEV', 'visitor': 'github:7',
           'molab': {'notebook_url': 'https://mine.molab.run/y', 'connection': '--token mine'}}
    (Path(web.JOBS_DIR) / jid / 'job.json').write_text(json.dumps(job))
    queued = []
    monkeypatch.setattr(web, '_queue', type('Q', (), {'put': lambda self, j: queued.append(j)})())
    client = ui_server.app.test_client()
    r = client.post('/api/runs/' + jid + '/data',
                    json={'url': 'https://huggingface.co/datasets/x/y'})
    assert r.status_code == 202, r.get_data(as_text=True)
    saved = json.loads((Path(web.JOBS_DIR) / jid / 'job.json').read_text())
    assert saved['molab'] == job['molab']
    assert 'operator-token' not in json.dumps(saved)
