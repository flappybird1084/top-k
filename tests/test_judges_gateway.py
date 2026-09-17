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
                       ('JUDGES_ALLOWED_GITHUB_LOGINS', 'tester,other,flappybird1084,andred1729'),
                       ('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations')),
                       ('JUDGES_ORIGIN', ORIGIN)):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(judges_server, 'NotebookPool', lambda *a, **k: None)
    monkeypatch.setattr(judges_server, 'validate_notebook_connection', lambda notebook: None)
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


def test_unlisted_github_session_cannot_use_any_protected_route(gateway):
    client = gateway.test_client()
    token = sign_in(gateway, 99, 'not-approved')
    assert client.get('/api/auth/me', headers=headers(token)).status_code == 403
    assert client.get('/api/health', headers=headers(token)).status_code == 403
    allowed = sign_in(gateway, 100, 'andred1729')
    assert client.get('/api/health', headers=headers(allowed)).status_code == 200


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


def test_expired_pair_prompt_is_rejected_before_it_is_saved(gateway, monkeypatch):
    import judges_server
    monkeypatch.setattr(judges_server, 'validate_notebook_connection',
                        lambda notebook: (_ for _ in ()).throw(
                            ValueError('That marimo notebook is no longer active.')))
    r = gateway.test_client().post('/api/integrations', headers=headers(sign_in(gateway)),
                                   json={'pair_prompt': 'connect to '
                                         'https://notebook.molab.run/abc '
                                         '--token secret-token-value'})
    assert r.status_code == 400
    assert r.json['error'] == 'That marimo notebook is no longer active.'


def test_expired_saved_notebook_is_rejected_before_a_run(gateway, monkeypatch):
    import judges_server
    client = gateway.test_client()
    token = sign_in(gateway)
    assert client.post('/api/integrations', headers=headers(token), json={
        'pair_prompt': 'connect to https://notebook.molab.run/abc '
                       '--token secret-token-value'}).status_code == 200
    monkeypatch.setattr(judges_server, 'validate_notebook_connection',
                        lambda notebook: (_ for _ in ()).throw(
                            ValueError('That marimo notebook is no longer active.')))
    r = client.post('/api/runs', headers=headers(token, **{'Idempotency-Key': 'x'}),
                    json={'repo': 'https://github.com/a/b'})
    assert r.status_code == 428
    assert r.json['error'] == 'That marimo notebook is no longer active.'


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


# ---- finding 1: only marimo's own notebook hosting may be paired ----

@pytest.mark.parametrize('prompt', [
    'connect to https://evil.example.com/abc --token secret-token-value',
    'connect to https://169.254.169.254/latest/meta-data --token secret-token-value',
    'connect to https://sb-abc.sb.molab.run.attacker.test/x --token secret-token-value',
    'connect to https://sb-abc.sb.molab.run:8443/x --token secret-token-value',
    'connect to https://user:pw@sb-abc.sb.molab.run --token secret-token-value',
    'connect to http://sb-abc.sb.molab.run --token secret-token-value',
    'connect to https://sb-abc.sb.molab.run/a/../../x --token secret-token-value',
])
def test_only_marimo_notebook_addresses_are_accepted(prompt):
    """The gateway connects to whatever address is pasted here, so an
    arbitrary one turns this server into a request forwarder into its own
    network."""
    import judges_server
    with pytest.raises(ValueError):
        judges_server.parse_notebook(prompt)


def test_a_real_molab_sandbox_url_is_accepted():
    import judges_server
    assert judges_server.parse_notebook(
        'Pair with agent: https://sb-9f2c1a.sb.molab.run --token abc123DEF456') == {
            'url': 'https://sb-9f2c1a.sb.molab.run', 'token': 'abc123DEF456'}


def test_a_query_or_fragment_is_refused(gateway):
    r = gateway.test_client().post('/api/integrations', headers=headers(sign_in(gateway)),
                                   json={'pair_prompt': 'https://sb-a.sb.molab.run/x?next=1'
                                                        ' --token secret-token-value'})
    assert r.status_code == 400


# ---- finding 9: the secret file is never briefly world-readable ----

def test_integration_records_are_private_from_the_first_byte(tmp_path):
    import stat
    from kernelevo.integrations import IntegrationStore
    store = IntegrationStore(tmp_path / 'integrations')
    store.save('github:7', {'wandb': {'api_key': 'k' * 40}})
    path = store._path('github:7')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


# ---- finding 6: a public run reports to its owner's W&B, never the server's ----

def test_a_public_run_never_borrows_the_servers_wandb_account(tmp_path, monkeypatch):
    import web
    from kernelevo.integrations import IntegrationStore
    monkeypatch.setenv('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations'))
    monkeypatch.setenv('WANDB_API_KEY', 'operator-key-value')
    monkeypatch.setenv('WANDB_ENTITY', 'operator-entity')
    job = {'id': 'd' * 32, 'visitor': 'github:7', 'judge_expires_at': 1, 'wandb': {}}
    # owner has connected nothing yet: the run reports nowhere
    assert 'WANDB_API_KEY' not in web._job_env(job, own_only=True)
    IntegrationStore(tmp_path / 'integrations').save(
        'github:7', {'wandb': {'api_key': 'visitor-key-value', 'entity': 'visitor'}})
    env = web._job_env(job, own_only=True)
    assert env['WANDB_API_KEY'] == 'visitor-key-value'
    assert env['WANDB_ENTITY'] == 'visitor'
    assert 'operator-key-value' not in json.dumps(env)


def test_the_observer_runs_with_the_owners_credentials_only(tmp_path, monkeypatch):
    import web
    from kernelevo.integrations import IntegrationStore
    from kernelevo.judges_pool import NotebookPool
    monkeypatch.setattr(web, 'JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setenv('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations'))
    monkeypatch.setenv('WANDB_API_KEY', 'operator-key-value')
    jid = 'e' * 32
    (Path(web.JOBS_DIR) / jid).mkdir(parents=True)
    IntegrationStore(tmp_path / 'integrations').save(
        'github:7', {'wandb': {'api_key': 'visitor-key-value', 'project': 'mine'}})
    launched = []
    monkeypatch.setattr('subprocess.Popen',
                        lambda *a, **k: launched.append(k.get('env')) or None)
    pool = NotebookPool(None, lambda j: None, lambda j: {}, lambda j: None,
                        time.time() + 3600)
    pool._start_observer(jid, {'id': jid, 'visitor': 'github:7'})
    assert launched and launched[0]['WANDB_API_KEY'] == 'visitor-key-value'
    assert launched[0]['WANDB_PROJECT'] == 'mine'
    # a visitor with no W&B account of their own gets no observer at all
    launched.clear()
    pool._start_observer(jid, {'id': jid, 'visitor': 'github:99'})
    assert launched == []


def test_wandb_metrics_are_read_with_the_owners_credentials(monkeypatch):
    """The live chart reader must not fall back to the gateway operator's
    ambient W&B account after the owner-specific observer has published it."""
    import sys
    import types
    import ui_server
    import kernelevo.integrations as integrations

    jid = 'f' * 32
    used = []

    class FakeApi:
        def __init__(self, api_key=None, timeout=None):
            used.append((api_key, timeout))

        def run(self, path):
            assert path == 'visitor/project/run1'
            return types.SimpleNamespace(name='run', state='running',
                                         summary={'loss': 1.25})

    monkeypatch.setitem(sys.modules, 'wandb', types.SimpleNamespace(Api=FakeApi))
    monkeypatch.setattr(ui_server, 'snapshot', lambda _jid: {
        'mode': 'kernel',
        'integrations': {
            'wandb_url': 'https://wandb.ai/visitor/project/runs/run1'
        },
    })
    monkeypatch.setattr(ui_server, 'read_json',
                        lambda path, default=None: {'visitor': 'github:7'})
    monkeypatch.setattr(integrations, 'wandb_env',
                        lambda owner: {'WANDB_API_KEY': 'visitor-key'})
    ui_server.cache.pop('wandb:' + jid, None)

    with ui_server.app.test_request_context():
        result = ui_server.wandb_metrics(jid)

    assert used == [('visitor-key', 10)]
    assert result['metrics'] == [{'name': 'loss', 'value': 1.25}]


# ---- finding 5: one visitor's run does not wait behind strangers ----

def _byo_pool(tmp_path, monkeypatch, jobs, run_job, concurrency=2):
    import web
    from kernelevo.judges_pool import NotebookPool
    monkeypatch.setattr(web, 'JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setenv('JUDGES_INTEGRATION_DIR', str(tmp_path / 'integrations'))
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    pool = NotebookPool(None, run_job, lambda jid: jobs[jid], lambda job: None,
                        time.time() + 3600, concurrency=concurrency)
    pool.start()
    return pool


def test_runs_on_different_notebooks_do_not_queue_behind_each_other(tmp_path, monkeypatch):
    import threading
    started, release = threading.Semaphore(0), threading.Event()
    jobs = {jid: {'id': jid, 'visitor': 'github:%s' % jid[0], 'mode': 'recipe',
                  'molab': {'connection': '--token theirs'}} for jid in ('1' * 32, '2' * 32)}

    def run_job(jid):
        started.release()
        release.wait(5)

    pool = _byo_pool(tmp_path, monkeypatch, jobs, run_job)
    for jid in jobs:
        pool.put(jid)
    try:
        assert started.acquire(timeout=5) and started.acquire(timeout=5), \
            'the second visitor waited for the first'
    finally:
        release.set()


def test_one_account_cannot_occupy_the_whole_pool(tmp_path, monkeypatch):
    import threading
    started, release = threading.Semaphore(0), threading.Event()
    jobs = {jid: {'id': jid, 'visitor': 'github:7', 'mode': 'recipe',
                  'molab': {'connection': '--token theirs'}} for jid in ('3' * 32, '4' * 32)}

    def run_job(jid):
        started.release()
        release.wait(5)

    pool = _byo_pool(tmp_path, monkeypatch, jobs, run_job)
    for jid in jobs:
        pool.put(jid)
    try:
        assert started.acquire(timeout=5), 'nothing started'
        assert not started.acquire(timeout=1), 'one account ran two jobs at once'
    finally:
        release.set()


# ---- finding 10: the relay directory is sticky ----

def test_the_relay_directory_denies_cross_deletion():
    import inspect
    from kernelevo import judges_sandbox
    source = inspect.getsource(judges_sandbox.user_command)
    assert '0o1733' in source, 'the relay directory must be sticky (01733)'
