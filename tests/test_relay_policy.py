"""The boundary around work a notebook asks the operator's machine to do.

A run executes on a GPU notebook that, in the public deployment, belongs to a
stranger. Whatever that notebook drops into the relay directory is answered by
this machine's own logins, so these tests pin down what it is allowed to ask
for.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from kernelevo.relay_policy import OwnerLedger, RelayPolicy  # noqa: E402


def policy(tmp_path, **job):
    job.setdefault('id', 'j' * 32)
    return RelayPolicy(job, 'relay-secret',
                       OwnerLedger(tmp_path / 'ledger.json'))


def ask(**extra):
    return dict(relay_token='relay-secret',
                messages=[{'role': 'user', 'content': 'optimize this'}], **extra)


def test_a_public_run_gets_no_operator_subscription_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', raising=False)
    p = policy(tmp_path, visitor='github:7', judge_expires_at=1)
    assert 'cannot borrow' in p.check_llm(ask())
    assert 'not available' in p.check_search(dict(relay_token='relay-secret', query='triton'))


def test_an_operator_run_on_their_own_notebook_is_served(tmp_path, monkeypatch):
    monkeypatch.delenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', raising=False)
    assert policy(tmp_path).check_llm(ask()) is None


def test_requests_from_outside_the_active_run_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    assert 'does not belong' in p.check_llm(dict(ask(), relay_token='guessed'))
    assert 'does not belong' in p.check_llm({'messages': ask()['messages']})
    assert p.check_llm(ask()) is None
    p.finished()
    assert 'already finished' in p.check_llm(ask())


def test_only_configured_models_may_be_named(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    assert p.check_llm(ask(model='haiku')) is None
    assert 'not available' in p.check_llm(ask(model='some-expensive-model'))


def test_malformed_and_oversized_prompts_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    assert 'Malformed' in p.check_llm(dict(relay_token='relay-secret', messages=[]))
    assert 'Malformed' in p.check_llm(dict(relay_token='relay-secret',
                                           messages=[{'role': 'root', 'content': 'x'}]))
    assert 'Malformed' in p.check_llm(dict(relay_token='relay-secret',
                                           messages=[{'role': 'user', 'content': {'a': 1}}]))
    huge = [{'role': 'user', 'content': 'x' * (p.limits['max_prompt_bytes'] + 1)}]
    assert 'larger than' in p.check_llm(dict(relay_token='relay-secret', messages=huge))


def test_a_run_stops_being_served_once_it_spends_its_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    p.limits['max_requests'] = 2
    assert p.check_llm(ask()) is None
    assert p.check_llm(ask()) is None
    assert 'request budget' in p.check_llm(ask())
    # and the closure sticks: nothing further from this run is served
    assert 'request budget' in p.check_llm(ask())


def test_tokens_spent_close_the_relay_for_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    p.limits['max_tokens'] = 100
    assert p.check_llm(ask()) is None
    p.record_usage({'input_tokens': 90, 'output_tokens': 90})
    assert 'token budget' in p.check_llm(ask())


def test_one_account_cannot_multiply_its_budget_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    ledger = OwnerLedger(tmp_path / 'ledger.json', limits=dict(max_requests=3,
                                                              max_tokens=10 ** 9,
                                                              max_searches=10))
    spent = 0
    for _ in range(3):                      # three separate runs, one account
        p = RelayPolicy({'id': 'a' * 32, 'visitor': 'github:7'}, 'relay-secret', ledger)
        for _ in range(2):
            if p.check_llm(ask()) is None:
                spent += 1
    assert spent == 3, 'the per-account ledger did not carry across runs'
    other = RelayPolicy({'id': 'b' * 32, 'visitor': 'github:9'}, 'relay-secret', ledger)
    assert other.check_llm(ask()) is None, 'one account exhausted another account'


def test_search_queries_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_SEARCH_RELAY', '1')
    p = policy(tmp_path, visitor='github:7')
    p.limits['max_searches'] = 1
    assert p.check_search(dict(relay_token='relay-secret', query='triton softmax')) is None
    assert 'search budget' in p.check_search(dict(relay_token='relay-secret', query='again'))
    p = policy(tmp_path, visitor='github:7')
    long_query = 'x' * (p.limits['max_query_chars'] + 1)
    assert 'too long' in p.check_search(dict(relay_token='relay-secret', query=long_query))


# ---- the dispatcher honours the policy before it touches a credential ----

class _Notebook:
    """Stands in for the marimo kernel the answer is written back to."""

    def __init__(self):
        self.wrote = []

    def run(self, code):
        self.wrote.append(code)
        return True, 'OAUTH-OK', ''


def _relay(calls):
    from kernelevo.codex_oauth import Relay

    class TestRelay(Relay):
        label = 'Test'
        worker = staticmethod(lambda request: calls.append(request) or
                              dict(text='hi', input_tokens=1, output_tokens=1))
    return TestRelay()


def test_a_refused_request_never_reaches_the_provider(tmp_path, monkeypatch):
    monkeypatch.delenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', raising=False)
    calls, lines, notebook = [], [], _Notebook()
    relay = _relay(calls)
    p = policy(tmp_path, visitor='github:7', judge_expires_at=1)
    request = dict(ask(), id='a' * 32, kind='test')
    for _ in range(3):
        relay.service(notebook, '/tmp/w', [request], lines.append, policy=p)
    assert calls == [], 'the operator subscription was used anyway'
    assert any('refused' in line for line in lines)
    # the refusal is handed back once, so the run fails fast instead of waiting
    assert len(notebook.wrote) == 1
    assert 'cannot borrow' in notebook.wrote[0]


def test_an_allowed_request_is_served_and_counted(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_ALLOW_OPERATOR_LLM_RELAY', '1')
    import time as _time
    calls, lines, notebook = [], [], _Notebook()
    relay = _relay(calls)
    p = policy(tmp_path, visitor='github:7', judge_expires_at=1)
    request = dict(ask(), id='b' * 32, kind='test')
    deadline = _time.time() + 2
    while not any('delivered' in line for line in lines) and _time.time() < deadline:
        relay.service(notebook, '/tmp/w', [request], lines.append, policy=p)
        _time.sleep(0.02)
    assert len(calls) == 1
    assert p.requests == 1 and p.tokens == 2
