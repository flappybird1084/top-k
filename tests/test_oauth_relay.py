"""Relay hygiene and retry policy for the OAuth CLI providers."""
import json
import os
from pathlib import Path
import sys
import time
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from kernelevo import codex_oauth  # noqa: E402


def test_null_usage_fields_do_not_poison_the_token_ledger():
    assert codex_oauth.token_count(None) == 0
    assert codex_oauth.token_count(True) == 0
    assert codex_oauth.token_count(-5) == 0
    assert codex_oauth.token_count(17) == 17


def test_sweep_removes_only_abandoned_relay_files(tmp_path):
    fresh = tmp_path / 'a' * 1 if False else tmp_path / ('a' * 32 + '.req.json')
    stale = tmp_path / ('b' * 32 + '.req.json')
    orphan = tmp_path / ('c' * 32 + '.res.json')
    keep = tmp_path / 'notes.txt'
    for path in (fresh, stale, orphan, keep):
        path.write_text('{}')
    old = time.time() - codex_oauth.RELAY_STALE_S - 60
    for path in (stale, orphan, keep):
        os.utime(path, (old, old))
    codex_oauth.sweep_relay(tmp_path)
    assert fresh.exists() and keep.exists()
    assert not stale.exists() and not orphan.exists()


def test_sweep_survives_a_directory_it_cannot_list(tmp_path):
    codex_oauth.sweep_relay(tmp_path / 'missing')   # must not raise


@pytest.mark.parametrize('payload', [
    'not a dict', {'text': ''}, {'text': None}, {}, {'error': 'dispatcher said no'}])
def test_malformed_relay_responses_are_rejected(payload):
    with pytest.raises(RuntimeError):
        codex_oauth.relay_answer(payload)


def test_relay_answer_normalises_missing_counts():
    assert codex_oauth.relay_answer({'text': 'hi', 'input_tokens': None}) == {
        'text': 'hi', 'input_tokens': 0, 'output_tokens': 0}


def test_relay_round_trip_sweeps_and_validates(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_RELAY_DIR', str(tmp_path))
    stale = tmp_path / ('d' * 32 + '.req.json')
    stale.write_text('{}')
    old = time.time() - codex_oauth.RELAY_STALE_S - 60
    os.utime(stale, (old, old))
    monkeypatch.setattr(codex_oauth, 'RELAY_WAIT_S', 2)

    with pytest.raises(TimeoutError):
        codex_oauth.complete({'messages': []})
    assert not stale.exists()
    # the request it wrote is cleaned up on the way out
    assert not list(tmp_path.glob('*.req.json'))


def test_unreadable_response_file_is_an_error_not_a_completion(tmp_path, monkeypatch):
    monkeypatch.setenv('KEVO_RELAY_DIR', str(tmp_path))
    monkeypatch.setattr(codex_oauth, 'RELAY_WAIT_S', 5)
    import threading

    def answer():
        for _ in range(50):
            requests = list(tmp_path.glob('*.req.json'))
            if requests:
                rid = requests[0].name[:-len('.req.json')]
                (tmp_path / (rid + '.res.json')).write_text('{"text": "trunc')
                return
            time.sleep(0.05)

    threading.Thread(target=answer, daemon=True).start()
    with pytest.raises(RuntimeError, match='unreadable'):
        codex_oauth.complete({'messages': []})


def test_claude_retries_only_failures_that_could_pass(monkeypatch):
    from kernelevo import claude_oauth
    assert claude_oauth._transient(None) is True              # CLI died mid-stream
    assert claude_oauth._transient('error_rate_limit') is True
    assert claude_oauth._transient('success', 'socket hang up') is True
    assert claude_oauth._transient('error_max_turns') is False
    assert claude_oauth._transient('error_invalid_flag', 'not logged in') is False


def test_claude_permanent_failure_is_not_retried(tmp_path, monkeypatch):
    """A refusal the CLI reported cleanly must fail once, not burn a second
    five-minute attempt on the same answer."""
    import stat
    from kernelevo import claude_oauth
    marker = tmp_path / 'calls'
    payload = json.dumps({'type': 'result', 'subtype': 'error_max_turns', 'result': ''})
    exe = tmp_path / 'claude'
    # --help is the one-off flag probe, not a completion attempt
    exe.write_text("#!/bin/sh\n"
                   'if [ "$1" = "--help" ]; then echo "--allowedTools --disallowedTools '
                   '--strict-mcp-config --mcp-config --setting-sources"; exit 0; fi\n'
                   "cat > /dev/null\n"
                   f"echo x >> {marker}\necho '{payload}'\nexit 3\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setattr(claude_oauth, 'RETRY_DELAY_S', 0)
    with pytest.raises(RuntimeError):
        claude_oauth.complete_local({'messages': []})
    assert marker.read_text().count('x') == 1


def test_claude_isolation_flags_follow_what_the_cli_supports(monkeypatch):
    from kernelevo import claude_oauth
    monkeypatch.setattr(claude_oauth, 'cli_flags', lambda: frozenset(
        {'--allowedTools', '--disallowedTools', '--strict-mcp-config', '--mcp-config',
         '--setting-sources'}))
    flags = claude_oauth.isolation_flags()
    # an empty positive allowlist: no tool is permitted by name
    assert flags[flags.index('--allowedTools') + 1] == ''
    assert '--strict-mcp-config' in flags
    assert flags[flags.index('--mcp-config') + 1] == '{"mcpServers":{}}'
    assert flags[flags.index('--setting-sources') + 1] == 'project'


def test_a_build_without_the_isolation_switches_is_refused(monkeypatch):
    """Finding 3: an unknown-capability CLI used to be run anyway, with its
    full tool suite, on prompts full of untrusted repository text."""
    from kernelevo import claude_oauth
    monkeypatch.setattr(claude_oauth, 'cli_flags', lambda: frozenset({'--print'}))
    # The CLI being installed is not what this test is about; without this,
    # check_login() short-circuits on `which('claude')` wherever the CLI is
    # absent (e.g. CI) and never reaches the capability check under test.
    monkeypatch.setattr(claude_oauth.shutil, 'which', lambda _cmd: '/usr/bin/claude')
    with pytest.raises(claude_oauth.CapabilityError):
        claude_oauth.isolation_flags()
    with pytest.raises(ValueError, match='does not support'):
        claude_oauth.check_login()


def test_a_failed_capability_probe_is_fatal(tmp_path, monkeypatch):
    import stat
    from kernelevo import claude_oauth
    exe = tmp_path / 'claude'
    exe.write_text('#!/bin/sh\nexit 9\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    claude_oauth._flags_for.cache_clear()
    with pytest.raises(claude_oauth.CapabilityError):
        claude_oauth.cli_flags()
    claude_oauth._flags_for.cache_clear()
    with pytest.raises(RuntimeError):
        claude_oauth.complete_local({'messages': [{'role': 'user', 'content': 'x'}]})
    claude_oauth._flags_for.cache_clear()
