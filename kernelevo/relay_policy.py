"""The rule set for work a notebook asks the operator's machine to do.

Background. A run executes on a GPU notebook. When that notebook cannot reach
the operator's own logins — a Claude/Codex subscription session, a private
search service — the run writes a small request file into a relay directory and
the dispatcher on the operator's machine answers it. That is convenient for a
notebook the operator owns and dangerous for one a stranger owns: whoever
controls the notebook controls what lands in that directory, so an unchecked
relay is an open proxy to the operator's paid accounts and private network.

The boundary this module enforces:

* A run on a notebook the operator does not own ("untrusted") gets **no**
  operator credentials at all unless the operator explicitly opts in by setting
  `KEVO_ALLOW_OPERATOR_LLM_RELAY=1` / `KEVO_ALLOW_OPERATOR_SEARCH_RELAY=1`.
  Default is closed: those runs must bring their own API key.
* Every relay request — trusted or not — must carry the secret stamped into
  the launch environment of the run currently being dispatched. A request
  without it is not attributable to the active job and is refused.
* Requests are capped: how many per run, how many tokens per run, how large a
  single prompt may be, which models may be named, how many searches and how
  long a query. Untrusted runs get the smaller caps.
* A per-owner ledger on disk carries those caps across the runs of one account
  inside a rolling window, so one visitor cannot multiply their budget by
  starting run after run.

What this module cannot do: prove that a request file was written by the
harness rather than by the notebook's owner. Root on the notebook can forge
one. That is precisely why the untrusted default is closed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

# Model names a relayed request may ask for. The CLI aliases plus the exact ids
# this project configures; anything else (a costlier model, a typo, an attempt
# to steer spend) is refused rather than passed to the operator's session.
DEFAULT_MODELS = frozenset({
    'sonnet', 'opus', 'haiku',
    'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5-1', 'claude-haiku-4-5-20251001',
    'gpt-5', 'gpt-5-codex', 'gpt-5-mini', 'o3', 'o4-mini',
    'Qwen/Qwen3-235B-A22B-Instruct-2507',
})

TRUSTED_LIMITS = dict(max_requests=2000, max_tokens=50_000_000, max_searches=500,
                      max_prompt_bytes=400_000, max_query_chars=400)
UNTRUSTED_LIMITS = dict(max_requests=300, max_tokens=8_000_000, max_searches=60,
                        max_prompt_bytes=200_000, max_query_chars=200)

# Rolling per-owner window. One account's runs share it.
OWNER_WINDOW_S = int(os.getenv('KEVO_RELAY_OWNER_WINDOW_S', '86400'))
OWNER_LIMITS = dict(max_requests=int(os.getenv('KEVO_RELAY_OWNER_MAX_REQUESTS', '900')),
                    max_tokens=int(os.getenv('KEVO_RELAY_OWNER_MAX_TOKENS', '24000000')),
                    max_searches=int(os.getenv('KEVO_RELAY_OWNER_MAX_SEARCHES', '180')))


def _flag(name):
    return os.getenv(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def _int_env(name, default):
    try:
        return max(0, int(os.environ[name]))
    except (KeyError, ValueError):
        return default


def allowed_models():
    raw = os.getenv('KEVO_RELAY_MODELS', '').strip()
    if raw:
        return frozenset(part.strip() for part in raw.split(',') if part.strip())
    return DEFAULT_MODELS


# Distinct from a field name (a crossed cap) and from None (nothing crossed):
# the ledger could not be persisted, so usage is unknown and the relay must fail
# CLOSED rather than silently stop enforcing the per-owner budget.
LEDGER_UNAVAILABLE = 'ledger-unavailable'


class OwnerLedger:
    """Usage carried across the runs of one account.

    Kept in a single JSON file next to the other gateway state. Each dispatcher
    is its own process, so the file is read-modify-written under an exclusive
    lock; on a platform without `flock` the accounting still works, it is just
    not race-proof (the per-run caps remain authoritative there)."""

    def __init__(self, path=None, window=OWNER_WINDOW_S, limits=None):
        self.path = Path(path or os.getenv(
            'KEVO_RELAY_LEDGER',
            str(Path.home() / '.local/state/kernel-evolution/relay-usage.json')))
        self.window = window
        self.limits = dict(limits or OWNER_LIMITS)

    def _locked(self, mutate):
        # Let an I/O failure (mkdir/open/read/write) propagate: charge() turns it
        # into LEDGER_UNAVAILABLE so callers fail closed, instead of a bare None
        # that the callers read as "no cap crossed".
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            raw = os.read(fd, 4_000_000).decode('utf-8', 'replace')
            try:
                state = json.loads(raw) if raw.strip() else {}
            except ValueError:
                state = {}
            if not isinstance(state, dict):
                state = {}
            cutoff = time.time() - self.window
            state = {k: v for k, v in state.items()
                     if isinstance(v, dict) and v.get('started', 0) > cutoff}
            result = mutate(state)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(state).encode())
            return result
        finally:
            os.close(fd)

    def charge(self, owner, requests=0, tokens=0, searches=0):
        """Add usage and report which per-owner cap it crossed, if any."""
        def mutate(state):
            row = state.setdefault(owner, {'started': time.time(), 'requests': 0,
                                           'tokens': 0, 'searches': 0})
            row['requests'] += requests
            row['tokens'] += tokens
            row['searches'] += searches
            for field, cap in (('requests', 'max_requests'), ('tokens', 'max_tokens'),
                               ('searches', 'max_searches')):
                if self.limits.get(cap) is not None and row[field] > self.limits[cap]:
                    return field
            return None
        try:
            return self._locked(mutate)
        except OSError:
            return LEDGER_UNAVAILABLE


class RelayPolicy:
    """One instance per dispatched run."""

    def __init__(self, job, relay_token, ledger=None):
        self.owner = str(job.get('visitor') or job.get('owner') or 'operator')
        self.job_id = str(job.get('id') or '')
        # A run on someone else's notebook: the public deployment stamps every
        # job with a judging expiry and the GitHub owner who submitted it. An
        # operator pasting their own pair prompt locally is not this case.
        self.untrusted = bool(job.get('judge_expires_at') or job.get('visitor'))
        self.token = relay_token or ''
        base = UNTRUSTED_LIMITS if self.untrusted else TRUSTED_LIMITS
        self.limits = {k: _int_env('KEVO_RELAY_' + k.upper(), v) for k, v in base.items()}
        self.models = allowed_models()
        self.ledger = ledger if ledger is not None else OwnerLedger()
        self.requests = self.tokens = self.searches = 0
        self.active = True
        self.stopped = ''

    # -- what the operator's accounts may be used for at all ----------------
    def operator_llm_allowed(self):
        if not self.untrusted or _flag('KEVO_ALLOW_OPERATOR_LLM_RELAY'):
            return None
        return ('This run is on a notebook this server does not own, so it cannot '
                'borrow the server\'s Claude/Codex subscription. Configure an API '
                'key for the run instead.')

    def operator_search_allowed(self):
        if not self.untrusted or _flag('KEVO_ALLOW_OPERATOR_SEARCH_RELAY'):
            return None
        return ('Web search through the server is not available to a run on an '
                'unowned notebook.')

    def finished(self):
        """Called once the run has exited: nothing more is attributable to it."""
        self.active = False

    # -- per-request admission ---------------------------------------------
    def _attributable(self, request):
        if not self.active:
            return 'The run that could have made this request has already finished.'
        if not self.token:
            return 'This dispatcher has no relay secret for the active run.'
        presented = request.get('relay_token')
        if not isinstance(presented, str) or not _equal(presented, self.token):
            return 'This request does not belong to the run being dispatched.'
        return None

    def check_llm(self, request):
        """None if the dispatcher may answer this completion request."""
        if self.stopped:
            return self.stopped
        reason = self.operator_llm_allowed() or self._attributable(request)
        if reason:
            return reason
        model = request.get('model') or ''
        if not isinstance(model, str) or (model and model not in self.models):
            return 'That model is not available through this relay.'
        messages = request.get('messages')
        if not isinstance(messages, list) or not messages:
            return 'Malformed relay request: no messages.'
        size = 0
        for message in messages:
            if not isinstance(message, dict):
                return 'Malformed relay request: bad message.'
            if message.get('role') not in ('system', 'user', 'assistant'):
                return 'Malformed relay request: unknown role.'
            content = message.get('content')
            if not isinstance(content, str):
                return 'Malformed relay request: non-text content.'
            size += len(content)
        if size > self.limits['max_prompt_bytes']:
            return ('That request is larger than the relay accepts '
                    f"({size} > {self.limits['max_prompt_bytes']} characters).")
        if self.requests >= self.limits['max_requests']:
            return self._stop('this run reached its relay request budget')
        self.requests += 1
        crossed = self.ledger.charge(self.owner, requests=1)
        if crossed == LEDGER_UNAVAILABLE:
            return self._stop('relay usage accounting is unavailable')
        if crossed:
            return self._stop(f'this account reached its relay {crossed} budget')
        return None

    def record_usage(self, answer):
        """Count what an answered request actually cost, and stop the relay for
        the rest of the run once the token budget is gone."""
        if not isinstance(answer, dict):
            return
        used = sum(v for v in (answer.get('input_tokens'), answer.get('output_tokens'))
                   if isinstance(v, int) and v > 0)
        self.tokens += used
        crossed = self.ledger.charge(self.owner, tokens=used)
        if crossed == LEDGER_UNAVAILABLE:
            self._stop('relay usage accounting is unavailable')
        elif crossed:
            self._stop(f'this account reached its relay {crossed} budget')
        elif self.tokens > self.limits['max_tokens']:
            self._stop('this run reached its relay token budget')

    def check_search(self, request):
        if self.stopped:
            return self.stopped
        reason = self.operator_search_allowed() or self._attributable(request)
        if reason:
            return reason
        query = request.get('query')
        if not isinstance(query, str) or not query.strip():
            return 'Malformed search request.'
        if len(query) > self.limits['max_query_chars']:
            return 'That search query is too long for the relay.'
        if self.searches >= self.limits['max_searches']:
            return self._stop('this run reached its relay search budget')
        self.searches += 1
        crossed = self.ledger.charge(self.owner, searches=1)
        if crossed == LEDGER_UNAVAILABLE:
            return self._stop('relay usage accounting is unavailable')
        if crossed:
            return self._stop(f'this account reached its relay {crossed} budget')
        return None

    def _stop(self, why):
        self.stopped = 'Relay closed for this run: ' + why + '.'
        return self.stopped


def _equal(a, b):
    import hmac
    return hmac.compare_digest(a, b)


def new_token():
    import secrets
    return secrets.token_urlsafe(32)
