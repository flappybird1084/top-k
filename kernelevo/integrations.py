"""Per-user notebook and Weights & Biases settings for the public deployment.

These records hold the only two secrets a signed-in visitor gives us — their
marimo notebook token and their W&B API key — so they live here and nowhere
else. They never enter `job.json`, a log line, or an API response; a run that
needs one reads it from this store at launch and passes it through the child
process environment only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

DEFAULT_DIR = str(Path.home() / '.local/state/kernel-evolution/integrations')


def store_dir():
    return os.getenv('JUDGES_INTEGRATION_DIR', DEFAULT_DIR)


class IntegrationStore:
    """Files are private to the gateway user and named by a hash of the GitHub
    user id, so the directory listing names no accounts."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.lock = threading.Lock()

    def _path(self, user_id):
        return self.root / (hashlib.sha256(user_id.encode()).hexdigest() + '.json')

    def load(self, user_id):
        try:
            return json.loads(self._path(user_id).read_text())
        except (OSError, ValueError):
            return {}

    def save(self, user_id, record):
        path = self._path(user_id)
        with self.lock:
            temporary = path.with_suffix('.tmp')
            # O_EXCL + mode 0600 in the open itself: a write_text()-then-chmod
            # leaves the file world-readable for the length of the write.
            temporary.unlink(missing_ok=True)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, json.dumps(record).encode())
            finally:
                os.close(fd)
            temporary.replace(path)

    @staticmethod
    def public(record):
        """What the browser is allowed to see: whether each integration is
        connected, plus the non-secret labels needed to confirm the right one.
        Notebook tokens and W&B API keys never appear here."""
        notebook = record.get('molab') or {}
        wandb = record.get('wandb') or {}
        return dict(
            notebook=dict(configured=bool(notebook.get('url') and notebook.get('token')),
                          url=notebook.get('url', '')),
            wandb=dict(configured=bool(wandb.get('api_key')),
                       entity=wandb.get('entity', ''), project=wandb.get('project', '')))


def secrets_for(user_id):
    """Resolve an owner's secrets at launch time. Returns {} for an unknown
    owner rather than falling back to anything operator-owned."""
    if not user_id:
        return {}
    try:
        return IntegrationStore(store_dir()).load(user_id)
    except OSError:
        return {}


def wandb_env(user_id):
    """The job owner's own W&B credentials, as process environment. A public
    run never reports into the operator's W&B account."""
    record = (secrets_for(user_id).get('wandb') or {})
    env = {}
    for key, name in (('api_key', 'WANDB_API_KEY'), ('entity', 'WANDB_ENTITY'),
                      ('project', 'WANDB_PROJECT')):
        if record.get(key):
            env[name] = record[key]
    return env


def notebook_connection(user_id):
    """The owner's notebook URL + token, resolved at launch. The token is never
    written into the job record."""
    notebook = (secrets_for(user_id).get('molab') or {})
    if not (notebook.get('url') and notebook.get('token')):
        return None
    return {'notebook_url': notebook['url'], 'connection': '--token ' + notebook['token']}
