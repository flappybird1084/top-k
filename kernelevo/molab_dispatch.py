"""Molab dispatch as a subprocess, so a long-running web server never runs a
stale cached copy of the dispatch code — each job execs this fresh from disk
(symmetric with how local jobs exec search.py).

Usage: python -m kernelevo.molab_dispatch <job.json> <project_root> <artifacts_dir>
Env: KEVO_REMOTE_ENV = JSON dict of env vars for the remote run.
Streams log lines to stdout; exits with the remote run's exit code.
"""

import json
import os
import sys

from kernelevo.molab import MolabTarget
from kernelevo.dispatch_once import run_once


def main():
    job = json.load(open(sys.argv[1]))
    project_root, artifacts_dir = sys.argv[2], sys.argv[3]
    env_updates = json.loads(os.environ.get("KEVO_REMOTE_ENV", "{}"))
    # A run owned by a signed-in visitor carries no notebook token in its job
    # file; the server passes it here, in this process's environment only.
    if os.environ.get("KEVO_MOLAB_CONNECTION"):
        job["molab"] = json.loads(os.environ["KEVO_MOLAB_CONNECTION"])

    def write_line(line):
        print(line, flush=True)

    try:
        rc = run_once(sys.argv[1], lambda: MolabTarget(job.get("molab")).dispatch(
            job, project_root, env_updates, write_line,
            artifacts_dir=artifacts_dir))
    except Exception as e:  # noqa: BLE001 — surface, don't traceback-spam the log
        write_line(f"[molab] dispatch error: {type(e).__name__}: {e}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
