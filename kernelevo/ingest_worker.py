"""Ingest verification worker: runs the §4.2 checks on a (possibly LLM-written)
adapter in its own process, so a broken adapter can't corrupt the orchestrator
and its raw traceback becomes the repair feedback for the adapter-writing agent.

Usage: python -m kernelevo.ingest_worker <job.json>
job = {adapter, device, seed}
Prints 'KEVO_RESULT {json}' on success; raw traceback on stderr on failure.
"""

import json
import os
import sys

from kernelevo import ingest, patch


def main():
    job = json.load(open(sys.argv[1]))
    patch.install()
    adapter, _ = ingest.load_adapter(job["adapter"])
    info = ingest.ingest(adapter, dict(seed=job.get("seed", 1234), device=job["device"]))
    print("KEVO_RESULT " + json.dumps(info))
    # Hard-exit: adapters may leave non-daemon threads behind (e.g. a
    # `datasets` streaming pool), and interpreter shutdown would join them
    # forever — the result is already flushed, so skip finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
