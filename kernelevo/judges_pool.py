"""GPU run queue.

Two shapes:

* An operator deployment passes a connection file and gets one queue per
  operator notebook — those notebooks are a fixed, shared resource, so a run
  waits for a free one.
* The public deployment passes None. There every run brings the notebook its
  own signed-in user connected, so runs do **not** contend for one GPU and
  serializing them only made each visitor wait behind strangers. They execute
  concurrently up to a bounded worker count, with at most one run in flight per
  owner so no single account can take the whole pool.

Operator notebook credentials are never attached to a public user's job.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

# How many bring-your-own-notebook runs may execute at once, and how long a run
# waits when its own account already has one in flight.
DEFAULT_CONCURRENCY = 4
OWNER_BUSY_RETRY_S = 5


class NotebookPool:
    def __init__(self, connection_file, run_job, load_job, save_job, expires_at,
                 concurrency=None):
        if connection_file is None:
            connections = []      # bring-your-own-notebook deployment
        else:
            connections = json.loads(Path(connection_file).read_text())
            if not connections or any(not c.get('url') or not c.get('token') for c in connections):
                raise ValueError('Configure at least one GPU notebook')
            if len({c['url'].rstrip('/') for c in connections}) != len(connections):
                raise ValueError('Each worker needs a different notebook')
        self.connections = connections
        self.run_job, self.load_job, self.save_job = run_job, load_job, save_job
        self.expires_at = float(expires_at)
        self.lock = threading.Lock()
        if connections:
            self.queues = [queue.Queue() for _ in connections]
            self.workers = len(connections)
        else:
            self.queues = [queue.Queue()]
            self.workers = max(1, int(concurrency or
                                      os.getenv('JUDGES_CONCURRENT_RUNS', DEFAULT_CONCURRENCY)))
        # slot -> job id for the operator shape; job id -> owner for the shared
        # queue, where a slot number means nothing.
        self.active = [None] * len(self.queues)
        self.running = {}
        import web
        self.next_uid = max([200000]+[j.get('judge_uid',200000) for j in web.list_jobs()])+1

    def put(self, jid):
        with self.lock:
            job = self.load_job(jid)
            # Keep the architecture and kernel siblings on different GPUs.
            slot = 1 if job.get('mode') == 'kernel' and len(self.queues) > 1 else 0
            if self.connections:
                c = self.connections[slot]
                job['molab'] = {'notebook_url': c['url'], 'connection': '--token ' + c['token']}
            elif not self.owner_notebook(job):
                raise ValueError('This run has no notebook connected.')
            job['judge_expires_at'] = self.expires_at
            job['judge_uid'] = self.next_uid
            self.next_uid += 1
            job['recipe'] = {'phases':[{'kind':'architecture','generations':2,'candidates':2,'train_seconds':60},{'kind':'hyperparam','generations':1,'candidates':2,'train_seconds':60}], 'finals_top_k':1,'finals_train_seconds':120,'subagent_parallelism':2}
            job['stage'] = ('Queued for GPU ' + str(slot + 1) if self.connections
                            else 'Queued for your notebook GPU')
            self.save_job(job)
            self.queues[slot].put(jid)

    @staticmethod
    def owner_notebook(job):
        """The notebook this job's owner connected, resolved from the
        integration store — the token is deliberately not in the job file."""
        from kernelevo.integrations import notebook_connection
        if (job.get('molab') or {}).get('connection'):
            return job['molab']
        return notebook_connection(job.get('visitor'))

    def start(self):
        if self.connections:
            for slot in range(len(self.queues)):
                threading.Thread(target=self._worker, args=(slot,), daemon=True).start()
            return
        for _ in range(self.workers):
            threading.Thread(target=self._worker, args=(0,), daemon=True).start()

    def _worker(self, slot):
        # The worker thread must outlive any single bad job: the original
        # handler re-called load_job inside `except`, so one corrupt job.json
        # killed the thread and silently orphaned its queue (audit finding 25).
        while True:
            jid = self.queues[slot].get()
            try:
                if self._defer_for_owner(slot, jid):
                    continue
                try:
                    self._run_one(slot, jid)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    try:
                        job = self.load_job(jid)
                        job.update(status='failed', stage='GPU worker failed. See the run log.')
                        self.save_job(job)
                    except Exception:  # noqa: BLE001 — job file itself unreadable
                        print(f'[judges] job {jid} unreadable after failure; skipped', flush=True)
            finally:
                with self.lock:
                    if self.active[slot] == jid:
                        self.active[slot] = None
                    self.running.pop(jid, None)
                self.queues[slot].task_done()

    def _defer_for_owner(self, slot, jid):
        """One run at a time per account. Anything else goes back on the queue
        rather than holding a worker thread hostage."""
        if self.connections:
            return False
        try:
            owner = self.load_job(jid).get('visitor')
        except Exception:  # noqa: BLE001 — handled as a normal job failure below
            return False
        with self.lock:
            busy = owner and owner in self.running.values()
            if not busy:
                self.running[jid] = owner
                return False
        time.sleep(OWNER_BUSY_RETRY_S)
        self.queues[slot].put(jid)
        return True

    def _run_one(self, slot, jid):
        job = self.load_job(jid)
        if time.time() >= self.expires_at:
            job.update(status='cancelled', stage='The judging window has ended.')
            self.save_job(job)
            return
        with self.lock:
            self.active[slot] = jid
        self._start_observer(jid, job)
        self.run_job(jid)

    def _start_observer(self, jid, job):
        """Mirror the run's measured rows into the *owner's* W&B account.

        A public job must never report into the operator's account, so the
        observer is started only when its owner supplied their own key, and
        that key reaches it through the child environment — never the job file
        or the log (audit finding 6)."""
        import subprocess, sys
        import web
        from kernelevo.integrations import wandb_env
        credentials = (wandb_env(job.get('visitor')) if job.get('visitor')
                       else {k: os.environ[k] for k in ('WANDB_API_KEY', 'WANDB_ENTITY',
                                                        'WANDB_PROJECT')
                             if os.environ.get(k)})
        if not credentials.get('WANDB_API_KEY'):
            return
        environment = {k: v for k, v in os.environ.items()
                       if k not in ('WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT')}
        environment.update(credentials)
        with open(Path(web.JOBS_DIR) / jid / 'observer.log', 'a') as log:
            subprocess.Popen([sys.executable, 'judges_observer.py', jid],
                             stdout=log, stderr=log, env=environment)

    def status(self):
        with self.lock:
            if self.connections:
                return [{'worker': i + 1, 'busy': jid is not None,
                         'queued': self.queues[i].qsize()}
                        for i, jid in enumerate(self.active)]
            running, queued = len(self.running), self.queues[0].qsize()
        return [{'worker': 1, 'busy': running > 0, 'running': running,
                 'capacity': self.workers, 'queued': queued}]
