"""GPU run queue.

Two shapes: an operator deployment passes a connection file and gets one queue
per operator notebook, and the public deployment passes None, where every run
carries the notebook its own signed-in user connected. Operator notebook
credentials are never attached to a public user's job.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path


class NotebookPool:
    def __init__(self, connection_file, run_job, load_job, save_job, expires_at):
        if connection_file is None:
            connections = []      # bring-your-own-notebook deployment
        else:
            connections = json.loads(Path(connection_file).read_text())
            if not connections or any(not c.get('url') or not c.get('token') for c in connections):
                raise ValueError('Configure at least one GPU notebook')
            if len({c['url'].rstrip('/') for c in connections}) != len(connections):
                raise ValueError('Each worker needs a different notebook')
        self.connections = connections
        self.queues = [queue.Queue() for _ in connections or [None]]
        self.run_job, self.load_job, self.save_job = run_job, load_job, save_job
        self.expires_at = float(expires_at)
        self.active = [None] * len(self.queues)
        self.lock = threading.Lock()
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
            elif not (job.get('molab') or {}).get('connection'):
                raise ValueError('This run has no notebook connected.')
            job['judge_expires_at'] = self.expires_at
            job['judge_uid'] = self.next_uid
            self.next_uid += 1
            job['recipe'] = {'phases':[{'kind':'architecture','generations':2,'candidates':2,'train_seconds':60},{'kind':'hyperparam','generations':1,'candidates':2,'train_seconds':60}], 'finals_top_k':1,'finals_train_seconds':120,'subagent_parallelism':2}
            job['stage'] = ('Queued for GPU ' + str(slot + 1) if self.connections
                            else 'Queued for your notebook GPU')
            self.save_job(job)
            self.queues[slot].put(jid)

    def start(self):
        for slot in range(len(self.queues)):
            threading.Thread(target=self._worker, args=(slot,), daemon=True).start()

    def _worker(self, slot):
        # The worker thread must outlive any single bad job: the original
        # handler re-called load_job inside `except`, so one corrupt job.json
        # killed the thread and silently orphaned its queue (audit finding 25).
        while True:
            jid = self.queues[slot].get()
            try:
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
                    self.active[slot] = None
                self.queues[slot].task_done()

    def _run_one(self, slot, jid):
        import web
        job = self.load_job(jid)
        if time.time() >= self.expires_at:
            job.update(status='cancelled', stage='The judging window has ended.')
            self.save_job(job)
            return
        with self.lock:
            self.active[slot] = jid
        import os, subprocess, sys
        if os.environ.get('WANDB_API_KEY'):
            with open(Path(web.JOBS_DIR) / jid / 'observer.log', 'a') as log:
                subprocess.Popen([sys.executable, 'judges_observer.py', jid],
                                 stdout=log, stderr=log)
        self.run_job(jid)

    def status(self):
        with self.lock:
            active = list(self.active)
        return [{'worker': i + 1, 'busy': jid is not None, 'queued': self.queues[i].qsize()}
                for i, jid in enumerate(active)]
