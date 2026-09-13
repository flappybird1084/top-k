"""One queue per GPU notebook; connection secrets remain outside the checkout."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path


class NotebookPool:
    def __init__(self, connection_file, run_job, load_job, save_job, expires_at):
        connections = json.loads(Path(connection_file).read_text())
        if not connections or any(not c.get('url') or not c.get('token') for c in connections):
            raise ValueError('Configure at least one GPU notebook')
        if len({c['url'].rstrip('/') for c in connections}) != len(connections):
            raise ValueError('Each worker needs a different notebook')
        self.connections = connections
        self.queues = [queue.Queue() for _ in connections]
        self.run_job, self.load_job, self.save_job = run_job, load_job, save_job
        self.expires_at = float(expires_at)
        self.active = [None] * len(connections)
        self.lock = threading.Lock()
        import web
        self.next_uid = max([200000]+[j.get('judge_uid',200000) for j in web.list_jobs()])+1

    def put(self, jid):
        with self.lock:
            job = self.load_job(jid)
            # Keep the architecture and kernel siblings on different GPUs.
            slot = 1 if job.get('mode') == 'kernel' and len(self.queues) > 1 else 0
            c = self.connections[slot]
            job['molab'] = {'notebook_url': c['url'], 'connection': '--token ' + c['token']}
            job['judge_expires_at'] = self.expires_at
            job['judge_uid'] = self.next_uid
            self.next_uid += 1
            job['recipe'] = {'phases':[{'kind':'architecture','generations':2,'candidates':2,'train_seconds':60},{'kind':'hyperparam','generations':1,'candidates':2,'train_seconds':60}], 'finals_top_k':1,'finals_train_seconds':120,'subagent_parallelism':2}
            job['stage'] = 'Queued for GPU ' + str(slot + 1)
            self.save_job(job)
            self.queues[slot].put(jid)

    def start(self):
        for slot in range(len(self.queues)):
            threading.Thread(target=self._worker, args=(slot,), daemon=True).start()

    def _worker(self, slot):
        while True:
            jid = self.queues[slot].get()
            try:
                job = self.load_job(jid)
                if time.time() >= self.expires_at:
                    job.update(status='cancelled', stage='The judging window has ended.')
                    self.save_job(job)
                    continue
                self.active[slot] = jid
                import os, subprocess, sys
                if os.environ.get('WANDB_API_KEY'):
                    with open(Path(__file__).parents[1]/'jobs'/jid/'observer.log','a') as log:
                        subprocess.Popen([sys.executable,'judges_observer.py',jid],stdout=log,stderr=log)
                self.run_job(jid)
            except Exception:
                job = self.load_job(jid)
                job.update(status='failed', stage='GPU worker failed. See the run log.')
                self.save_job(job)
            finally:
                self.active[slot] = None
                self.queues[slot].task_done()

    def status(self):
        return [{'worker': i + 1, 'busy': jid is not None, 'queued': self.queues[i].qsize()}
                for i, jid in enumerate(self.active)]
