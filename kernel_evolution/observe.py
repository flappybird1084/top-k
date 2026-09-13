"""Optional one-way telemetry. Archive commits always precede mirror enqueue."""
import queue
import threading
from .tracing import Traces

try:
    import weave
    op=weave.op
except ImportError:
    def op(fn=None,**kwargs): return fn if fn else lambda f:f


class Mirror:
    def __init__(self,config,archive):
        self.queue=queue.Queue(maxsize=512)
        self.url=None
        self.archive=archive
        self.enabled=config.get('wandb_mode')=='online'
        client = None
        if self.enabled:
            try:
                import weave
                client = weave.init(config['wandb_entity']+'/'+config['wandb_project'])
            except Exception as exc:
                archive.event('weave_unavailable', {'error': str(exc)})
                if config.get('require_traces'): raise RuntimeError('Required Weave initialization failed') from exc
        self.traces = Traces(archive, client, enabled=self.enabled)
        archive.traces = self.traces
        if self.enabled and config.get('require_traces'):
            import time
            stamp = time.time()
            probe = self.traces.record('platform_trace_readiness',
                {'purpose': 'Verify platform trace delivery before any LLM call; no LLM used'},
                {'ready': True}, started_at=stamp, ended_at=time.time(),
                attributes={'span_kind': 'platform_self_test'})
            self.traces.flush(timeout=30)
            if not self.traces.url(probe):
                self.traces.close(timeout=1)
                raise RuntimeError('Required Weave trace read-back failed; no LLM calls may start')
            archive.event('weave_ready', {'url': self.traces.url(probe)})
        elif config.get('require_traces'):
            raise RuntimeError('Required Weave traces need wandb_mode=online')
        if self.enabled:
            self.thread=threading.Thread(target=self._loop,args=(config,),daemon=True)
            self.thread.start()

    def _loop(self,config):
        try:
            import wandb
            prior=self.archive.rows("SELECT payload_json FROM events WHERE kind='wandb_connected' ORDER BY id DESC LIMIT 1")
            import json
            run_id=json.loads(prior[0]['payload_json'])['url'].rstrip('/').split('/')[-1] if prior else None
            run=wandb.init(entity=config['wandb_entity'],project=config['wandb_project'],config=config,
                id=run_id,resume='allow' if run_id else None,
                name=config.get('run_name'),settings=wandb.Settings(init_timeout=30,disable_git=True))
            self.url=run.url
            self.archive.event('wandb_connected',{'url':self.url})
            while True:
                item=self.queue.get()
                if item is None: break
                if item.get('artifact'):
                    artifact=wandb.Artifact(item['name'],type='triton-kernel')
                    artifact.add_file(item['artifact'])
                    run.log_artifact(artifact)
                elif item.get('table'):
                    run.log({item['name']:wandb.Table(columns=item['columns'],data=item['table'])})
                else: run.log(item)
            run.finish()
        except Exception as exc:
            self.archive.event('mirror_unavailable',{'error':str(exc)})

    def log(self,item):
        if self.enabled:
            try: self.queue.put_nowait(item)
            except queue.Full: self.archive.event('mirror_dropped',{'reason':'queue_full'})

    def finish(self):
        self.traces.close()
        if self.enabled:
            self.log_sentinel()
            self.thread.join(timeout=15)

    def log_sentinel(self):
        try:self.queue.put_nowait(None)
        except queue.Full:pass
