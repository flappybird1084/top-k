"""Durable, explicitly parented Weave spans. SQLite is committed before export."""
import datetime as dt
import json
import threading
import time
import uuid


class Traces:
    def __init__(self, archive, client=None, *, enabled=False):
        self.archive, self.client, self.enabled = archive, client, enabled
        self.calls = {}
        self.lock = threading.RLock()
        self.wake, self.stop = threading.Event(), threading.Event()
        archive.execute('''CREATE TABLE IF NOT EXISTS trace_outbox (
            id TEXT PRIMARY KEY, parent_id TEXT, name TEXT NOT NULL, inputs_json TEXT,
            attributes_json TEXT, started_at REAL NOT NULL, ended_at REAL,
            output_json TEXT, error TEXT, url TEXT, exported_end INTEGER DEFAULT 0,
            last_error TEXT)''')
        self.thread = None
        if enabled and client is not None:
            self.thread = threading.Thread(target=self._loop, daemon=True, name='weave-outbox')
            self.thread.start()
            self.wake.set()

    def start(self, name, inputs=None, *, parent_id=None, attributes=None, started_at=None):
        span_id = str(uuid.uuid4())
        if parent_id and not self.archive.rows('SELECT id FROM trace_outbox WHERE id=?', (parent_id,)):
            raise ValueError('Unknown trace parent: '+parent_id)
        self.archive.put('trace_outbox', id=span_id, parent_id=parent_id, name=name,
                         inputs_json=json.dumps(inputs or {}, default=str),
                         attributes_json=json.dumps(attributes or {}, default=str),
                         started_at=time.time() if started_at is None else started_at)
        self.wake.set()
        return span_id

    def finish(self, span_id, output=None, *, error=None, ended_at=None):
        end = time.time() if ended_at is None else ended_at
        rows = self.archive.rows('SELECT started_at,ended_at FROM trace_outbox WHERE id=?', (span_id,))
        if not rows: raise ValueError('Unknown trace span: '+span_id)
        if end < rows[0]['started_at']: raise ValueError('Trace ends before it starts')
        if rows[0]['ended_at'] is not None: raise ValueError('Trace already finished')
        self.archive.execute('UPDATE trace_outbox SET ended_at=?,output_json=?,error=? WHERE id=?',
                             (end, json.dumps(output, default=str), str(error) if error else None, span_id))
        self.wake.set()

    def record(self, name, inputs=None, output=None, *, started_at, ended_at,
               parent_id=None, attributes=None, error=None):
        span_id = self.start(name, inputs, parent_id=parent_id, attributes=attributes, started_at=started_at)
        self.finish(span_id, output, error=error, ended_at=ended_at)
        return span_id

    def url(self, span_id):
        rows = self.archive.rows('SELECT url FROM trace_outbox WHERE id=?', (span_id,))
        return rows[0]['url'] if rows else None

    def _create(self, row):
        if row['id'] in self.calls: return self.calls[row['id']]
        parent = None
        if row['parent_id']:
            parent_row = self.archive.rows('SELECT * FROM trace_outbox WHERE id=?', (row['parent_id'],))[0]
            parent = self._create(parent_row)
        call = self.client.create_call(row['name'], json.loads(row['inputs_json']), parent=parent,
            attributes=json.loads(row['attributes_json']), use_stack=False,
            _call_id_override=row['id'], started_at=dt.datetime.fromtimestamp(row['started_at'], dt.timezone.utc))
        self.calls[row['id']] = call
        return call

    def flush(self, timeout=None):
        """Export persisted spans and verify server visibility before marking delivered."""
        if self.client is None: return
        if timeout is not None:
            worker = threading.Thread(target=self.flush, daemon=True)
            worker.start()
            worker.join(timeout)
            return not worker.is_alive()
        with self.lock:
            rows = self.archive.rows('SELECT * FROM trace_outbox WHERE exported_end=0 ORDER BY started_at,id')
            for row in rows:
                if row['url'] and row['ended_at'] is None:
                    continue
                try:
                    call = self._create(row)
                    if row['ended_at'] is not None:
                        self.client.finish_call(call, output=json.loads(row['output_json']),
                            exception=RuntimeError(row['error']) if row['error'] else None,
                            ended_at=dt.datetime.fromtimestamp(row['ended_at'], dt.timezone.utc))
                    self.client.flush()
                    remote = self.client.get_call(row['id'])
                    if remote is None: raise RuntimeError('Weave did not return the exported call')
                    # Only an actual SDK call URL is stored, after read-back succeeds.
                    url = call.ui_url
                    if callable(url): url = url()
                    if row['ended_at'] is not None and getattr(remote, 'ended_at', None) is None:
                        raise RuntimeError('Weave call end is not yet visible')
                    self.archive.execute('UPDATE trace_outbox SET url=?,exported_end=?,last_error=NULL WHERE id=?',
                        (url, int(row['ended_at'] is not None), row['id']))
                    attrs = json.loads(row['attributes_json'])
                    if attrs.get('candidate_id') and attrs.get('span_kind') == 'candidate_lifecycle':
                        self.archive.execute('UPDATE candidates SET weave_trace_url=? WHERE id=?',
                                             (url, attrs['candidate_id']))
                except Exception as exc:
                    self.archive.execute('UPDATE trace_outbox SET last_error=? WHERE id=?', (str(exc), row['id']))
                    # Drop cached call so a retry replays both start and end under the same ID.
                    self.calls.pop(row['id'], None)

    def _loop(self):
        while not self.stop.is_set():
            self.wake.wait(2)
            self.wake.clear()
            self.flush()

    def close(self, timeout=15):
        self.stop.set()
        self.wake.set()
        if self.thread: self.thread.join(timeout=timeout)
        # If network is still blocked, leave the durable outbox for the next process.
        if self.thread is None or not self.thread.is_alive(): self.flush()
