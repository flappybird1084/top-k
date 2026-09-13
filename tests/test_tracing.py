import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from kernel_evolution.archive import Archive
from kernel_evolution.tracing import Traces


class Client:
    def __init__(self, archive):
        self.archive, self.calls, self.fail = archive, {}, False
    def create_call(self, name, inputs, **kw):
        span_id = kw['_call_id_override']
        assert self.archive.rows('SELECT id FROM trace_outbox WHERE id=?', (span_id,))
        parent = kw['parent']
        call = SimpleNamespace(id=span_id, ended_at=None, inputs=inputs,
            parent_id=parent.id if parent else None, started_at=kw['started_at'],
            ui_url='https://wandb.ai/test/project/weave/calls/'+span_id)
        self.calls[span_id] = call
        return call
    def finish_call(self, call, output=None, exception=None, ended_at=None):
        call.ended_at, call.output, call.exception = ended_at, output, exception
    def flush(self):
        if self.fail: raise RuntimeError('network down')
    def get_call(self, span_id): return self.calls[span_id]


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = Archive(Path(self.tmp.name)/'archive.sqlite')
        self.client = Client(self.archive)
        self.traces = Traces(self.archive, self.client)
    def tearDown(self):
        self.archive.db.close()
        self.tmp.cleanup()
    def test_parent_timestamps_prompt_usage_and_link(self):
        self.archive.put('candidates', id='candidate')
        root = self.traces.start('candidate_lifecycle', {'strategy': 'agent supplied'}, started_at=10,
            attributes={'span_kind':'candidate_lifecycle', 'candidate_id':'candidate'})
        child = self.traces.record('llm_complete', {'messages':[{'content':'prompt'}]},
            {'response':'source', 'usage':{'input_tokens':42}}, parent_id=root, started_at=11, ended_at=12)
        self.traces.finish(root, {'accepted':False}, ended_at=13)
        self.traces.flush()
        self.assertEqual(self.client.calls[child].parent_id, root)
        self.assertEqual(self.client.calls[child].started_at.timestamp(), 11)
        self.assertEqual(self.client.calls[child].output['usage']['input_tokens'],42)
        self.assertEqual(self.archive.rows('SELECT weave_trace_url FROM candidates')[0]['weave_trace_url'],self.traces.url(root))
    def test_durable_replay_after_network_failure(self):
        root = self.traces.record('gate_2', {}, {'correct':False}, started_at=2, ended_at=3, error='mismatch')
        self.client.fail=True
        self.traces.flush()
        self.assertIsNone(self.traces.url(root))
        self.assertEqual(self.archive.rows('SELECT exported_end FROM trace_outbox')[0]['exported_end'],0)
        self.client.fail=False
        replay = Traces(self.archive, self.client)
        replay.flush()
        self.assertIsNotNone(replay.url(root))
        self.assertEqual(self.archive.rows('SELECT exported_end FROM trace_outbox')[0]['exported_end'],1)
    def test_open_span_then_finish(self):
        span = self.traces.start('live', started_at=2)
        self.traces.flush()
        self.assertIsNotNone(self.traces.url(span))
        self.assertEqual(self.archive.rows('SELECT exported_end FROM trace_outbox')[0]['exported_end'],0)
        self.traces.finish(span, ended_at=3)
        self.traces.flush()
        self.assertEqual(self.archive.rows('SELECT exported_end FROM trace_outbox')[0]['exported_end'],1)
    def test_reject_invalid_parent_and_time(self):
        with self.assertRaises(ValueError): self.traces.start('bad', parent_id='absent')
        span=self.traces.start('valid', started_at=4)
        with self.assertRaises(ValueError): self.traces.finish(span, ended_at=3)
