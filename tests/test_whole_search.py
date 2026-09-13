"""Integration checks for routing whole-model agent jobs, without model calls."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_evolution.archive import Archive
from search import compile_offline, source_prompt, validated_jobs


class WholeSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = Archive(self.root/'archive.sqlite')
        self.source = self.root/'parent.py'
        self.source.write_text('# Archived agent source used only as prompt data\n')
        self.archive.put('lineages', id='whole_model', op_name='whole_model', retired=0, incumbent_id='parent')
        self.archive.put('candidates', id='parent', lineage_id='whole_model', accepted=1, correct_ok=1,
                         code_path=str(self.source), step_time_ms=20.)

    def tearDown(self):
        self.archive.db.close()
        self.temp.cleanup()

    def test_whole_prompt_exposes_external_work_and_parent_source(self):
        target = {'id':'whole_model', 'contract':'whole_model',
                  'gpu_profile':{'kernels':[{'name':'external_gemm', 'mapping':'unmapped', 'cuda_time_ms':12.}]}}
        job = {'lineage':'whole_model', 'parents':['parent'], 'strategy':'opaque agent proposal'}
        prompt = json.loads(source_prompt(job, self.archive, [target]))
        self.assertEqual(prompt['target']['gpu_profile'], target['gpu_profile'])
        self.assertEqual(prompt['parents'][0]['source'], self.source.read_text())
        self.assertEqual(prompt['strategy'], job['strategy'])
        self.assertIn('install(model, optimizer)', prompt['task'])
        self.assertIn('backward', prompt['task'])
        self.assertIn('optimizer', prompt['task'])

    def test_whole_job_can_compose_accepted_complete_installations(self):
        self.archive.put('candidates', id='second', lineage_id='whole_model', accepted=1, correct_ok=1)
        raw = {'jobs':[{'lineage':'whole_model', 'parents':['parent','second'], 'strategy':'FUSE: opaque agent proposal'}]}
        with self.assertRaisesRegex(ValueError, 'generation'):
            validated_jobs(raw, self.archive, 1, 4)
        selected = validated_jobs(raw, self.archive, 2, 4)
        self.assertTrue(selected[0]['fusion'])
        self.assertEqual(selected[0]['parents'], ['parent','second'])
        self.archive.execute('UPDATE candidates SET accepted=0 WHERE id=?', ('second',))
        with self.assertRaisesRegex(ValueError, 'accepted parents'):
            validated_jobs(raw, self.archive, 2, 4)

    def test_whole_source_preflight_never_calls_op_signature_compile_worker(self):
        candidate = {'code_path':str(self.source)}
        with patch('search.run_worker') as worker:
            result = compile_offline(candidate, {'search_scope':'whole_model'}, self.root, {}, time.monotonic()+10, None)
            self.assertEqual(result['status'], 'deferred')
            worker.assert_not_called()
        self.source.write_text('def broken(:\n')
        result = compile_offline(candidate, {'search_scope':'whole_model'}, self.root, {}, time.monotonic()+10, None)
        self.assertEqual(result['status'], 'compile_error')
        self.assertIn('SyntaxError', result['failure_note'])
