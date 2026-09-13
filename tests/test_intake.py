import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_evolution.intake import load_adapter, repository_files, repository_context, resolve_input, validate_source

SOURCE = 'def build_model(): return 1\ndef get_dataloader(split): return [1]\ndef loss_fn(model,batch): return 1\n'


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.run = Path(self.tmp.name) / 'run'
        self.cfg = {'llm': 'codex', 'planner_llm': 'gpt-6-astra', 'max_repairs': 1}
        self.args = SimpleNamespace(repo=str(self.repo), run_dir=str(self.run), adapter='adapter.py', prompt='Optimize training')
        self.archive = Mock()
        (self.repo / 'adapter.py').write_text(SOURCE)

    def resolve(self):
        return resolve_input(self.args, self.cfg, self.archive, self.repo)

    def test_file_adapter_ingestion_resume_and_changed_source(self):
        with patch('kernel_evolution.intake.run_worker', return_value={'status': 'ready'}) as worker:
            adapter = self.resolve()
            self.assertEqual(adapter, str((self.repo / 'adapter.py').resolve()))
            self.assertEqual(worker.call_args.args[1]['action'], 'ingest')
            self.assertEqual(self.resolve(), adapter)
            self.assertEqual(worker.call_count, 1)
            (self.repo / 'model.py').write_text('changed = True')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.resolve()

    def test_generated_adapter_repairs_raw_feedback_and_provenance(self):
        self.args.adapter = None
        llm = Mock(model='gpt-6-astra')
        llm.complete.side_effect = [{'source': SOURCE}, {'source': SOURCE + '# repaired\n'}]
        with patch('kernel_evolution.intake.CodexOAuthLLM', return_value=llm) as factory, patch(
                'kernel_evolution.intake.run_worker', side_effect=[
                    {'status': 'crash', 'failure_note': 'raw CUDA assertion'}, {'status': 'ready'}]):
            adapter = self.resolve()
        self.assertEqual(factory.call_args.args[3], 'adapter')
        self.assertEqual(llm.complete.call_count, 2)
        feedback = llm.complete.call_args.args[0][-1]['content']
        self.assertIn('raw CUDA assertion', feedback)
        self.assertNotIn('diagnosis', feedback)
        manifest = json.loads((self.run / 'input/manifest.json').read_text())
        self.assertTrue(manifest['generated'])
        self.assertEqual(manifest['model_name'], 'gpt-6-astra')
        self.assertEqual(Path(adapter).read_text(), SOURCE + '# repaired\n')
        self.assertTrue((self.run / 'input/adapter_attempt_0.py').exists())
        with patch('kernel_evolution.intake.run_worker') as worker:
            self.assertEqual(self.resolve(), adapter)
            worker.assert_not_called()
        Path(adapter).write_text(SOURCE)
        with self.assertRaisesRegex(ValueError, 'Adapter changed'):
            self.resolve()

    def test_ingestion_failure_never_becomes_valid_input(self):
        with patch('kernel_evolution.intake.run_worker', return_value={'status': 'crash', 'failure_note': 'missing dataset'}):
            with self.assertRaisesRegex(RuntimeError, 'missing dataset'):
                self.resolve()
        self.assertFalse((self.run / 'input/manifest.json').exists())

    def test_generated_optimization_kernel_rejected_before_gpu(self):
        self.args.adapter = None
        self.cfg['max_repairs'] = 0
        llm = Mock(model='gpt-6-astra')
        llm.complete.return_value = {'source': 'import triton\n' + SOURCE}
        with patch('kernel_evolution.intake.CodexOAuthLLM', return_value=llm), patch('kernel_evolution.intake.run_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'optimization kernels'):
                self.resolve()
            worker.assert_not_called()

    def test_context_excludes_secrets_data_links_and_is_bounded(self):
        (self.repo / 'credentials.json').write_text('PRIVATE')
        (self.repo / 'data').mkdir()
        (self.repo / 'data/shard.txt').write_text('PRIVATE DATA')
        (self.repo / 'linked.py').symlink_to(self.repo / 'adapter.py')
        (self.repo / 'large.py').write_text('x' * 100000)
        (self.repo / 'fixtures.py').write_text('HANDWRITTEN KERNEL')
        manifest = repository_files(self.repo)
        self.assertEqual(set(manifest), {'adapter.py', 'large.py', 'fixtures.py'})
        context = repository_context(self.repo, manifest, max_chars=200)
        self.assertLessEqual(len(context), 200)
        self.assertNotIn('PRIVATE', context)
        self.assertNotIn('HANDWRITTEN', context)

    def test_load_file_generic_and_contract_validation(self):
        module = load_adapter(str(self.repo / 'adapter.py'), self.repo)
        self.assertEqual(module.build_model(), 1)
        validate_source(SOURCE)
        with self.assertRaisesRegex(ValueError, 'Missing adapter functions'):
            validate_source('def build_model(): pass')

    def test_nested_run_outputs_do_not_invalidate_input(self):
        self.args.run_dir = str(self.repo / 'experiments/current')
        with patch('kernel_evolution.intake.run_worker', return_value={'status': 'ready'}):
            adapter = self.resolve()
        (Path(self.args.run_dir) / 'output.py').write_text('anything = 1')
        self.assertEqual(self.resolve(), adapter)

    def test_stub_requires_supplied_adapter(self):
        self.args.adapter = None
        self.cfg['llm'] = 'stub'
        with self.assertRaisesRegex(ValueError, 'Stub intake requires'):
            self.resolve()

    def test_worker_source_mutation_fails_closed(self):
        def worker(*args, **kwargs):
            (self.repo / 'adapter.py').write_text(SOURCE + '# changed\n')
            return {'status': 'ready'}
        with patch('kernel_evolution.intake.run_worker', side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, 'source changed during'):
                self.resolve()


if __name__ == '__main__':
    unittest.main()
