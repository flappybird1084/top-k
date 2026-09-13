import copy
import tempfile
import unittest
from pathlib import Path
from kernel_evolution.provenance import identity,assert_compatible


class ProvenanceTests(unittest.TestCase):
    def test_model_edits_invalidate_calibration_but_prompt_edits_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'adapters').mkdir();(root/'kernel_evolution').mkdir()
            model=root/'adapters/demo.py';model.write_text('MODEL_WIDTH=192')
            prompt=root/'kernel_evolution/llm.py';prompt.write_text('PROMPT="first"')
            initial=identity(root)
            prompt.write_text('PROMPT="second"')
            changed_prompt=identity(root)
            self.assertNotEqual(initial['source_files'],changed_prompt['source_files'])
            assert_compatible(initial,changed_prompt)
            model.write_text('MODEL_WIDTH=384')
            with self.assertRaisesRegex(ValueError,'benchmark_digest'):
                assert_compatible(initial,identity(root))

    def test_runtime_upgrade_and_missing_identity_require_recalibration(self):
        with tempfile.TemporaryDirectory() as directory:
            original=identity(directory)
            changed=copy.deepcopy(original)
            changed['runtime_versions']['torch']='different'
            with self.assertRaisesRegex(ValueError,'runtime_versions'):assert_compatible(original,changed)
            with self.assertRaisesRegex(ValueError,'no source identity'):assert_compatible(None,original)
