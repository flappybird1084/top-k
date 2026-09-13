import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from kernel_evolution.fixtures import EMA,fixture


@unittest.skipUnless(importlib.util.find_spec('triton') and importlib.util.find_spec('torch'),'Triton compiler required')
class OfflineCompileTests(unittest.TestCase):
    def test_compiles_and_caches_without_cuda_and_rejects_bad_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            candidate=root/'candidate.py';candidate.write_text(EMA)
            descriptor=dict(shape=[192],stride=[1],dtype='float32',requires_grad=False)
            request=dict(code_path=str(candidate),args=[descriptor,descriptor,.996],
                         gpu_target=dict(backend='cuda',arch=80,warp_size=32),cache_dir=str(root/'cache'))
            inp=root/'input.json';out=root/'output.json';inp.write_text(json.dumps(request))
            def run():
                subprocess.run([sys.executable,'-m','kernel_evolution.compile_worker',str(inp),str(out)],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,check=True,timeout=60)
                return json.loads(out.read_text())
            first=run()
            self.assertEqual(first['status'],'compiled',first)
            self.assertFalse(first['cuda_initialized'])
            self.assertFalse(first['cache_hit'])
            self.assertTrue(run()['cache_hit'])
            tuned=EMA.replace('@triton.jit',"@triton.autotune(configs=[triton.Config({'B':1024}),triton.Config({'B':128})],key=['N'])\n@triton.jit")
            tuned=tuned.replace('    i=tl.program_id(0)*B',"    tl.static_assert(B<256)\n    i=tl.program_id(0)*B")
            tuned=tuned.replace('out,target.numel(),decay,1024)','out,target.numel(),decay)')
            candidate.write_text(tuned)
            self.assertEqual(run()['status'],'compiled')
            candidate.write_text(fixture(1,'ema_update')[0])
            broken=run()
            self.assertEqual(broken['status'],'compile_error',broken)
            self.assertIn('api_does_not_exist',broken['failure_note'])
