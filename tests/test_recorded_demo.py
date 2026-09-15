import json, statistics, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class RecordedEvidenceTests(unittest.TestCase):
    def test_kernel_headline_uses_paired_reductions(self):
        s=json.loads((ROOT/'ui/assets/recorded/kernel.json').read_text());v=s['verification']
        paired=statistics.median(1-b['candidate']/b['baseline'] for b in v['blocks'])
        self.assertAlmostEqual(paired,v['summary']['paired_median_time_reduction'])
        self.assertEqual(round(paired*100,2),3.30)
        self.assertGreaterEqual(min(b['gain'] for b in v['blocks']),.0311)
        self.assertEqual(len([r for r in s['candidates'] if r['generation']>0]),37)
        self.assertEqual(max(r['generation'] for r in s['candidates']),5)
    def test_architecture_budgets_and_lineage(self):
        s=json.loads((ROOT/'ui/assets/recorded/architecture.json').read_text());rows=s['architecture']['candidates']
        self.assertEqual(len([r for r in rows if r['generation'] in [1,2,3]]),24)
        baseline=next(r['val_loss'] for r in rows if r['phase']=='baseline' and r['train_secs']==300)
        winner=min((r for r in rows if r['phase']=='finals'),key=lambda r:r['val_loss'])
        self.assertEqual(round(100*(1-winner['val_loss']/baseline),2),7.21)
        self.assertEqual(winner['parent_id'],24)
        self.assertEqual(next(r for r in rows if r['id']==24)['parent_id'],16)
        self.assertEqual(next(r for r in rows if r['id']==16)['parent_id'],8)
        self.assertEqual(sum(r['val_loss'] is None for r in rows),7)
