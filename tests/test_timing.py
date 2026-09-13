import unittest

try:
    from kernel_evolution.verifier import paired_measure
except ImportError:
    paired_measure=None


@unittest.skipIf(paired_measure is None,'PyTorch environment required')
class TimingTests(unittest.TestCase):
    def test_clear_win_has_four_balanced_pairs(self):
        order=[]
        def measure(name):
            order.append(name)
            return 10 if name=='baseline' else 8
        result=paired_measure(measure,'baseline','candidate',.03)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(len(result['blocks']),4)
        self.assertEqual(order,['baseline','candidate','candidate','baseline']*2)

    def test_borderline_win_stops_after_eight_pairs(self):
        result=paired_measure(lambda name:10 if name=='baseline' else 9.9,'baseline','candidate',.03)
        self.assertEqual(result['status'],'inconclusive')
        self.assertEqual(len(result['blocks']),8)

    def test_slow_candidate_never_gets_remeasured_until_lucky(self):
        result=paired_measure(lambda name:10 if name=='baseline' else 11,'baseline','candidate',.03)
        self.assertEqual(result['status'],'slower')
        self.assertEqual(len(result['blocks']),4)
