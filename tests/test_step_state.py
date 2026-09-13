import copy
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch required')
class StepStateTests(unittest.TestCase):
    def state(self):
        return (torch.tensor(1.),{'weight':torch.tensor([2.])},
                {'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor([.1])}},
                 'param_groups':[{'capturable':False,'lr':1e-4}]},
                {'weight':torch.tensor([.5])})

    def test_execution_flag_normalization_preserves_inputs(self):
        from kernel_evolution.verifier import compare_step_state
        expected=self.state();actual=copy.deepcopy(expected)
        actual[2]['param_groups'][0]['capturable']=True
        result=compare_step_state(actual,expected,1e-4,1e-5)
        self.assertEqual(result['normalized_execution_metadata'],['capturable_execution_flag'])
        self.assertTrue(actual[2]['param_groups'][0]['capturable'])
        self.assertFalse(expected[2]['param_groups'][0]['capturable'])

    def test_algorithm_state_and_gradients_remain_strict(self):
        from kernel_evolution.verifier import compare_step_state
        for field in ('lr','step','moment','gradient','parameter'):
            expected=self.state();actual=copy.deepcopy(expected)
            if field=='lr':actual[2]['param_groups'][0]['lr']=.1
            elif field=='step':actual[2]['state'][0]['step'].add_(1)
            elif field=='moment':actual[2]['state'][0]['exp_avg'].add_(1)
            elif field=='gradient':actual[3]['weight'].add_(1)
            else:actual[1]['weight'].add_(1)
            with self.subTest(field=field),self.assertRaises(AssertionError):
                compare_step_state(actual,expected,1e-4,1e-5)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
    def test_relocated_step_counter_must_match_exactly(self):
        from kernel_evolution.verifier import compare_step_state
        expected=self.state();actual=copy.deepcopy(expected)
        actual[2]['state'][0]['step']=actual[2]['state'][0]['step'].cuda()
        result=compare_step_state(actual,expected,1e-4,1e-5)
        self.assertEqual(result['normalized_execution_metadata'],['step_counter_device'])
        actual[2]['state'][0]['step'].add_(1e-6)
        with self.assertRaises(AssertionError):compare_step_state(actual,expected,1e-4,1e-5)
