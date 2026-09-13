import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'PyTorch required')
class FusionContractTests(unittest.TestCase):
    def make_model(self):
        from kernel_evolution.ops import EMARegion
        model=torch.nn.Module()
        model.register_parameter('target_a',torch.nn.Parameter(torch.zeros(7),requires_grad=False))
        model.register_parameter('target_b',torch.nn.Parameter(torch.ones(11),requires_grad=False))
        model.register_parameter('source_a',torch.nn.Parameter(torch.ones(7)))
        model.register_parameter('source_b',torch.nn.Parameter(torch.full((11,),3.)))
        model.ema=EMARegion()
        return model

    @staticmethod
    def hook(model):
        with torch.no_grad():
            model.target_a.copy_(model.ema(model.target_a,model.source_a,.996))
            model.target_b.copy_(model.ema(model.target_b,model.source_b,.996))

    def test_trace_preserves_state_then_replays_independent_updates(self):
        from kernel_evolution.fusion import capture_ema_bindings,apply,reference
        model=self.make_model()
        before={k:v.clone() for k,v in model.state_dict().items()}
        bindings=capture_ema_bindings(model,self.hook)
        self.assertEqual(len(bindings),2)
        for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        self.hook(model)
        expected={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(before)
        apply(bindings,reference)
        for key,value in expected.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)

    def test_other_post_step_mutations_are_rejected_and_restored(self):
        from kernel_evolution.fusion import capture_ema_bindings
        model=self.make_model()
        def side_effect(m):
            self.hook(m)
            m.source_a.add_(1)
        with self.assertRaisesRegex(ValueError,'outside the traced EMA'):
            capture_ema_bindings(model,side_effect)
        torch.testing.assert_close(model.source_a,torch.ones(7),rtol=0,atol=0)

    def test_dependent_updates_cannot_be_fused_as_independent(self):
        from kernel_evolution.fusion import capture_ema_bindings
        model=self.make_model()
        def dependent(m):
            m.target_a.copy_(m.ema(m.target_a,m.source_a,.996))
            m.source_a.copy_(m.ema(m.source_a,m.target_a,.996))
        with self.assertRaisesRegex(ValueError,'alias or depend'):
            capture_ema_bindings(model,dependent)

    def test_profile_spans_are_absent_from_normal_step_calls(self):
        model=self.make_model()
        with patch('torch.profiler.record_function',side_effect=AssertionError('Profiling overhead in normal call')):
            self.hook(model)
