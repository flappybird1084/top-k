import unittest
try:
    import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'PyTorch required')
class FunctionalGraphTests(unittest.TestCase):
    def make_model(self,*,bias=True,side_effect=False):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection=torch.nn.Linear(8,8)
                self.weight=torch.nn.Parameter(torch.ones(8))
                self.bias=torch.nn.Parameter(torch.zeros(8)) if bias else None
                self.register_buffer('calls',torch.tensor(0))
                self.python_calls=0
            def forward(self,x):
                if side_effect:
                    self.calls.add_(1)
                    self.python_calls+=1
                x=torch.nn.functional.gelu(self.projection(x),approximate='tanh')
                return torch.nn.functional.layer_norm(x,(8,),self.weight,self.bias)
        return Model()

    def test_functional_regions_preserve_model_parameters_and_gradients(self):
        from kernel_evolution.graph import discover_functional,validate_rewrite
        for bias in (True,False):
            model=self.make_model(bias=bias)
            original=model.forward
            identities=[id(p) for p in model.parameters()]
            report,rollback=discover_functional(model)
            self.assertEqual({r['op'] for r in report['regions']},{'layer_norm_backward','gelu_mlp'},report)
            self.assertEqual([id(p) for p in model.parameters()],identities)
            validate_rewrite(model,original,lambda m,b:m(b).square().mean(),torch.randn(3,8),
                             {'seed':1729,'rtol':1e-4,'atol':1e-5})
            self.assertEqual(model.python_calls,0)
            rollback()
            self.assertFalse(any(name.startswith('_kernel_evolution_functional_') for name in model._modules))

    def test_unrecorded_buffer_and_python_side_effects_reject_rewrite(self):
        from kernel_evolution.graph import discover_functional,validate_rewrite
        model=self.make_model(side_effect=True)
        original=model.forward
        report,rollback=discover_functional(model)
        self.assertTrue(report['regions'],report)
        self.assertEqual(model.python_calls,0)
        self.assertEqual(int(model.calls),0)
        with self.assertRaises(AssertionError):
            validate_rewrite(model,original,lambda m,b:m(b).square().mean(),torch.randn(3,8),
                             {'seed':1729,'rtol':1e-4,'atol':1e-5})
        rollback()
        self.assertEqual(model.python_calls,0)
        self.assertEqual(int(model.calls),0)

    def test_fx_cannot_freeze_random_tensor_factories(self):
        from kernel_evolution.graph import discover_functional
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(8,8)
            def forward(self,x):
                return torch.nn.functional.gelu(self.linear(x+torch.rand(8)),approximate='tanh')
        model=Model();original=model.forward
        report,_=discover_functional(model)
        self.assertFalse(report['regions'])
        self.assertIn('tensor constant',report['fallback_reason'])
        self.assertEqual(model.forward,original)

    def test_shared_linear_output_is_not_duplicated_by_fusion(self):
        from kernel_evolution.graph import discover_functional
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(8,8)
            def forward(self,x):
                z=self.linear(x)
                return torch.nn.functional.gelu(z,approximate='tanh')+z
        report,_=discover_functional(Model())
        self.assertFalse(report['regions'])
