import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required for Inductor extraction')
class SeedTests(unittest.TestCase):
    def test_extracted_seed_preserves_gradients_and_never_reenters_dynamo(self):
        from kernel_evolution.seeds import InductorSeed
        from kernel_evolution.ops import REFERENCES
        from kernel_evolution.verifier import evaluate_with_grads,compare
        x=torch.randn(2,8,32,device='cuda',requires_grad=True)
        w=torch.randn(64,32,device='cuda',requires_grad=True)
        b=torch.randn(64,device='cuda',requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            seed=InductorSeed('gelu_mlp',directory)
            seed.specialize((x,w,b))
            # Fresh data at the same specialization; compilation is now forbidden.
            with patch('torch.compile',side_effect=AssertionError('Dynamo used during timing')):
                actual=evaluate_with_grads(seed,(x,w,b))
            expected=evaluate_with_grads(REFERENCES['gelu_mlp'],(x,w,b))
            compare(actual,expected,1e-4,1e-4)

    def test_saved_statistic_argument_order_and_unseen_shape(self):
        from kernel_evolution.seeds import InductorSeed
        from kernel_evolution.verifier import gate2
        x=torch.randn(2,8,32,device='cuda')
        dy=torch.randn_like(x);w=torch.randn(32,device='cuda')
        _,mean,rstd=torch.native_layer_norm(x,(32,),w,None,1e-5)
        with tempfile.TemporaryDirectory() as directory:
            seed=InductorSeed('layer_norm_backward',directory)
            result=gate2('layer_norm_backward',seed,[{'args':(x,dy,w,mean,rstd),'count':1}],
                         {'seed':1729,'rtol':1e-4,'atol':1e-4})
            self.assertEqual(result['unseen_shapes'],1)
