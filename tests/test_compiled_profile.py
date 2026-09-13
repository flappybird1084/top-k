import copy
import unittest
from kernel_evolution.compiled_profile import parse_source,classify,attribute,augment_targets


# This is generated-source metadata only, not an executable candidate kernel.
SOURCE = '''
# Topologically Sorted Source Nodes: [a, b, c], Original ATen: [aten.mul, aten.add, aten.copy_]
# %a = call_function[target=torch.ops.aten.mul.Tensor](args = (%x, 0.996), kwargs = {})
# %b = call_function[target=torch.ops.aten.mul.Tensor](args = (%y, 0.0040000000000000036), kwargs = {})
triton_poi_fused_ema_0 = async_compile.triton('triton_poi_fused_ema_0', "@triton.jit\\ndef metadata_only(): pass", device_str='cuda')
# Topologically Sorted Source Nodes: [norm_bwd, add], Original ATen: [aten.native_layer_norm_backward, aten.add]
triton_red_norm_1 = async_compile.triton('triton_red_norm_1', "@triton.jit\\ndef metadata_only(): pass", device_str='cuda')
'''


class CompiledProfileTests(unittest.TestCase):
    def test_metadata_parsing_does_not_execute_source(self):
        regions = parse_source(SOURCE + '\nraise RuntimeError("must not execute")', 'code.py')
        self.assertEqual(len(regions),2)
        self.assertEqual(regions[0]['original_aten'],['aten.mul','aten.add','aten.copy_'])
        self.assertEqual(regions[1]['source_nodes'],['norm_bwd','add'])
        self.assertEqual(regions[0]['source_path'],'code.py')
        self.assertTrue(regions[0]['has_triton_jit'])

    def test_generic_add_mul_is_not_ema_without_observed_decay(self):
        ema=parse_source(SOURCE)[0]
        self.assertEqual(classify(ema,['ema_update']),[])
        self.assertEqual(classify(ema,['ema_update'],.9),[])
        self.assertEqual(classify(ema,['ema_update'],.996),['ema_update'])
        other=copy.deepcopy(ema);other['original_aten'].append('aten.sqrt')
        self.assertEqual(classify(other,['ema_update'],.996),[])

    def test_leaf_costs_unknown_and_fused_bound_are_explicit(self):
        report=attribute(parse_source(SOURCE),[
            {'name':'triton_poi_fused_ema_0','duration_us':200},
            {'name':'triton_poi_fused_ema_0','duration_us':100},
            {'name':'triton_red_norm_1','duration_us':100},
            {'name':'external_gemm','duration_us':600},
        ],steps=2,step_time_ms=1,known_ops=['ema_update','layer_norm_backward'],ema_decay=.996)
        self.assertAlmostEqual(report['total_cuda_time_ms'],.5)
        self.assertAlmostEqual(report['unmapped_cuda_time_ms'],.3)
        self.assertAlmostEqual(report['operations']['ema_update']['region_time_ms'],.15)
        self.assertEqual(report['operations']['layer_norm_backward']['attribution'],'associated_fused_region_upper_bound')

    def test_conflicting_kernel_names_fail_closed_but_duplicate_captures_match(self):
        regions=parse_source(SOURCE)
        event=[{'name':'triton_poi_fused_ema_0','duration_us':10}]
        kwargs=dict(steps=1,step_time_ms=1,known_ops=['ema_update'],ema_decay=.996)
        duplicate=copy.deepcopy(regions[0]);duplicate['source_path']='another.py'
        report=attribute(regions+[duplicate],event,**kwargs)
        self.assertEqual(report['kernels'][0]['mapping'],'matched')
        self.assertTrue(report['kernels'][0]['attribution_unambiguous'])
        duplicate['source_hash']='different'
        report=attribute(regions+[duplicate],event,**kwargs)
        self.assertEqual(report['kernels'][0]['mapping'],'ambiguous')
        self.assertEqual(report['operations']['ema_update']['region_time_ms'],0)

    def test_eligibility_uses_compiled_evidence_never_eager_fallback(self):
        report=attribute(parse_source(SOURCE),[{'name':'triton_poi_fused_ema_0','duration_us':10}],
                         steps=1,step_time_ms=1,known_ops=['ema_update','gelu_mlp'],ema_decay=.996)
        profile={'step_time_ms':10,'targets':[dict(id=op,pct_step_time=50,op_time_ms=5,eligible=True)
                                             for op in ['ema_update','gelu_mlp']]}
        augment_targets(profile,report,allowed_ops=['ema_update','gelu_mlp'],min_pct_step_time=5)
        self.assertFalse(any(t['eligible'] for t in profile['targets']))
        self.assertEqual(profile['targets'][0]['eager_pct_step_time'],50)
        self.assertEqual(profile['targets'][0]['pct_step_time'],1)
        self.assertEqual(profile['targets'][1]['op_time_ms'],0)

    def test_non_triton_metadata_never_qualifies(self):
        regions=parse_source(SOURCE)
        regions[0]['has_triton_jit']=False
        report=attribute(regions,[dict(name='triton_poi_fused_ema_0',duration_us=100)],
                         steps=1,step_time_ms=1,known_ops=['ema_update'],ema_decay=.996)
        self.assertEqual(report['operations']['ema_update']['region_time_ms'],0)
        self.assertFalse(report['kernels'][0]['emits_triton'])

    def test_invalid_timing_rejected(self):
        with self.assertRaises(ValueError):
            attribute([],[],steps=0,step_time_ms=1,known_ops=[])
        with self.assertRaises(ValueError):
            attribute([],[dict(name='kernel',duration_us=float('nan'))],steps=1,step_time_ms=1,known_ops=[])
