import concurrent.futures
import json
import tempfile
import unittest
import threading
import time
from unittest.mock import patch,Mock
from types import SimpleNamespace
from pathlib import Path
from kernel_evolution.archive import Archive
from kernel_evolution.budget import Budget,BudgetExceeded,cost
from kernel_evolution.llm import usage_tokens
from search import validated_jobs,load_prepared,subagent_model,verify


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.archive=Archive(Path(self.tmp.name)/'archive.sqlite')
    def tearDown(self):
        self.archive.db.close()
        self.tmp.cleanup()

    def test_parallel_reservations_cannot_overspend(self):
        b=Budget(self.archive,10)
        def attempt(_):
            try:return b.reserve('subagent','gpt-5.6-sol',1,3)
            except BudgetExceeded:return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            accepted=[x for x in pool.map(attempt,range(8)) if x]
        self.assertEqual(len(accepted),3)
        self.assertEqual(self.archive.rows('SELECT SUM(reserved_usd) AS n FROM llm_calls')[0]['n'],9)

    def test_completion_releases_unused_reservation_and_records_cached_tokens(self):
        b=Budget(self.archive,3)
        cid=b.reserve('planner','gpt-5.6-sol',1,3)
        b.finish(cid,'gpt-5.6-sol',dict(input_tokens=1000,cached_input_tokens=500,output_tokens=100))
        self.assertAlmostEqual(b.spent,.0042)
        self.assertTrue(b.reserve('curator','gpt-5.6-sol',1,2))

    def test_missing_usage_does_not_become_free(self):
        b=Budget(self.archive,3)
        cid=b.reserve('subagent','gpt-5.6-sol',1,3)
        b.finish(cid,'gpt-5.6-sol',None)
        self.assertEqual(b.spent,3)
        with self.assertRaises(BudgetExceeded):b.reserve('curator','gpt-5.6-sol',1,.01)

    def test_usage_counts_all_tool_turns_not_only_final_turn(self):
        data={'total':{'input_tokens':3000,'cached_input_tokens':2000,'output_tokens':500},
              'last':{'input_tokens':500,'output_tokens':5}}
        self.assertEqual(usage_tokens(data)['output_tokens'],500)
        self.assertAlmostEqual(cost('gpt-5.6-sol',3000,2000,500),.0148)
        self.assertIsNone(usage_tokens({'total':{}}))
        self.assertIsNone(usage_tokens({'total':{'input_tokens':123}}))

    def test_planner_cannot_parent_from_incorrect_candidate(self):
        self.archive.put('lineages',id='norm',incumbent_id='seed',retired=0)
        self.archive.put('candidates',id='seed',lineage_id='norm',correct_ok=1,accepted=1)
        self.archive.put('candidates',id='wrong',lineage_id='norm',correct_ok=0,accepted=0)
        with self.assertRaises(ValueError):
            validated_jobs({'jobs':[{'lineage':'norm','parents':['wrong'],'strategy':'reuse'}]},self.archive,1,4)
        jobs=validated_jobs({'jobs':[{'lineage':'norm','parents':['seed'],'strategy':'tile'}]*8},self.archive,1,4)
        self.assertEqual(len(jobs),4)

    def test_diversity_keeps_slow_correct_strategy_without_acceptance(self):
        self.archive.put('lineages',id='norm',incumbent_id='seed',retired=0)
        self.archive.put('candidates',id='seed',lineage_id='norm',generation=0,strategy='base',correct_ok=1,accepted=1,step_time_ms=10)
        self.archive.put('candidates',id='slow',lineage_id='norm',generation=1,strategy='new reduction',correct_ok=1,accepted=0,step_time_ms=12)
        self.archive.put('candidates',id='wrong',lineage_id='norm',generation=1,strategy='incorrect',correct_ok=0,accepted=0)
        ids={r['id'] for r in self.archive.parents()['norm']}
        self.assertEqual(ids,{'seed','slow'})

    def test_same_operation_recombination_is_not_cross_region_fusion(self):
        self.archive.put('lineages',id='ema',incumbent_id='fast',retired=0)
        self.archive.put('candidates',id='fast',lineage_id='ema',correct_ok=1,accepted=1)
        self.archive.put('candidates',id='slow',lineage_id='ema',correct_ok=1,accepted=0)
        jobs=validated_jobs({'jobs':[{'lineage':'ema','parents':['fast','slow'],
            'strategy':'Recombine fixed-alpha arithmetic with size-class dispatch within EMA'}]},self.archive,2,4)
        self.assertEqual(jobs[0]['parents'],['fast','slow'])
        self.assertFalse(jobs[0]['fusion'])
        with self.assertRaises(ValueError):
            validated_jobs({'jobs':[{'lineage':'ema','parents':['fast','slow'],
                'strategy':'FUSE: both operations'}]},self.archive,2,4)

    def test_changed_baseline_cannot_reuse_old_calibration(self):
        root=Path(self.tmp.name)
        (root/'prepared.json').write_text(json.dumps({'model':{'adapter_path':'demo'},'config':{}}))
        with self.assertRaisesRegex(ValueError,'Benchmark protocol changed'):
            load_prepared(SimpleNamespace(run_dir=str(root),adapter='demo'),
                          {'benchmark_protocol':'direct_inductor_v2'},self.archive)

    def test_audit_revokes_acceptance_but_preserves_original_evidence(self):
        self.archive.put('lineages',id='ema',incumbent_id='candidate')
        self.archive.put('generations',id=1,n_accepted=1)
        self.archive.put('candidates',id='seed',lineage_id='ema',generation=0,accepted=1)
        self.archive.put('candidates',id='candidate',lineage_id='ema',generation=1,accepted=1,step_time_ms=9)
        self.archive.invalidate_acceptance('candidate','baseline overhead',{'status':'inconclusive'})
        self.assertEqual(self.archive.rows('SELECT incumbent_id FROM lineages')[0]['incumbent_id'],'seed')
        self.assertEqual(self.archive.rows('SELECT n_accepted FROM generations')[0]['n_accepted'],0)
        audit=self.archive.rows('SELECT * FROM candidate_audits')[0]
        self.assertEqual(json.loads(audit['original_row_json'])['accepted'],1)
        self.assertEqual(json.loads(audit['original_row_json'])['step_time_ms'],9)
        self.archive.invalidate_acceptance('candidate','baseline overhead',{})
        self.assertEqual(len(self.archive.rows('SELECT * FROM candidate_audits')),1)

    def test_configured_agent_models_alternate_across_generations(self):
        cfg={'llm':'codex_oauth','subagent_llm':'gpt-5.6-sol',
             'subagent_models':['gpt-5.6-sol','gpt-6-astra'],'candidates_per_gen':3}
        self.assertEqual([subagent_model(cfg,g,i) for g in (1,2) for i in range(3)],
                         ['gpt-5.6-sol','gpt-6-astra']*3)

    def test_repairs_do_not_hold_gpu_lock_and_keep_the_candidate_model(self):
        root=Path(self.tmp.name);source=root/'candidate.py';source.write_text('def kernel(x): return x')
        candidate=dict(id='test',generation=1,lineage_id='ema',code_path=str(source),model_name='gpt-5.6-sol')
        self.archive.put('lineages',id='ema',incumbent_id='seed')
        self.archive.put('candidates',id='seed',lineage_id='ema',source_kind='inductor')
        gpu=threading.Lock()
        llm=Mock()
        def complete(*args,**kwargs):
            self.assertFalse(gpu.locked())
            return {'source':'def kernel(x): return x'}
        llm.complete.side_effect=complete
        cfg=dict(llm='codex_oauth',max_repairs=1,max_call_seconds=10)
        prepared=dict(config={},targets=[{'id':'ema'}],model={'flops_per_sample':1,'peak_flops':1})
        failure=dict(gate_reached=2,compile_ok=1,correct_ok=0,accepted=0,failure_note='raw mismatch')
        success=dict(gate_reached=3,compile_ok=1,correct_ok=1,accepted=0,failure_note='gate3_slower')
        with patch('search.run_worker',side_effect=[failure,success]),patch('search.CodexOAuthLLM',return_value=llm) as provider:
            result=verify(candidate,self.archive,cfg,root,SimpleNamespace(adapter='demo'),prepared,
                          time.monotonic()+10,gpu,threading.BoundedSemaphore(1))
        self.assertEqual(result['repairs_used'],1)
        self.assertEqual(provider.call_args.kwargs['model'],'gpt-5.6-sol')



if __name__=='__main__':unittest.main()
