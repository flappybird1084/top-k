import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from kernel_evolution.archive import Archive
from kernel_evolution.llm import CodexOAuthLLM
from kernel_evolution.tracing import Traces
from search import plan_jobs


class AgentPlatformTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.archive=Archive(self.root/'archive.sqlite')
        self.archive.traces=Traces(self.archive)
        self.cfg=dict(subagent_llm='gpt-6-astra',spend_cap_usd=100,max_call_reservation_usd=10,
                      max_call_seconds=10,max_repairs=1,candidates_per_gen=1)
    def tearDown(self):
        self.archive.db.close();self.tmp.cleanup()

    def test_agent_response_and_usage_are_children_of_candidate_lifecycle(self):
        parent=self.archive.traces.start('candidate_lifecycle',{'candidate_id':'c1'})
        llm=CodexOAuthLLM(self.archive,self.cfg,self.root,'subagent')
        llm.candidate_id='c1';llm.parent_trace_id=parent;llm.generation=2;llm.repair=1
        response={'text':'{"source":"agent response"}','usage':{'input_tokens':100,'output_tokens':20},'status':'completed'}
        with patch('kernel_evolution.llm.run_worker',return_value=response):
            self.assertEqual(llm.complete([{'role':'user','content':'raw compiler feedback'}],json_mode=True),
                             {'source':'agent response'})
        row=self.archive.rows('SELECT * FROM llm_calls')[0]
        self.assertEqual((row['model'],row['candidate_id'],row['repair']),('gpt-6-astra','c1',1))
        span=self.archive.rows("SELECT * FROM trace_outbox WHERE name='agent.subagent'")[0]
        self.assertEqual(span['parent_id'],parent)
        self.assertIn('raw compiler feedback',span['inputs_json'])
        self.assertEqual(json.loads(span['output_json'])['response'],response['text'])
        self.assertIsNotNone(span['ended_at'])

    def test_invalid_plan_gets_raw_feedback_and_automatic_repair(self):
        self.archive.put('lineages',id='op',incumbent_id='seed',retired=0)
        self.archive.put('candidates',id='seed',lineage_id='op',correct_ok=1,accepted=1)
        llm=Mock()
        llm.complete.side_effect=[{'jobs':[{'lineage':'missing','parents':[],'strategy':'agent proposal'}]},
                                 {'jobs':[{'lineage':'op','parents':['seed'],'strategy':'agent proposal'}]}]
        with patch('search.CodexOAuthLLM',return_value=llm):
            jobs=plan_jobs(self.archive,self.cfg,self.root,{'targets':[]},1,time.monotonic()+10)
        self.assertEqual(jobs[0]['lineage'],'op')
        self.assertEqual(llm.complete.call_count,2)
        self.assertIn('unknown/retired lineage',llm.complete.call_args.args[0][-1]['content'])
        self.assertEqual(len(self.archive.rows("SELECT * FROM events WHERE kind='planner_output'")),2)
