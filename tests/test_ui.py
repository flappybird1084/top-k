import json,tempfile,unittest,sqlite3
from pathlib import Path
from unittest.mock import patch
import ui_server as ui
from kernelevo.archive import SCHEMA


def finish_discovery(jid):
 job=ui.web.load_job(jid);job['status']='awaiting_data';ui.web.save_job(job)

class UI(unittest.TestCase):
 def setUp(self):
  self.discovery=patch.object(ui,'start_discovery',side_effect=finish_discovery);self.discovery.start();self.addCleanup(self.discovery.stop)
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name);self.old=ui.web.JOBS_DIR;ui.web.JOBS_DIR=str(self.p);self.client=ui.app.test_client()
 def tearDown(self):ui.web.JOBS_DIR=self.old;self.tmp.cleanup()
 def test_create_idempotent_and_data_queue(self):
  with patch.object(ui,'read_json',wraps=ui.read_json) as read,patch.object(ui.web._queue,'put') as put:
   r=self.client.post('/api/runs',json={'repo':'https://github.com/a/b/tree/feature/model'},headers={'Idempotency-Key':'abc'})
   self.assertEqual(r.status_code,202);jid=r.json['id']
   again=self.client.post('/api/runs',json={'repo':'https://github.com/a/b/tree/feature/model'},headers={'Idempotency-Key':'abc'})
   self.assertEqual(r.json,again.json)
   job=ui.web.load_job(jid);job['execution_target']='local';job['llm']='stub';ui.web.save_job(job)
   for _ in range(2):self.assertEqual(self.client.post('/api/runs/'+jid+'/data',json={'url':'https://huggingface.co/datasets/a/b'}).status_code,202)
   put.assert_called_once_with(jid)
 def test_missing_provider_preserves_draft_without_queueing(self):
  with patch.dict(ui.os.environ,{},clear=True),patch.object(ui.web._queue,'put') as put:
   jid=self.client.post('/api/runs',json={'repo':'https://github.com/a/b'},headers={'Idempotency-Key':'missing-key'}).json['id']
   response=self.client.post('/api/runs/'+jid+'/data',json={'url':'https://huggingface.co/datasets/a/b'})
   self.assertEqual(response.status_code,400)
   self.assertIn('ANTHROPIC_API_KEY',response.json['error'])
   self.assertEqual(ui.web.load_job(jid)['status'],'awaiting_data')
   self.assertEqual(ui.web.load_job(jid)['data'],'https://huggingface.co/datasets/a/b')
   put.assert_not_called()
 def test_actual_rian_schema_and_no_secret_leak(self):
  jid='a'*32;root=self.p/jid; (root/'run').mkdir(parents=True)
  ui.web.save_job(dict(id=jid,status='done',created_at=1,repo='https://github.com/a/b',data='https://huggingface.co/datasets/a/b',molab={'connection':'secret'},wandb={'api_key':'secret'}))
  with sqlite3.connect(root/'run/archive.sqlite') as db:
   db.executescript(SCHEMA)
   db.execute("insert into models(id) values(1)");db.execute("insert into lineages(id,model_id,op_name) values(1,1,'layer_norm')")
   db.execute("insert into candidates(id,lineage_id,generation,accepted,step_time_ms,incumbent_step_time_ms,gate_reached) values(1,1,1,1,9,10,4)")
  db.close()
  data=self.client.get('/api/runs/'+jid).json
  self.assertEqual(data['baseline_ms'],10);self.assertEqual(data['candidates'][0]['lineage_id'],'layer_norm')
  self.assertNotIn('secret',json.dumps(data));self.assertNotIn('secret',self.client.get('/api/jobs/'+jid).text)
 def test_no_source_or_cross_origin(self):
  self.assertEqual(self.client.get('/ui_server.py').status_code,404)
  self.assertEqual(self.client.post('/api/runs',json={},headers={'Origin':'https://evil.example'}).status_code,403)
 def test_branch_parser(self):
  self.assertEqual(ui.repo_url('github.com/a/b/tree/feature/model'),'https://github.com/a/b/tree/feature/model')
  for value in ('https://github.com/a/../x','http://github.com/a/b','https://github.com/a/b?token=x'):
   with self.assertRaises(ValueError):ui.repo_url(value)
if __name__=='__main__':unittest.main()
