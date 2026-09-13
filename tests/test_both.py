import tempfile
from unittest import TestCase
from unittest.mock import patch
import ui_server as ui


def finish_discovery(jid):
 job=ui.web.load_job(jid);job['status']='awaiting_data';ui.web.save_job(job)

class Both(TestCase):
 def setUp(self):
  mock=patch.object(ui,'start_discovery',side_effect=finish_discovery);mock.start();self.addCleanup(mock.stop)
 def test_both_queues_two_linked_jobs_once(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp),patch.dict(ui.os.environ,{'KEVO_UI_LLM':'stub','KEVO_UI_TARGET':'local'}),patch.object(ui.web._queue,'put') as queue:
   client=ui.app.test_client();payload={'repo':'https://github.com/a/b','mode':'both'};headers={'Idempotency-Key':'both'}
   jid=client.post('/api/runs',json=payload,headers=headers).json['id']
   for _ in range(2):
    response=client.post('/api/runs/'+jid+'/data',json={'url':'https://huggingface.co/datasets/a/b'})
    self.assertEqual(response.status_code,202)
   parent=ui.web.load_job(jid);kid=parent['related_runs']['kernel'];child=ui.web.load_job(kid)
   self.assertEqual(parent['mode'],'recipe');self.assertEqual(child['mode'],'kernel')
   self.assertEqual(child['repo'],parent['repo']);self.assertEqual(child['data'],parent['data'])
   self.assertEqual(child['related_runs'],parent['related_runs'])
   self.assertEqual([call.args[0] for call in queue.call_args_list],[jid,kid])
   self.assertEqual(client.post('/api/runs',json=payload,headers=headers).json['id'],jid)
   self.assertEqual(client.get('/api/runs/'+jid).json['related_runs'],{'architecture':jid,'kernel':kid})
