import tempfile
from unittest import TestCase
from unittest.mock import patch
import ui_server as ui

class Discovery(TestCase):
 def test_discovery_is_idempotent_and_waits_for_user(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp),patch.object(ui,'start_discovery') as start,patch.object(ui.web._queue,'put') as queue:
   client=ui.app.test_client();headers={'Idempotency-Key':'discovery'}
   jid=client.post('/api/runs',json={'repo':'https://github.com/karpathy/nanogpt'},headers=headers).json['id']
   client.post('/api/runs',json={'repo':'https://github.com/karpathy/nanogpt'},headers=headers)
   start.assert_called_once_with(jid)
   self.assertEqual(ui.snapshot(jid)['status'],'exploring')
   self.assertEqual(client.post('/api/runs/'+jid+'/data',json={'url':'https://example.com/data'}).status_code,409)
   finding={'summary':'Two training configurations found.','question':'Which training dataset?','options':[{'name':'Text','url':'https://example.com/data','reason':'Training config uses text','evidence':'https://github.com/karpathy/nanogpt/blob/master/README.md'},{'name':'Unsafe','url':'javascript:alert(1)','evidence':'https://example.com'}]}
   with patch('kernelevo.repo_discovery.discover',return_value=finding):ui.discover_repository(jid)
   state=ui.snapshot(jid);self.assertEqual(state['status'],'awaiting_data');self.assertEqual(len(state['dataset_options']),1)
   self.assertEqual(state['discovery_summary'],finding['summary']);queue.assert_not_called()
 def test_search_failure_falls_back_to_question(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp):
   jid='d'*32;ui.web.save_job(dict(id=jid,status='exploring',repo='https://github.com/a/b',created_at=1))
   with patch('kernelevo.repo_discovery.discover',side_effect=RuntimeError('secret')):ui.discover_repository(jid)
   state=ui.snapshot(jid);self.assertEqual(state['status'],'awaiting_data');self.assertEqual(state['dataset_options'],[]);self.assertNotIn('secret',str(state))
