import json,tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import ui_server as ui
class Pending(TestCase):
 def test_disconnected_and_finished_unarchived_work_is_retained(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp):
   jid='f'*32;ui.web.save_job(dict(id=jid,status='failed',stage='exit 1',created_at=1))
   events=[{'id':'2-0','strategy':'A','started_at':1},{'id':'2-1','strategy':'B','started_at':1},{'id':'2-0','finished':True}]
   root=Path(temp)/jid
   (root/'log.txt').write_text('\n'.join('[evaluation] '+json.dumps(e) for e in events)+'\n[molab] lost the notebook session (HTTP 410)')
   s=ui.snapshot(jid)
   self.assertTrue(s['connection_lost']);self.assertEqual(len(s['pending_evaluations']),2)
   self.assertTrue(all(e['state']=='disconnected' for e in s['pending_evaluations']))
   self.assertTrue(all(e['generation']==2 for e in s['pending_evaluations']))
 def test_connected_work_remains_running(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp):
   jid='e'*32;ui.web.save_job(dict(id=jid,status='running',stage='optimizing',created_at=1))
   (Path(temp)/jid/'log.txt').write_text('[evaluation] '+json.dumps({'id':'1-0','strategy':'A','started_at':1}))
   self.assertEqual(ui.snapshot(jid)['pending_evaluations'][0]['state'],'running')
