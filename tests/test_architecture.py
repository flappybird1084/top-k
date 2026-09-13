import json
import sqlite3
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import ui_server as ui
from kernelevo.archive import SCHEMA
from kernelevo.recipes import is_better_final

class Architecture(TestCase):
 def test_separate_recipe_rows_and_budget_baselines(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp):
   jid='b'*32;root=Path(temp)/jid/'run';root.mkdir(parents=True)
   ui.web.save_job(dict(id=jid,status='done',mode='recipe',created_at=1))
   with sqlite3.connect(root/'archive.sqlite') as db:
    db.executescript(SCHEMA)
    db.execute('insert into models(id) values(1)')
    db.execute("insert into lineages(id,model_id,op_name) values(1,1,'architecture')")
    for loss,phase,budget in [(4.1,'baseline',60),(3.5,'baseline',300),(3.8,'architecture',60),(3.4,'finals',300)]:
     db.execute('insert into candidates(lineage_id,generation,val_loss,phase,train_secs,model_params,accepted) values(1,1,?,?,?,?,1)',(loss,phase,budget,23000000))
   db.close()
   state=ui.snapshot(jid)
   self.assertEqual(state['candidates'],[])
   self.assertEqual(state['mode'],'recipe')
   rows=state['architecture']['candidates']
   self.assertEqual([(r['train_secs'],r['val_loss']) for r in rows if r['phase']=='baseline'],[(60,4.1),(300,3.5)])
   self.assertNotIn('baseline_ms',state)
 def test_mode_validation_and_default(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(ui.web,'JOBS_DIR',temp):
   client=ui.app.test_client();headers={'Idempotency-Key':'mode'}
   response=client.post('/api/runs',json={'repo':'https://github.com/a/b'},headers=headers)
   self.assertEqual(ui.web.load_job(response.json['id'])['mode'],'recipe')
   self.assertEqual(client.post('/api/runs',json={'repo':'https://github.com/a/b','mode':'kernel'},headers=headers).status_code,400)
   self.assertEqual(client.post('/api/runs',json={'repo':'https://github.com/a/b','mode':'invalid'},headers=headers).status_code,400)
 def test_final_winner_requires_improvement_at_final_budget(self):
  winner={'val_loss':2.0,'final_val_loss':3.5}
  self.assertTrue(is_better_final(3.4,True,winner))
  self.assertFalse(is_better_final(3.6,True,winner))
  self.assertFalse(is_better_final(3.0,False,None))
