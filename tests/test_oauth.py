import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from kernelevo.codex_oauth import complete_local,check_login,Relay
from kernelevo.llm import LLMPool
import config

class OAuth(TestCase):
 def test_local_response_and_usage(self):
  def execute(command,**kwargs):
   self.assertIn('--ignore-user-config',command)
   self.assertEqual(command[command.index('--sandbox')+1],'read-only')
   self.assertNotIn('ANTHROPIC_API_KEY',kwargs['input'])
   Path(command[command.index('-o')+1]).write_text('{"ok":true}')
   return SimpleNamespace(returncode=0,stdout=json.dumps({'type':'turn.completed','usage':{'input_tokens':11,'output_tokens':4}}))
  with patch('kernelevo.codex_oauth.subprocess.run',side_effect=execute):
   answer=complete_local({'messages':[{'role':'user','content':'Return JSON.'}],'json_mode':True})
  self.assertEqual(answer,dict(text='{"ok":true}',input_tokens=11,output_tokens=4))
 def test_oauth_routes_all_roles_without_api_keys(self):
  cfg=config.load('DEV');cfg['llm']='codex_oauth'
  pool=LLMPool(cfg)
  for role in (pool.adapter,pool.planner,pool.curator,*pool.subagents,pool.researcher):
   self.assertEqual(role.provider,'codex_oauth')
   self.assertEqual(role.usage_usd(),0)
 def test_login_rejects_api_key_auth(self):
  with patch('kernelevo.codex_oauth.shutil.which',return_value='/bin/codex'),patch('kernelevo.codex_oauth.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='Logged in using API key',stderr='')):
   with self.assertRaises(ValueError):check_login()
 def test_remote_relay_round_trip_cleans_files(self):
  import tempfile
  from kernelevo.codex_oauth import complete
  with tempfile.TemporaryDirectory() as temp,patch.dict('os.environ',{'KEVO_RELAY_DIR':temp}):
   def respond(_):
    request=next(Path(temp).glob('*.req.json'))
    payload=json.loads(request.read_text())
    self.assertEqual(payload['kind'],'codex_oauth')
    self.assertEqual(set(payload),{'messages','kind'})
    request.with_name(request.name.replace('.req.','.res.')).write_text(json.dumps({'text':'connected','input_tokens':2,'output_tokens':1}))
   with patch('kernelevo.codex_oauth.time.sleep',side_effect=respond):
    self.assertEqual(complete({'messages':[]})['text'],'connected')
   self.assertEqual(list(Path(temp).iterdir()),[])
 def test_connection_parser_defers_token_validity_to_server(self):
  from kernelevo.molab import parse_connection
  self.assertEqual(parse_connection({'notebook_url':'https://test.molab.run','connection':'--token *valid-token'}),('https://test.molab.run','*valid-token'))
