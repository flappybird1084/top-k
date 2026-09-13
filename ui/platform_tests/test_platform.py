import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'platform_backend'))
import worker
import app

class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.old=app.STORE;app.STORE=Path(self.temp.name)
    def tearDown(self):app.STORE=self.old;self.temp.cleanup()
    def request(self,path,method='GET',data=None,**headers):
        raw=json.dumps(data or {}).encode();env={'PATH_INFO':path,'REQUEST_METHOD':method,'REMOTE_ADDR':'127.0.0.1','wsgi.url_scheme':'http','HTTP_HOST':'localhost','CONTENT_LENGTH':str(len(raw)),'wsgi.input':io.BytesIO(raw),**headers};status=[]
        result=b''.join(app.application(env,lambda code,h:status.append(code)))
        return status[0],json.loads(result)
    def test_repository_404_has_specific_error(self):
        import urllib.error
        text=worker.failure_message(urllib.error.HTTPError('https://api.github.com/repos/owner/repo',404,'Not Found',{},None),'exploring')
        self.assertIn('repository',text)
        self.assertNotIn('dataset',text)

    def test_urls_reject_network_and_shell_injection(self):
        for value in ('http://github.com/a/b','https://github.com/a/b?x=y','https://github.com/a/..','https://127.0.0.1/a/b','https://github.com/a/b;touch'):
            with self.assertRaises(ValueError):worker.github_url(value)
        for value in ('file:///etc/passwd','https://huggingface.co.evil/datasets/a/b','https://huggingface.co/datasets/a/b?token=secret'):
            with self.assertRaises(ValueError):worker.dataset_url(value)
    def test_hf_tree_revision_is_preserved_and_sampled(self):
        url='https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09f'
        self.assertEqual(worker.dataset_url(url),url)
        with patch.object(worker,'get_json',return_value={'sha':'f54c09full'}) as get, patch.object(worker,'pinned_sample',return_value={'config':'default','split':'train','features':[{'name':'text'}]}) as sample:
            result=worker.verify_dataset(url)
            get.assert_called_once_with('https://huggingface.co/api/datasets/roneneldan/TinyStories/revision/f54c09f')
            sample.assert_called_once_with('roneneldan/TinyStories','f54c09full')
            self.assertEqual(result['revision'],'f54c09full')
    def test_hf_revision_path_traversal_rejected(self):
        with self.assertRaises(ValueError):worker.dataset_url('https://huggingface.co/datasets/a/b/tree/..')

    def test_dataset_needs_real_training_sample(self):
        with patch.object(worker,'get_json',return_value={'sha':'rev'}), patch.object(worker,'pinned_sample',side_effect=ValueError('No readable sample')):
            with self.assertRaises(ValueError):worker.verify_dataset('https://huggingface.co/datasets/a/b')
    def test_dataset_verification_records_revision(self):
        with patch.object(worker,'get_json',return_value={'sha':'rev'}), patch.object(worker,'pinned_sample',return_value={'features':['text']}):
            self.assertEqual(worker.verify_dataset('https://huggingface.co/datasets/a/b')['revision'],'rev')
    def test_file_selection_preserved(self):
        with patch.object(worker,'get_json',return_value={'sha':'rev'}), patch.object(worker,'pinned_sample',return_value={'features':['text']}) as sample:
            worker.verify_dataset('https://huggingface.co/datasets/a/b/blob/main/data/train.csv?download=true')
            sample.assert_called_once_with('a/b','rev',file='data/train.csv')
    def test_direct_data_sample(self):
        with patch.object(worker,'read_public_file',return_value=b'text,label\nhello,1\n'):
            result=worker.verify_dataset('https://example.org/train.csv')
            self.assertEqual(result['features'],['text','label'])
            self.assertEqual(len(result['sha256']),64)
    def test_private_download_blocked(self):
        with self.assertRaises(ValueError):worker.read_public_file('https://127.0.0.1/train.csv')
    def test_github_revision_file(self):
        with patch.object(worker,'get_json',side_effect=[{'default_branch':'main'},{'sha':'abc'},{'tree':[{'type':'blob','path':'data/train.jsonl'}]}]), patch.object(worker,'read_public_file',return_value=b'{"text":"hello"}') as read:
            result=worker.verify_dataset('https://github.com/a/b/blob/branch/data/train.jsonl')
            self.assertEqual(result['revision'],'abc')
            read.assert_called_once_with('https://raw.githubusercontent.com/a/b/abc/data/train.jsonl')
    def test_streaming_sample_uses_selected_revision_and_split(self):
        from types import SimpleNamespace
        def configs(path, *, revision):
            self.assertEqual(revision,'commit');return ['default']
        def splits(path, *, config_name, revision):
            self.assertEqual(config_name,'default');return ['train','validation']
        def load(path, *, name, revision, split, streaming):
            self.assertEqual((revision,split,streaming),('commit','train',True));return iter([{'text':'hello'}])
        with patch.dict(sys.modules,{'datasets':SimpleNamespace(get_dataset_config_names=configs,get_dataset_split_names=splits,load_dataset=load)}):
            self.assertEqual(worker.pinned_sample('a/b','commit')['features'][0]['name'],'text')
    def test_empty_repository_stops_before_data_and_gpu(self):
        for files in ([], ['README.md'], ['config.json']):
            with self.assertRaisesRegex(ValueError,'No Python model source'):
                worker.require_model_source(files)
        worker.require_model_source(['src/model.py'])
    def test_repository_branch_links(self):
        for ref in ('feature/model', 'feature%2Fmodel', 'main'):
            value='https://github.com/team/model/tree/'+ref
            self.assertEqual(worker.github_url(value),'https://github.com/team/model/tree/'+ref.replace('%2F','/'))
        for ref in ('../main','-bad;echo','main?token=x','%2e%2e'):
            with self.assertRaises(ValueError):worker.github_url('https://github.com/team/model/tree/'+ref)
    def test_checkout_pins_selected_branch_commit(self):
        sha='a'*40
        with patch.object(worker,'get_json',side_effect=[{'default_branch':'main'}, {'sha':sha}]) as get, patch.object(worker.subprocess,'run') as run, patch.object(worker.subprocess,'check_output',return_value=sha+'\n'):
            branch,commit=worker.checkout_repository('https://github.com/team/model/tree/feature/model',Path('/tmp/source'))
            self.assertEqual((branch,commit),('feature/model',sha))
            self.assertEqual(get.call_args.args[0],'https://api.github.com/repos/team/model/commits/feature%2Fmodel')
            self.assertIn(['git','-c','core.hooksPath=/dev/null','-C','/tmp/source','fetch','--depth','1','origin',sha],[call.args[0] for call in run.call_args_list])
    def test_api_preserves_branch(self):
        with patch.object(app,'TOKEN',''),patch.object(app,'watch') as watch:
            code,result=self.request('/api/runs','POST',{'repo':'https://github.com/team/model/tree/feature/model'},HTTP_IDEMPOTENCY_KEY='c'*32)
            self.assertTrue(code.startswith(('200','201','202')))
            self.assertEqual(watch.call_args.args[1]['repo'],'https://github.com/team/model/tree/feature/model')
    def test_profile_without_candidates_is_reported_separately(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import serve_run
        path=Path(self.temp.name)
        (path/'profile.json').write_text(json.dumps({'step_time_ms':26.69,'eager_step_time_ms':30.17,'targets':[{'eligible':False}]}))
        result=serve_run.profile_summary(path/'archive.sqlite')
        self.assertEqual(result['compiled_step_ms'],26.69)
        self.assertEqual(result['eligible_operations'],0)
        self.assertNotIn('baseline_ms',result)
    def test_runtime_is_run_scoped_and_redacted(self):
        from types import SimpleNamespace
        root=Path(self.temp.name);run='d'*32;job=root/'jobs'/run;job.mkdir(parents=True)
        (job/'search.log').write_text('hello\ntoken=private\nBearer private\n')
        with patch.object(worker,'ROOT',root),patch.object(worker.subprocess,'run',return_value=SimpleNamespace(stdout='Test GPU, 23, 2048, 8192\n')):
            result=worker.runtime_snapshot(run)
        self.assertEqual(result['gpus'][0]['utilization_pct'],23)
        self.assertNotIn('private',result['terminal'])
        self.assertIn('hello',result['terminal'])
    def test_runtime_requires_owned_run(self):
        with patch.object(app,'TOKEN',''),patch.object(app,'remote') as remote:
            code,_=self.request('/api/runs/'+('e'*32)+'/runtime')
            self.assertTrue(code.startswith('404'));remote.assert_not_called()
    def test_create_is_idempotent(self):
        with patch.object(app,'TOKEN',''),patch.object(app,'watch') as watch:
            a=self.request('/api/runs','POST',{'repo':'github.com/team/model'},HTTP_IDEMPOTENCY_KEY='a'*32)
            b=self.request('/api/runs','POST',{'repo':'github.com/team/model'},HTTP_IDEMPOTENCY_KEY='a'*32)
            self.assertEqual(a,b);self.assertEqual(len([p for p in app.STORE.glob('*.json') if p.name!='current.json']),1)
    def test_cross_site_post_rejected(self):
        code,_=self.request('/api/runs','POST',HTTP_ORIGIN='https://evil.example');self.assertTrue(code.startswith('403'))
    def test_auth_and_private_source(self):
        with patch.object(app,'TOKEN','required'):
            self.assertTrue(self.request('/api/runs/'+'a'*32)[0].startswith('401'))
        self.assertTrue(self.request('/platform_backend/app.py')[0].startswith('404'))
    def test_data_requires_waiting_state(self):
        run='a'*32;app.write(app.STORE/(run+'.json'),{'owner':'workspace','status':'running'})
        with patch.object(app,'TOKEN',''):
            self.assertTrue(self.request('/api/runs/'+run+'/data','POST',{'url':'https://huggingface.co/datasets/a/b'})[0].startswith('409'))
    def test_report_override_is_persisted(self):
        run='b'*32;app.write(app.STORE/(run+'.json'),{'owner':'workspace','status':'complete'})
        with patch.object(app,'TOKEN',''):
            code,_=self.request('/api/runs/'+run+'/integrations','POST',{'wandb_report_url':'https://wandb.ai/team/project/reports/test'})
            self.assertTrue(code.startswith('200'))
            _,result=self.request('/api/runs/'+run)
            self.assertIn('wandb_embed_url',result['integrations'])
            self.assertIn('marimo_embed_url',result['integrations'])
    def test_job_budget_reservation_cap(self):
        with patch.object(app,'TOKEN',''),patch.object(app,'watch'):
            app.write(app.STORE/('f'*32+'.json'),{'status':'complete','created_at':__import__('time').time(),'spend_cap_usd':30})
            code,_=self.request('/api/runs','POST',{'repo':'github.com/team/model'},HTTP_IDEMPOTENCY_KEY='b'*32)
            self.assertTrue(code.startswith('400'))
    def test_notebook_auth(self):
        from starlette.testclient import TestClient
        import asgi
        with patch.object(app,'TOKEN','required'):
            client=TestClient(asgi.app)
            result=client.get('/notebook/',follow_redirects=False)
            self.assertEqual(result.status_code,307)
            self.assertEqual(result.headers['location'],'/login.html')

if __name__=='__main__':unittest.main()
