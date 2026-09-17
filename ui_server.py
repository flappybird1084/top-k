"""Modern UI adapter for Rian's Flask job queue and kernelevo archives."""
from contextlib import closing
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from flask import request, jsonify, send_from_directory
import web

app=web.app
ROOT=Path(__file__).parent
UI=ROOT/'ui'
lock=threading.RLock()
cache={}
discovery_slots=threading.BoundedSemaphore(2)

def job_path(jid):
    if not re.fullmatch(r'[a-f0-9]{32}',jid):raise ValueError('Invalid run ID')
    return Path(web.JOBS_DIR)/jid

def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def redact(text):
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    text=re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+',r'\1[redacted]',text)
    return re.sub(r'(?i)(Bearer\s+\S+|sk-[\w-]+)','[redacted]',text)

def log_tail(root):
    try:
        with (root/'log.txt').open('rb') as f:
            f.seek(0,2);offset=max(0,f.tell()-24000);f.seek(offset);raw=f.read().decode(errors='replace')
            if offset:raw='[Earlier output omitted]\n'+raw.partition('\n')[2]
        return redact('\n'.join(raw.splitlines()[-100:]))
    except OSError:return ''

def repo_url(value):
    u=urlsplit(value.strip() if '://' in value else 'https://'+value.strip())
    if u.scheme!='https' or u.hostname!='github.com' or u.username or u.password or u.port or u.query or u.fragment:raise ValueError('Use a GitHub repository or branch HTTPS link')
    parts=u.path.strip('/').split('/')
    if len(parts)!=2 and (len(parts)<4 or parts[2]!='tree'):raise ValueError('Use a repository or /tree/branch link')
    if not all(re.fullmatch(r'[\w.-]+',p) and '..' not in p and not p.startswith('-') for p in parts):raise ValueError('Invalid repository link')
    return 'https://github.com/'+'/'.join(parts)

def data_url(value):
    u=urlsplit(value.strip())
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.port:raise ValueError('Use an HTTPS dataset link without credentials')
    return value.strip()

SETTINGS_PROVIDERS = {'stub', 'anthropic', 'openai', 'wandb', 'codex_oauth', 'claude_oauth'}


def sanitize_settings(payload):
    """Validated per-job overrides from the front page's Settings dialog.
    Everything is optional; server env (KEVO_UI_*) is the fallback. In the
    public judging deployment, server policy is absolute — client settings
    are ignored wholesale."""
    s = payload.get('settings') or {}
    if not isinstance(s, dict) or os.getenv('JUDGES_EXPIRES_AT'):
        return {}
    out = {}
    llm = str(s.get('llm') or '').strip()
    if llm and len(llm) < 80 and llm.partition(':')[0] in SETTINGS_PROVIDERS:
        out['llm'] = llm
    if s.get('profile') in ('DEV', 'RUN'):
        out['profile'] = s['profile']

    def num(key, lo, hi, cast):
        try:
            v = cast(s[key])
        except (KeyError, TypeError, ValueError):
            return None
        return v if lo <= v <= hi else None

    for key, lo, hi, cast in (('spend_cap', 1, 500, float),
                              ('max_debug_turns', 1, 10, int),
                              ('max_generations', 1, 50, int)):
        v = num(key, lo, hi, cast)
        if v is not None:
            out[key] = v
    conn = str(s.get('molab_connection') or '').strip()
    if conn and len(conn) < 4000:
        out['molab_connection'] = conn
    recipe = {}
    for key, lo, hi, cast in (('arch_gens', 0, 10, int), ('arch_cands', 1, 16, int),
                              ('arch_secs', 10, 3600, int),
                              ('hp_gens', 0, 10, int), ('hp_cands', 1, 16, int),
                              ('hp_secs', 10, 3600, int),
                              ('finals_k', 1, 8, int), ('finals_secs', 10, 7200, int),
                              ('parallelism', 1, 16, int), ('parent_pool', 1, 16, int),
                              ('eval_batches', 1, 64, int),
                              ('loss_margin', 0.0, 0.2, float),
                              ('param_ratio', 1.0, 3.0, float)):
        v = num(key, lo, hi, cast)
        if v is not None:
            recipe[key] = v
    if recipe:
        out['recipe'] = dict(
            phases=[dict(kind='architecture', generations=recipe.get('arch_gens', 2),
                         candidates=recipe.get('arch_cands', 8),
                         train_seconds=recipe.get('arch_secs', 60)),
                    dict(kind='mixed', generations=0, candidates=8, train_seconds=180),
                    dict(kind='hyperparam', generations=recipe.get('hp_gens', 1),
                         candidates=recipe.get('hp_cands', 8),
                         train_seconds=recipe.get('hp_secs', 120))],
            finals_top_k=recipe.get('finals_k', 2),
            finals_train_seconds=recipe.get('finals_secs', 300),
            param_budget_ratio=recipe.get('param_ratio', 1.10),
            loss_margin_rel=recipe.get('loss_margin', 0.003),
            eval_batches=recipe.get('eval_batches', 8),
            subagent_parallelism=recipe.get('parallelism', 8),
            parent_pool=recipe.get('parent_pool', 4))
    return out


# A signed-in run must not wait on the server's own CLI logins. The live
# checks — `codex login status`, and for Claude Code a real (billed) one-token
# completion — took up to 90 seconds inside the request that starts a run
# (audit finding 7). Now the request does only the free local check and reads
# the last background result; the live probe runs on its own thread.
PROVIDER_PROBE_TTL = 900
PROVIDER_BINARIES = {
    'codex_oauth': ('codex', 'Install Codex CLI on the server and sign in with ChatGPT.'),
    'claude_oauth': ('claude', 'Install Claude Code on the server (npm i -g '
                               '@anthropic-ai/claude-code) and sign in with `claude`.')}
_provider_probe = {}        # provider -> (checked_at, error message or None)
_probe_lock = threading.Lock()


def _live_probe(provider):
    try:
        if provider == 'codex_oauth':
            from kernelevo.codex_oauth import check_login
            check_login()
        else:
            from kernelevo.claude_oauth import check_login
            check_login(live=True)
        error = None
    except Exception as e:  # noqa: BLE001 — any failure is a readiness answer
        error = str(e)
    with _probe_lock:
        _provider_probe[provider] = (time.time(), error)


def provider_ready(provider):
    """Fast readiness for an OAuth CLI provider: is it installed, and did the
    last background probe find it usable? Never blocks on the model."""
    binary, message = PROVIDER_BINARIES[provider]
    if not shutil.which(binary):
        raise ValueError(message)
    with _probe_lock:
        entry = _provider_probe.get(provider)
        stale = not entry or time.time() - entry[0] > PROVIDER_PROBE_TTL
        if stale:
            # Stamp before launching so a burst of requests starts one probe,
            # not one per request.
            _provider_probe[provider] = (time.time(), None)
    if stale:
        threading.Thread(target=_live_probe, args=(provider,), daemon=True).start()
        return
    if entry[1]:
        raise ValueError(entry[1])


def validate_provider(job):
    import config
    cfg=config.load(job['profile'])
    if job.get('llm'):
        cfg['llm']=job['llm']
        cfg['planner_llm']=cfg['subagent_llm']=cfg['curator_llm']=None
    specs=[cfg['llm']]+[cfg.get(role+'_llm') for role in ('planner','subagent','curator','adapter','researcher')]
    env=web._job_env(job)
    for spec in specs:
        for value in (spec if isinstance(spec,list) else [spec]):
            if not value:continue
            provider=value.partition(':')[0]
            if provider in PROVIDER_BINARIES:
                provider_ready(provider)
            keys={'anthropic':('ANTHROPIC_API_KEY',),'openai':('OPENAI_API_KEY',),'wandb':('WANDB_INFERENCE_API_KEY','WANDB_API_KEY')}.get(provider,())
            if keys and not any(env.get(key) for key in keys):
                raise ValueError('Training is not configured yet. Set '+ ' or '.join(keys)+' on the server, then retry. Your links are saved.')

def discover_repository(jid):
    from kernelevo.repo_discovery import discover
    with discovery_slots:
        with lock:
            job=read_json(job_path(jid)/'job.json')
            if not job or job['status']!='exploring':return
            if os.getenv('JUDGES_EXPIRES_AT') or job.get('visitor'):
                # Public deployment: never spend the operator's Codex OAuth session
                # or live web search on a visitor's arbitrary repository. That path
                # runs before the job reaches the visitor's notebook and bypasses
                # the relay opt-in, active-run token, model allowlist, and per-owner
                # budget. Ask the visitor for the data link directly instead.
                job.update(status='awaiting_data',
                           stage='Add your training data link to continue.',
                           dataset_options=[],
                           discovery_summary='Add the Hugging Face or dataset URL this run should use.')
                job['discovery_activity']=[{'message':job['stage'],'created_at':time.time()}]
                web.save_job(job)
                return
            job['stage']='Inspecting repository training code and searching for dataset sources…'
            job['discovery_activity']=[{'message':job['stage'],'created_at':time.time()}]
            web.save_job(job)
        try:
            finding=discover(job['repo'])
            options=[]
            for option in finding.get('options',[])[:4]:
                try:
                    options.append(dict(name=str(option['name'])[:100],url=data_url(option['url']),
                        reason=redact(str(option.get('reason','')))[:500],evidence=data_url(option['evidence'])))
                except (KeyError,ValueError,TypeError):continue
            summary=redact(str(finding.get('summary','Repository review complete.')))[:1000]
            question=redact(str(finding.get('question','Which dataset would you like to use?')))[:400]
        except Exception:
            options=[];summary='Repository search could not identify a dataset. Add your training data link to continue.'
            question='Which dataset should this run use?'
        with lock:
            job=read_json(job_path(jid)/'job.json')
            if not job or job['status']!='exploring':return
            job.update(status='awaiting_data',stage=question,dataset_options=options,discovery_summary=summary)
            job.setdefault('discovery_activity',[]).append({'message':summary,'created_at':time.time()})
            web.save_job(job)

def start_discovery(jid):
    threading.Thread(target=discover_repository,args=(jid,),daemon=True).start()

def evaluation_history(root):
    """Retain started work even if its completion never reaches the archive."""
    events={}
    try:
        with (root/'log.txt').open() as stream:
            for line in stream:
                if not line.startswith('[evaluation] '):continue
                try:
                    event=json.loads(line[len('[evaluation] '):]);eid=str(event['id'])
                    if event.get('finished'):
                        if eid in events:events[eid]['finished']=True
                    else:events[eid]={**event,'id':eid}
                except (ValueError,KeyError,TypeError):continue
    except OSError:pass
    return list(events.values())

def snapshot(jid):
    root=job_path(jid);job=read_json(root/'job.json')
    if not job:raise FileNotFoundError()
    status={'done':'complete','interrupted':'failed'}.get(job['status'],job['status'])
    result=dict(id=jid,repo=job.get('repo'),data=job.get('data'),status=status,mode=job.get('mode','kernel'),message=job.get('stage',''),candidates=[],traces=[],activity=[],integrations=dict(job.get('integrations',{})))
    result['integrations'].update(read_json(root/'observability.json',{}))
    result['related_runs']=job.get('related_runs',{})
    result['dataset_options']=job.get('dataset_options',[])
    result['discovery_summary']=job.get('discovery_summary','')
    result['active_evaluations']=list(job.get('active_evaluations',{}).values()) if status=='running' else []
    log=log_tail(root)
    result['activity']=job.get('discovery_activity',[])+[{'message':line,'created_at':job['created_at']} for line in log.splitlines() if line.startswith(('[recipe]','[baseline]','[finals]','[agent]','[adapter]','[ingest]','[profile]','[calibrate]','[planner]','[gates]','[gate4]','[loop]'))][-12:]
    db_path=root/'run/archive.sqlite'
    if db_path.exists():
        with closing(sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT c.*,l.op_name FROM candidates c JOIN lineages l ON c.lineage_id=l.id WHERE l.model_id=(SELECT MAX(id) FROM models) ORDER BY c.generation,c.id')]
            for row in rows:
                candidate={k:row.get(k) for k in ('id','generation','strategy','accepted','gate_reached','step_time_ms','incumbent_step_time_ms','created_at','val_loss','phase','train_secs','model_params','parent_id','parents_json','model_name','failure_note','correct_ok','repairs_used','tokens_in','tokens_out')}
                # basename only: evaluation-event dedupe key, never a full path
                candidate['code_file']=str(row.get('code_path') or '').rsplit('/',1)[-1] or None
                candidate['lineage_id']=row['op_name'];result['candidates'].append(candidate)
                if row.get('weave_trace_url'):result['traces'].append(dict(id=str(row['id']),name=row['op_name'],url=row['weave_trace_url'],ended_at=0,error=None))
            result['architecture']={'candidates':[r for r in result['candidates'] if r.get('phase')]}
            result['candidates']=[r for r in result['candidates'] if not r.get('phase')]
            if result['architecture']['candidates']:result['mode']='recipe'
            baseline=next((r['incumbent_step_time_ms'] for r in rows if r.get('incumbent_step_time_ms')),None)
            if baseline:result['baseline_ms']=baseline
            gens=db.execute('SELECT id,stop_reason FROM generations WHERE model_id=(SELECT MAX(id) FROM models) ORDER BY id DESC LIMIT 1').fetchone()
            if gens:result.update(generation=gens['id'],stop_reason=gens['stop_reason'])
            try:  # per-generation metadata (older archives lack the token columns)
                result['generations']=[
                    dict(n=i+1,n_candidates=g['n_candidates'],n_accepted=g['n_accepted'],
                         llm_usd=g['llm_usd'],tokens_in=g['tokens_in'],tokens_out=g['tokens_out'])
                    for i,g in enumerate(db.execute(
                        'SELECT n_candidates,n_accepted,llm_usd,tokens_in,tokens_out '
                        'FROM generations WHERE model_id=(SELECT MAX(id) FROM models) ORDER BY id'))]
            except sqlite3.Error:
                result['generations']=[]
    urls=re.findall(r'https://wandb.ai/[\w.-]+/[\w.-]+/runs/[\w]+',log)
    if urls:result['integrations']['wandb_url']=urls[-1]
    weave=re.findall(r'https://wandb.ai/[\w.-]+/[\w.-]+/weave',log)
    if weave:result['integrations']['weave_url']=weave[-1]
    if job.get('judge_expires_at'):
        # Notebook authentication stays with the notebook owner, never in a public iframe.
        result['integrations']['marimo_url']=job.get('molab',{}).get('notebook_url','')
        result['integrations'].pop('marimo_embed_url',None)
    else:
        result['integrations'].update(marimo_url='/notebook/?run='+jid,marimo_embed_url='/notebook/?run='+jid)
    targets=read_json(root/'run/targets.json',{})
    if result.get('baseline_ms') and targets.get('step_time_ms'):
        result['profiling']=dict(compiled_step_ms=result['baseline_ms'],eager_step_ms=targets['step_time_ms'],eligible_operations=len(targets.get('lineages',[])),operations_inspected=len(targets.get('lineages',[])))
    disconnected='lost the notebook session' in log or job.get('connection_lost',False)
    result['connection_lost']=disconnected
    archived=result.get('candidates',[])+result.get('architecture',{}).get('candidates',[])
    pending=[]
    for event in evaluation_history(root):
        # authoring load-checks are plumbing, never diagram nodes (older runs'
        # logs still contain their events)
        if event.get('stage')=='Checking recipe':continue
        kernel=str(event.get('kernel') or '')
        # baselines are archived as generation-0 rows the moment they finish
        if kernel=='BASELINE' and event.get('finished'):continue
        match=re.match(r'^(\d+)-',event['id'])
        generation=event.get('generation',int(match[1]) if match else None)
        if any(r.get('strategy')==event.get('strategy') and (generation is None or r.get('generation')==generation) for r in archived):continue
        # recipe events name the candidate FILE; older ones carried a generic
        # 'Ns training budget' strategy that never matched an archived row, so
        # every evaluation lingered as a phantom bubble — dedupe by code_path
        if kernel.endswith('.py') and any(r.get('code_file')==kernel for r in archived):continue
        event['generation']=generation
        event['state']='disconnected' if disconnected else 'awaiting_sync' if event.get('finished') else 'running' if status=='running' else 'interrupted'
        pending.append(event)
    result['pending_evaluations']=pending
    result['bpd_comparison']=read_json(root/'run/result.json',{}).get('bpd_comparison',{})
    return result

@app.before_request
def guard():
    if request.remote_addr not in ('127.0.0.1','::1',None):return jsonify(error='This development server is local only'),403
    if request.method=='POST' and request.headers.get('Origin') not in (None,request.host_url.rstrip('/')):return jsonify(error='Origin rejected'),403

@app.errorhandler(FileNotFoundError)
def missing(_):return jsonify(error='Run not found'),404
@app.errorhandler(ValueError)
def invalid(error):return jsonify(error=str(error)),400

app.view_functions['index']=lambda:send_from_directory(UI,'index.html')
@app.get('/<name>')
def static_ui(name):
    if name not in {'evolution.js','demo.js','index.html','front.js','front.css','workspace.html','run.js','run.css','results.html','results.js'}:return jsonify(error='Not found'),404
    response=send_from_directory(UI,name);response.headers['Cache-Control']='no-store';return response
@app.get('/assets/<path:name>')
def assets(name):return send_from_directory(UI/'assets',name)
@app.get('/demo/')
@app.get('/demo/<path:name>')
def recorded_demo(name='index.html'):
    if Path(name).suffix.lower() not in {'.html','.js','.css','.png','.svg','.json','.woff2','.ttf'}:
        return jsonify(error='Not found'),404
    return send_from_directory(UI/'demo',name)
@app.get('/api/runs/<jid>')
def run_api(jid):return snapshot(jid)

@app.post('/api/runs')
def create():
    payload=request.get_json() or {}
    repo=repo_url(payload.get('repo',''))
    mode=payload.get('mode','recipe')
    if mode not in ('recipe','kernel','both'):raise ValueError('Choose architecture, kernels, or both')
    key=request.headers.get('Idempotency-Key','')
    if not key:raise ValueError('Missing request identifier')
    jid=hashlib.sha256(key.encode()).hexdigest()[:32]
    with lock:
        old=read_json(job_path(jid)/'job.json')
        if old and (old.get('repo')!=repo or old.get('requested_mode',old.get('mode','kernel'))!=mode):raise ValueError('Request identifier already used')
        if not old:
            job=dict(id=jid,created_at=time.time(),status='exploring',mode=mode,requested_mode=mode,stage='Repository agent queued for dataset discovery.',repo=repo,adapter=None,comments='',max_debug_turns=5,profile=os.getenv('KEVO_UI_PROFILE','RUN'),llm=os.getenv('KEVO_UI_LLM') or None,execution_target=os.getenv('KEVO_UI_TARGET','molab'),molab={},wandb={})
            if os.getenv('KEVO_UI_MAX_GENERATIONS'):job['max_generations']=int(os.environ['KEVO_UI_MAX_GENERATIONS'])
            job.update(sanitize_settings(payload))  # user settings win over env
            web.save_job(job)
            start_discovery(jid)
    return {'id':jid},202


@app.get('/api/settings')
def settings_defaults():
    """Server-side defaults the Settings dialog shows as placeholders."""
    connection=read_json(os.getenv('KEVO_MOLAB_CONNECTION_FILE',str(Path.home()/'.local/state/kernel-evolution/molab.json')), {})
    return dict(llm=os.getenv('KEVO_UI_LLM') or '(config profile default)',
                profile=os.getenv('KEVO_UI_PROFILE','RUN'),
                execution_target=os.getenv('KEVO_UI_TARGET','molab'),
                connection_file=bool(connection.get('url') and connection.get('token')))

@app.post('/api/runs/<jid>/data')
def submit_data(jid):
    root=job_path(jid)
    with lock:
        job=read_json(root/'job.json')
        if not job:raise FileNotFoundError()
        url=data_url((request.get_json() or {}).get('url',''))
        if job['status']!='awaiting_data':
            if job.get('data')==url:return {'id':jid},202
            return jsonify(error='Run is no longer waiting for data'),409
        job['data']=url
        web.save_job(job)
        # llm/profile/max_generations were resolved at create time with
        # user-settings > KEVO_UI_* env > default precedence; no re-override here
        validate_provider(job)
        if job['execution_target']=='molab':
            if (job.get('molab') or {}).get('connection') or job.get('visitor'):
                # Already tied to its owner's own notebook (public gateway
                # runs, whose token is resolved from the integration store at
                # launch): never replace it with the operator's connection file.
                pass
            elif job.get('molab_connection'):
                # per-job pasted "Pair with agent" prompt from the Settings dialog
                job['molab']={'notebook_url':'','connection':job.pop('molab_connection')}
            else:
                connection=read_json(os.getenv('KEVO_MOLAB_CONNECTION_FILE',str(Path.home()/'.local/state/kernel-evolution/molab.json')), {})
                if not connection.get('url') or not connection.get('token'):raise ValueError('No molab connection: paste the notebook’s "Pair with agent" prompt in Settings, or configure KEVO_MOLAB_CONNECTION_FILE on the server')
                job['molab']={'notebook_url':connection['url'],'connection':'--token '+connection['token']}
        job.update(data=url,comments='Use this training data link, preserving its revision and split: '+url+'. Verify data compatibility before optimization. Do not substitute synthetic data.',status='queued',stage='Queued for repository inspection and data verification.')
        choice=(request.get_json() or {}).get('dataset_choice')
        selected=next((o for o in job.get('dataset_options',[]) if o['name']==choice and o['url']==url),None)
        if selected:
            job['dataset_choice']=selected['name']
            job['comments']+=' User selected dataset/configuration: '+selected['name']+'. Repository evidence: '+selected['evidence']+'. '+selected['reason']
        companion=None
        if job.get('requested_mode',job.get('mode'))=='both':
            import copy
            kernel_id=hashlib.sha256((jid+':kernel').encode()).hexdigest()[:32]
            job.update(mode='recipe',related_runs={'architecture':jid,'kernel':kernel_id})
            companion=copy.deepcopy(job)
            companion.update(id=kernel_id,mode='kernel',requested_mode='kernel',stage='Queued after architecture search.')
            web.save_job(companion)
        web.save_job(job)
        web._queue.put(jid)
        if companion:web._queue.put(companion['id'])
    return {'id':jid},202

@app.get('/api/runs/<jid>/runtime')
def runtime(jid):
    root=job_path(jid);job=read_json(root/'job.json')
    if not job:raise FileNotFoundError()
    cached=cache.get(jid)
    if cached and time.time()-cached['sampled_at']<5:return {**cached,'terminal':log_tail(root)}
    code="import subprocess,json\np=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)\nprint('GPU_JSON:'+json.dumps(p.stdout))"
    try:
        if job.get('execution_target')=='molab':
            from kernelevo.molab import MolabClient,parse_connection
            connection=job.get('molab') or {}
            if job.get('visitor') and not connection.get('connection'):
                from kernelevo.integrations import notebook_connection
                connection=notebook_connection(job['visitor']) or connection
            ok,out,_=MolabClient(*parse_connection(connection)).run(code)
        else:out=subprocess.check_output([os.sys.executable,'-c',code],text=True,timeout=6)
        raw=json.loads(next(line[9:] for line in out.splitlines() if line.startswith('GPU_JSON:')))
        gpus=[]
        for row in csv.reader(io.StringIO(raw)):
            if len(row)==4:gpus.append(dict(name=row[0].strip(),utilization_pct=float(row[1]),memory_used_mb=float(row[2]),memory_total_mb=float(row[3])))
        result=dict(gpus=gpus,sampled_at=time.time(),terminal=log_tail(root))
    except Exception:result=dict(gpus=[],sampled_at=time.time(),terminal=log_tail(root),gpu_error='GPU telemetry unavailable')
    cache[jid]=result;return result

@app.get('/api/runs/<jid>/wandb')
def wandb_metrics(jid):
    """Return a small, browser-native W&B view for the owner’s run.

    We deliberately read a bounded history instead of embedding W&B’s page.
    W&B can restrict framing and an embedded page is not a useful live view on
    smaller screens. The full run remains available through its direct link.
    """
    state=snapshot(jid);url=state['integrations'].get('wandb_url')
    if not url:return {'metrics':[], 'series':[]}
    key='wandb:'+jid
    if key in cache and time.time()-cache[key]['sampled_at']<30:return cache[key]
    try:
        import wandb
        # Public runs report to the visitor's W&B account. Read the resulting
        # run with that same account instead of silently falling back to the
        # gateway operator's ambient WANDB_API_KEY.
        job=read_json(job_path(jid)/'job.json',{})
        api_key=None
        if job.get('visitor'):
            from kernelevo.integrations import wandb_env
            api_key=wandb_env(job['visitor']).get('WANDB_API_KEY')
            if not api_key:return {'metrics':[], 'series':[]},503
        parts=urlsplit(url).path.strip('/').split('/')
        api=wandb.Api(api_key=api_key,timeout=10) if api_key else wandb.Api(timeout=10)
        run=api.run('/'.join([parts[0],parts[1],parts[3]]))
        numeric=lambda value: isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        metrics=[dict(name=k,value=v) for k,v in dict(run.summary).items()
                 if not k.startswith('_') and numeric(v)]
        prefix='recipe/' if state.get('mode')=='recipe' else None
        if prefix:metrics=[m for m in metrics if m['name'].startswith(prefix)]

        # History is capped so refreshing the Top-K page never turns into a
        # bulk W&B export. Keep the most legible learning/performance curve.
        rows=[]
        try:
            rows=list(run.history(samples=240, pandas=False) or [])
        except (AttributeError, TypeError):
            rows=[]
        candidates={}
        for index,row in enumerate(rows):
            if not isinstance(row,dict):continue
            step=row.get('_step', index)
            if not numeric(step):step=index
            for name,value in row.items():
                if name.startswith('_') or not numeric(value):continue
                lowered=name.lower()
                if any(token in lowered for token in ('loss','accuracy','step_time','throughput','learning_rate','reward')):
                    candidates.setdefault(name,[]).append(dict(step=step,value=value))
        ranked=sorted(candidates.items(), key=lambda item: (
            0 if 'loss' in item[0].lower() else 1,
            -len(item[1]), item[0]))
        series=[dict(name=name, points=points[-120:]) for name,points in ranked[:2] if len(points)>=2]
        result=dict(name=run.name,state=run.state,metrics=metrics[:6],series=series,sampled_at=time.time())
        cache[key]=result
        return result
    except Exception:return {'metrics':[], 'series':[]},503

def start_worker():
    for job in sorted(web.list_jobs(),key=lambda j:(j["created_at"],j.get("related_runs",{}).get("kernel")==j["id"])):
        if job['status']=='running':
            job.update(status='interrupted',stage='Server restarted; inspect remote execution before restarting.');web.save_job(job)
        elif job['status']=='queued':web._queue.put(job['id'])
        elif job['status']=='exploring':start_discovery(job['id'])
    threading.Thread(target=web._worker,daemon=True).start()
