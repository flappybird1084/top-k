"""Authenticated single-workspace WSGI API. Use gunicorn for hosted deployments."""
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import unquote, urlsplit
from worker import github_url, dataset_url

UI=Path(__file__).resolve().parents[1]
WORKSPACE=UI.parent
STORE=Path(os.environ.get('TOPK_STATE_DIR',str(Path.home()/'.local/state/topkernel-platform')))
STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
os.chmod(STORE,0o700)
ENGINE=Path(os.environ.get('TOPK_LOCAL_ENGINE',str(WORKSPACE if (WORKSPACE/'search.py').exists() else WORKSPACE/'top-k')))
TOKEN=os.environ.get('TOPK_ACCESS_TOKEN','')
REMOTE_ROOT='/marimo/top-k-platform'
SIGN_FILE=STORE/'session-key'
if not SIGN_FILE.exists():
    fd=os.open(SIGN_FILE,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(secrets.token_hex(32))
SIGN=SIGN_FILE.read_text().encode()
remote_lock=threading.Lock(); state_lock=threading.RLock()
active=set()
runtime_cache={}
wandb_cache={}


def write(path,value):
    with state_lock:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)


def remote(request):
    code="""import sys,json
sys.path.insert(0,'/marimo/top-k-platform')
import importlib,worker,archive_bridge,control
for module in (worker,archive_bridge,control):importlib.reload(module)
print('TOPK_RESPONSE:'+json.dumps(control.handle(REQUEST)))
""".replace('REQUEST',repr(request))
    with remote_lock:
        proc=subprocess.run([sys.executable,str(ENGINE/'scripts/molab.py')],input=code,text=True,capture_output=True,timeout=90)
    if proc.returncode:raise RuntimeError('GPU runtime could not be reached')
    line=next((x for x in proc.stdout.splitlines() if x.startswith('TOPK_RESPONSE:')),None)
    if not line:raise RuntimeError('GPU runtime returned no status')
    return json.loads(line.split(':',1)[1])


def watch(run_id,initial=None):
    with state_lock:
        if run_id in active:return
        active.add(run_id)
    def work():
        path=STORE/(run_id+'.json')
        try:
            saved=json.loads(path.read_text())
            action=initial or ({'action':'status','id':run_id} if saved.get('remote_created') else {'action':'create','id':run_id,'repo':saved['repo'],'spend_cap_usd':saved.get('spend_cap_usd',3)})
            while True:
                try:
                    result=remote(action)
                    current=json.loads(path.read_text());result['owner']=current['owner'];result['created_at']=current.get('created_at',time.time());result['spend_cap_usd']=current.get('spend_cap_usd',0)
                    result['remote_created']=True
                    result['integration_overrides']=current.get('integration_overrides',{})
                    write(path,result)
                    action={'action':'status','id':run_id}
                    if result['status'] in ('awaiting_data','complete','failed','cancelled'):break
                except Exception:
                    current=json.loads(path.read_text());current['connection_error']='GPU runtime is unavailable. Reconnecting…';write(path,current)
                time.sleep(5)
        finally:
            with state_lock:active.discard(run_id)
    threading.Thread(target=work,daemon=True).start()


def cookie_value():
    stamp=str(int(time.time()))
    return stamp+'.'+hmac.new(SIGN,stamp.encode(),hashlib.sha256).hexdigest()


def authorized(environ):
    if not TOKEN:
        return environ.get('REMOTE_ADDR') in ('127.0.0.1','::1')
    cookie=SimpleCookie()
    try:
        cookie.load(environ.get('HTTP_COOKIE',''));value=cookie['topk_session'].value
        stamp,signature=value.split('.')
        return 0<=time.time()-int(stamp)<86400 and hmac.compare_digest(signature,hmac.new(SIGN,stamp.encode(),hashlib.sha256).hexdigest())
    except (KeyError,ValueError):return False


def origin_ok(environ):
    origin=environ.get('HTTP_ORIGIN')
    if not origin:return environ.get('HTTP_SEC_FETCH_SITE') not in ('cross-site',)
    expected=os.environ.get('TOPK_PUBLIC_ORIGIN') or environ.get('wsgi.url_scheme','http')+'://'+environ.get('HTTP_HOST','')
    return origin==expected


def body(environ):
    length=int(environ.get('CONTENT_LENGTH') or 0)
    if length>8192:raise ValueError('Request is too large')
    return json.loads(environ['wsgi.input'].read(length))


def application(environ,start_response):
    def response(value,status='200 OK',headers=()):
        raw=json.dumps(value).encode();start_response(status,[('Content-Type','application/json'),('Cache-Control','no-store'),('Content-Length',str(len(raw))),*headers]);return [raw]
    method=environ['REQUEST_METHOD'];path=environ.get('PATH_INFO','/')
    try:
        if method=='POST' and not origin_ok(environ):return response({'error':'Request origin rejected'},'403 Forbidden')
        if path=='/api/session' and method=='POST':
            value=body(environ).get('token','')
            if not TOKEN or not hmac.compare_digest(str(value),TOKEN):return response({'error':'Invalid access token'},'401 Unauthorized')
            secure='; Secure' if os.environ.get('TOPK_PUBLIC_ORIGIN','').startswith('https://') else ''
            return response({'ok':True},headers=[('Set-Cookie','topk_session='+cookie_value()+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400'+secure)])
        if path.startswith('/api/') and not authorized(environ):return response({'error':'Sign in to this workspace'},'401 Unauthorized')
        if path=='/api/runs/current' and method=='GET':
            pointer=STORE/'current.json'
            if not pointer.exists():return response({'status':'waiting','candidates':[],'traces':[]})
            path='/api/runs/'+json.loads(pointer.read_text())['id']
        if path=='/api/runs' and method=='POST':
            request=body(environ);repo=github_url(request.get('repo',''))
            cap=float(os.environ.get('TOPK_RUN_SPEND_CAP','3'))
            if not 0<cap<=100:raise ValueError('Invalid configured spend cap')
            key=environ.get('HTTP_IDEMPOTENCY_KEY','')
            if not re.fullmatch(r'[a-f0-9-]{16,64}',key):raise ValueError('Missing request identifier')
            run_id=hashlib.sha256(('workspace:'+key).encode()).hexdigest()[:32];file=STORE/(run_id+'.json')
            with state_lock:
                if not file.exists():
                    jobs=[json.loads(p.read_text()) for p in STORE.glob('*.json') if p.name!='current.json']
                    if sum(j.get('status') not in ('complete','failed','cancelled') for j in jobs)>=3:raise ValueError('This workspace already has three active runs')
                    reserved=sum(j.get('spend_cap_usd',0) for j in jobs if time.time()-j.get('created_at',0)<86400)
                    if reserved+cap>float(os.environ.get('TOPK_DAILY_SPEND_CAP','30')):raise ValueError('Workspace daily run budget reached')
                    write(file,dict(id=run_id,repo=repo,status='exploring',activity=[],owner='workspace',created_at=time.time(),spend_cap_usd=cap))
                elif json.loads(file.read_text())['repo']!=repo:raise ValueError('Request identifier already used')
            write(STORE/'current.json',{'id':run_id})
            watch(run_id,dict(action='create',id=run_id,repo=repo,spend_cap_usd=cap))
            return response({'id':run_id},'202 Accepted')
        match=re.fullmatch(r'/api/runs/([a-f0-9]{32})(/data|/cancel|/integrations|/runtime|/wandb)?',path)
        if match:
            run_id,action=match.groups();file=STORE/(run_id+'.json')
            if not file.exists():return response({'error':'Run not found'},'404 Not Found')
            state=json.loads(file.read_text())
            if state.get('owner')!='workspace':return response({'error':'Run not found'},'404 Not Found')
            if method=='GET' and action=='/wandb':
                url=state.get('integrations',{}).get('wandb_url','')
                parsed=urlsplit(url);parts=parsed.path.strip('/').split('/')
                if parsed.hostname!='wandb.ai' or len(parts)!=4 or parts[2]!='runs':return response({'metrics':[]})
                cached=wandb_cache.get(url)
                if not cached or time.time()-cached['sampled_at']>30:
                    cached=remote({'action':'wandb','id':run_id,'path':'/'.join([parts[0],parts[1],parts[3]])});wandb_cache[url]=cached
                return response(cached)
            if method=='GET' and action=='/runtime':
                cached=runtime_cache.get(run_id)
                if not cached or time.time()-cached['sampled_at']>5:
                    cached=remote({'action':'runtime','id':run_id});runtime_cache[run_id]=cached
                return response(cached)
            if method=='POST' and action=='/integrations':
                value=body(environ).get('wandb_report_url','')
                parsed=urlsplit(value)
                if parsed.scheme!='https' or parsed.hostname not in ('wandb.ai','www.wandb.ai') or '/reports/' not in parsed.path or parsed.username or parsed.password:raise ValueError('Use an approved W&B report HTTPS URL')
                with state_lock:
                    state=json.loads(file.read_text());state['integration_overrides']={'wandb_embed_url':value};write(file,state)
                return response({'ok':True})
            if method=='POST' and action=='/cancel':
                result=remote({'action':'cancel','id':run_id});result['owner']='workspace';write(file,result)
                return response({'id':run_id})
            if method=='POST' and action=='/data':
                url=dataset_url(body(environ).get('url',''))
                with state_lock:
                    state=json.loads(file.read_text())
                    if state['status']!='awaiting_data':return response({'error':'Run is not waiting for data'},'409 Conflict')
                    state.update(status='validating',message='Checking dataset…');write(file,state)
                watch(run_id,dict(action='data',id=run_id,url=url))
                return response({'id':run_id},'202 Accepted')
            if method=='GET' and not action:
                if state['status'] not in ('complete','failed','awaiting_data','cancelled'):watch(run_id)
                state.pop('owner',None)
                state.setdefault('integrations',{}).update(state.pop('integration_overrides',{}))
                state['integrations'].update(marimo_url='/notebook/?run='+run_id,marimo_embed_url='/notebook/?run='+run_id)
                return response(state)
        if path.startswith('/api/'):return response({'error':'Not found'},'404 Not Found')
        if method!='GET':return response({'error':'Method not allowed'},'405 Method Not Allowed')
        # Explicit static allowlist: backend source, state, and credentials never served.
        name=unquote(path).lstrip('/') or 'index.html'
        allowed={'results.html','results.js','index.html','front.js','front.css','workspace.html','run.js','run.css','login.html','login.js'}
        target=UI/name
        if name not in allowed and not (name.startswith('assets/') and target.suffix in ('.png','.jpg','.svg')):return response({'error':'Not found'},'404 Not Found')
        if not target.resolve().is_relative_to(UI.resolve()) or not target.is_file():return response({'error':'Not found'},'404 Not Found')
        raw=target.read_bytes();start_response('200 OK',[('Content-Type',mimetypes.guess_type(name)[0] or 'application/octet-stream'),('Content-Length',str(len(raw))),('Cache-Control','no-cache'),('X-Content-Type-Options','nosniff'),('Referrer-Policy','strict-origin-when-cross-origin')]);return [raw]
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:return response({'error':str(exc)},'400 Bad Request')
    except Exception:return response({'error':'Unable to process this request'},'503 Service Unavailable')


if __name__=='__main__':
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import make_server,WSGIServer
    class Threaded(ThreadingMixIn,WSGIServer):daemon_threads=True
    port=int(os.environ.get('PORT','8766'))
    print(f'http://127.0.0.1:{port}/',flush=True)
    make_server('127.0.0.1',port,application,server_class=Threaded).serve_forever()
