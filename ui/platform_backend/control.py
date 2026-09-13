"""Small control surface executed through the authenticated marimo connection."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from worker import ROOT, atomic, jobdir
from archive_bridge import snapshot


def handle(request):
    if request['action']=='wandb':
        import re,math,wandb
        path=request['path']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9]+',path):raise ValueError('Invalid W&B run')
        run=wandb.Api(timeout=15).run(path)
        metrics=[{'name':k,'value':v} for k,v in dict(run.summary).items() if not k.startswith('_') and isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)]
        metrics.sort(key=lambda m:(not any(word in m['name'].lower() for word in ('step_time','loss','speedup','accepted','generation')),m['name']))
        return dict(name=run.name,state=run.state,metrics=metrics[:6],sampled_at=time.time())
    if request['action']=='runtime':
        from worker import runtime_snapshot
        return runtime_snapshot(request['id'])
    root=jobdir(request['id']); action=request['action']
    if action=='create':
        if not (root/'state.json').exists():
            atomic(root/'request.json',request)
            atomic(root/'state.json',dict(id=request['id'],repo=request['repo'],status='exploring',activity=[],updated_at=time.time()))
            with (root/'worker.log').open('ab') as log:
                proc=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'explore',request['id']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (root/'worker.pid').write_text(str(proc.pid))
    elif action=='cancel':
        import signal
        state=json.loads((root/'state.json').read_text())
        if state['status'] not in ('complete','failed','cancelled'):
            for pid in (state.get('search_pid'), int((root/'worker.pid').read_text()) if (root/'worker.pid').exists() else None):
                if pid:
                    try: os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
            state.update(status='cancelled',message='Run cancelled.',updated_at=time.time());atomic(root/'state.json',state)
    elif action=='data':
        state=json.loads((root/'state.json').read_text())
        if state['status']!='awaiting_data':
            previous=json.loads((root/'data-request.json').read_text()) if (root/'data-request.json').exists() else {}
            if previous.get('url')!=request['url']:raise ValueError('This run is not waiting for data')
            request=dict(request,action='status')
            return handle(request)
        atomic(root/'data-request.json',{'url':request['url']})
        state.update(status='validating',message='Checking the dataset…');atomic(root/'state.json',state)
        with (root/'worker.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'verify',request['id']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'worker.pid').write_text(str(proc.pid))
    state=json.loads((root/'state.json').read_text())
    state.pop('search_pid',None);state.pop('worker_pid',None);state.pop('source_files',None)
    archive=root/'run/archive.sqlite'
    view=snapshot(archive,state['repo'],state.get('data'))
    if archive.exists():
        import sqlite3
        with sqlite3.connect(archive.resolve().as_uri()+'?mode=ro',uri=True) as db:
            ready=db.execute("SELECT 1 FROM events WHERE kind='input_ready' LIMIT 1").fetchone()
            if ready and state['status']=='preparing': state['status']='running'
    for trace in view['traces']:
        if trace.get('error'):trace['error']='This operation failed. Open the trace for details.'
    view.update(state)
    return view
