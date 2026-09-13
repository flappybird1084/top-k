"""Mirror only this run's measured archive rows from the trusted server."""
import json, os, sys, time
from pathlib import Path
import ui_server

def observe(jid):
    import wandb
    root=ui_server.job_path(jid)
    job=ui_server.read_json(root/'job.json',{})
    run=wandb.init(project='top-k-judges', entity=os.environ.get('WANDB_ENTITY') or None,
                   id=jid, name=job['mode']+'-'+jid[:8],
                   group=job.get('related_runs',{}).get('architecture',jid),
                   config={'repo':job.get('repo'),'data':job.get('data'),'mode':job['mode']})
    integrations={'wandb_url':run.url}
    record=lambda row:row
    try:
        import weave
        project=run.entity+'/top-k-judges'
        weave.init(project)
        @weave.op()
        def measured_evaluation(run_id, repository, measurement):
            return measurement
        record=lambda row:measured_evaluation(jid,job.get('repo'),row)
        integrations['weave_url']='https://wandb.ai/'+project+'/weave'
    except Exception:
        pass
    p=root/'observability.json';p.write_text(json.dumps(integrations));p.chmod(0o600)
    seen=set()
    while True:
        state=ui_server.snapshot(jid)
        rows=state.get('architecture',{}).get('candidates',[]) if job['mode']=='recipe' else state.get('candidates',[])
        for row in rows:
            key=json.dumps(row,sort_keys=True,default=str)
            if key in seen:continue
            seen.add(key)
            metrics={k:v for k,v in row.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            if metrics:run.log(metrics)
            record(row)
        if state['status'] in ('complete','failed','cancelled'):
            run.summary['job_status']=state['status'];run.finish();return
        time.sleep(10)
if __name__=='__main__':observe(sys.argv[1])
