"""Build browser-safe recorded fixtures from a W&B CLI pull and immutable Astra archive.

No credentials, prompts, model reasoning, or source code are copied into the UI.
"""
import argparse, json, re, sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
def build(architecture, checkpoint, evaluation):
    calls=json.loads((architecture/'calls.json').read_text())['calls']
    authors={c['inputs']['job']['strategy']:c for c in calls if 'job' in c['inputs']}
    records=[]
    for i,(budget,loss) in enumerate([(60,6.90625),(120,6.5390625),(300,5.84765625)]):
        records.append(dict(id=i+1,generation=0,phase='baseline',strategy='Original model + AdamW',train_secs=budget,val_loss=loss,model_params=480881212,accepted=1))
    for c in calls:
        d=c['inputs'].get('gen_digest')
        if not d:continue
        for row in d['candidates']:
            a=authors[row['strategy']];out=a['output'];g=d['generation']
            records.append({**row,'id':len(records)+1,'generation':g,'phase':d['phase'],'train_secs':120 if g==3 else 60,'accepted':int(row['val_loss'] is not None and row['val_loss']<d['baseline_val_loss']),'correct_ok':int(row['val_loss'] is not None),'parent_id':a['inputs']['job'].get('parent'),'model_name':out.get('model_name'),'trace_url':f"https://wandb.ai/rianbutala-ucla/kernel-evolution/r/call/{a['id']}"})
    for source,loss in [(27,5.578125),(24,5.42578125)]:
        parent=next(r for r in records if r['id']==source)
        records.append({**parent,'id':len(records)+1,'generation':4,'phase':'finals','train_secs':300,'val_loss':loss,'parent_id':source})
    traces=[]
    for c in calls:
        inp=c['inputs'];description=(inp.get('job') or {}).get('strategy') or ('Generation '+str(inp['gen_digest']['generation'])+' outcomes reviewed' if 'gen_digest' in inp else 'Propose the next generation')
        traces.append(dict(id=c['id'],name='author_recipe' if 'job' in inp else 'curate' if 'gen_digest' in inp else 'planner',started_at=datetime.fromisoformat(c['started_at'].replace('Z','+00:00')).timestamp(),ended_at=datetime.fromisoformat(c['ended_at'].replace('Z','+00:00')).timestamp(),error=c.get('exception'),description=description,url=f"https://wandb.ai/rianbutala-ucla/kernel-evolution/r/call/{c['id']}"))
    report='https://wandb.ai/rianbutala-ucla/kernel-evolution/reports/Kernel-Evolution-Recipe-75890bd0-Budget-Matched-Results--VmlldzoxNzkyNTEzOQ'
    arch=dict(id='75890bd0',mode='recipe',status='complete',model='modern-lm · 480.9M',agent_model='moonshotai/Kimi-K2.7-Code',subagent_model='moonshotai/Kimi-K2.7-Code',repo=None,data='https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu',architecture={'candidates':records},candidates=[],traces=traces,message='7.21% lower validation loss · equal 300s training budget',terminal=re.sub(r'\x1b\[[0-9;]*m','',(architecture/'output.log').read_text()),integrations={'wandb_url':'https://wandb.ai/rianbutala-ucla/kernel-evolution/runs/pwn5or77','wandb_embed_url':report+'?view=embedded','weave_url':'https://wandb.ai/rianbutala-ucla/kernel-evolution/weave/traces'},provenance={'run':'pwn5or77','report':report,'source':'W&B CLI output.log and Weave generation digests','stop_reason':'recipe_complete'})
    con=sqlite3.connect(f'file:{checkpoint}/archive.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
    fields=['id','generation','parent_id','parents_json','strategy','model_name','gate_reached','compile_ok','correct_ok','accepted','failure_note','step_time_ms','incumbent_step_time_ms','weave_trace_url']
    rows=[{k:r[k] for k in fields} for r in con.execute('select * from candidates order by generation,created_at,id')]
    for r in rows:
        r['lineage_id']='Whole training step'
        m=re.match(r'cand_(\d+)_(\d+)_',r['id'])
        if m:r['display_id']=f'{int(m[1])}.{int(m[2])+1}'
    result=json.loads((evaluation/'result.json').read_text());summary=json.loads((evaluation/'summary.json').read_text());summary.pop('durable_snapshot',None)
    kernel_traces=[dict(id=str(i),name='gate.'+s['name'],started_at=s['started_at'],ended_at=s['ended_at'],error=None,description=json.dumps(s['output']),url=summary['final_trace_url']) for i,s in enumerate(result['gate_spans'])]
    kernel=dict(id='astra-final',mode='kernel',status='complete',model='adapters.language · 21.6M',repo='https://github.com/flappybird1084/top-k',data='https://huggingface.co/datasets/roneneldan/TinyStories',agent_model='gpt-6-astra',subagent_model='gpt-6-astra',candidates=rows,architecture={'candidates':[]},generation=5,baseline_ms=result['incumbent_step_time_ms'],traces=kernel_traces,verification={'blocks':result['details']['gate4']['blocks'],'summary':summary,'result':{k:result[k] for k in ['step_time_ms','incumbent_step_time_ms','correct_ok']}},message='3.30% lower training-step time · all correctness checks passed',integrations={'wandb_url':'https://wandb.ai/stephenslee0127-acme/kernel-evolution/runs/7j5vlp9k','weave_url':summary['final_trace_url']},provenance={'commit':'660b6d9','source_sha256':summary['source_sha256'],'stop_reason':'max_generations'},terminal='\n'.join(f"Gen {r['generation']} | {r['id']} | {'accepted' if r['accepted'] else 'not accepted'} | {r['strategy']}\n{r['failure_note'] or ''}" for r in rows))
    integration_file=ROOT/'ui/assets/recorded/integrations.json'
    configured=json.loads(integration_file.read_text()) if integration_file.exists() else {}
    for side,s in [('architecture',arch),('kernel',kernel)]:
        s['integrations'].update(configured.get(side,{}))
        s['integrations'].update(marimo_url=f'/assets/recorded/{side}-notebook.html',marimo_embed_url=f'/assets/recorded/{side}-notebook.html')
        (ROOT/f'ui/assets/recorded/{side}.json').write_text(json.dumps(s,indent=2,allow_nan=False))
    (ROOT/'ui/assets/recorded-data.js').write_text('window.RECORDED_RUNS='+json.dumps({'architecture':arch,'kernel':kernel},allow_nan=False)+';\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();build(a.architecture,a.checkpoint,a.evaluation)
