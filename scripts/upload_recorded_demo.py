"""Upload a separate evidence run; never resume or alter the original runs."""
from pathlib import Path
import json, os, subprocess
from dotenv import dotenv_values
import wandb

root=Path(__file__).resolve().parents[1]
credentials=dotenv_values(Path.home()/'Documents/ChatGPT/w')
os.environ['WANDB_API_KEY']=credentials['WANDB_API_KEY']
entity='rianbutala-ucla';project='kernel-evolution';run_id='topk-evidence-660b6d9'
arch=json.loads((root/'ui/assets/recorded/architecture.json').read_text())
kernel=json.loads((root/'ui/assets/recorded/kernel.json').read_text())
folder=root/'jobs/video-source/upload';folder.mkdir(parents=True,exist_ok=True)
run=wandb.init(entity=entity,project=project,id=run_id,name='Top-Kernel · recorded verification',mode='offline',dir=str(folder),config={'recorded_evidence':True,'kernel_commit':'660b6d9','architecture_source_run':'pwn5or77','architecture_job':'75890bd0','new_training':False})
for budget in [60,120,300]:
 rows=[r for r in arch['architecture']['candidates'] if r['train_secs']==budget]
 baseline=next(r['val_loss'] for r in rows if r['phase']=='baseline')
 for i,r in enumerate(r for r in rows if r['phase']!='baseline' and r['val_loss'] is not None):
  run.log({f'architecture/{budget}s/evaluation':i+1,f'architecture/{budget}s/baseline':baseline,f'architecture/{budget}s/candidate':r['val_loss']})
for i,b in enumerate(kernel['verification']['blocks']):
 run.log({'kernel/pair':i+1,'kernel/baseline_ms':b['baseline'],'kernel/astra_ms':b['candidate'],'kernel/paired_reduction_pct':100*b['gain']})
run.summary.update({'kernel/median_paired_reduction_pct':100*kernel['verification']['summary']['paired_median_time_reduction'],'kernel/min_paired_reduction_pct':100*kernel['verification']['summary']['paired_minimum_time_reduction'],'kernel/correctness_checks':8,'kernel/search_generations':5,'architecture/final_baseline_loss':5.84765625,'architecture/final_best_loss':5.42578125})
run_dir=Path(run.dir).parent;run.finish()
subprocess.run([str(root/'.venv/bin/wandb'),'sync',str(run_dir)],check=True,env=os.environ)
print(f'https://wandb.ai/{entity}/{project}/runs/{run_id}')
