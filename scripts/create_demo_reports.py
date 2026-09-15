"""Prepare private report drafts for the recorded demo. Sharing is a separate action."""
from pathlib import Path
import os,json
from dotenv import dotenv_values
os.environ['WANDB_API_KEY']=dotenv_values(Path.home()/'Documents/ChatGPT/w')['WANDB_API_KEY']
import wandb_workspaces.reports.v2 as wr
root=Path(__file__).resolve().parents[1]
urls={}
for side in ['architecture','kernel']:
    panels=[wr.LinePlot(title='Compiled PyTorch vs Astra',x='kernel/pair',y=['kernel/baseline_ms','kernel/astra_ms'],title_x='Paired block',title_y='Full training-step time (ms)',smoothing_type='none',line_colors={'kernel/baseline_ms':'#89927f','kernel/astra_ms':'#417d64'})] if side=='kernel' else [wr.LinePlot(title=f'Validation loss · equal {b}s budget',x=f'architecture/{b}s/evaluation',y=[f'architecture/{b}s/baseline',f'architecture/{b}s/candidate'],title_x='Candidate evaluation',title_y='Validation loss',smoothing_type='none') for b in [300,60,120]]
    report=wr.Report(entity='rianbutala-ucla',project='kernel-evolution',title='Top-Kernel · '+side+' recorded results',description='Historical evidence for a video demonstration; no new training.',blocks=[wr.PanelGrid(runsets=[wr.Runset(entity='rianbutala-ucla',project='kernel-evolution',filters='Name == "topk-evidence-660b6d9"')],panels=panels,hide_run_sets=True)])
    report.save(draft=True);urls[side]=report.url
(root/'jobs/video-source/report-drafts.json').write_text(json.dumps(urls,indent=2))
print(json.dumps(urls))
