'use strict';
const q=s=>document.querySelector(s);
const params=new URLSearchParams(location.search);
// Related runs must come from the selected run's API response.
params.delete('architecture_run');params.delete('kernel_run');
if(document.querySelector('#search-page-link'))document.querySelector('#search-page-link').href='assets/search.html?'+params;
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function safeURL(value,brand=false){try{const url=new URL(value);if(!['https:','http:'].includes(url.protocol)||url.username||url.password)return null;if(brand&&(url.protocol!=='https:'||!['wandb.ai','www.wandb.ai'].includes(url.hostname)))return null;return url.href}catch{return null}}
function identity(repo,data){if(repo){const url=safeURL(repo);if(url){const parsed=new URL(url);q('#repo-row').hidden=false;q('#repository').textContent=parsed.href;q('#repository').href=url}}if(data&&safeURL(data)){const url=new URL(data);q('#dataset').hidden=false;q('#dataset-link').textContent=url.href;q('#dataset-link').href=url.href}}
identity(params.get('repo')||params.get('requested_repo'),params.get('data')||params.get('requested_data'));
function link(selector,value){const node=q(selector),url=safeURL(value,true);node.hidden=!url;if(url)node.href=url;else node.removeAttribute('href')}
function embed(name,url){const frame=q(`#${name}-frame`),valid=safeURL(url,true);frame.hidden=!valid;q(`#${name}-empty`).hidden=!!valid;if(valid&&frame.getAttribute('src')!==valid)frame.src=valid;if(!valid)frame.removeAttribute('src')}
function chart(rows){const vals=rows.map(r=>r.step_time_ms),low=Math.min(...vals)*.95,high=Math.max(...vals)*1.05,range=high-low||1;const points=vals.map((v,i)=>[45+i*920/Math.max(1,vals.length-1),15+(high-v)/range*170]);const path=points.map((p,i)=>`${i?'L':'M'}${p.join(',')}`).join(' ');q('#performance-chart').innerHTML=[0,.5,1].map(t=>`<line x1="45" x2="980" y1="${15+t*170}" y2="${15+t*170}" stroke="#e5e9df"/><text x="0" y="${19+t*170}" fill="#89927f" font-size="10">${(high-t*range).toFixed(1)}</text>`).join('')+`<path d="${path}" fill="none" stroke="#417d64" stroke-width="2"/>`+points.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${rows[i].generation===0?'#89927f':rows[i].accepted?'#417d64':'#bf795a'}"><title>${escapeHTML(rows[i].id)}${rows[i].generation===0?' · baseline':rows[i].accepted?' · verified':' · rejected'}: ${vals[i].toFixed(2)} ms</title></circle>`).join('')}
function replaySnapshot(snapshot){return snapshot}
function renderKernel(snapshot){
 identity(snapshot.repo,snapshot.data);
 const active=['complete','failed','cancelled'].includes(snapshot.status)?[]:(snapshot.active_evaluations||[]);
 q('#active-kernels').hidden=!active.length;q('#evaluation-count').textContent=active.length+' in progress';
 q('#evaluation-list').innerHTML=active.slice(0,6).map(e=>`<div class="evaluation"><span class="live-dot"></span><div><strong>${escapeHTML(e.kernel)}</strong><p>${escapeHTML(e.stage)} · ${Math.max(0,Math.floor(Date.now()/1000-e.started_at))}s</p><small>${escapeHTML((e.strategy||'').split('. ')[0])}</small></div></div>`).join('');

 const rows=Array.isArray(snapshot.candidates)?snapshot.candidates:[],traces=Array.isArray(snapshot.traces)?snapshot.traces:[];
 q('#connection').textContent=({awaiting_data:'Waiting for data',waiting:'Waiting for a run',running:'Run in progress',complete:'Run complete',failed:'Run stopped',cancelled:'Run cancelled',preparing:'Preparing the run',queued:'Waiting for GPU',exploring:'Exploring repository',validating:'Verifying data'})[snapshot.status]||'Waiting for a run';q('#connection').className=snapshot.status==='running'?'connected':'';if(snapshot.preview_notice)q('#connection').textContent=snapshot.status==='running'?'Recorded replay':'Replay complete';
 const accepted=rows.filter(r=>r.accepted&&r.generation!==0&&Number.isFinite(r.step_time_ms)&&r.step_time_ms>0);
 const profile=snapshot.profiling;
 q('#profiling-data').hidden=!profile&&!Number.isFinite(snapshot.baseline_ms)&&!accepted.length;
 if(profile){q('#profile-compiled').textContent=profile.compiled_step_ms.toFixed(2)+' ms';q('#profile-eager').textContent=Number.isFinite(profile.eager_step_ms)?profile.eager_step_ms.toFixed(2)+' ms':'—';q('#profile-eligible').textContent=profile.eligible_operations+' / '+profile.operations_inspected;q('#profile-note').textContent=snapshot.stop_reason==='no_targets'?'Profiling complete. No operations met the kernel replacement criteria.':'';}
 const measured=accepted.length||Number.isFinite(snapshot.baseline_ms);q('#performance-data').hidden=!measured;q('.performance-metrics').hidden=!measured;q('#empty-performance').hidden=!!measured||!!profile;
 if(accepted.length){const best=Math.min(...accepted.map(r=>r.step_time_ms));q('#best').textContent=best.toFixed(2)+' ms';q('#accepted').textContent=String(accepted.length);q('#speedup').textContent=Number.isFinite(snapshot.baseline_ms)&&snapshot.baseline_ms>0?(snapshot.baseline_ms/best).toFixed(2)+'×':'—';chart(accepted)}else if(Number.isFinite(snapshot.baseline_ms)){q('#best').textContent=snapshot.baseline_ms.toFixed(2)+' ms';q('#speedup').textContent='1.00×';q('#accepted').textContent='0';chart([{id:'Measured baseline',step_time_ms:snapshot.baseline_ms}])}
 const timed=rows.filter(r=>Number.isFinite(r.step_time_ms)&&r.step_time_ms>0);if(timed.length)chart(timed);
 q('#empty-performance p').textContent=snapshot.status==='complete'&&!timed.length?'No timed kernel evaluations recorded':snapshot.stop_reason==='no_targets'?'No eligible kernels found':'Waiting for the first measurement';
 q('#empty-performance>span').textContent=snapshot.status==='complete'&&!timed.length?'This run finished before a kernel reached full training-step measurement.':snapshot.stop_reason==='no_targets'?'Profiling completed. No operations met the criteria for kernel replacement.':'Training-step performance will appear as kernels are verified.';
 q('#generation').textContent=Number.isInteger(snapshot.generation)?'Generation '+snapshot.generation:'';
 q('#kernels').hidden=!rows.length;q('#candidate-count').textContent=rows.length+' evaluated';
 q('#candidates').innerHTML=(demo?rows:rows.slice(-12)).slice().reverse().map(r=>`<tr><td>${escapeHTML(r.lineage_id)}<small>${escapeHTML(r.id)}</small></td><td><details><summary>${escapeHTML((r.strategy||"Candidate").split(". ")[0])}</summary><p>${escapeHTML(r.strategy)}</p></details></td><td class="${r.accepted?'accepted':'rejected'}">${r.generation===0?'Baseline':r.accepted?'✓ Verified':'Passed '+escapeHTML(r.gate_reached)+' / 4 · rejected'}</td><td>${Number.isFinite(r.step_time_ms)?r.step_time_ms.toFixed(2)+' ms':'—'}</td></tr>`).join('');
 const integration=snapshot.integrations||{};
 if(!integration.wandb_url){q('#wandb-empty').classList.remove('has-metrics');q('#wandb-empty').textContent='No metrics recorded for this run.';}
 const notebook=safeURL(/^\/(notebook|assets\/recorded)\//.test(integration.marimo_url||'')?new URL(integration.marimo_url,location.href).href:integration.marimo_url);
 q('#marimo-panel').hidden=!notebook;
 const notebookMeasured=rows.filter(r=>Number.isFinite(r.step_time_ms)&&r.step_time_ms>0);
 const recipeRows=snapshot.architecture?.candidates||[];
 q('#notebook-stats').innerHTML=`<div><span>Candidates</span><strong>${rows.length}</strong></div><div><span>Timed evaluations</span><strong>${notebookMeasured.length}</strong></div><div><span>Compiled profile</span><strong>${Number.isFinite(snapshot.profiling?.compiled_step_ms)?snapshot.profiling.compiled_step_ms.toFixed(2)+' ms':'—'}</strong></div>`;

 if(snapshot.mode==='recipe')q('#notebook-stats').innerHTML=`<div><span>Recipes</span><strong>${recipeRows.filter(r=>r.phase!=='baseline').length}</strong></div><div><span>Held-out evaluations</span><strong>${recipeRows.filter(r=>r.phase!=='baseline'&&Number.isFinite(r.val_loss)).length}</strong></div><div><span>Improved</span><strong>${recipeRows.filter(r=>r.phase!=='baseline'&&r.accepted).length}</strong></div>`;
 if(notebook)q('#marimo-link').href=notebook;
 const notebookEmbed=safeURL(/^\/(notebook|assets\/recorded)\//.test(integration.marimo_embed_url||'')?new URL(integration.marimo_embed_url,location.href).href:integration.marimo_embed_url);
 q('#marimo-frame').hidden=!notebookEmbed;q('#marimo-note').hidden=!!notebookEmbed;
 if(notebookEmbed){if(q('#marimo-frame').getAttribute('src')!==notebookEmbed)q('#marimo-frame').src=notebookEmbed}else q('#marimo-frame').removeAttribute('src');
 const wandbRun=safeURL(integration.wandb_url,true);
 const projectURL=wandbRun?new URL(wandbRun).origin+'/'+new URL(wandbRun).pathname.split('/').filter(Boolean).slice(0,2).join('/'):null;
 link('#wandb-project',projectURL);
 link('#wandb-link',integration.wandb_url);link('#weave-link',integration.weave_url);embed('wandb',integration.wandb_embed_url);embed('weave',integration.weave_embed_url);
 if(integration.wandb_url&&!integration.wandb_embed_url&&!q('#wandb-empty').classList.contains('has-metrics'))q('#wandb-empty').textContent='Run connected. Open W&B to inspect the recorded metrics.';
 if(!integration.weave_embed_url){q('#weave-empty').hidden=!!traces.length;const open=new Set([...document.querySelectorAll('.trace[open]')].map(n=>n.dataset.trace));q('#traces').innerHTML=traces.slice(-6).reverse().map(t=>{const url=safeURL(t.url,true);return `<details class="trace" data-trace="${escapeHTML(t.id)}" ${open.has(String(t.id))?'open':''}><summary><span>${escapeHTML(t.name)}</span><span>${t.error?'Failed':t.ended_at==null?'In progress':Number.isFinite(t.started_at)?Math.max(0,t.ended_at-t.started_at).toFixed(1)+'s':'Complete'}</span></summary><p>${escapeHTML(t.error||t.description||'Trace recorded for this run.')}</p>${url?`<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">Open trace ↗</a>`:''}</details>`}).join('')}else q('#traces').replaceChildren();
}
let latestSnapshot=null;
const demo=false; // Stats never use replay measurements.
let runId=params.get('run'),retryTimer;
const emptyRun=()=>({status:'waiting',repo:params.get('requested_repo'),data:params.get('requested_data'),candidates:[],traces:[],integrations:{},message:'No measurements recorded for this run.'});
q('#final-summary-link').hidden=false;q('#final-summary-link').href='assets/final.html?'+params;
function render(snapshot){
 latestSnapshot=snapshot;
 if(!snapshot.candidates?.some(r=>Number.isFinite(r.step_time_ms))&&!Number.isFinite(snapshot.baseline_ms)){q('#performance-chart').replaceChildren();for(const id of ['best','speedup','accepted'])q('#'+id).textContent='—';}
 q('#recorded-logs').hidden=true;q('#recorded-evidence').hidden=true;
 q('#aria-panel').hidden=!snapshot.integrations?.wandb_url;
 q('#copy-aria-prompt').onclick=async()=>{await navigator.clipboard.writeText('Analyze this run: '+snapshot.integrations.wandb_url+'. Use only its recorded measurements and logs.');q('#aria-copy-status').textContent='Copied';};
 if(snapshot.integrations?.wandb_url)link('#aria-link',snapshot.integrations.wandb_url);
 renderKernel(snapshot);architectureRender(snapshot);
 q('#architecture-view').hidden=evolution.view!=='architecture';q('#kernel-view').hidden=evolution.view!=='kernel';
 if(evolution.view==='architecture')q('#architecture-view').append(q('#active-kernels'));
 else q('#kernel-view').prepend(q('#active-kernels'));
 for(const view of ['architecture','kernel']){const b=q('#tab-'+view);b.setAttribute('aria-selected',String(evolution.view===view));b.tabIndex=evolution.view===view?0:-1}
 q('#view-status').textContent=snapshot.message||'';
 q('#runtime-panel').hidden=demo||!runId;
 q('#demo-label').hidden=!demo;q('#replay-demo').hidden=!demo;

}
async function refresh(){
 clearTimeout(retryTimer);

 const view=evolution.view,id=params.get(view+'_run')||params.get('run');runId=id;
 if(!id){render(emptyRun());return}
 try{
  const response=await fetch('/api/runs/'+encodeURIComponent(id),{signal:AbortSignal.timeout(8000),cache:'no-store'});
  if(!response.ok)throw Error('Unavailable');let data=await response.json();if(data.id!==id)throw Error('Run mismatch');
  if(view!==evolution.view)return;
  if(data.related_runs){
   for(const kind of ['architecture','kernel'])if(data.related_runs[kind])params.set(kind+'_run',data.related_runs[kind]);
   const target=params.get(view+'_run');
   if(target&&target!==id){refresh();return;}
  }
  if(!evolution.initialized&&!params.has('view'))evolution.view=data.mode==='recipe'?'architecture':'kernel';
  evolution.initialized=true;
  const mode=data.mode==='recipe'?'architecture':'kernel';
  if(mode!==evolution.view){data={status:'waiting',repo:data.repo,data:data.data,candidates:[],traces:[],message:'No '+evolution.view+' run connected to this view.'};runId=null;}
  render(data);q('#results-link').hidden=true;q('#run-notice').hidden=true;
 }catch{render(emptyRun());q('#view-status').textContent='Run unavailable · no measurements to display';}
 retryTimer=setTimeout(refresh,3000);
}
setupEvolution(()=>{if((latestSnapshot?.mode==='recipe'?'architecture':'kernel')!==evolution.view)render(emptyRun());q('#gpu-readings').replaceChildren();q('#terminal-output').textContent='';q('#wandb-empty').classList.remove('has-metrics');q('#wandb-empty').textContent='Waiting for metrics from this view.';refresh();clearTimeout(wandbTimer);setTimeout(refreshWandb,250)});
q('#replay-demo').addEventListener('click',()=>{location.href='assets/search.html?demo=1'});
render(emptyRun());
refresh();
const demoTimer=null;
window.addEventListener('pagehide',()=>{clearTimeout(retryTimer);clearInterval(demoTimer)});

let runtimeTimer;
async function refreshRuntime(){
 if(demo||!runId){runtimeTimer=setTimeout(refreshRuntime,5000);return}
 const requestedRun=runId;
 if(document.hidden){runtimeTimer=setTimeout(refreshRuntime,5000);return}
 try{
  const response=await fetch('/api/runs/'+encodeURIComponent(runId)+'/runtime',{cache:'no-store',signal:AbortSignal.timeout(20000)});
  if(!response.ok)throw Error('Unavailable');
  const data=await response.json();
  if(requestedRun!==runId)return;
  q('#gpu-freshness').textContent='Live node usage · '+new Date(data.sampled_at*1000).toLocaleTimeString();
  q('#gpu-readings').innerHTML=(data.gpus||[]).map(g=>`<div><span>${escapeHTML(g.name)}</span><strong>${Number.isFinite(g.utilization_pct)?g.utilization_pct.toFixed(0)+'%':'—'}</strong></div><div><span>GPU memory</span><strong>${Number.isFinite(g.memory_used_mb)&&Number.isFinite(g.memory_total_mb)?(g.memory_used_mb/1024).toFixed(1)+' / '+(g.memory_total_mb/1024).toFixed(1)+' GB':'—'}</strong></div>`).join('');
  if(!data.gpus?.length)q('#gpu-freshness').textContent=data.gpu_error||'No GPU detected';
  const terminal=q('#terminal-output'),atEnd=terminal.scrollHeight-terminal.scrollTop-terminal.clientHeight<40;
  terminal.textContent=data.terminal||'No console output recorded for this run.';
  if(atEnd)terminal.scrollTop=terminal.scrollHeight;
 }catch{if(requestedRun===runId){q('#gpu-readings').replaceChildren();q('#terminal-output').textContent='';q('#gpu-freshness').textContent='Telemetry unavailable · retrying';}}
 runtimeTimer=setTimeout(refreshRuntime,5000);
}
if(runId&&['http:','https:'].includes(location.protocol))refreshRuntime();
window.addEventListener('pagehide',()=>clearTimeout(runtimeTimer));

let wandbTimer;
function renderWandbNative(data){
 const node=q('#wandb-native'),series=(data.series||[])[0];
 if(!series?.points?.length){node.hidden=true;node.replaceChildren();return;}
 const points=series.points.filter(p=>Number.isFinite(p.step)&&Number.isFinite(p.value));
 if(points.length<2){node.hidden=true;node.replaceChildren();return;}
 const values=points.map(p=>p.value),low=Math.min(...values),high=Math.max(...values),range=high-low||1;
 const position=(point,index)=>[40+index*940/Math.max(1,points.length-1),12+(high-point.value)/range*128];
 const line=points.map((point,index)=>`${index?'L':'M'}${position(point,index).join(',')}`).join(' ');
 const labels=[0,.5,1].map(t=>`<text x="0" y="${16+t*128}" fill="#89927f" font-size="10">${(high-t*range).toPrecision(4)}</text>`).join('');
 const grid=[0,.5,1].map(t=>`<line x1="40" x2="980" y1="${12+t*128}" y2="${12+t*128}" stroke="#e5e9df"/>`).join('');
 const final=values.at(-1),first=values[0],delta=first===0?'':`${((final-first)/Math.abs(first)*100).toFixed(1)}%`;
 node.hidden=false;
 node.innerHTML=`<div class="wandb-native-heading"><span>${escapeHTML(series.name)}</span><strong>${escapeHTML(Number(final).toPrecision(5))}${delta?` · ${escapeHTML(delta)}`:''}</strong></div><svg viewBox="0 0 1000 155" role="img" aria-label="${escapeHTML(series.name)} recorded by Weights and Biases">${grid}${labels}<path d="${line}" fill="none" stroke="#417d64" stroke-width="2" vector-effect="non-scaling-stroke"/><circle cx="${position(points.at(-1),points.length-1)[0]}" cy="${position(points.at(-1),points.length-1)[1]}" r="3.5" fill="#417d64"/></svg><div class="wandb-native-axis"><span>First recorded step</span><span>Latest · W&amp;B</span></div>`;
}
async function refreshWandb(){
 if(demo||!runId){wandbTimer=setTimeout(refreshWandb,30000);return}
 const requestedRun=runId,requestedView=evolution.view;
 try{
  const response=await fetch('/api/runs/'+encodeURIComponent(runId)+'/wandb',{cache:'no-store',signal:AbortSignal.timeout(30000)});
  if(!response.ok)throw Error('Unavailable');
  const data=await response.json(),box=q('#wandb-empty');
  if(requestedRun!==runId||requestedView!==evolution.view)return;
  box.classList.toggle('has-metrics',!!data.metrics?.length);
  box.innerHTML=data.metrics?.length?`<div class="wandb-summary"><p>${escapeHTML(data.name)} · ${escapeHTML(data.state)}</p>${data.metrics.map(m=>`<div><span>${escapeHTML(m.name)}</span><strong>${escapeHTML(Number.isInteger(m.value)?m.value:m.value.toFixed(4))}</strong></div>`).join('')}</div>`:'Metrics will appear when a W&B run is connected and logs data.';
  renderWandbNative(data);
 }catch{if(requestedRun===runId&&requestedView===evolution.view){q('#wandb-empty').classList.remove('has-metrics');q('#wandb-empty').textContent='No metrics available for this run.';renderWandbNative({});}}
 wandbTimer=setTimeout(refreshWandb,30000);
}
if(runId&&['http:','https:'].includes(location.protocol))refreshWandb();
window.addEventListener('pagehide',()=>clearTimeout(wandbTimer));
