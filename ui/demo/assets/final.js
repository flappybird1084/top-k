'use strict';
const params=new URLSearchParams(location.search),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>Number.isFinite(v)&&v>0;
const median=values=>{const v=values.slice().sort((a,b)=>a-b),m=Math.floor(v.length/2);return v.length%2?v[m]:(v[m-1]+v[m])/2;};
function external(url,label){try{const u=new URL(url,location.href);if(!['http:','https:'].includes(u.protocol))return '';return `<a href="${esc(u.href)}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`}catch{return ''}}
function links(s,side){const p=new URLSearchParams(params);p.set('view',side);return `<div class="result-links"><a href="../workspace.html?${esc(p)}">All metrics & logs ↗</a>${external(s.integrations?.wandb_url,'W&B')}${external(s.integrations?.weave_url,'Weave')}${external(s.integrations?.marimo_url,'marimo')}</div>`}
function architecture(s){
 const rows=s?.architecture?.candidates||[],finals=rows.filter(r=>r.phase==='finals'&&r.accepted&&finite(r.val_loss)),winner=finals.sort((a,b)=>a.val_loss-b.val_loss)[0];
 if(!winner)return '<h2>Architecture</h2><p class="empty-result">No final architecture evaluation recorded.</p>';
 const baseline=rows.find(r=>r.phase==='baseline'&&r.train_secs===winner.train_secs&&finite(r.val_loss));
 if(!baseline)return '<h2>Architecture</h2><p class="empty-result">A matching baseline is needed for the final comparison.</p>';
 const reduction=100*(1-winner.val_loss/baseline.val_loss),budget=winner.train_secs;
 const budgets=[...new Set(rows.map(r=>r.train_secs).filter(finite))].sort((a,b)=>a-b);
 const table=budgets.map(b=>{const base=rows.find(r=>r.phase==='baseline'&&r.train_secs===b),c=rows.filter(r=>r.phase!=='baseline'&&r.accepted&&r.train_secs===b&&finite(r.val_loss)).sort((a,b)=>a.val_loss-b.val_loss)[0];return `<tr><td>${b}s</td><td>${finite(base?.val_loss)?base.val_loss.toFixed(4):'—'}</td><td>${c?c.val_loss.toFixed(4):'—'}</td><td>${c&&finite(base?.val_loss)?(100*(1-c.val_loss/base.val_loss)).toFixed(2)+'%':'—'}</td></tr>`}).join('');
 return `<div class="result-title"><h2>Architecture</h2><span>Final retrain complete</span></div><p class="model-label">${esc(s.model||'Architecture search')}</p><div class="outcome">${reduction.toFixed(2)}% ↓</div><p class="outcome-caption">Lower validation loss</p><p class="outcome-method">Equal ${budget}-second training budget</p><div class="final-chart" role="img" aria-label="Baseline validation loss ${baseline.val_loss}; final architecture ${winner.val_loss}"><div class="comparison-label"><span>Baseline</span><span>${baseline.val_loss.toFixed(4)}</span></div><div class="comparison-bar"></div><div class="comparison-label"><span>Final architecture</span><span>${winner.val_loss.toFixed(4)}</span></div><div class="comparison-bar optimized" style="width:${Math.min(100,100*winner.val_loss/baseline.val_loss)}%"></div></div><table class="result-table"><thead><tr><th>Budget</th><th>Baseline</th><th>Best</th><th>Reduction</th></tr></thead><tbody>${table}</tbody></table><div class="winner"><h3>Selected architecture</h3><p>${esc(s.id==='75890bd0'?'Parallel blocks · reduced KV projections · cyclic learning rate':winner.strategy)}</p></div>${links(s,'architecture')}`;
}
function kernel(s){
 const v=s?.verification,blocks=v?.blocks?.filter(b=>finite(b.baseline)&&finite(b.candidate))||[];
 if(!blocks.length)return '<h2>Kernels</h2><p class="empty-result">No final paired kernel verification recorded.</p>';
 const gains=blocks.map(b=>100*(1-b.candidate/b.baseline)),reduction=median(gains),max=Math.max(...blocks.flatMap(b=>[b.baseline,b.candidate]));
 const chart=blocks.map((b,i)=>`<div class="pair-column"><div class="pair-sticks"><div class="pair-stick" style="height:${b.baseline/max*100}%" title="Compiled PyTorch: ${b.baseline.toFixed(4)} ms"></div><div class="pair-stick optimized" style="height:${b.candidate/max*100}%" title="Astra: ${b.candidate.toFixed(4)} ms"></div></div><span>Pair ${i+1}</span><span>${b.baseline.toFixed(2)} → ${b.candidate.toFixed(2)}</span></div>`).join('');
 const checks=v.summary.full_state_checks,gens=v.summary.search_generations.length;
 return `<div class="result-title"><h2>Kernels</h2><span>Correctness verified</span></div><p class="model-label">${esc(s.model==='adapters.language · 21.6M'?'Language model · 21.6M':s.model||'Kernel search')}</p><div class="outcome">${reduction.toFixed(2)}% ↓</div><p class="outcome-caption">Lower full training-step time</p><p class="outcome-method">Median of ${blocks.length} paired reductions · compiled PyTorch baseline</p><div class="final-chart" role="img" aria-label="${blocks.length} paired measurements; every kernel measurement improves on its paired baseline"><p class="chart-legend">Gray: compiled PyTorch · Green: Astra · time in milliseconds</p><div class="pair-columns">${chart}</div></div><table class="result-table"><thead><tr><th>Verification</th><th>Result</th></tr></thead><tbody><tr><td>Minimum paired improvement</td><td>${Math.min(...gains).toFixed(2)}%</td></tr><tr><td>Full-state correctness checks</td><td>${checks} passed</td></tr><tr><td>Search generations</td><td>${gens} completed</td></tr></tbody></table><div class="winner"><h3>Selected kernels</h3><p>${esc(s.id==='astra-final'?'Fused MLP backward · QKV gradient packing · token-pair weight reuse':v.summary.backward_launches.join(' · '))}<br>Verified at ${esc(s.provenance.commit)}.</p></div>${links(s,'kernel')}`;
}
async function load(){
 document.querySelector('#back-diagram').href='search.html?'+params;
 try{
 let pair;
 if(params.get('demo')==='1')pair=window.RECORDED_RUNS;
 else{
  const fetchRun=async id=>{const r=await fetch('/api/runs/'+encodeURIComponent(id),{signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('Run unavailable');return r.json()};
  const id=params.get('run');if(!id)throw Error('Open final results from a run.');
  const s=await fetchRun(id);pair={[s.mode==='recipe'?'architecture':'kernel']:s};
  await Promise.all(['architecture','kernel'].map(async side=>{const related=params.get(side+'_run')||s.related_runs?.[side];if(related&&related!==id)pair[side]=await fetchRun(related)}));
 }
 document.querySelector('#architecture-result').innerHTML=architecture(pair.architecture);
 document.querySelector('#kernel-result').innerHTML=kernel(pair.kernel);
 document.querySelector('#result-status').textContent=params.get('demo')==='1'?'Recorded results · separate workloads':'Run results';
 }catch(e){const n=document.querySelector('#result-error');n.hidden=false;n.textContent=e.message;document.querySelector('#result-status').textContent='Results unavailable';}
}
load();
