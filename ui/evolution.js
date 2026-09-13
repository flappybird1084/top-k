'use strict';
const evolution={view:new URLSearchParams(location.search).get('view')==='kernel'?'kernel':'architecture',budget:null,budgetPinned:false,initialized:false};
function architectureRender(snapshot){
 const rows=snapshot.architecture?.candidates||[], budgets=[...new Set(rows.map(r=>r.train_secs).filter(Number.isFinite))].sort((a,b)=>a-b);
 if(!evolution.budgetPinned)evolution.budget=rows.filter(r=>r.phase!=='baseline'&&Number.isFinite(r.val_loss)).at(-1)?.train_secs??budgets[0]??null;
 if(!budgets.includes(evolution.budget))evolution.budget=budgets[0]??null;
 const select=q('#architecture-budget');select.innerHTML=budgets.map(b=>`<option value="${b}" ${b===evolution.budget?'selected':''}>${b}s</option>`).join('');select.disabled=!budgets.length;
 const atBudget=rows.filter(r=>r.train_secs===evolution.budget),base=atBudget.find(r=>r.phase==='baseline'),evaluated=atBudget.filter(r=>r.phase!=='baseline'),measured=evaluated.filter(r=>Number.isFinite(r.val_loss)),best=measured.length?measured.reduce((a,b)=>a.val_loss<b.val_loss?a:b):null;
 const improvement=base?.val_loss>0&&best?100*(base.val_loss-best.val_loss)/base.val_loss:null;
 const fmt=v=>Number.isFinite(v)?v.toFixed(4):'—';
 q('#architecture-metrics').hidden=!base&&!evaluated.length;
 q('#architecture-metrics').innerHTML=`<div><span>Baseline loss</span><strong>${fmt(base?.val_loss)}</strong><small>${evolution.budget??'—'}s training budget</small></div><div><span>Best candidate loss</span><strong>${fmt(best?.val_loss)}</strong><small>${best?escapeHTML(best.phase):'Awaiting evaluation'}</small></div><div><span>Loss reduction</span><strong>${improvement===null?'—':improvement.toFixed(2)+'%'}</strong><small>At the same training budget</small></div><div><span>Improved candidates</span><strong>${evaluated.filter(r=>r.accepted).length}</strong><small>${evaluated.length} evaluated</small></div>`;
 q('#architecture-empty').hidden=!!base||!!evaluated.length;
 q('#architecture-plot').hidden=!base&&!measured.length;
 const allChanges=rows.filter(r=>r.phase!=='baseline');
 q('#architecture-changes').hidden=!allChanges.length;q('#architecture-count').textContent=allChanges.length+' evaluated';
 q('#architecture-candidates').innerHTML=allChanges.slice(-12).reverse().map(r=>`<tr><td>${escapeHTML(r.phase)}<small>Generation ${r.generation} · ${r.train_secs}s</small></td><td><details><summary>${escapeHTML(r.strategy)}</summary><p>${escapeHTML(r.failure_note|| (r.parent_id?'Derived from candidate '+r.parent_id:'Evaluated against the original model.'))}</p></details></td><td>${r.model_params?(r.model_params/1e6).toFixed(1)+'M':'—'}</td><td>${fmt(r.val_loss)}</td><td class="${r.accepted?'accepted':'rejected'}">${r.accepted?'Improved':Number.isFinite(r.val_loss)?'Not improved':'Failed check'}</td></tr>`).join('');
 if(base||measured.length){
  const values=[...measured.map(r=>r.val_loss),...(base?[base.val_loss]:[])],low=Math.min(...values)*.97,high=Math.max(...values)*1.03,range=high-low||1,y=v=>15+(high-v)/range*170;
  const points=measured.map((r,i)=>[50+i*920/Math.max(1,measured.length-1),y(r.val_loss)]);
  q('#architecture-chart').innerHTML=[0,.5,1].map(t=>`<line x1="50" x2="980" y1="${15+t*170}" y2="${15+t*170}" stroke="#e5e9df"/><text x="0" y="${19+t*170}" fill="#89927f" font-size="10">${(high-t*range).toFixed(2)}</text>`).join('')+(base?`<path d="M50 ${y(base.val_loss)} H980" stroke="#89927f" stroke-dasharray="5 5" fill="none"><title>Baseline ${base.val_loss.toFixed(4)} · ${base.train_secs}s</title></path>`:'')+`<path d="${points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ')}" stroke="#417d64" stroke-width="2" fill="none"/>`+points.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${measured[i].accepted?'#417d64':'#bf795a'}"><title>${escapeHTML(measured[i].strategy)} · ${measured[i].val_loss.toFixed(4)}</title></circle>`).join('');
 }
}
function setupEvolution(onChange){
 for(const view of ['architecture','kernel']){
  const button=document.querySelector('#tab-'+view);
  button.addEventListener('click',()=>{evolution.view=view;evolution.budget=null;evolution.budgetPinned=false;const url=new URL(location.href);url.searchParams.set('view',view);history.replaceState(null,'',url);onChange()});
  button.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();document.querySelector('#tab-'+(e.key==='Home'?'architecture':e.key==='End'?'kernel':view==='kernel'?'architecture':'kernel')).click();document.querySelector('#tab-'+evolution.view).focus()}});
 }
 document.querySelector('#architecture-budget').addEventListener('change',e=>{evolution.budget=Number(e.target.value);evolution.budgetPinned=true;onChange()});
}
