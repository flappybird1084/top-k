'use strict';
const evolution={view:new URLSearchParams(location.search).get('view')==='kernel'?'kernel':'architecture',gen:null,genPinned:false,initialized:false};
function architectureRender(snapshot){
 const rows=snapshot.architecture?.candidates||[];
 const fmt=v=>Number.isFinite(v)?v.toFixed(4):'—';
 const baseByBudget=new Map(rows.filter(r=>r.phase==='baseline'&&Number.isFinite(r.val_loss)).map(r=>[r.train_secs,r.val_loss]));
 const meta=new Map((snapshot.generations||[]).map(g=>[g.n,g]));
 const groups=new Map();
 for(const r of rows){if(r.phase==='baseline')continue;const g=r.generation??0;if(!groups.has(g))groups.set(g,[]);groups.get(g).push(r)}
 const gens=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([n,cands])=>({n,cands,
  finals:cands.every(c=>c.phase==='finals'),
  budget:cands.find(c=>Number.isFinite(c.train_secs))?.train_secs??null}));
 const genLabel=g=>g.finals?'Finals':'Gen '+g.n;
 if(!evolution.genPinned||!gens.some(g=>g.n===evolution.gen))
  evolution.gen=(gens.filter(g=>g.cands.some(c=>Number.isFinite(c.val_loss))).at(-1)??gens.at(-1))?.n??null;
 const picker=q('#generation-picker');
 picker.hidden=!gens.length;
 picker.innerHTML=gens.map(g=>`<button type="button" data-gen="${g.n}" aria-pressed="${g.n===evolution.gen}">${genLabel(g)}${g.budget?`<small>${g.budget}s</small>`:''}</button>`).join('');
 const sel=gens.find(g=>g.n===evolution.gen);
 const base=sel?.budget!=null?baseByBudget.get(sel.budget):null;
 const evaluated=sel?.cands??[],measured=evaluated.filter(r=>Number.isFinite(r.val_loss)),best=measured.length?measured.reduce((a,b)=>a.val_loss<b.val_loss?a:b):null;
 const improvement=base>0&&best?100*(base-best.val_loss)/base:null;
 q('#architecture-metrics').hidden=!gens.length;
 q('#architecture-metrics').innerHTML=`<div><span>Baseline loss</span><strong>${fmt(base)}</strong><small>${sel?.budget??'—'}s training budget</small></div><div><span>Best candidate loss</span><strong>${fmt(best?.val_loss)}</strong><small>${sel?genLabel(sel):'Awaiting evaluation'}</small></div><div><span>Loss reduction</span><strong>${improvement===null?'—':improvement.toFixed(2)+'%'}</strong><small>At the same training budget</small></div><div><span>Improved candidates</span><strong>${evaluated.filter(r=>r.accepted).length}</strong><small>${evaluated.length} evaluated in ${sel?genLabel(sel).toLowerCase():'—'}</small></div>`;
 q('#architecture-empty').hidden=!!gens.length||baseByBudget.size>0;
 q('#architecture-plot').hidden=!measured.length&&!Number.isFinite(base);
 if(measured.length||Number.isFinite(base)){
  const values=[...measured.map(r=>r.val_loss),...(Number.isFinite(base)?[base]:[])],low=Math.min(...values)*.97,high=Math.max(...values)*1.03,range=high-low||1,y=v=>15+(high-v)/range*170;
  const points=measured.map((r,i)=>[50+i*920/Math.max(1,measured.length-1),y(r.val_loss)]);
  q('#architecture-chart').innerHTML=[0,.5,1].map(t=>`<line x1="50" x2="980" y1="${15+t*170}" y2="${15+t*170}" stroke="#e5e9df"/><text x="0" y="${19+t*170}" fill="#89927f" font-size="10">${(high-t*range).toFixed(2)}</text>`).join('')
   +(Number.isFinite(base)?`<path d="M50 ${y(base)} H980" stroke="#89927f" stroke-dasharray="5 5" fill="none"><title>Baseline ${base.toFixed(4)} · ${sel?.budget}s</title></path>`:'')
   +(points.length>1?`<path d="${points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ')}" stroke="#417d64" stroke-width="2" fill="none"/>`:'')
   +points.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${measured[i].accepted?'#417d64':'#bf795a'}"><title>${escapeHTML(measured[i].strategy)} · ${measured[i].val_loss.toFixed(4)}</title></circle>`).join('');
 }
 // Generations panels — the barebones UI's collapsible per-generation view.
 const allChanges=rows.filter(r=>r.phase!=='baseline');
 q('#architecture-changes').hidden=!allChanges.length;q('#architecture-count').textContent=allChanges.length+' evaluated';
 const panels=q('#generation-panels');
 const open=new Set([...panels.querySelectorAll('details[open]')].map(d=>d.dataset.key));
 const firstRender=!panels.childElementCount;
 panels.innerHTML=gens.map((g,gi)=>{
  const acc=g.cands.filter(c=>c.accepted).length,m=meta.get(g.n);
  const tok=m&&m.tokens_in?` · ${Number(m.tokens_in).toLocaleString()} tok in / ${Number(m.tokens_out||0).toLocaleString()} out`:'';
  const isOpen=firstRender?gi===gens.length-1:open.has('g'+g.n);
  return `<details class="gen-panel" data-key="g${g.n}"${isOpen?' open':''}><summary>${genLabel(g)} — ${g.cands.length} candidate(s), ${acc} improved${g.budget?` · ${g.budget}s budget`:''}${tok}</summary>`+g.cands.map(r=>{
   const b=baseByBudget.get(r.train_secs);
   const delta=Number.isFinite(r.val_loss)&&b?100*(b-r.val_loss)/b:null;
   const cls=r.accepted?'improved':Number.isFinite(r.val_loss)?'not-improved':'failed';
   const label=r.accepted?'Improved':Number.isFinite(r.val_loss)?'Not improved':'Failed';
   const head=Number.isFinite(r.val_loss)
    ?`val ${r.val_loss.toFixed(4)}${delta!=null?` (${delta>=0?'+':''}${delta.toFixed(2)}% vs baseline)`:''} — ${r.strategy||''}`
    :`${(r.failure_note||'failed to evaluate')} — ${r.strategy||''}`;
   return `<details class="cand-row" data-key="c${r.id}"${open.has('c'+r.id)?' open':''}><summary><span class="pill ${cls}">${label}</span><span class="cand-head">${escapeHTML(head)}</span></summary><div class="cand-body"><p>${escapeHTML(r.strategy||'')}</p><dl>${r.model_params?`<div><dt>Parameters</dt><dd>${(r.model_params/1e6).toFixed(1)}M</dd></div>`:''}${Number.isFinite(r.val_loss)?`<div><dt>Val loss</dt><dd>${r.val_loss.toFixed(4)} @${r.train_secs}s</dd></div>`:''}${r.model_name?`<div><dt>Agent model</dt><dd>${escapeHTML(r.model_name)}</dd></div>`:''}${r.parent_id?`<div><dt>Parent</dt><dd>#${r.parent_id}</dd></div>`:''}${r.failure_note?`<div><dt>Failure</dt><dd>${escapeHTML(String(r.failure_note).slice(0,600))}</dd></div>`:''}</dl></div></details>`;
  }).join('')+`</details>`;
 }).join('');
}
function setupEvolution(onChange){
 for(const view of ['architecture','kernel']){
  const button=document.querySelector('#tab-'+view);
  button.addEventListener('click',()=>{evolution.view=view;evolution.gen=null;evolution.genPinned=false;const url=new URL(location.href);url.searchParams.set('view',view);history.replaceState(null,'',url);onChange()});
  button.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();document.querySelector('#tab-'+(e.key==='Home'?'architecture':e.key==='End'?'kernel':view==='kernel'?'architecture':'kernel')).click();document.querySelector('#tab-'+evolution.view).focus()}});
 }
 document.querySelector('#generation-picker').addEventListener('click',e=>{
  const b=e.target.closest('button[data-gen]');if(!b)return;
  evolution.gen=Number(b.dataset.gen);evolution.genPinned=true;onChange();
 });
}
