'use strict';
// Reuse the dashboard's architectureRender without its live polling/submission controllers.
const q=selector=>document.querySelector(selector);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const steps=[
  ['repo','Repository','Bring a repo. Find a better recipe.','Agents adapt your training code, then explore better model designs and training strategies.'],
  ['adapter','Adapter','Make the code ready to experiment.','The adapter agent connects model, data, and loss — and repairs errors using raw feedback.'],
  ['plan','Research & plan','Research first. Then divide the work.','The planner turns evidence and past lessons into concrete experiments.'],
  ['agents','Coding agents','Build, check, repair. In parallel.','Each coding agent owns a candidate. Click one to follow its work.'],
  ['evaluate','Evaluation','Measure. Learn. Evolve.','The harness evaluates candidates; the curator carries useful lessons into the next generation.'],
  ['finals','Finals','Give the finalists a longer run.','Retrain the strongest candidates and compare their final validation losses.'],
  ['aria','ARIA review','Make the experiments understandable.','ARIA helps us compare training runs, understand traces, and review the evidence.']
];
let pitchState=PitchDemo.initialState();
const hashStep=()=>steps.findIndex(step=>step[0]===location.hash.slice(1));
if(hashStep()>=0)pitchState=PitchDemo.transition(pitchState,{type:'step',value:hashStep()});
function dispatch(action,focusHeading=false){
  pitchState=PitchDemo.transition(pitchState,action);
  if(pitchState.step===3 && !['architecture','hyperparam'].includes(pitchState.phase))pitchState=PitchDemo.transition(pitchState,{type:'phase',value:'architecture'});
  renderPitch();
  if(focusHeading)q('#stage-title').focus({preventScroll:true});
}
function renderAgents(){
  const candidates=PitchDemo.visibleCandidates(pitchState);
  q('#generation-buttons').hidden=pitchState.phase!=='architecture';
  q('#agent-cards').innerHTML=candidates.map(r=>`<button class="agent-block has-avatar ${r.accepted?'improved':r.correct_ok?'dropped':'failed'}" data-candidate="${r.id}" aria-pressed="${r.id===pitchState.selectedId}"><span class="agent-header"><span class="agent-avatar"><img src="ui/assets/agent-${r.model_name}.png" alt=""></span><span class="agent-id">Candidate ${r.id}</span></span><span class="state-mark" aria-hidden="true">${r.accepted?'✓':'×'}</span><span class="agent-label">${escapeHTML(r.strategy)}</span><span class="candidate-status">${r.accepted?'Improved':r.correct_ok?'Not improved':'Failed check'}</span></button>`).join('');
  const row=candidates.find(r=>r.id===pitchState.selectedId)||candidates[0];
  if(!row){q('#candidate-detail').textContent='No candidates in this phase.';return;}
  const parent=PitchDemo.rows.find(r=>r.id===row.parent_id);
  q('#candidate-detail').innerHTML=`<p class="section-label">Candidate ${row.id}</p><h2>${escapeHTML(row.strategy)}</h2><div class="candidate-process"><span>Build</span><i>→</i><span>Check</span><i>→</i><span>${row.repair?'Repair':'Evaluate'}</span></div><dl><div><dt>Parent recipe</dt><dd>${parent?'#'+parent.id+' · '+escapeHTML(parent.strategy):'Adapter source'}</dd></div><div><dt>Training budget</dt><dd>${row.train_secs} seconds</dd></div><div><dt>Validation loss</dt><dd>${row.val_loss===null?'Not evaluated':row.val_loss.toFixed(4)}</dd></div></dl><p class="detail-note">${escapeHTML(row.failure_note||row.repair||'The candidate passed its check, then trained from fresh weights for evaluation.')}</p>`;
}
function renderEvaluation(){
  const mixed=pitchState.phase==='mixed';
  q('#mixed-empty').hidden=!mixed;q('#evaluation-content').hidden=mixed;
  if(mixed)return;
  evolution.budget=pitchState.budget;evolution.budgetPinned=true;
  architectureRender(PitchDemo.snapshot(pitchState));
  // These dashboard components are rendered above; the walkthrough supplies a phase-focused disclosure table.
  q('#architecture-changes').hidden=true;
  const hp=pitchState.phase==='hyperparam';
  q('#phase-caption').textContent=hp?'1 generation × 8 candidates · 2 minutes each':'2 generations × 8 candidates · 1 minute each';
  q('#curator-lesson').textContent=hp?'Warmup and weight decay helped this parent recipe. Carry the strongest candidates into the finals.':'RMSNorm and tied embeddings improved loss. Build the next generation from these recipes.';
  const chain=hp?[10,18]:[3,9];
  q('#lineage').innerHTML=chain.map((id,index)=>{const row=PitchDemo.rows.find(r=>r.id===id);return `${index?'<span class="lineage-arrow" aria-hidden="true">↓</span>':''}<button class="lineage-node" data-inspect="${id}"><span>#${id} · ${row.phase==='finals'?'Finalist':'Gen '+row.generation}</span><strong>${escapeHTML(row.strategy)}</strong></button>`}).join('');
  const rows=PitchDemo.rows.filter(r=>r.phase===pitchState.phase);
  q('#table-count').textContent=rows.length+' candidates · expand';
  q('#phase-results').innerHTML=rows.map(r=>`<tr><td><button class="table-recipe" data-inspect="${r.id}">#${r.id} · ${escapeHTML(r.strategy)}</button></td><td>${r.parent_id?'#'+r.parent_id:'Adapter'}</td><td>${r.val_loss?.toFixed(4)??'—'}</td><td class="${r.accepted?'accepted':'rejected'}">${r.accepted?'Improved':r.correct_ok?'Not improved':'Failed check'}</td></tr>`).join('');
}
function renderFinals(){
  q('#finalist-cards').innerHTML=PitchDemo.finalists().map((r,index)=>`<div class="finalist ${index===0?'winner':''}"><div class="finalist-top"><span class="status ${index===0?'kept':''}">${index===0?'Selected winner':'Runner-up'}</span><span>Candidate ${r.id}</span></div><h2>${escapeHTML(r.strategy)}</h2><div class="final-loss">${r.val_loss.toFixed(4)}<span>validation loss</span></div><div class="final-comparison"><span>Baseline</span><b>3.6600</b></div><button class="text-link" data-inspect="${r.parent_id}">Inspect parent recipe #${r.parent_id} ↗</button></div>`).join('');
}
function renderPitch(){
  const [hash,,title,description]=steps[pitchState.step];
  document.querySelectorAll('[data-panel]').forEach(panel=>panel.hidden=Number(panel.dataset.panel)!==pitchState.step);
  document.querySelectorAll('[data-step]').forEach(button=>{if(Number(button.dataset.step)===pitchState.step)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});
  document.querySelectorAll('[data-phase]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.phase===pitchState.phase)));
  document.querySelectorAll('[data-generation]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.generation)===pitchState.generation)));
  q('#stage-title').textContent=title;q('#stage-description').textContent=description;q('#step-count').textContent=(pitchState.step+1)+' / 7';
  q('#previous').disabled=pitchState.step===0;q('#next').hidden=pitchState.step===6;
  q('#next').textContent=pitchState.step<6?'Next: '+steps[pitchState.step+1][1]+' →':'';
  q('#stage-hint').textContent=pitchState.step===6?'Explore the report, or revisit any stage.':'Click any stage to explore · ← → to move';
  if(pitchState.step===3)renderAgents();
  if(pitchState.step===4)renderEvaluation();
  if(pitchState.step===5)renderFinals();
  history.replaceState(null,'','#'+hash);
}
document.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.step!==undefined||button.dataset.go!==undefined)dispatch({type:'step',value:Number(button.dataset.step??button.dataset.go)},true);
  else if(button.dataset.phase)dispatch({type:'phase',value:button.dataset.phase});
  else if(button.dataset.generation)dispatch({type:'generation',value:Number(button.dataset.generation)});
  else if(button.dataset.candidate){dispatch({type:'select',value:Number(button.dataset.candidate)});q('[data-candidate="'+button.dataset.candidate+'"]').focus({preventScroll:true});}
  else if(button.dataset.inspect){
    const row=PitchDemo.rows.find(r=>r.id===Number(button.dataset.inspect));
    if(row.phase==='finals'){dispatch({type:'step',value:5},true);return;}
    pitchState=PitchDemo.transition(pitchState,{type:'phase',value:row.phase});
    if(row.phase==='architecture')pitchState=PitchDemo.transition(pitchState,{type:'generation',value:row.generation});
    pitchState=PitchDemo.transition(pitchState,{type:'select',value:row.id});
    dispatch({type:'step',value:3},true);
  }
});
q('#next').addEventListener('click',()=>dispatch({type:'next'},true));
q('#previous').addEventListener('click',()=>dispatch({type:'back'},true));
q('#restart').addEventListener('click',()=>{
  document.querySelectorAll('details').forEach(detail=>detail.open=detail.classList.contains('research'));
  dispatch({type:'restart'},true);
});
q('#architecture-budget').addEventListener('change',event=>dispatch({type:'budget',value:Number(event.target.value)}));
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.target.closest('input,select,textarea,button,summary'))return;
  if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();dispatch({type:event.key==='ArrowLeft'?'back':'next'},true);}
});
window.addEventListener('hashchange',()=>{const index=hashStep();if(index>=0)dispatch({type:'step',value:index},true);});
const report=PitchDemo.safeReportURL(PitchDemo.reportURL);
q('#report-link').hidden=!report;if(report)q('#report-link').href=report;
renderPitch();
