'use strict';
const q=selector=>document.querySelector(selector);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const steps=[
  ['repo','Repository','Bring a repo. Find a better recipe.','Agents adapt your training code, then explore better model designs and training strategies.'],
  ['flow','The loop','Every generation starts with evidence.',''],
  ['agents','Sub-agents','Build, check, repair. In parallel.','Each coding agent owns a candidate. Click one to follow its work.'],
  ['aria','ARIA review','We use ARIA to Understand Results','ARIA helps us compare training runs, investigate errors, and share the evidence.']
];
let pitchState=PitchDemo.initialState();
const hashStep=()=>steps.findIndex(step=>step[0]===location.hash.slice(1));
if(hashStep()>=0)pitchState=PitchDemo.transition(pitchState,{type:'step',value:hashStep()});
function dispatch(action,focusHeading=false){
  pitchState=PitchDemo.transition(pitchState,action);
  if(pitchState.step===2 && !['architecture','hyperparam'].includes(pitchState.phase))pitchState=PitchDemo.transition(pitchState,{type:'phase',value:'architecture'});
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
  q('#candidate-detail').innerHTML=`<p class="section-label">Candidate ${row.id}</p><h2>${escapeHTML(row.strategy)}</h2><div class="candidate-process-loop"><div class="candidate-process"><span>Build</span><i>→</i><span>Check</span><i>→</i><span>${row.repair?'Repair':'Evaluate'}</span></div><svg viewBox="0 0 300 24" preserveAspectRatio="none" aria-hidden="true"><path d="M 266 1 V 7 Q 266 18 255 18 H 28 Q 17 18 17 7 V 1 M 12 7 L 17 1 L 22 7"/></svg></div><dl><div><dt>Parent recipe</dt><dd>${parent?'#'+parent.id+' · '+escapeHTML(parent.strategy):'Adapter source'}</dd></div><div><dt>Training budget</dt><dd>${row.train_secs} seconds</dd></div>${row.val_loss===null?'':`<div><dt>Validation loss</dt><dd>${row.val_loss.toFixed(4)}</dd></div>`}</dl><p class="detail-note">${escapeHTML(row.failure_note||row.repair||'The candidate passed its check, then trained from fresh weights for evaluation.')}</p>`;
}
function renderPitch(){
  const [hash,,title,description]=steps[pitchState.step];
  document.querySelectorAll('[data-panel]').forEach(panel=>panel.hidden=Number(panel.dataset.panel)!==pitchState.step);
  document.querySelectorAll('[data-step]').forEach(button=>{if(Number(button.dataset.step)===pitchState.step)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});
  document.querySelectorAll('[data-phase]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.phase===pitchState.phase)));
  document.querySelectorAll('[data-generation]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.generation)===pitchState.generation)));
  q('#stage-title').textContent=title;
  q('#stage-description').textContent=description;q('#stage-description').hidden=!description;q('#step-count').textContent=(pitchState.step+1)+' / '+steps.length;
  q('#previous').disabled=pitchState.step===0;q('#next').hidden=pitchState.step===3;
  q('#next').textContent=pitchState.step<steps.length-1?'Next: '+steps[pitchState.step+1][1]+' →':'';
  q('#stage-hint').textContent=pitchState.step===3?'Explore the report, or revisit any stage.':'Click any stage to explore · ← → to move';
  if(pitchState.step===2)renderAgents();
  history.replaceState(null,'','#'+hash);
}
document.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.step!==undefined||button.dataset.go!==undefined)dispatch({type:'step',value:Number(button.dataset.step??button.dataset.go)},true);
  else if(button.dataset.phase)dispatch({type:'phase',value:button.dataset.phase});
  else if(button.dataset.generation)dispatch({type:'generation',value:Number(button.dataset.generation)});
  else if(button.dataset.candidate){dispatch({type:'select',value:Number(button.dataset.candidate)});q('[data-candidate="'+button.dataset.candidate+'"]').focus({preventScroll:true});}

});
q('#next').addEventListener('click',()=>dispatch({type:'next'},true));
q('#previous').addEventListener('click',()=>dispatch({type:'back'},true));
q('#restart').addEventListener('click',()=>{
  document.querySelectorAll('details').forEach(detail=>detail.open=detail.classList.contains('research'));
  dispatch({type:'restart'},true);
});
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.target.closest('input,select,textarea,button,summary'))return;
  if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();dispatch({type:event.key==='ArrowLeft'?'back':'next'},true);}
});
window.addEventListener('hashchange',()=>{const index=hashStep();if(index>=0)dispatch({type:'step',value:index},true);});
const report=PitchDemo.safeReportURL(PitchDemo.reportURL);
q('#report-link').hidden=!report;if(report)q('#report-link').href=report;
renderPitch();
