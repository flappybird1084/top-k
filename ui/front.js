'use strict';
const form = document.querySelector('#repo-form');
const input = document.querySelector('#repository');
const note = document.querySelector('#form-note');
// Reset the default after browser form-state restoration on reload.
window.addEventListener('pageshow',()=>{document.querySelector('#search-mode').value='kernel'});
const paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
document.body.classList.toggle('still', paused);
const sprites = [...document.querySelectorAll('.sprite')];
document.querySelector('main').addEventListener('pointermove', event => {
  if (paused || event.pointerType === 'touch') return;
  const x = (event.clientX / innerWidth - .5) * 12;
  const y = (event.clientY / innerHeight - .5) * 9;
  sprites.forEach((sprite, i) => {
    sprite.style.setProperty('--dx', `${x * (i + 1) / 2}px`);
    sprite.style.setProperty('--dy', `${y * (i + 1) / 2}px`);
  });
});

const flow=document.querySelector('#agent-flow'), events=document.querySelector('#agent-events');
const progress=document.querySelector('#agent-progress'), submit=form.querySelector('button[type="submit"]');
const dataDialog=document.querySelector('#data-dialog'), dataForm=document.querySelector('#data-form');
const dataInput=document.querySelector('#data-link'), dataError=document.querySelector('#data-error');
const dataCheck=document.querySelector('#data-check'), dataCheckText=document.querySelector('#data-check-text');
const stop=document.querySelector('#stop-flow');
let selectedDataset=null;
dataInput.addEventListener('input',()=>{selectedDataset=null});
let runId=null,timer=null,posting=false,lastState=null,requestKey=crypto.randomUUID();
document.querySelector('#search-mode').addEventListener('change',()=>{requestKey=crypto.randomUUID()});
input.addEventListener('input',()=>{if(!runId)requestKey=crypto.randomUUID()});
// AbortSignal.timeout() is absent on older Firefox, where it throws instead of
// aborting; an AbortController plus a timer behaves the same everywhere.
function deadline(ms){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),ms);
  return {signal:controller.signal,done:()=>clearTimeout(timer)};
}
async function api(path,options={}) {
  const limit=deadline(15000);
  let response;
  try{response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers},signal:limit.signal});}
  finally{limit.done();}
  if(response.status===401)throw Error('Sign in with GitHub to continue.');
  if(response.status===428)throw Error('Connect your notebook and W&B account to start a run.');
  const result=await response.json();if(!response.ok)throw Error(result.error||'Request failed');return result;
}
function message(text){note.textContent=text;note.classList.toggle('error',!!text)}
function renderActivity(state){
  flow.hidden=false;events.replaceChildren();
  const terminal=['failed','cancelled'].includes(state.status);
  const activity=[...(state.activity||[])];
  if(terminal&&activity.length>1&&activity.at(-1).message===state.message)activity.pop();
  activity.forEach((event,index)=>{
    const last=index===activity.length-1;
    const status=last?(terminal?'failed':['exploring','validating','queued','preparing'].includes(state.status)?'working':'done'):'done';
    const li=document.createElement('li');li.className=status;
    const marker=document.createElement('span');marker.className='event-marker';marker.textContent=status==='failed'?'×':status==='working'?'·':'✓';
    const content=document.createElement('div');const label=document.createElement('span');label.className='event-agent';label.textContent='Repository agent';
    const text=document.createElement('p');text.textContent=event.message;content.append(label,text);li.append(marker,content);events.append(li);
  });
  const stage={exploring:1,awaiting_data:2,validating:3,queued:4,preparing:5,running:6,complete:6}[state.status]||0;
  progress.setAttribute('aria-valuenow',String(stage));progress.querySelector('span').style.width=`${stage/6*100}%`;
  events.scrollTop=events.scrollHeight;
  stop.hidden=['complete','failed','cancelled'].includes(state.status);
}
async function poll(){
  clearTimeout(timer);
  try{
    const state=await api('/api/runs/'+runId);lastState=state;input.value=state.repo||input.value;input.readOnly=!['failed','cancelled','complete'].includes(state.status);renderActivity(state);
    if(state.connection_error)message(state.connection_error);else message('');
    if(state.status==='awaiting_data'){
      document.querySelector('#data-caption').textContent=state.message||'Which dataset would you like to use?';
      document.querySelector('#discovery-summary').textContent=state.discovery_summary||'';
      const choices=document.querySelector('#dataset-options');choices.replaceChildren();
      for(const option of state.dataset_options||[]){
        const button=document.createElement('button');button.type='button';button.className='dataset-option';
        const title=document.createElement('strong');title.textContent=option.name;
        const reason=document.createElement('span');reason.textContent=option.reason;button.append(title,reason);
        button.addEventListener('click',()=>{dataInput.value=option.url;selectedDataset=option.name;dataInput.focus()});
        const source=document.createElement('a');source.textContent='Source ↗';
        try{const url=new URL(option.evidence);if(url.protocol==='https:'){source.href=url.href;source.target='_blank';source.rel='noopener noreferrer'}}catch{}
        const row=document.createElement('div');row.append(button,source);choices.append(row);
      }
      dataForm.hidden=false;dataCheck.hidden=true;dataForm.querySelector('button').disabled=false;
      dataError.textContent='';
      if(!dataDialog.open)dataDialog.showModal();
      submit.disabled=false;submit.setAttribute('aria-label','Add training data');return;
    }
    if(['validating','queued','preparing'].includes(state.status)){
      if(dataDialog.open){dataForm.hidden=true;dataCheck.hidden=false;dataCheckText.textContent=state.message||'Preparing the run…'}
    }
    if(['running','complete'].includes(state.status)||(state.related_runs&&['failed','cancelled'].includes(state.status))){
      location.href='assets/search.html?run='+runId;return;
    }
    if(['failed','cancelled'].includes(state.status)){
      message(state.message||'The run could not start.');submit.disabled=false;submit.textContent='↻';if(dataDialog.open)dataDialog.close();return;
    }
  }catch(err){message(err.message+'. Retrying…')}
  timer=setTimeout(poll,2500);
}
form.addEventListener('submit',async e=>{
  e.preventDefault();if(posting)return;
  if(lastState&&['failed','cancelled'].includes(lastState.status)){requestKey=crypto.randomUUID();runId=null;lastState=null}
  if(runId&&lastState?.status==='awaiting_data'){dataDialog.showModal();return}
  posting=true;submit.disabled=true;message('');
  try{
    const result=await api('/api/runs',{method:'POST',headers:{'Idempotency-Key':requestKey},body:JSON.stringify({repo:input.value.trim(),mode:document.querySelector('#search-mode').value,settings:gatherSettings()})});
    runId=result.id;history.replaceState(null,'','?intake='+runId);submit.textContent='···';poll();
  }catch(err){message(err.message);submit.disabled=false}finally{posting=false}
});
dataForm.addEventListener('submit',async e=>{
  e.preventDefault();const button=dataForm.querySelector('button');button.disabled=true;dataError.textContent='';
  try{
    await api('/api/runs/'+runId+'/data',{method:'POST',body:JSON.stringify({url:dataInput.value.trim(),dataset_choice:selectedDataset})});
    dataForm.hidden=true;dataCheck.hidden=false;dataCheckText.textContent='Checking data access and the training sample…';poll();
  }catch(err){dataError.textContent=err.message;button.disabled=false}
});
document.querySelector('#close-data').addEventListener('click',()=>dataDialog.close());
stop.addEventListener('click',async()=>{try{await api('/api/runs/'+runId+'/cancel',{method:'POST',body:'{}'});clearTimeout(timer);poll()}catch(err){message(err.message)}});
const resume=new URLSearchParams(location.search).get('intake');
if(resume&&/^[a-f0-9]{32}$/.test(resume)){runId=resume;submit.disabled=true;poll()}
window.addEventListener('pagehide',()=>clearTimeout(timer));

// Settings dialog: optional per-run overrides; blank fields defer to the
// server's KEVO_UI_* defaults. Persisted locally so choices survive reloads.
const settingsDialog=document.querySelector('#settings-dialog');
const S_FIELDS={llm:'#s-llm',profile:'#s-profile',spend_cap:'#s-spend',max_debug_turns:'#s-debug',molab_connection:'#s-molab',arch_gens:'#s-arch-gens',arch_cands:'#s-arch-cands',arch_secs:'#s-arch-secs',hp_gens:'#s-hp-gens',hp_cands:'#s-hp-cands',hp_secs:'#s-hp-secs',finals_k:'#s-finals-k',finals_secs:'#s-finals-secs',parallelism:'#s-par',parent_pool:'#s-pool',eval_batches:'#s-evalb',loss_margin:'#s-margin'};
const TEXT_KEYS=new Set(['llm','profile','molab_connection']);
function gatherSettings(){
  const out={};
  for(const[key,sel]of Object.entries(S_FIELDS)){
    const v=document.querySelector(sel).value.trim();
    if(v)out[key]=TEXT_KEYS.has(key)?v:Number(v);
  }
  return out;
}
try{
  const saved=JSON.parse(localStorage.getItem('topk-settings')||'{}');
  for(const[key,sel]of Object.entries(S_FIELDS))if(saved[key]!=null&&saved[key]!=='')document.querySelector(sel).value=saved[key];
}catch{}
document.querySelector('#open-settings').addEventListener('click',()=>settingsDialog.showModal());
document.querySelector('#close-settings').addEventListener('click',()=>settingsDialog.close());
document.querySelector('#settings-form').addEventListener('submit',e=>{
  e.preventDefault();
  localStorage.setItem('topk-settings',JSON.stringify(gatherSettings()));
  settingsDialog.close();
});
document.querySelector('#reset-settings').addEventListener('click',()=>{
  localStorage.removeItem('topk-settings');
  for(const sel of Object.values(S_FIELDS))document.querySelector(sel).value='';
});
api('/api/settings').then(d=>{
  document.querySelector('#settings-conn-note').textContent=d.connection_file
    ?'Server connection file: configured ✓'
    :'No server connection file — paste the pair prompt above before launching on molab.';
  document.querySelector('#s-llm').options[0].textContent='server default ('+d.llm+')';
  document.querySelector('#s-profile').options[0].textContent='server default ('+d.profile+')';
}).catch(()=>{});
