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
async function api(path,options={}) {
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers},signal:AbortSignal.timeout(15000)});
  if(response.status===401){location.href='/login.html';throw Error('Sign in required')}
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
    if(['running','complete'].includes(state.status)){
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
  if(new URLSearchParams(location.search).get('demo')==='1'){replayIntake();return;}
  if(lastState&&['failed','cancelled'].includes(lastState.status)){requestKey=crypto.randomUUID();runId=null;lastState=null}
  if(runId&&lastState?.status==='awaiting_data'){dataDialog.showModal();return}
  posting=true;submit.disabled=true;message('');
  try{
    const result=await api('/api/runs',{method:'POST',headers:{'Idempotency-Key':requestKey},body:JSON.stringify({repo:input.value.trim(),mode:document.querySelector('#search-mode').value})});
    runId=result.id;history.replaceState(null,'','?intake='+runId);submit.textContent='···';poll();
  }catch(err){message(err.message);submit.disabled=false}finally{posting=false}
});
dataForm.addEventListener('submit',async e=>{
  e.preventDefault();const button=dataForm.querySelector('button');button.disabled=true;dataError.textContent='';
  if(new URLSearchParams(location.search).get('demo')==='1'){finishReplayData(button);return;}
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

function replayIntake(){
 try{const u=new URL(input.value.trim());if(u.protocol!=='https:'||u.hostname!=='github.com'||u.pathname.split('/').filter(Boolean).length<2)throw Error();}catch{message('Paste a GitHub repository URL.');return;}
 posting=true;submit.disabled=true;input.readOnly=true;stop.hidden=true;
 const steps=['Explore repository structure and training entry points.','Review data preparation scripts and training configurations.','Choose a dataset for this training run.'];
 let step=0;
 function advance(){
  renderActivity({status:step===2?'awaiting_data':'exploring',activity:steps.slice(0,step+1).map(message=>({message}))});
  note.textContent='Recorded demo';note.classList.remove('error');stop.hidden=true;
  if(++step<steps.length)timer=setTimeout(advance,1500);else showReplayData();
 }
 advance();
}
function showReplayData(){
 document.querySelector('#data-caption').textContent='Which dataset would you like to use?';
 document.querySelector('#discovery-summary').textContent='Choose a dataset to preview the flow. Results use the recorded benchmark runs.';
 const choices=document.querySelector('#dataset-options');choices.replaceChildren();
 for(const option of [{name:'Tiny Shakespeare',url:'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',reason:'Small text corpus for a quick nanoGPT run.'},{name:'TinyStories',url:'https://huggingface.co/datasets/roneneldan/TinyStories',reason:'Short stories for language-model training.'}]){
  const button=document.createElement('button');button.type='button';button.className='dataset-option';
  const title=document.createElement('strong');title.textContent=option.name;const reason=document.createElement('span');reason.textContent=option.reason;
  button.append(title,reason);button.onclick=()=>{dataInput.value=option.url;selectedDataset=option.name;dataInput.focus()};choices.append(button);
 }
 dataForm.hidden=false;dataCheck.hidden=true;dataInput.value='';dataForm.querySelector('button').disabled=false;
 dataDialog.showModal();posting=false;submit.disabled=false;submit.onclick=e=>{e.preventDefault();dataDialog.showModal()};
}
function finishReplayData(button){
 try{const u=new URL(dataInput.value.trim());if(u.protocol!=='https:')throw Error();}catch{dataError.textContent='Enter an HTTPS dataset link.';button.disabled=false;return;}
 dataForm.hidden=true;dataCheck.hidden=false;dataCheckText.textContent='Restoring the recorded evaluations…';
 const p=new URLSearchParams({demo:'1',flow:'1',requested_repo:input.value.trim(),requested_data:dataInput.value.trim()});
 timer=setTimeout(()=>{location.href='assets/search.html?'+p},1800);
}
if(new URLSearchParams(location.search).get('demo')==='1')note.textContent='Recorded demo';
