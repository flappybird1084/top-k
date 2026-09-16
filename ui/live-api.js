'use strict';
(() => {
 const backend='https://api.top-k.dev';
 const original=window.fetch.bind(window), key='top-k-github-session-v1';
 const isLanding=/\/(?:index\.html)?$/.test(location.pathname);
 let user=null, popup=null, config=null, setup=null;
 let accountButton, accountStatus, accountSetup, gate, gateButton, gateStatus, setupPanel, setupStatus;
 const criticalStyle=document.createElement('style');
 criticalStyle.textContent=isLanding
  ? 'html.topk-auth-pending body>.github-account,html.topk-auth-locked body>.github-account{visibility:hidden!important}'
  : 'html.topk-auth-pending body>*,html.topk-auth-locked body>*{visibility:hidden!important}html.topk-auth-pending body>.github-gate,html.topk-auth-locked body>.github-gate{visibility:visible!important}';
 document.head.append(criticalStyle);
 document.documentElement.classList.add('topk-auth-pending');
 document.documentElement.classList.toggle('topk-landing',isLanding);

 // AbortSignal.timeout() is missing on older Firefox (and on any browser that
 // ships AbortController without it), which turns every call into an
 // uncatchable TypeError. An AbortController plus a timer works everywhere.
 function deadline(ms){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),ms);
  return {signal:controller.signal,done:()=>clearTimeout(timer)};
 }
 async function endpoint(path,opts={},ms=15000){
  const limit=deadline(ms);
  try{return await original(backend+path,{...opts,credentials:'omit',signal:limit.signal});}
  finally{limit.done();}
 }
 const headers=()=>({'Authorization':'Bearer '+(localStorage.getItem(key)||'')});
 const jsonHeaders=()=>({...headers(),'Content-Type':'application/json'});
 function setStatus(message=''){
  if(accountStatus)accountStatus.textContent=message;
  if(gateStatus)gateStatus.textContent=message;
 }
 function paint(message=''){
  if(accountButton)accountButton.textContent=user?'Sign out · @'+user.login:'Sign in with GitHub';
  if(gateButton){gateButton.disabled=!config?.enabled;gateButton.textContent=config?.enabled?'Sign in with GitHub':'GitHub sign-in unavailable';}
  if(accountSetup)accountSetup.hidden=!user;
  setStatus(message);
 }
 const ready=state=>!!(state&&state.notebook?.configured&&state.wandb?.configured);
 function unlock(){
  document.documentElement.classList.remove('topk-auth-pending','topk-auth-locked');
  document.documentElement.classList.add('topk-authenticated');
  gate?.remove();gate=null;
  window.dispatchEvent(new Event('topk-auth-changed'));
 }
 function lock(message){
  document.documentElement.classList.remove('topk-auth-pending','topk-authenticated');
  document.documentElement.classList.add('topk-auth-locked');
  paint(message);
 }
 // The run cannot start without the visitor's own notebook and W&B account, so
 // the tool stays closed until both are connected rather than failing later.
 async function loadSetup(){
  try{
   const response=await endpoint('/api/integrations',{headers:headers()});
   if(!response.ok)return null;
   return await response.json();
  }catch{return null;}
 }
 async function gateOnSetup(){
  setup=await loadSetup();
  if(ready(setup)){unlock();paint();return;}
  showSetup(setup?'':'Could not load your setup. Check your connection and retry.');
 }
 async function refreshIdentity(){
  try{
   const response=await endpoint('/api/auth/config');if(!response.ok)throw Error();
   config=await response.json();
   if(!config.enabled){lock('Access is protected. GitHub sign-in is awaiting setup.');return;}
   const saved=localStorage.getItem(key);
   if(saved){
    const me=await endpoint('/api/auth/me',{headers:headers()});
    if(me.ok){
     const body=await me.json();
     // The check extends the session server-side; the token itself is only
     // replaced by a fresh GitHub sign-in, so two open tabs cannot revoke
     // each other's credential.
     user=body.user;paint();await gateOnSetup();return;
    }
    if(me.status===401)localStorage.removeItem(key);
   }
   lock('Sign in with GitHub to use Top-Kernel.');
  }catch{config=null;lock('Access is protected. The sign-in service is unavailable.');}
 }
 function beginSignIn(){
  if(!config?.enabled)return;
  popup=window.open(backend+'/auth/github/login','topk-github-signin','popup,width=600,height=730');
  if(!popup)paint('Allow the sign-in window, then try again.');
 }
 window.addEventListener('message',event=>{
  if(event.origin!==backend||event.source!==popup||event.data?.type!=='topk-github-auth')return;
  if(event.data.error){paint(event.data.error);return;}
  if(typeof event.data.token!=='string'||!event.data.user)return;
  localStorage.setItem(key,event.data.token);user=event.data.user;popup=null;paint();gateOnSetup();
 });

 function field(label,hint,node){
  const wrap=document.createElement('label');
  const title=document.createElement('span');title.className='setup-label';title.textContent=label;
  const note=document.createElement('small');note.textContent=hint;
  wrap.append(title,note,node);return wrap;
 }
 function buildSetup(){
  const panel=document.createElement('section');panel.className='github-gate topk-setup';
  panel.setAttribute('aria-labelledby','topk-setup-title');
  const card=document.createElement('div');card.className='github-gate-card setup-card';
  card.innerHTML='<p class="github-gate-kicker">Top-Kernel</p>'+
   '<h1 id="topk-setup-title">Connect your own GPU</h1>'+
   '<p class="github-gate-copy">Top-Kernel runs on a notebook you own, and reports to your own Weights &amp; Biases account. Nothing here is shared with other accounts.</p>'+
   '<ol class="setup-steps">'+
   '<li>Create or sign in to a <a href="https://molab.marimo.io" target="_blank" rel="noopener noreferrer">marimo</a> account.</li>'+
   '<li>Start a notebook on a GPU runtime and keep its browser tab open.</li>'+
   '<li>Choose <b>Pair with agent</b> and copy the whole prompt — the token on screen is masked, only the copied text carries it.</li>'+
   '<li>Add your Weights &amp; Biases account so the run reports to you.</li>'+
   '</ol>';
  const form=document.createElement('form');form.className='setup-form';form.noValidate=true;
  const pair=document.createElement('textarea');pair.rows=3;pair.required=true;
  pair.placeholder='Paste the whole "Pair with agent" prompt';
  const wandbKey=document.createElement('input');wandbKey.type='password';wandbKey.required=true;
  wandbKey.autocomplete='off';wandbKey.placeholder='W&B API key';
  const entity=document.createElement('input');entity.type='text';entity.placeholder='W&B entity';
  const project=document.createElement('input');project.type='text';project.placeholder='W&B project';
  const pairRow=field('Your marimo notebook','From the notebook’s "Pair with agent" prompt.',pair);
  const wandbRow=field('Your Weights & Biases account','From wandb.ai/authorize. Stored on the server, never shown again.',wandbKey);
  const names=document.createElement('div');names.className='setup-names';names.append(entity,project);
  const submit=document.createElement('button');submit.type='submit';submit.textContent='Save and continue';
  setupStatus=document.createElement('p');setupStatus.className='github-gate-status';setupStatus.setAttribute('role','status');
  form.append(pairRow,wandbRow,names,submit,setupStatus);
  card.append(form);panel.append(card);
  form.addEventListener('submit',async event=>{
   event.preventDefault();
   if(!pair.value.trim()||!wandbKey.value.trim()){
    setupStatus.textContent='Add the pair prompt and your W&B API key to continue.';return;
   }
   submit.disabled=true;setupStatus.textContent='Connecting…';
   try{
    const response=await endpoint('/api/integrations',{method:'POST',headers:jsonHeaders(),
     body:JSON.stringify({pair_prompt:pair.value.trim(),wandb_api_key:wandbKey.value.trim(),
      wandb_entity:entity.value.trim(),wandb_project:project.value.trim()})},20000);
    const body=await response.json();
    if(!response.ok)throw Error(body.error||'Could not save your setup.');
    setup=body;
    // Never keep the notebook token or the API key in the browser.
    pair.value='';wandbKey.value='';
    if(ready(setup)){panel.remove();setupPanel=null;unlock();paint();return;}
    setupStatus.textContent='Add both your notebook and your W&B account to continue.';
   }catch(error){setupStatus.textContent=error.message||'Could not save your setup.';}
   finally{submit.disabled=false;}
  });
  return panel;
 }
 function showSetup(message=''){
  document.documentElement.classList.remove('topk-auth-pending','topk-authenticated');
  document.documentElement.classList.add('topk-auth-locked');
  if(!setupPanel){setupPanel=buildSetup();document.body.prepend(setupPanel);}
  gate?.remove();gate=null;
  setupPanel.hidden=false;
  if(setupStatus)setupStatus.textContent=message;
 }

 document.addEventListener('DOMContentLoaded',()=>{
  const style=document.createElement('link');style.rel='stylesheet';style.href='/github-auth.css?v=7';document.head.append(style);
  gate=document.createElement('section');gate.className='github-gate';gate.setAttribute('aria-labelledby','github-gate-title');
  gate.classList.toggle('landing-gate',isLanding);
  gate.innerHTML='<div class="github-gate-card"><button type="button" class="github-gate-close" aria-label="Close sign-in">×</button><p class="github-gate-kicker">Top-Kernel</p><h1 id="github-gate-title">Sign in to continue</h1><p class="github-gate-copy">Repository analysis, agent traces, and performance results are private.</p></div>';
 gateButton=document.createElement('button');gateButton.type='button';gateButton.textContent='Checking GitHub sign-in…';gateButton.disabled=true;
  gateStatus=document.createElement('p');gateStatus.className='github-gate-status';gateStatus.setAttribute('role','status');
  gate.querySelector('.github-gate-card').append(gateButton,gateStatus);document.body.prepend(gate);
  if(isLanding)gate.hidden=true;
  gateButton.addEventListener('click',beginSignIn);
  gate.querySelector('.github-gate-close').addEventListener('click',()=>{gate.hidden=true;setStatus('');});

  const repoForm=document.querySelector('#repo-form');
  repoForm?.addEventListener('submit',event=>{
   if(user||!isLanding)return;
   event.preventDefault();event.stopImmediatePropagation();
   gate.hidden=false;paint('Sign in with GitHub to enter.');
   if(config?.enabled)beginSignIn();
  },true);

  const panel=document.createElement('div');panel.className='github-account';
  accountSetup=document.createElement('button');accountSetup.type='button';accountSetup.className='ghost';
  accountSetup.textContent='Notebook setup';accountSetup.hidden=true;
  accountButton=document.createElement('button');accountButton.type='button';
  accountStatus=document.createElement('span');accountStatus.setAttribute('role','status');
  panel.append(accountSetup,accountButton,accountStatus);document.body.prepend(panel);
  accountSetup.addEventListener('click',()=>showSetup(''));
  accountButton.addEventListener('click',async()=>{
   if(!user){beginSignIn();return;}
   try{const r=await endpoint('/api/auth/logout',{method:'POST',headers:headers()});if(!r.ok)throw Error();
    localStorage.removeItem(key);user=null;location.href='/';
   }catch{paint('Could not sign out. Please retry.');}
  });
  paint('Checking access…');refreshIdentity();
 });

 // Recorded demo evidence carries private trace links, so it is fetched from
 // the authenticated API rather than shipped in the public static bundle.
 window.topkRecorded=async()=>{
  if(window.RECORDED_RUNS)return window.RECORDED_RUNS;
  const response=await endpoint('/api/recorded/runs',{headers:headers()});
  if(!response.ok)throw Error('Recorded evidence is unavailable.');
  window.RECORDED_RUNS=await response.json();
  return window.RECORDED_RUNS;
 };

 window.fetch=async(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin!==location.origin||!url.pathname.startsWith('/api/'))return original(input,options);
  const token=localStorage.getItem(key);
  if(!token)return new Response(JSON.stringify({error:'Sign in with GitHub to continue.'}),{status:401,headers:{'Content-Type':'application/json'}});
  const h=new Headers(options.headers);h.set('Authorization','Bearer '+token);
  const limit=deadline(30000);
  let response;
  try{response=await original(backend+url.pathname+url.search,{...options,headers:h,credentials:'omit',signal:options.signal||limit.signal});}
  finally{limit.done();}
  if(response.status===401){localStorage.removeItem(key);user=null;lock('Your session expired. Sign in again.');}
  if(response.status===428){setup=null;showSetup('Connect your notebook and W&B account to start a run.');}
  return response;
 };
})();
