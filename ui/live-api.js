'use strict';
(() => {
 const backend='https://top-k.andredlcruz.com';
 const original=window.fetch.bind(window), key='top-k-github-session-v1';
 let user=null, popup=null, config=null, accountButton, accountStatus, gate, gateButton, gateStatus;
 const criticalStyle=document.createElement('style');
 criticalStyle.textContent='html.topk-auth-pending body>*,html.topk-auth-locked body>*{visibility:hidden!important}html.topk-auth-pending body>.github-gate,html.topk-auth-locked body>.github-gate{visibility:visible!important}';
 document.head.append(criticalStyle);
 document.documentElement.classList.add('topk-auth-pending');

 const headers=()=>({'Authorization':'Bearer '+(localStorage.getItem(key)||'')});
 const endpoint=(path,opts={})=>original(backend+path,{...opts,credentials:'omit'});
 function setStatus(message=''){
  if(accountStatus)accountStatus.textContent=message;
  if(gateStatus)gateStatus.textContent=message;
 }
 function paint(message=''){
  if(accountButton)accountButton.textContent=user?'Sign out · @'+user.login:'Sign in with GitHub';
  if(gateButton){gateButton.disabled=!config?.enabled;gateButton.textContent=config?.enabled?'Sign in with GitHub':'GitHub sign-in unavailable';}
  setStatus(message);
 }
 function unlock(){
  document.documentElement.classList.remove('topk-auth-pending','topk-auth-locked');
  document.documentElement.classList.add('topk-authenticated');
  gate?.remove();
  window.dispatchEvent(new Event('topk-auth-changed'));
 }
 function lock(message){
  document.documentElement.classList.remove('topk-auth-pending','topk-authenticated');
  document.documentElement.classList.add('topk-auth-locked');
  paint(message);
 }
 async function refreshIdentity(){
  try{
   const response=await endpoint('/api/auth/config');if(!response.ok)throw Error();
   config=await response.json();
   if(!config.enabled){lock('Access is protected. GitHub sign-in is awaiting setup.');return;}
   const saved=localStorage.getItem(key);
   if(saved){
    const me=await endpoint('/api/auth/me',{headers:headers()});
    if(me.ok){user=(await me.json()).user;unlock();paint();return;}
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
  localStorage.setItem(key,event.data.token);user=event.data.user;popup=null;unlock();paint();
 });
 document.addEventListener('DOMContentLoaded',()=>{
  const style=document.createElement('link');style.rel='stylesheet';style.href='/github-auth.css?v=2';document.head.append(style);
  gate=document.createElement('section');gate.className='github-gate';gate.setAttribute('aria-labelledby','github-gate-title');
  gate.innerHTML='<div class="github-gate-card"><span class="github-gate-mark" aria-hidden="true">K</span><p class="github-gate-kicker">Top-Kernel</p><h1 id="github-gate-title">Sign in to continue</h1><p class="github-gate-copy">Repository analysis, agent traces, and performance results are private.</p></div>';
  gateButton=document.createElement('button');gateButton.type='button';gateButton.textContent='Checking GitHub sign-in…';gateButton.disabled=true;
  gateStatus=document.createElement('p');gateStatus.className='github-gate-status';gateStatus.setAttribute('role','status');
  gate.querySelector('.github-gate-card').append(gateButton,gateStatus);document.body.prepend(gate);
  gateButton.addEventListener('click',beginSignIn);

  const panel=document.createElement('div');panel.className='github-account';
  accountButton=document.createElement('button');accountButton.type='button';
  accountStatus=document.createElement('span');accountStatus.setAttribute('role','status');
  panel.append(accountButton,accountStatus);document.body.prepend(panel);
  accountButton.addEventListener('click',async()=>{
   if(!user){beginSignIn();return;}
   try{const r=await endpoint('/api/auth/logout',{method:'POST',headers:headers()});if(!r.ok)throw Error();
    localStorage.removeItem(key);user=null;location.href='/';
   }catch{paint('Could not sign out. Please retry.');}
  });
  paint('Checking access…');refreshIdentity();
 });
 window.fetch=async(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin!==location.origin||!url.pathname.startsWith('/api/'))return original(input,options);
  const token=localStorage.getItem(key);
  if(!token)return new Response(JSON.stringify({error:'Sign in with GitHub to continue.'}),{status:401,headers:{'Content-Type':'application/json'}});
  const h=new Headers(options.headers);h.set('Authorization','Bearer '+token);
  const response=await original(backend+url.pathname+url.search,{...options,headers:h,credentials:'omit'});
  if(response.status===401){localStorage.removeItem(key);user=null;lock('Your session expired. Sign in again.');}
  return response;
 };
})();
