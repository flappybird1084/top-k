'use strict';
(() => {
 const backend='https://cancel-permanent-enlarge-sent.trycloudflare.com';
 const original=window.fetch.bind(window), key='top-k-github-session-v1';
 let user=null, popup=null, config=null, accountButton, accountStatus;
 const headers=()=>({'Authorization':'Bearer '+(localStorage.getItem(key)||'')});
 const endpoint=(path,opts={})=>original(backend+path,{...opts,credentials:'omit',signal:AbortSignal.timeout(15000)});
 function paint(message=''){
  if(!accountButton)return;
  accountButton.textContent=user?'Sign out · @'+user.login:'Sign in with GitHub';
  accountStatus.textContent=message;
 }
 async function refreshIdentity(){
  try{
   const response=await endpoint('/api/auth/config');if(!response.ok)throw Error();
   config=await response.json();
   if(!config.enabled){paint('GitHub sign-in is awaiting setup.');return;}
   if(localStorage.getItem(key)){
    const me=await endpoint('/api/auth/me',{headers:headers()});
    if(me.ok)user=(await me.json()).user;else if(me.status===401)localStorage.removeItem(key);
   }
   paint();
  }catch{paint('Sign-in service is unavailable. Please try again later.');}
 }
 window.addEventListener('message',event=>{
  if(event.origin!==backend||event.source!==popup||event.data?.type!=='topk-github-auth')return;
  if(event.data.error){paint(event.data.error);return;}
  if(typeof event.data.token!=='string'||!event.data.user)return;
  localStorage.setItem(key,event.data.token);user=event.data.user;popup=null;paint();
  window.dispatchEvent(new Event('topk-auth-changed'));
 });
 document.addEventListener('DOMContentLoaded',()=>{
  const panel=document.createElement('div');panel.className='github-account';
  accountButton=document.createElement('button');accountButton.type='button';
  accountStatus=document.createElement('span');accountStatus.setAttribute('role','status');
  panel.append(accountButton,accountStatus);document.body.prepend(panel);
  const style=document.createElement('link');style.rel='stylesheet';style.href='/github-auth.css';document.head.append(style);
  accountButton.addEventListener('click',async()=>{
   if(user){
    try{const r=await endpoint('/api/auth/logout',{method:'POST',headers:headers()});if(!r.ok)throw Error();
     localStorage.removeItem(key);user=null;location.href='/';
    }catch{paint('Could not sign out. Please retry.');}return;
   }
   popup=window.open(backend+'/auth/github/login','topk-github-signin','popup,width=600,height=730');
   if(!popup)paint('Allow the sign-in window, then try again.');
  });
  paint();refreshIdentity();
 });
 window.fetch=async(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin!==location.origin||!url.pathname.startsWith('/api/'))return original(input,options);
  const token=localStorage.getItem(key);
  if(!token)return new Response(JSON.stringify({error:'Sign in with GitHub to continue.'}),{status:401,headers:{'Content-Type':'application/json'}});
  const h=new Headers(options.headers);h.set('Authorization','Bearer '+token);
  const response=await original(backend+url.pathname+url.search,{...options,headers:h,credentials:'omit'});
  if(response.status===401){localStorage.removeItem(key);user=null;paint('Your session expired. Sign in again.');}
  return response;
 };
})();
