'use strict';
(() => {
 const backend='https://cancel-permanent-enlarge-sent.trycloudflare.com';
 const original=window.fetch.bind(window);
 const key='top-k-judge-session-v1';
 let session;
 async function token(){
  const saved=localStorage.getItem(key);if(saved)return saved;
  if(!session)session=original(backend+'/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(async r=>{if(!r.ok)throw Error('The live server is unavailable. Please retry.');const data=await r.json();localStorage.setItem(key,data.token);return data.token;}).finally(()=>{session=null});
  return session;
 }
 window.fetch=async(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin!==location.origin||!url.pathname.startsWith('/api/'))return original(input,options);
  const headers=new Headers(options.headers);headers.set('Authorization','Bearer '+await token());
  const response=await original(backend+url.pathname+url.search,{...options,headers,credentials:'omit'});
  if(response.status===401)localStorage.removeItem(key);
  return response;
 };
})();
