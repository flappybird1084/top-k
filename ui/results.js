'use strict';
const run=new URLSearchParams(location.search).get('run');
document.querySelector('#back').href='workspace.html?run='+encodeURIComponent(run||'');
async function refresh(){
 try{
  const response=await fetch('/api/runs/'+encodeURIComponent(run),{cache:'no-store'});
  if(response.status===401){location.href='/login.html';return}
  if(!response.ok)throw Error('Unable to load run results.');
  const state=await response.json(),bpd=state.bpd_comparison||{};
  const baseline=bpd.baseline,optimized=bpd.kernel;
  const valid=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
  document.querySelector('#baseline').textContent=valid(baseline)?baseline.toFixed(4):'—';
  document.querySelector('#kernel').textContent=valid(optimized)?optimized.toFixed(4):'—';
  document.querySelector('#result-note').textContent=valid(baseline)&&valid(optimized)?'Recorded BPD comparison.':state.status==='complete'?'This run has finished. Baseline and kernel BPD have not both been recorded.':'BPD results will appear when evaluation finishes.';
  document.querySelector('#difference').textContent=valid(baseline)&&valid(optimized)?'Δ BPD '+(optimized-baseline>=0?'+':'')+(optimized-baseline).toFixed(4):'';
 }catch(error){document.querySelector('#result-note').textContent=error.message}
 setTimeout(refresh,5000);
}
if(run)refresh();else document.querySelector('#result-note').textContent='Open results from a run.';
