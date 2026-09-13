'use strict';
// A presentation clock only. It never changes a recorded metric or launches work.
function recordedReplay(side,elapsed){
 const s=structuredClone(window.RECORDED_RUNS[side]),all=side==='architecture'?s.architecture.candidates:s.candidates;
 const max=side==='architecture'?3:5,slot=24000/max;
 s.rows=all.filter(r=>r.generation>0&&r.generation<=max&&elapsed>=(r.generation-1)*slot).map(r=>{
  const siblings=all.filter(a=>a.generation===r.generation),i=siblings.findIndex(a=>a.id===r.id);
  const finished=elapsed>=(r.generation-1)*slot+slot*(.46+.42*(i+1)/siblings.length);
  return {...r,state:finished?(r.accepted?'improved':r.failure_note&&!r.correct_ok?'failed':'dropped'):'running'};
 });
 if(elapsed>=24000){
  const winner=side==='architecture'?all.filter(r=>r.phase==='finals'&&r.accepted).sort((a,b)=>a.val_loss-b.val_loss)[0]:all.find(r=>r.id==='cand_04_04_3f280b');
  s.final_result={title:side==='architecture'?'Final architecture':'Final kernel',parent_id:side==='architecture'?winner.parent_id:winner.id,description:side==='architecture'?'Parallel block + reduced KV projections + cyclic learning rate. Final retrain: 5.4258 vs 5.8477 validation loss at 300s.':'QKV packing + fused MLP backward + token-pair weight reuse. Generation 5 retained the generation-4 winner; final paired verification passed all correctness checks.',metric:side==='architecture'?'7.21% lower loss':'3.30% lower step time',state:elapsed<26000?'running':'improved',note:side==='architecture'?'300s final retrain':'Retained through Gen 5'};
 }
 s.status=elapsed>=26000?'complete':'running';
 const g=Math.min(max,Math.floor(elapsed/slot)+1);
 s.message=elapsed>=24000?s.message:(side==='architecture'?['Explore architecture proposals.','Recombine the strongest parents.','Tune learning rates and regularization.'][g-1]:['Explore backward kernels.','Evaluate alternative reductions.','Fuse backward work.','Reuse weight panels.','Verify remaining proposals.'][g-1]);
 return s;
}
