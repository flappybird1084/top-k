/* Deterministic illustrative data. No training, network calls, or observability writes. */
(function(root, factory) {
  const api=factory();
  if(typeof module==='object' && module.exports) module.exports=api;
  else root.PitchDemo=api;
})(typeof window==='object'?window:globalThis, function() {
  'use strict';
  const phases=['architecture','mixed','hyperparam','finals'];
  const reportURL='https://wandb.ai/rianbutala-ucla/kernel-evolution/reports/Evolution-of-b3c7b7aa--VmlldzoxNzkyNTIyMw==';
  const strategies=[
    'Replace dense MLP with SwiGLU','Tie input and output embeddings','Use RMSNorm before attention','Reduce depth, widen hidden layers',
    'Increase attention head count','Reduce feed-forward expansion','Change residual projection','Use grouped-query attention',
    'Combine RMSNorm with tied embeddings','Refine SwiGLU expansion','Reduce KV projections','Tune residual scaling',
    'Increase hidden width','Share attention projections','Remove intermediate normalization','Balance depth and width',
    'Tune learning rate + cosine decay','Tune warmup + weight decay','Reduce AdamW beta₂','Increase gradient clipping',
    'Use constant learning rate','Lengthen warmup','Change optimizer implementation','Tune decay + clipping',
    'Warmup + weight decay finalist','Cosine-decay finalist'
  ];
  const losses=[3.91,3.78,3.67,4.24,4.18,3.96,null,4.08,3.64,3.58,3.62,3.72,4.19,3.80,4.15,3.69,3.51,3.49,3.62,3.54,3.96,3.65,null,3.57,3.34,3.38];
  const parents=[null,null,null,null,null,null,null,null,3,1,8,3,4,2,5,8,10,10,11,9,16,10,11,9,18,17];
  const baseline={60:4.12,120:3.85,300:3.66};
  const rows=losses.map((loss,index)=>{
    const id=index+1,phase=id<=16?'architecture':id<=24?'hyperparam':'finals',budget=id<=16?60:id<=24?120:300;
    return Object.freeze({id,phase,generation:id<=8?1:id<=16?2:id<=24?3:4,strategy:strategies[index],train_secs:budget,
      val_loss:loss,parent_id:parents[index],model_params:23400000,model_name:['gpt','claude','glm'][index%3],
      accepted:loss!==null && loss<baseline[budget]*.997?1:0,correct_ok:loss!==null?1:0,
      failure_note:loss===null?'Shape mismatch persisted after the repair budget; candidate not evaluated.':'',
      repair:id===3?'A shape mismatch was returned to the coding agent. The revised candidate passed its two-step check.':null});
  });
  function initialState(){return {step:0,phase:'architecture',generation:1,selectedId:1,budget:60};}
  function visibleCandidates(state){return rows.filter(r=>r.phase===state.phase && (state.phase!=='architecture'||r.generation===state.generation));}
  function transition(state,action){
    let next={...state};
    if(action.type==='restart')return initialState();
    if(action.type==='next')next.step=Math.min(3,state.step+1);
    if(action.type==='back')next.step=Math.max(0,state.step-1);
    if(action.type==='step' && Number.isInteger(action.value))next.step=Math.max(0,Math.min(3,action.value));
    if(action.type==='phase' && phases.includes(action.value)){
      next.phase=action.value;next.generation=1;next.budget=action.value==='finals'?300:action.value==='hyperparam'?120:60;
      next.selectedId=visibleCandidates(next)[0]?.id??null;
    }
    if(action.type==='generation' && [1,2].includes(action.value) && state.phase==='architecture'){
      next.generation=action.value;next.selectedId=visibleCandidates(next)[0].id;
    }
    if(action.type==='select' && visibleCandidates(state).some(r=>r.id===action.value))next.selectedId=action.value;
    if(action.type==='budget' && [60,120,300].includes(action.value))next.budget=action.value;
    return next;
  }
  function snapshot(state){
    if(state.phase==='mixed')return {mode:'recipe',architecture:{candidates:[]}};
    const maxBudget=state.phase==='finals'?300:state.phase==='hyperparam'?120:60;
    const bases=Object.entries(baseline).filter(([budget])=>Number(budget)<=maxBudget).map(([budget,loss],i)=>({id:-i-1,phase:'baseline',generation:0,train_secs:Number(budget),val_loss:loss,model_params:23400000,strategy:'Original model + AdamW',accepted:1}));
    return {mode:'recipe',architecture:{candidates:[...bases,...rows.filter(r=>r.train_secs<=maxBudget).map(r=>({...r}))]}};
  }
  function finalists(){return rows.filter(r=>r.phase==='finals').slice().sort((a,b)=>a.val_loss-b.val_loss);}
  function safeReportURL(value){try{const u=new URL(value);return u.protocol==='https:'&&!u.username&&!u.password?u.href:null}catch{return null}}
  return {initialState,transition,visibleCandidates,snapshot,finalists,safeReportURL,rows:Object.freeze(rows),reportURL};
});
