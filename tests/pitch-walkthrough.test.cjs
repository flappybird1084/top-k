const {test} = require('node:test');
const assert = require('node:assert/strict');
const demo = require('../ui/pitch-model.js');

test('navigation is bounded, direct selection works, and restart clears exploration', () => {
  let state = demo.initialState();
  state = demo.transition(state, {type:'back'});
  assert.equal(state.step, 0);
  for (let i=0;i<10;i++) state=demo.transition(state,{type:'next'});
  assert.equal(state.step, 3);
  state=demo.transition(state,{type:'step',value:2});
  state=demo.transition(state,{type:'generation',value:2});
  state=demo.transition(state,{type:'select',value:10});
  assert.equal(state.selectedId,10);
  assert.equal(state.generation,2);
  state=demo.transition(state,{type:'restart'});
  assert.deepEqual(state,demo.initialState());
});

test('phase changes choose a valid budget and candidate; skipped phase has no fake results', () => {
  let state=demo.transition(demo.initialState(),{type:'phase',value:'hyperparam'});
  assert.equal(state.budget,120);
  assert.equal(state.selectedId,17);
  assert.equal(demo.visibleCandidates(state).length,8);
  state=demo.transition(state,{type:'phase',value:'mixed'});
  assert.equal(demo.visibleCandidates(state).length,0);
  assert.equal(demo.snapshot(state).architecture.candidates.length,0);
});

test('snapshot includes only reached phases and comparisons keep budget-specific baselines', () => {
  const early=demo.snapshot(demo.initialState()).architecture.candidates;
  assert.equal(early.filter(r=>r.phase==='architecture').length,16);
  assert.equal(early.some(r=>r.phase==='hyperparam'||r.phase==='finals'),false);
  const late=demo.snapshot(demo.transition(demo.initialState(),{type:'phase',value:'hyperparam'})).architecture.candidates;
  assert.equal(late.find(r=>r.phase==='baseline'&&r.train_secs===120).val_loss,3.85);
  assert.equal(late.filter(r=>r.phase==='hyperparam').length,8);
});

test('every parent exists earlier and finals rank fresh final losses', () => {
  const rows=demo.rows;
  for(const row of rows.filter(r=>r.parent_id)) {
    const parent=rows.find(r=>r.id===row.parent_id);
    assert.ok(parent);
    assert.ok(parent.generation<row.generation);
  }
  assert.equal(demo.finalists()[0].id,25);
  assert.equal(demo.finalists()[0].val_loss,3.34);
  assert.equal(demo.finalists()[0].train_secs,300);
  assert.equal(demo.finalists()[1].val_loss,3.38);
});

test('invalid selections do not corrupt the walkthrough and snapshots cannot mutate fixtures', () => {
  const state=demo.initialState();
  assert.deepEqual(demo.transition(state,{type:'phase',value:'bogus'}),state);
  assert.deepEqual(demo.transition(state,{type:'step',value:NaN}),state);
  assert.deepEqual(demo.transition(state,{type:'select',value:999}),state);
  const copy=demo.snapshot(state);copy.architecture.candidates[0].val_loss=-1;
  assert.equal(demo.snapshot(state).architecture.candidates[0].val_loss,4.12);
});

test('report link uses supplied HTTPS URL and rejects unsafe destinations', () => {
  assert.equal(demo.safeReportURL(demo.reportURL),'https://wandb.ai/rianbutala-ucla/kernel-evolution/reports/Evolution-of-b3c7b7aa--VmlldzoxNzkyNTIyMw==');
  for(const url of ['', 'javascript:alert(1)','http://example.com','https://user:secret@example.com']) assert.equal(demo.safeReportURL(url),null);
});
