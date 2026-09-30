const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const replay=require('../ui/project-replay-model.js');
const projects=JSON.parse(fs.readFileSync('ui/assets/project-replays.json','utf8')).projects;

test('replays reveal measurements only after completion and never change evidence',()=>{
  for(const project of projects){
    const original=JSON.stringify(project), events=replay.schedule(project);
    const first=replay.snapshot(project,0);
    assert.equal(first.complete,false);
    for(const mode of Object.values(first.modes)){assert.deepEqual(mode.candidates,[]);assert.deepEqual(mode.result,{});}
    for(const event of events){
      const working=replay.snapshot(project,event.start+1).modes[event.side];
      if(event.row){
        const row=working.candidates.find(r=>r.ordinal===event.row.ordinal);
        assert.equal(row.state,'running');
        for(const key of ['accepted','val_loss','step_time_ms','correct_ok'])assert.equal(row[key],undefined);
        const done=replay.snapshot(project,event.end).modes[event.side].candidates.find(r=>r.ordinal===event.row.ordinal);
        for(const [key,value] of Object.entries(event.row))assert.deepEqual(done[key],value);
      }
    }
    const final=replay.snapshot(project,replay.duration);
    assert.equal(final.complete,true);
    for(const side of ['architecture','kernel'])assert.deepEqual(final.modes[side].result,project.modes[side].result);
    assert.equal(JSON.stringify(project),original);
  }
});

test('generation playback is sequential across domains, and zero loss is retained',()=>{
  for(const project of projects){const events=replay.schedule(project);events.slice(1).forEach((e,i)=>assert.equal(e.start,events[i].end));}
  const lit=projects.find(p=>p.repo==='Lightning-AI/litgpt');
  assert.equal(replay.snapshot(lit,replay.duration).modes.architecture.result.candidate_val_loss,0);
  assert.equal(lit.caveat,'loss-floor');
});

test('different project replays retain their own values and missing domains',()=>{
  const timm=projects.find(p=>p.name==='timm'),nano=projects.find(p=>p.name==='nanochat');
  assert.notEqual(timm.modes.kernel.result.baseline_ms,nano.modes.kernel.result.baseline_ms);
  assert.equal(nano.modes.architecture.result.measured,false);
  assert.equal(replay.snapshot(nano,replay.duration).modes.architecture.candidates.length,0);
});

test('chart series include every measured point once and separate training budgets',()=>{
  for(const project of projects){
    const snapshot=replay.snapshot(project,replay.duration);
    for(const side of ['architecture','kernel']){
      const mode=snapshot.modes[side], metric=side==='architecture'?'val_loss':'step_time_ms';
      const groups=replay.series(mode,side);
      assert.deepEqual(groups.flatMap(g=>g.rows).map(r=>r.ordinal).sort(),mode.candidates.filter(r=>Number.isFinite(r[metric])).map(r=>r.ordinal).sort());
      for(const group of groups)if(side==='architecture')assert.ok(group.rows.every(r=>r.train_secs===group.budget));
    }
    const initial=replay.snapshot(project,0);
    assert.ok(replay.series(initial.modes.architecture,'architecture').every(g=>g.rows.length===0));
    assert.equal(initial.modes.architecture.baseline.baseline_val_loss,project.modes.architecture.result.baseline_val_loss);
    assert.deepEqual(initial.modes.architecture.result,{});
    const pair=replay.runPair(project,replay.duration);
    for(const side of ['architecture','kernel']){
      assert.equal(pair[side].rows.length,project.modes[side].candidates.length);
      if(pair[side].final_result?.parent_id)assert.ok(pair[side].rows.some(r=>r.id===pair[side].final_result.parent_id&&r.accepted));
      assert.ok(pair[side].rows.every(r=>r.parent_id===undefined));
    }
  }
});

test('graph completion uses the rendered snapshot boundary',()=>{
 const project=projects.find(p=>p.name==='timm');
 assert.ok(Object.values(replay.runPair(project,replay.duration-1)).every(s=>!s.replay_complete));
 assert.ok(Object.values(replay.runPair(project,replay.duration)).every(s=>s.replay_complete&&s.final_result));
});
