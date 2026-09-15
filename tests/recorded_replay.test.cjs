const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={window:{},structuredClone};vm.createContext(context);
for(const file of ['ui/assets/recorded-data.js','ui/assets/replay.js','ui/demo.js'])vm.runInContext(fs.readFileSync(file,'utf8'),context);
const before=JSON.stringify(context.window.RECORDED_RUNS);
for(const side of ['architecture','kernel']){
 const get=t=>vm.runInContext(`recordedReplay('${side}',${t})`,context);
 assert.ok(get(0).rows.every(r=>r.generation===1&&r.state==='running'));
 const slot=24000/(side==='architecture'?3:5);
 const next=get(slot+1);assert.ok(next.rows.some(r=>r.generation===2&&r.state==='running'));
 assert.ok(next.rows.filter(r=>r.generation===1).every(r=>r.state!=='running'));
 assert.equal(get(26000).status,'complete');assert.equal(get(23999).final_result,undefined);assert.equal(get(24000).final_result.state,'running');assert.equal(get(26000).final_result.state,'improved');assert.ok(get(24000).rows.every(r=>r.state!=='running'));
 assert.equal(JSON.stringify(vm.runInContext(`demoSnapshot('${side}',0)`,context)),JSON.stringify(vm.runInContext(`demoSnapshot('${side}',99)`,context)));
}
assert.equal(JSON.stringify(context.window.RECORDED_RUNS),before);
console.log('Recorded playback sequencing and immutable results passed');

assert.equal(vm.runInContext("recordedReplay('architecture',26000).final_result.parent_id",context),24);
assert.equal(vm.runInContext("recordedReplay('kernel',26000).final_result.parent_id",context),'cand_04_04_3f280b');
