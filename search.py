"""Owns search orchestration; GPU gates execute serially in disposable processes."""
import argparse
import ast
import concurrent.futures
import inspect
import json
import os
import signal
import threading
import sys
import time
import traceback
import uuid
from pathlib import Path
from config import active_config
from kernel_evolution.archive import Archive,source_hash
from kernel_evolution.budget import Budget,BudgetExceeded
from kernel_evolution.fixtures import fixture
from kernel_evolution.llm import CodexOAuthLLM,JOB_SCHEMA,SOURCE_SCHEMA,LESSON_SCHEMA
from kernel_evolution.observe import Mirror,op
from kernel_evolution.processes import run_worker


def emit(archive,kind,**payload):
    archive.event(kind,payload)
    print(json.dumps(dict(event=kind,**payload),default=str),flush=True)


def load_prepared(args,cfg,archive):
    root=Path(args.run_dir)
    cached=root/'prepared.json'
    if cached.exists():
        prepared=json.loads(cached.read_text())
        if prepared['model']['adapter_path']!=args.adapter:
            raise ValueError('Run directory belongs to another adapter')
        if prepared['config'].get('benchmark_protocol')!=cfg['benchmark_protocol']:
            raise ValueError('Benchmark protocol changed. Preserve this archive and prepare a new run directory; '
                             'old calibration and acceptances cannot be reused with a different baseline.')
        # Profile, precision, and calibration identity cannot silently change on resume.
        for key in ['seed','batch_size','dtype','allowed_ops','min_pct_step_time']:
            if prepared['config'][key]!=cfg[key]: raise ValueError('Prepared configuration differs: '+key)
        return prepared
    emit(archive,'prepare_started',adapter=args.adapter,lineage=args.lineage)
    request=dict(action='prepare',adapter=args.adapter,config=cfg,run_dir=str(root),lineage=args.lineage)
    result=run_worker('kernel_evolution.gpu_worker',request,root/'workers',cfg['run_wallclock_s'])
    if result.get('status')!='ready':
        emit(archive,'prepare_failed',result=result)
        raise RuntimeError(result.get('failure_note',result.get('status')))
    cached.write_text(json.dumps(result,indent=2))
    (root/'targets.json').write_text(json.dumps(result['targets'],indent=2))
    emit(archive,'prepare_finished',targets=[t['id'] for t in result['targets']],calibration=result['calibration'])
    return result


def register(prepared,archive):
    model=prepared['model']
    cfg=prepared['config']
    archive.put('models',**model)
    archive.put('calibration',model_id=model['id'],noise_spread=prepared['calibration']['noise_spread'],
        gate3_margin=cfg['gate3_margin'],gate4_margin=cfg['gate4_margin'],tol_json=json.dumps({'rtol':cfg['rtol'],'atol':cfg['atol']}),
        cheats_rejected=1,details_json=json.dumps(prepared['calibration']))
    if archive.rows('SELECT id FROM lineages'): return
    for t in prepared['targets']:
        cid='seed_'+t['id']
        archive.put('lineages',id=t['id'],model_id=model['id'],op_name=t['op'],shapes_json=json.dumps(t['shapes']),
            pct_step_time=t['pct_step_time'],incumbent_id=None if t.get('fusion_of') else cid,
            fusion_of=t.get('fusion_of'),call_sites=t.get('call_sites'),barren_generations=0,retired=0)
        if t.get('fusion_of'):continue
        archive.put('candidates',id=cid,lineage_id=t['id'],generation=0,strategy='Inductor generation-zero incumbent',
            source_kind='inductor',code_path=t['seed_paths'][0],source_hash=source_hash(Path(t['seed_paths'][0]).read_text()),
            model_name='inductor',gate_reached=2,compile_ok=1,correct_ok=1,accepted=1,created_at=time.time(),
            step_time_ms=prepared['step_time_ms'],details_json=json.dumps({'seed_paths':t['seed_paths'],'baseline':True}))


def incumbents(archive):
    return {r['id']:r['code_path'] if r['source_kind']!='inductor' else None for r in archive.rows(
        'SELECT l.id,c.code_path,c.source_kind FROM lineages l JOIN candidates c ON c.id=l.incumbent_id')}


def validated_jobs(raw,archive,generation,limit):
    active={r['id']:r for r in archive.rows('SELECT * FROM lineages WHERE retired=0')}
    selected=[]
    for job in raw.get('jobs',[])[:limit]:
        lineage=job.get('lineage')
        if lineage not in active: raise ValueError('Planner named unknown/retired lineage: '+str(lineage))
        fusion_of=active[lineage].get('fusion_of')
        fusion=bool(fusion_of) or str(job.get('strategy','')).lstrip().upper().startswith('FUSE:')
        default_parent=active[lineage]['incumbent_id']
        if default_parent is None and fusion_of:
            default_parent=archive.rows('SELECT incumbent_id FROM lineages WHERE id=?',(fusion_of,))[0]['incumbent_id']
        parents=job.get('parents') or [default_parent]
        if fusion and generation<2:raise ValueError('Fusion requires generation >=2')
        if fusion and (not fusion_of or (active[lineage].get('call_sites') or 0)<2):
            raise ValueError('Fusion requires a discovered executable contract with at least two call sites')
        for parent in parents:
            rows=archive.rows('SELECT * FROM candidates WHERE id=?',(parent,))
            if not rows or not rows[0]['correct_ok']: raise ValueError('Invalid parent: '+parent)
            if fusion and not rows[0]['accepted']: raise ValueError('Fusion requires accepted parents')
            if rows[0]['lineage_id'] not in ({lineage,fusion_of} if fusion else {lineage}):
                raise ValueError('Parent is outside this executable operation/fusion contract')
        selected.append(dict(lineage=lineage,strategy=str(job['strategy']),parents=parents,fusion=fusion))
    if not selected: raise ValueError('Planner returned no valid jobs')
    return selected


def source_prompt(job,archive,targets):
    target=next(t for t in targets if t['id']==job['lineage'])
    parents=[]
    for cid in job['parents']:
        p=archive.rows('SELECT * FROM candidates WHERE id=?',(cid,))[0]
        parents.append(dict(id=cid,source=Path(p['code_path']).read_text()[:45000],latency_us=p['latency_us']))
    reference=Path(__file__).parent/'kernel_evolution/ops.py'
    manifest=archive.path.parent/'references/manifest.json'
    reference_snippets=json.loads(manifest.read_text()) if manifest.exists() else []
    return json.dumps(dict(task='Implement exactly the named strategy as one Python source file exposing kernel(*args). '
        'Use Triton for computation; torch may allocate tensors and support autograd scaffolding. No output caching, file I/O, '
        'torch.compile, subprocesses, hardcoded batch sizes, or calls back into the harness. Support variable batch dimension. '
        'For layer_norm_backward: kernel(x,dy,weight,mean,rstd) returns (dx,dweight,dbias). Inputs x,dy have shape [...,D]; '
        'mean,rstd are saved forward statistics, flattened in row order. xhat=(x-mean)*rstd. '
        'dx=rstd*(dy*w-mean(dy*w)-xhat*mean(dy*w*xhat)); dweight=sum(dy*xhat); dbias=sum(dy), sums across rows. '
        'These returned tensors ARE the gradients; no second-order derivative is requested. For other operations expose '
        'autograd through all differentiable inputs. For ema_update the contract is kernel(target,source,decay)->Tensor; '
        'return target*decay + source*(1-decay), preserve shape/dtype, and do not mutate inputs. '
        'For fused_ema_update: kernel(*args) takes target0,source0,target1,source1,...,decay; '
        'returns a tuple with one EMA result per target/source pair in order. All pairs are independent. '
        'Fuse their computation into Triton launches; preserve each shape/dtype and never mutate inputs. '
        'The number of pairs is the number of selected call sites, not a single tensor dimension. '
        'Return JSON with source only. Do not execute tools or grade your own output.',
        strategy=job['strategy'],shapes=target['shapes'],parents=parents,
        eager_reference=reference.read_text()[:4000],lessons=archive.lessons(),fetched_triton_reference=reference_snippets,
        triton_reference='tl.load(ptr, mask, other); tl.store(ptr,value,mask); tl.arange(0,B) requires power-of-two B; '
        'tl.sum(x,axis); tl.dot(a,b,acc); tl.make_block_ptr(base,shape,strides,offsets,block_shape,order). '
        'Mask padded elements before reductions. Reduce fp32. tl.program_id(axis) selects the program; '
        'triton.cdiv and triton.next_power_of_2 are host helpers. See https://triton-lang.org/main/python-api/triton.language.html'))


@op
def implement(job,index,generation,archive,cfg,root,targets,deadline):
    cid=f'cand_{generation:02d}_{index:02d}_{uuid.uuid4().hex[:6]}'
    model=subagent_model(cfg,generation,index)
    if cfg['llm']=='stub': source,kind=fixture((generation-1)*cfg['candidates_per_gen']+index,job['lineage'])
    else:
        llm=CodexOAuthLLM(archive,cfg,root,'subagent',model=model);llm.generation=generation
        source=llm.complete([{'role':'user','content':source_prompt(job,archive,targets)}],json_mode=True,
            schema=SOURCE_SCHEMA,timeout=min(cfg['max_call_seconds'],max(1,deadline-time.monotonic())))['source']
        kind='fusion' if job.get('fusion') else 'mutation'
    path=root/'candidates'/f'{cid}.py'
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return dict(id=cid,lineage_id=job['lineage'],generation=generation,parent_id=job['parents'][0],
        parents_json=json.dumps(job['parents']),strategy=job['strategy'],source_kind=kind,code_path=str(path),
        source_hash=source_hash(source),model_name=model,created_at=time.time())


def subagent_model(cfg,generation,index):
    if cfg['llm']=='stub':return 'stub'
    models=cfg.get('subagent_models') or [cfg['subagent_llm']]
    return models[((generation-1)*cfg['candidates_per_gen']+index)%len(models)]


def save_result(candidate,result,repairs,archive):
    candidate.update({k:v for k,v in result.items() if k in {
        'gate_reached','compile_ok','correct_ok','accepted','failure_note','latency_us','incumbent_latency_us',
        'step_time_ms','incumbent_step_time_ms','samples_per_s','mfu'}})
    candidate['repairs_used']=repairs
    candidate['details_json']=json.dumps(result.get('details',{}))
    archive.put('candidates',**candidate)
    if candidate.get('accepted'):
        archive.execute('UPDATE lineages SET incumbent_id=? WHERE id=?',(candidate['id'],candidate['lineage_id']))
    return candidate


@op
def compile_offline(candidate,cfg,root,prepared,deadline,compile_lock):
    target=next(t for t in prepared['targets'] if t['id']==candidate['lineage_id'])
    gpu_target=prepared['config'].get('gpu_target')
    if not gpu_target:return dict(status='deferred',reason='No recorded GPU compilation target')
    with compile_lock:
        remaining=deadline-time.monotonic()
        if remaining<=0:return dict(status='timeout',failure_note='gen_timeout')
        return run_worker('kernel_evolution.compile_worker',dict(code_path=candidate['code_path'],
            args=target['shapes'][0]['args'],gpu_target=gpu_target,cache_dir=str(root/'compile_cache')),
            root/'compile_workers',remaining,cuda=False)


@op
def verify(candidate,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock):
    repairs=0
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            return save_result(candidate,dict(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,
                                              failure_note='gen_timeout'),repairs,archive)
        compiled=compile_offline(candidate,cfg,root,prepared,deadline,compile_lock)
        if compiled.get('status')=='compile_error':
            result=dict(gate_reached=1,compile_ok=0,correct_ok=0,accepted=0,
                        failure_note=compiled['failure_note'],details={'cpu_compile':compiled})
        elif compiled.get('status')=='timeout':
            return save_result(candidate,dict(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,
                                              failure_note='gen_timeout'),repairs,archive)
        else:
            with gpu_lock:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    return save_result(candidate,dict(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,
                                                      failure_note='gen_timeout'),repairs,archive)
                request=dict(action='evaluate',config=cfg,run_dir=str(root),adapter=args.adapter,
                    op=candidate['lineage_id'],code_path=candidate['code_path'],incumbents=incumbents(archive),
                    flops_per_sample=prepared['model']['flops_per_sample'],peak_flops=prepared['model']['peak_flops'])
                result=run_worker('kernel_evolution.gpu_worker',request,root/'workers',remaining)
                result.setdefault('details',{})['cpu_compile']=compiled
                if result.get('status') in {'timeout','crash'}:
                    result.update(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0)
                    return save_result(candidate,result,repairs,archive)
                if result.get('correct_ok') or repairs>=cfg['max_repairs'] or cfg['llm']=='stub':
                    # Commit the new incumbent before the next GPU worker selects its baseline.
                    return save_result(candidate,result,repairs,archive)
        if repairs>=cfg['max_repairs'] or cfg['llm']=='stub':
            return save_result(candidate,result,repairs,archive)
        llm=CodexOAuthLLM(archive,cfg,root,'subagent',model=candidate['model_name'])
        llm.generation=candidate['generation']
        prompt=json.dumps({'task':'Repair this candidate. Return source only as JSON. Keep the same strategy and signature. '
            'Do not execute commands. Raw external verifier feedback follows.',
            'source':Path(candidate['code_path']).read_text(),'feedback':result.get('failure_note')})
        try:
            remaining=deadline-time.monotonic()
            if remaining<=0:return save_result(candidate,result,repairs,archive)
            repaired=llm.complete([{'role':'user','content':prompt}],json_mode=True,schema=SOURCE_SCHEMA,
                                 timeout=min(cfg['max_call_seconds'],remaining))['source']
        except BudgetExceeded:return save_result(candidate,result,repairs,archive)
        original=Path(candidate['code_path'])
        original.with_suffix(f'.repair{repairs}.py').write_text(original.read_text())
        original.write_text(repaired)
        candidate['source_hash']=source_hash(repaired)
        repairs+=1


@op
def candidate_lifecycle(job,index,generation,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock):
    candidate=implement(job,index,generation,archive,cfg,root,prepared['targets'],deadline)
    return verify(candidate,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock)


def run(args,cfg,archive,prepared):
    root=Path(args.run_dir)
    # Preserve selected providers/budget while applying calibrated numerical/performance gates.
    cfg={**cfg,**{k:prepared['config'][k] for k in ['rtol','atol','gate3_margin','gate4_margin']}}
    archive.put('run_config',created_at=time.time(),config_json=json.dumps(cfg))
    register(prepared,archive)
    mirror=Mirror(cfg,archive)
    budget=Budget(archive,cfg['spend_cap_usd'])
    deadline=time.monotonic()+cfg['run_wallclock_s']
    systemic=0
    previous=archive.rows('SELECT MAX(id) AS n FROM generations')[0]['n'] or 0
    stop=None
    try:
        for gen in range(previous+1,cfg['max_generations']+1):
            if time.monotonic()>=deadline:stop='run_deadline';break
            if budget.spent>=budget.limit:stop='spend_cap_usd';break
            active=archive.rows('SELECT * FROM lineages WHERE retired=0 AND (fusion_of IS NULL OR ? >= 2)',(gen,))
            if not active:stop='all_lineages_retired';break
            gen_deadline=min(deadline,time.monotonic()+cfg['gen_wallclock_s'])
            archive.put('generations',id=gen,model_id=args.adapter,started_at=time.time(),n_candidates=0,n_accepted=0,llm_usd=budget.spent)
            emit(archive,'generation_started',generation=gen,usd=budget.spent)
            results=[]
            try:
                if cfg['llm']=='stub':
                    jobs=[dict(lineage=active[i%len(active)]['id'],strategy=['Triton fixture','Broken compiler API','Incorrect gradient','Cached-output cheat'][((gen-1)*cfg['candidates_per_gen']+i)%4],
                        parents=[active[i%len(active)]['incumbent_id']]) for i in range(cfg['candidates_per_gen'])]
                else:
                    planner=CodexOAuthLLM(archive,cfg,root,'planner');planner.generation=gen
                    cached_plan=archive.rows('SELECT results_json FROM search_cache WHERE query=?',(f'recovery_planner:{gen}',))
                    raw=json.loads(cached_plan[0]['results_json']) if cached_plan else planner.complete([{'role':'user','content':json.dumps(dict(
                        task=f'Propose {cfg["candidates_per_gen"]} distinct specific Triton optimization strategies. '
                        'Use existing lineage IDs and parent IDs. From generation 2, targets with fusion_of support FUSE jobs '
                        'across the listed call sites; one accepted parent may seed multiple call sites. '
                        'Do not fuse unrelated operations without a listed executable contract. '
                        'You may use native web search for prior art. JSON only.',
                        generation=gen,targets=prepared['targets'],archive=archive.summary(gen),lessons=archive.lessons()))}],
                        json_mode=True,schema=JOB_SCHEMA,timeout=min(cfg['max_call_seconds'],gen_deadline-time.monotonic()))
                    jobs=validated_jobs(raw,archive,gen,cfg['candidates_per_gen'])
                gpu_lock=threading.Lock()
                compile_lock=threading.BoundedSemaphore(cfg['compile_workers'])
                with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['llm_concurrency']) as pool:
                    futures={pool.submit(candidate_lifecycle,j,i,gen,archive,cfg,root,args,prepared,gen_deadline,gpu_lock,compile_lock):i for i,j in enumerate(jobs)}
                    for f in concurrent.futures.as_completed(futures):
                        i=futures[f]
                        try:c=f.result()
                        except Exception as exc:
                            c=dict(id=f'failed_{gen}_{i}',lineage_id=jobs[i]['lineage'],generation=gen,
                                strategy=jobs[i]['strategy'],source_kind='mutation',model_name=cfg.get('subagent_llm','stub'),
                                gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,failure_note=str(exc),created_at=time.time())
                            archive.put('candidates',**c)
                        results.append(c)
                        emit(archive,'candidate_result',id=c['id'],generation=gen,gate=c.get('gate_reached'),accepted=c.get('accepted'),
                            failure_note=c.get('failure_note'),step_time_ms=c.get('step_time_ms'),usd=budget.spent)
                        mirror.log({f'candidate/{k}':v for k,v in c.items() if isinstance(v,(int,float))})
                        if c.get('accepted'):mirror.log({'artifact':c['code_path'],'name':c['lineage_id']})
                if time.monotonic()<gen_deadline and budget.spent<budget.limit:
                    if cfg['llm']=='stub':lessons=['Fixture outcomes are determined by the verifier, not fixture labels.']
                    else:
                        curator=CodexOAuthLLM(archive,cfg,root,'curator');curator.generation=gen
                        lessons=curator.complete([{'role':'user','content':json.dumps(dict(task='Write 2–5 concise evidence-based lessons. '
                            'Do not infer performance from compilation failures. Return JSON lessons.',results=results))}],
                            json_mode=True,schema=LESSON_SCHEMA,timeout=min(cfg['max_call_seconds'],gen_deadline-time.monotonic()))['lessons']
                    for lesson in lessons[:5]:archive.put('lessons',generation_id=gen,model_id=args.adapter,text=lesson)
            except BudgetExceeded:
                stop='spend_cap_usd'
            except Exception:
                emit(archive,'generation_error',generation=gen,error=traceback.format_exc())
                if not results:stop='planning_error'
            for line in active:
                if not any(r['lineage_id']==line['id'] and r.get('gate_reached',0)>=1 for r in results):continue
                accepted=any(r.get('accepted') for r in results if r['lineage_id']==line['id'])
                barren=0 if accepted else line['barren_generations']+1
                archive.execute('UPDATE lineages SET barren_generations=?,retired=? WHERE id=?',
                    (barren,int(barren>=cfg['retire_after']),line['id']))
            attempts=[r for r in results if r.get('gate_reached',0)>=1]
            if attempts:systemic=0 if any(r.get('compile_ok') for r in attempts) else systemic+1
            if systemic>=cfg['systemic_halt_after']:stop=stop or 'systemic_halt'
            if gen==cfg['max_generations']:stop=stop or 'max_generations'
            if not archive.rows('SELECT id FROM lineages WHERE retired=0'):stop=stop or 'all_lineages_retired'
            if budget.spent>=budget.limit:stop='spend_cap_usd'
            archive.execute('UPDATE generations SET finished_at=?,n_candidates=?,n_accepted=?,llm_usd=?,stop_reason=? WHERE id=?',
                (time.time(),len(results),sum(bool(r.get('accepted')) for r in results),budget.spent,stop,gen))
            mirror.log({'generation':gen,'generation/accepted':sum(bool(r.get('accepted')) for r in results),'llm/api_equivalent_usd':budget.spent})
            emit(archive,'generation_finished',generation=gen,accepted=sum(bool(r.get('accepted')) for r in results),usd=budget.spent,stop_reason=stop)
            if stop:break
    finally:
        last=archive.rows('SELECT MAX(id) AS n FROM generations')[0]['n']
        if last:archive.execute('UPDATE generations SET stop_reason=? WHERE id=?',(stop or 'interrupted',last))
        emit(archive,'run_finished',stop_reason=stop or 'interrupted',usd=budget.spent,wandb_url=mirror.url)
        mirror.finish()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',default='adapters.jepa')
    p.add_argument('--profile',choices=['DEV','RUN'],default='DEV')
    p.add_argument('--llm',choices=['stub','codex_oauth'])
    p.add_argument('--lineage')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--candidates',type=int)
    args=p.parse_args()
    os.environ['KERNEL_EVOLUTION_PROFILE']=args.profile
    cfg=active_config()
    if args.llm:cfg['llm']=args.llm
    if args.candidates:cfg['candidates_per_gen']=args.candidates
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True);args.run_dir=str(root)
    cfg['run_name']=root.name
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache')
    archive=Archive(root/'archive.sqlite')
    if not archive.rows('SELECT id FROM run_config'):
        archive.put('run_config',created_at=time.time(),config_json=json.dumps(cfg))
    from kernel_evolution.references import fetch_references
    try:fetch_references(root)
    except Exception as exc:archive.event('reference_fetch_failed',{'error':str(exc)})
    prepared=load_prepared(args,cfg,archive)
    register(prepared,archive)
    if not args.prepare_only:run(args,cfg,archive,prepared)


if __name__=='__main__': main()
