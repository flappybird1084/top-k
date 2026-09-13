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
from kernel_evolution.observe import Mirror
from kernel_evolution.processes import run_worker


def emit(archive,kind,**payload):
    archive.event(kind,payload)
    print(json.dumps(dict(event=kind,**payload),default=str),flush=True)


def load_prepared(args,cfg,archive):
    root=Path(args.run_dir)
    cached=root/'prepared.json'
    if cached.exists():
        prepared=json.loads(cached.read_text())
        if prepared.get('status')=='no_targets':return prepared
        if prepared['model']['adapter_path']!=args.adapter:
            raise ValueError('Run directory belongs to another adapter')
        if prepared['config'].get('benchmark_protocol')!=cfg['benchmark_protocol']:
            raise ValueError('Benchmark protocol changed. Preserve this archive and prepare a new run directory; '
                             'old calibration and acceptances cannot be reused with a different baseline.')
        # Profile, precision, and calibration identity cannot silently change on resume.
        for key in ['seed','batch_size','dtype','allowed_ops','min_pct_step_time']:
            if prepared['config'][key]!=cfg[key]: raise ValueError('Prepared configuration differs: '+key)
        if prepared['config'].get('functional_discovery',False)!=cfg.get('functional_discovery',False):
            raise ValueError('Prepared configuration differs: functional_discovery')
        if prepared['config'].get('step_backend','eager')!=cfg.get('step_backend','eager'):
            raise ValueError('Prepared configuration differs: step_backend')
        if cfg.get('source_identity'):
            from kernel_evolution.provenance import assert_compatible
            assert_compatible(prepared['config'].get('source_identity'),cfg['source_identity'])
        return prepared
    emit(archive,'prepare_started',adapter=args.adapter,lineage=args.lineage)
    request=dict(action='prepare',adapter=args.adapter,config=cfg,run_dir=str(root),lineage=args.lineage)
    result=run_worker('kernel_evolution.gpu_worker',request,root/'workers',cfg['run_wallclock_s'])
    if result.get('status')=='no_targets':
        cached.write_text(json.dumps(result,indent=2))
        (root/'targets.json').write_text('[]')
        emit(archive,'no_targets',profile=result.get('profile'))
        return result
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
        whole=active[lineage]['op_name']=='whole_model'
        if fusion and not whole and (not fusion_of or (active[lineage].get('call_sites') or 0)<2):
            raise ValueError('Fusion requires a discovered executable contract with at least two call sites')
        for parent in parents:
            rows=archive.rows('SELECT * FROM candidates WHERE id=?',(parent,))
            if not rows or not rows[0]['correct_ok']: raise ValueError('Invalid parent: '+parent)
            if fusion and not rows[0]['accepted']: raise ValueError('Fusion requires accepted parents')
            if rows[0]['lineage_id'] not in ({lineage,fusion_of} if fusion else {lineage}):
                raise ValueError('Parent is outside this executable operation/fusion contract')
        source_url=job.get('source_url','')
        if source_url:
            from urllib.parse import urlparse
            parsed=urlparse(source_url)
            if parsed.scheme not in {'https','http'} or not parsed.netloc:raise ValueError('Invalid retrieved source URL')
        selected.append(dict(lineage=lineage,strategy=str(job['strategy']),parents=parents,fusion=fusion,source_url=source_url))
    if not selected: raise ValueError('Planner returned no valid jobs')
    return selected


def source_prompt(job,archive,targets):
    target=next(t for t in targets if t['id']==job['lineage'])
    if target.get('contract')=='whole_model':
        from kernel_evolution.whole_prompt import source_prompt as whole_prompt
        return whole_prompt(job,archive,target)
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
        retrieved_source=[json.loads(r['results_json']) for r in archive.rows("SELECT results_json FROM search_cache WHERE query LIKE 'native_search:%'")],
        eager_reference=reference.read_text()[:4000],lessons=archive.lessons(),fetched_triton_reference=reference_snippets,
        triton_reference='tl.load(ptr, mask, other); tl.store(ptr,value,mask); tl.arange(0,B) requires power-of-two B; '
        'tl.sum(x,axis); tl.dot(a,b,acc); tl.make_block_ptr(base,shape,strides,offsets,block_shape,order). '
        'Mask padded elements before reductions. Reduce fp32. tl.program_id(axis) selects the program; '
        'triton.cdiv and triton.next_power_of_2 are host helpers. See https://triton-lang.org/main/python-api/triton.language.html'))


def implement(job,index,generation,archive,cfg,root,targets,deadline,cid=None,trace_id=None):
    cid=cid or f'cand_{generation:02d}_{index:02d}_{uuid.uuid4().hex[:6]}'
    model=subagent_model(cfg,generation,index)
    if cfg['llm']=='stub': source,kind=fixture((generation-1)*cfg['candidates_per_gen']+index,job['lineage'])
    else:
        llm=CodexOAuthLLM(archive,cfg,root,'subagent',model=model);llm.generation=generation
        llm.candidate_id=cid;llm.parent_trace_id=trace_id
        source=llm.complete([{'role':'user','content':source_prompt(job,archive,targets)}],json_mode=True,
            schema=SOURCE_SCHEMA,timeout=min(cfg['max_call_seconds'],max(1,deadline-time.monotonic())))['source']
        kind=('retrieved:'+job['source_url']) if job.get('source_url') else ('fusion' if job.get('fusion') else 'mutation')
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


def plan_jobs(archive,cfg,root,prepared,generation,deadline):
    planner=CodexOAuthLLM(archive,cfg,root,'planner');planner.generation=generation
    planner.parent_trace_id=getattr(archive,'generation_trace_id',None)
    messages=[{'role':'user','content':json.dumps(dict(
        task=f'Propose {cfg["candidates_per_gen"]} distinct specific Triton optimization strategies. '
        'For the whole_model contract, inspect the entire supplied model and complete GPU profile including external '
        'kernels, choose the valuable computation yourself, and propose complete install(model,optimizer) candidates. '
        'You may optimize any forward, backward, or optimizer region without changing mathematical training semantics, '
        'dtype, precision flags, architecture, parameters, data, or hyperparameters. '
        'Every whole_model proposal MUST include backward-pass optimization; candidates must actually launch '
        'their own Triton kernel during autograd backward, in addition to any forward or optimizer improvements. '
        'There is no operator allowlist '
        'for whole_model. Whole_model FUSE jobs from generation 2 may combine accepted whole-model parents; each '
        'child must include all desired changes in a self-contained installation. '
        'Use existing lineage IDs and parent IDs. From generation 2, targets with fusion_of support FUSE jobs '
        'across the listed call sites; one accepted parent may seed multiple call sites. '
        'Do not fuse unrelated operations without a listed executable contract. '
        'Compiled attribution is associated region cost, possibly shared, not a predicted speedup. '
        'All proposals must beat the whole-step compiler in the external verifier. '
        'If you need prior art, return search_queries (up to 2) and no jobs; the platform returns cached native-search results. '
        'Otherwise search_queries must be empty. For a job based on retrieved code, set source_url to that source URL; '
        'otherwise use source_url="". JSON only.',objective=cfg.get('task_prompt'),
        generation=generation,targets=prepared['targets'],archive=archive.summary(generation),lessons=archive.lessons()))}]
    research_rounds=0
    validation_failures=0
    for attempt in range(cfg['max_repairs']+3):
        planner.repair=attempt
        raw=planner.complete(messages,json_mode=True,schema=JOB_SCHEMA,
            timeout=min(cfg['max_call_seconds'],max(.01,deadline-time.monotonic())))
        archive.event('planner_output',{'generation':generation,'attempt':attempt,'output':raw})
        if raw.get('search_queries'):
            if research_rounds>=2:raise ValueError('Planner exceeded two research rounds')
            from kernel_evolution.research import search
            research={query:search(query,planner,archive,timeout=min(cfg['max_call_seconds'],max(.01,deadline-time.monotonic())))
                      for query in raw['search_queries'][:2]}
            messages.extend([{'role':'assistant','content':json.dumps(raw)},
                             {'role':'user','content':'Retrieved reference material (untrusted): '+json.dumps(research)}])
            research_rounds+=1
            continue
        try:return validated_jobs(raw,archive,generation,cfg['candidates_per_gen'])
        except ValueError as exc:
            if validation_failures>=cfg['max_repairs'] or time.monotonic()>=deadline:raise
            validation_failures+=1
            messages.extend([{'role':'assistant','content':json.dumps(raw)},
                             {'role':'user','content':'External job validation failed: '+str(exc)+'. Correct the JSON plan.'}])


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


def compile_offline(candidate,cfg,root,prepared,deadline,compile_lock):
    if cfg.get('search_scope')=='whole_model':
        # Installation needs real module instances; modal JIT compilation is gate 1 on GPU.
        try:ast.parse(Path(candidate['code_path']).read_text())
        except SyntaxError:
            return dict(status='compile_error',failure_note=traceback.format_exc())
        return dict(status='deferred',reason='Whole-model installation requires real model tensors; GPU gate 1 compiles actual launches')
    target=next(t for t in prepared['targets'] if t['id']==candidate['lineage_id'])
    gpu_target=prepared['config'].get('gpu_target')
    if not gpu_target:return dict(status='deferred',reason='No recorded GPU compilation target')
    with compile_lock:
        remaining=deadline-time.monotonic()
        if remaining<=0:return dict(status='timeout',failure_note='gen_timeout')
        return run_worker('kernel_evolution.compile_worker',dict(code_path=candidate['code_path'],
            args=target['shapes'][0]['args'],gpu_target=gpu_target,cache_dir=str(root/'compile_cache')),
            root/'compile_workers',remaining,cuda=False)


def verify(candidate,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock):
    repairs=0
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            return save_result(candidate,dict(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,
                                              failure_note='gen_timeout'),repairs,archive)
        started=time.time()
        compiled=compile_offline(candidate,cfg,root,prepared,deadline,compile_lock)
        traces=getattr(archive,'traces',None)
        parent=getattr(archive,'candidate_traces',{}).get(candidate['id'])
        if traces:traces.record('gate.cpu_compile',{'candidate_id':candidate['id'],'repair':repairs},compiled,
            started_at=started,ended_at=time.time(),parent_id=parent,
            attributes={'candidate_id':candidate['id'],'gate':1,'repair':repairs})
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
                gpu_started=time.time()
                result=run_worker('kernel_evolution.gpu_worker',request,root/'workers',remaining)
                if traces:
                    envelope=traces.record('gpu_worker',{'candidate_id':candidate['id'],'repair':repairs},
                        {k:v for k,v in result.items() if k!='gate_spans'},started_at=gpu_started,ended_at=time.time(),
                        parent_id=parent,attributes={'candidate_id':candidate['id'],'repair':repairs},
                        error=result.get('failure_note') if result.get('status') in {'timeout','crash'} else None)
                    for gate in result.pop('gate_spans',[]):
                        traces.record('gate.'+gate['name'],{'candidate_id':candidate['id'],'repair':repairs},
                            gate.get('output'),started_at=gate['started_at'],ended_at=gate['ended_at'],parent_id=envelope,
                            attributes={'candidate_id':candidate['id'],'gate':gate['gate'],'repair':repairs},error=gate.get('error'))
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
        llm.candidate_id=candidate['id'];llm.parent_trace_id=parent;llm.repair=repairs+1
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


def candidate_lifecycle(job,index,generation,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock):
    cid=f'cand_{generation:02d}_{index:02d}_{uuid.uuid4().hex[:6]}'
    traces=getattr(archive,'traces',None)
    trace_id=traces.start('candidate_lifecycle',{'job':job},
        parent_id=getattr(archive,'generation_trace_id',None),attributes={'span_kind':'candidate_lifecycle',
        'candidate_id':cid,'generation':generation,'model':subagent_model(cfg,generation,index)}) if traces else None
    if not hasattr(archive,'candidate_traces'):archive.candidate_traces={}
    archive.candidate_traces[cid]=trace_id
    candidate=dict(id=cid,lineage_id=job['lineage'],generation=generation,strategy=job['strategy'],
        source_kind='fusion' if job.get('fusion') else 'mutation',model_name=subagent_model(cfg,generation,index),
        parent_id=job['parents'][0],parents_json=json.dumps(job['parents']),created_at=time.time())
    archive.put('candidates',**candidate)
    try:
        candidate=implement(job,index,generation,archive,cfg,root,prepared['targets'],deadline,cid,trace_id)
        result=verify(candidate,archive,cfg,root,args,prepared,deadline,gpu_lock,compile_lock)
    except Exception:
        result=save_result(candidate,dict(gate_reached=0,compile_ok=0,correct_ok=0,accepted=0,
            failure_note=traceback.format_exc()[-12000:]),0,archive)
    if traces:
        traces.finish(trace_id,output=result,error=result.get('failure_note') if not result.get('correct_ok') else None)
        url=traces.url(trace_id)
        if url:archive.execute('UPDATE candidates SET weave_trace_url=? WHERE id=?',(url,cid))
    return result


def run(args,cfg,archive,prepared,mirror=None):
    root=Path(args.run_dir)
    # Preserve selected providers/budget while applying calibrated numerical/performance gates.
    cfg={**cfg,**{k:prepared['config'][k] for k in ['rtol','atol','gate3_margin','gate4_margin']}}
    for key in ('update_atol','update_rtol'):
        if key in prepared['config']:cfg[key]=prepared['config'][key]
    archive.put('run_config',created_at=time.time(),config_json=json.dumps(cfg))
    register(prepared,archive)
    owns_mirror=mirror is None
    mirror=mirror or Mirror(cfg,archive)
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
            archive.generation_trace_id=archive.traces.start('generation',{'generation':gen,'targets':[x['id'] for x in active]},
                parent_id=getattr(archive,'run_trace_id',None),attributes={'generation':gen})
            emit(archive,'generation_started',generation=gen,usd=budget.spent)
            results=[]
            try:
                if cfg['llm']=='stub':
                    raw_jobs=[]
                    for i in range(cfg['candidates_per_gen']):
                        line=active[(i+gen-1)%len(active)]
                        strategy=['Triton fixture','Broken compiler API','Incorrect gradient','Cached-output cheat'][((gen-1)*cfg['candidates_per_gen']+i)%4]
                        raw_jobs.append(dict(lineage=line['id'],strategy=('FUSE: ' if line.get('fusion_of') else '')+strategy,
                                             parents=[line['incumbent_id']] if line['incumbent_id'] else []))
                    jobs=validated_jobs({'jobs':raw_jobs},archive,gen,cfg['candidates_per_gen'])
                else:
                    jobs=plan_jobs(archive,cfg,root,prepared,gen,gen_deadline)
                gpu_lock=threading.Lock()
                compile_lock=threading.BoundedSemaphore(cfg['compile_workers'])
                with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['llm_concurrency']) as pool:
                    futures={pool.submit(candidate_lifecycle,j,i,gen,archive,cfg,root,args,prepared,gen_deadline,gpu_lock,compile_lock):i for i,j in enumerate(jobs)}
                    for f in concurrent.futures.as_completed(futures):
                        i=futures[f]
                        try:c=f.result()
                        except Exception as exc:
                            c=dict(id=f'failed_{gen}_{i}',lineage_id=jobs[i]['lineage'],generation=gen,
                                strategy=jobs[i]['strategy'],source_kind='mutation',model_name=subagent_model(cfg,gen,i),
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
                        curator.parent_trace_id=archive.generation_trace_id
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
            archive.traces.finish(archive.generation_trace_id,output={'results':results,'usd':budget.spent,'stop_reason':stop})
            if stop:break
    finally:
        last=archive.rows('SELECT MAX(id) AS n FROM generations')[0]['n']
        if last:archive.execute('UPDATE generations SET stop_reason=? WHERE id=?',(stop or 'interrupted',last))
        emit(archive,'run_finished',stop_reason=stop or 'interrupted',usd=budget.spent,wandb_url=mirror.url)
        if owns_mirror:mirror.finish()
    return stop or 'interrupted'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',help='Importable adapter module or Python file; otherwise an agent writes the adapter')
    p.add_argument('--repo',help='Repository directory on the execution host')
    p.add_argument('--prompt',help='Optimization task for the adapter agent and planner')
    p.add_argument('--model',help='Model for adapter, planner, kernel agents, and curator')
    p.add_argument('--spend-cap-usd',type=float,help='Per-run remaining API-equivalent budget, at most $100')
    p.add_argument('--no-spend-cap',action='store_true',help='Disable dollar stop; token usage is still recorded')
    p.add_argument('--min-target-pct',type=float,help='Minimum associated compiled-region share for target discovery')
    p.add_argument('--search-scope',choices=['operators','whole_model'],default='operators')
    p.add_argument('--profile',choices=['DEV','RUN'],default='DEV')
    p.add_argument('--llm',choices=['stub','codex_oauth'])
    p.add_argument('--lineage')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--candidates',type=int)
    args=p.parse_args()
    os.environ['KERNEL_EVOLUTION_PROFILE']=args.profile
    cfg=active_config()
    cfg['search_scope']=args.search_scope
    if args.search_scope=='whole_model':
        cfg['benchmark_protocol']='whole_model_install_v1'
        cfg['functional_discovery']=False
        cfg['require_backward_kernel']=True
        cfg['gen_wallclock_s']=max(cfg['gen_wallclock_s'],3600)
        cfg['run_wallclock_s']=max(cfg['run_wallclock_s'],21600)
    if args.llm:cfg['llm']=args.llm
    if args.candidates:cfg['candidates_per_gen']=args.candidates
    if args.model:
        for role in ('adapter','planner','subagent','curator'):cfg[role+'_llm']=args.model
    if args.spend_cap_usd is not None:
        if not 0<args.spend_cap_usd<=100:p.error('--spend-cap-usd must be positive and at most 100')
        cfg['spend_cap_usd']=args.spend_cap_usd
    if args.no_spend_cap:
        if args.spend_cap_usd is not None:p.error('Choose --no-spend-cap or --spend-cap-usd')
        cfg['spend_cap_usd']=None
    if args.min_target_pct is not None:
        if not 0<args.min_target_pct<=100:p.error('--min-target-pct must be positive and at most 100')
        cfg['min_pct_step_time']=args.min_target_pct
    cfg['task_prompt']=args.prompt or 'Minimize the complete fixed-batch training step time while preserving numerical correctness.'
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True);args.run_dir=str(root)
    cfg['run_name']=root.name
    from kernel_evolution.provenance import identity
    cfg['source_identity']=identity(Path(__file__).parent)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache')
    archive=Archive(root/'archive.sqlite')
    if not archive.rows('SELECT id FROM run_config'):
        archive.put('run_config',created_at=time.time(),config_json=json.dumps(cfg))
    from kernel_evolution.references import fetch_references
    try:fetch_references(root)
    except Exception as exc:archive.event('reference_fetch_failed',{'error':str(exc)})
    mirror=Mirror(cfg,archive)
    archive.run_trace_id=mirror.traces.start('kernel_evolution_run',{'repo':args.repo,'adapter':args.adapter,
        'prompt':cfg['task_prompt'],'config':cfg},attributes={'run_name':root.name,'span_kind':'run'})
    status='interrupted'
    try:
        from kernel_evolution.intake import resolve_input
        args.adapter=resolve_input(args,cfg,archive,Path(__file__).parent)
        started=time.time()
        prepared=load_prepared(args,cfg,archive)
        mirror.traces.record('prepare',{'adapter':args.adapter},prepared,started_at=started,ended_at=time.time(),
            parent_id=archive.run_trace_id)
        if prepared.get('status')=='no_targets':
            status='no_targets'
            archive.put('generations',id=0,model_id=args.adapter,started_at=started,finished_at=time.time(),
                n_candidates=0,n_accepted=0,llm_usd=Budget(archive,cfg['spend_cap_usd']).spent,stop_reason=status)
        else:
            register(prepared,archive)
            status='prepared' if args.prepare_only else run(args,cfg,archive,prepared,mirror)
    except Exception:
        status='platform_error'
        emit(archive,'platform_error',error=traceback.format_exc())
        raise
    finally:
        summary={'stop_reason':status,'usd':Budget(archive,cfg['spend_cap_usd']).spent,
            'agent_candidates':archive.rows('SELECT id,generation,model_name,accepted,gate_reached,step_time_ms,weave_trace_url '
                                            'FROM candidates WHERE generation>0')}
        (root/'result.json').write_text(json.dumps(summary,indent=2))
        mirror.traces.finish(archive.run_trace_id,output=summary,error=status if status=='platform_error' else None)
        mirror.finish()


if __name__=='__main__': main()
