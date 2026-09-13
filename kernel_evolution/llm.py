import json
import time
from pathlib import Path
from typing import Protocol
from kernel_evolution.budget import Budget,PRICES
from kernel_evolution.processes import run_worker


class LLM(Protocol):
    def complete(self,messages,*,json_mode=False,tools=None): ...
    def usage_usd(self): ...


def usage_tokens(usage):
    if usage is None: return None
    # App-server reports both cumulative total and last-turn counts. Each lifecycle
    # starts a fresh thread, so use total to include native-search/tool turns.
    u=usage.get('total',usage)
    if not isinstance(u,dict) or not any(k in u for k in ('input_tokens','inputTokens')) or not any(k in u for k in ('output_tokens','outputTokens')):
        return None
    def get(snake,camel): return u.get(snake,u.get(camel,0))
    return dict(input_tokens=get('input_tokens','inputTokens'),
                cached_input_tokens=get('cached_input_tokens','cachedInputTokens'),
                output_tokens=get('output_tokens','outputTokens'))


class CodexOAuthLLM:
    def __init__(self,archive,config,run_dir,role,model=None):
        self.archive,self.config,self.root,self.role=archive,config,Path(run_dir),role
        self.model=model or config[role+'_llm']
        if self.model not in PRICES:raise ValueError('Configure verified token pricing before using '+self.model)
        self.budget=Budget(archive,config['spend_cap_usd'])
        self.generation=0
        self.candidate_id=None
        self.parent_trace_id=None
        self.repair=0

    def usage_usd(self): return self.budget.spent

    def complete(self,messages,*,json_mode=False,tools=None,schema=None,timeout=None):
        prompt='\n\n'.join(f"{m['role']}: {m['content']}" for m in messages)
        cid=self.budget.reserve(self.role,self.model,self.generation,self.config['max_call_reservation_usd'])
        traces=getattr(self.archive,'traces',None)
        trace_id=traces.start('agent.'+self.role,{'messages':messages,'schema':schema},
            parent_id=self.parent_trace_id,attributes={'role':self.role,'model':self.model,
            'generation':self.generation,'candidate_id':self.candidate_id,'llm_call_id':cid,'repair':self.repair}) if traces else None
        self.archive.execute('UPDATE llm_calls SET candidate_id=?,trace_id=?,repair=? WHERE id=?',
                             (self.candidate_id,trace_id,self.repair,cid))
        cwd=self.root/'llm_work'/cid
        cwd.mkdir(parents=True)
        request=dict(prompt=prompt,model=self.model,role=self.role,schema=schema,cwd=str(cwd),
                     effort=self.config.get(self.role+'_effort','medium'))
        try:
            result=run_worker('kernel_evolution.llm_worker',request,self.root/'llm_calls',
                              timeout or self.config['max_call_seconds'],cuda=False)
        except BaseException:
            self.budget.finish(cid,self.model,None,{'error':'worker interrupted'})
            if traces:traces.finish(trace_id,output={'usage':'unknown'},error='worker interrupted')
            raise
        usage=usage_tokens(result.get('usage'))
        self.budget.finish(cid,self.model,usage,dict(status=result.get('status'),error=result.get('error'),log_path=result.get('log_path')))
        self.archive.event('llm_call',dict(id=cid,role=self.role,model=self.model,usage=usage,total_usd=self.usage_usd()))
        error=result.get('error') or result.get('failure_note')
        decoded=None
        if result.get('text') and json_mode:
            try:decoded=json.loads(result['text'])
            except ValueError as exc:error='Invalid JSON response: '+str(exc)
        if traces:traces.finish(trace_id,output={'response':result.get('text'),'usage':usage,
            'model':self.model,'items':result.get('items',[]),'status':result.get('status'),
            'api_equivalent_usd':self.archive.rows('SELECT usd FROM llm_calls WHERE id=?',(cid,))[0]['usd']},
            error=str(error) if error else None)
        if not result.get('text'):
            raise RuntimeError(str(result.get('error') or result.get('failure_note') or result)[:4000])
        if error:raise RuntimeError(str(error)[:4000])
        return decoded if json_mode else result['text']


class StubLLM:
    def complete(self,messages,*,json_mode=False,tools=None):
        return {'lessons':['Fixture run: use real verifier results; never infer acceptance from fixture identity.']}
    def usage_usd(self): return 0.


JOB_SCHEMA={'type':'object','properties':{'jobs':{'type':'array','items':{'type':'object','properties':{
    'lineage':{'type':'string'},'strategy':{'type':'string'},'parents':{'type':'array','items':{'type':'string'}}},
    'required':['lineage','strategy','parents'],'additionalProperties':False}}},'required':['jobs'],'additionalProperties':False}
SOURCE_SCHEMA={'type':'object','properties':{'source':{'type':'string'}},'required':['source'],'additionalProperties':False}
LESSON_SCHEMA={'type':'object','properties':{'lessons':{'type':'array','items':{'type':'string'}}},'required':['lessons'],'additionalProperties':False}
