"""Optional API-key providers. The pilot uses Codex OAuth instead.

Prices for other models must be supplied explicitly; never invent zero-cost usage.
Native tools are passed through in each provider's own format.
"""
import json
from kernel_evolution.budget import cost,PRICES


class MeteredProvider:
    def __init__(self,model,prices=None):
        self.model=model
        if prices is not None:
            if len(prices)!=3 or any(p<0 for p in prices):raise ValueError('Expected input, cached-input, output prices per million')
            PRICES[model]=tuple(prices)
        if model not in PRICES:raise ValueError('Configure verified token prices for '+model)
        self._usd=0.
        self.last_usage=None

    def record(self,inputs,cached,outputs):
        self.last_usage=dict(input_tokens=inputs,cached_input_tokens=cached,output_tokens=outputs)
        self._usd+=cost(self.model,inputs,cached,outputs)

    def usage_usd(self):return self._usd


class OpenAILLM(MeteredProvider):
    def __init__(self,model,*,client=None,prices=None):
        super().__init__(model,prices)
        if client is None:
            from openai import OpenAI
            client=OpenAI()
        self.client=client

    def complete(self,messages,*,json_mode=False,tools=None,schema=None):
        kwargs=dict(model=self.model,input=messages)
        if tools:kwargs['tools']=tools
        if schema:
            kwargs['text']={'format':{'type':'json_schema','name':'response','strict':True,'schema':schema}}
        elif json_mode:kwargs['text']={'format':{'type':'json_object'}}
        response=self.client.responses.create(**kwargs)
        if response.usage is None:raise RuntimeError('Provider omitted usage; reconcile cost before continuing')
        u=response.usage
        cached=getattr(getattr(u,'input_tokens_details',None),'cached_tokens',0) or 0
        self.record(u.input_tokens,cached,u.output_tokens)
        return json.loads(response.output_text) if json_mode else response.output_text


class AnthropicLLM(MeteredProvider):
    def __init__(self,model,*,client=None,prices=None,max_tokens=8192):
        super().__init__(model,prices)
        if client is None:
            import anthropic
            client=anthropic.Anthropic()
        self.client,self.max_tokens=client,max_tokens

    def complete(self,messages,*,json_mode=False,tools=None):
        system='\n'.join(m['content'] for m in messages if m['role']=='system')
        if json_mode:system+='\nRespond with one valid JSON object and no Markdown fencing.'
        kwargs=dict(model=self.model,max_tokens=self.max_tokens,system=system,
                    messages=[m for m in messages if m['role']!='system'])
        if tools:kwargs['tools']=tools
        response=self.client.messages.create(**kwargs)
        u=response.usage
        cached=getattr(u,'cache_read_input_tokens',0) or 0
        cache_writes=getattr(u,'cache_creation_input_tokens',0) or 0
        self.record(u.input_tokens+cached+cache_writes,cached,u.output_tokens)
        # Cache creation is charged above ordinary input, not silently omitted.
        self._usd+=cache_writes*PRICES[self.model][0]*.25/1e6
        text=''.join(block.text for block in response.content if getattr(block,'type',None)=='text')
        return json.loads(text) if json_mode else text


class WandbInferenceLLM(MeteredProvider):
    def __init__(self,model,*,base_url,api_key=None,client=None,prices=None):
        super().__init__(model,prices)
        if client is None:
            from openai import OpenAI
            client=OpenAI(base_url=base_url,api_key=api_key)
        self.client=client

    def complete(self,messages,*,json_mode=False,tools=None):
        kwargs=dict(model=self.model,messages=messages)
        if json_mode:kwargs['response_format']={'type':'json_object'}
        # This endpoint has no assumed native web-search capability.
        response=self.client.chat.completions.create(**kwargs)
        u=response.usage
        if u is None:raise RuntimeError('Provider omitted usage; reconcile cost before continuing')
        cached=getattr(getattr(u,'prompt_tokens_details',None),'cached_tokens',0) or 0
        self.record(u.prompt_tokens,cached,u.completion_tokens)
        text=response.choices[0].message.content
        return json.loads(text) if json_mode else text
