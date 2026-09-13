import unittest
from types import SimpleNamespace as NS
from kernel_evolution.providers import OpenAILLM,WandbInferenceLLM


class ProviderTests(unittest.TestCase):
    def test_openai_reports_usage_before_rejecting_malformed_json(self):
        response=NS(output_text='invalid json',usage=NS(input_tokens=1000,output_tokens=100,input_tokens_details=NS(cached_tokens=500)))
        provider=OpenAILLM('gpt-5.6-sol',client=NS(responses=NS(create=lambda **kwargs:response)))
        with self.assertRaises(ValueError):provider.complete([{'role':'user','content':'JSON'}],json_mode=True)
        self.assertAlmostEqual(provider.usage_usd(),.0042)

    def test_wandb_compatible_endpoint_does_not_assume_native_search(self):
        seen={}
        def create(**kwargs):
            seen.update(kwargs)
            return NS(choices=[NS(message=NS(content='{"ok":true}'))],usage=NS(prompt_tokens=100,completion_tokens=10,prompt_tokens_details=None))
        provider=WandbInferenceLLM('gpt-5.6-sol',base_url='https://example.invalid',client=NS(chat=NS(completions=NS(create=create))))
        self.assertEqual(provider.complete([],json_mode=True,tools=[{'type':'web_search'}]),{'ok':True})
        self.assertNotIn('tools',seen)
        self.assertAlmostEqual(provider.usage_usd(),.0006)
