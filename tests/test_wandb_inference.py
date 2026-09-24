"""Keep W&B Inference requests bounded and attributed to the intended project."""

import sys
import types

import config
from kernelevo.llm import make_wandb_inference_llm


def test_wandb_inference_sends_output_limit_and_project(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="OK"))],
                usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_INFERENCE_PROJECT", "team/benchmarks")
    llm = make_wandb_inference_llm("example/model", max_tokens=123)
    answer = llm.complete([{"role": "user", "content": "hello"}])

    assert calls[0]["default_headers"] == {"OpenAI-Project": "team/benchmarks"}
    assert calls[1]["max_tokens"] == 123
    assert answer.text == "OK"
    assert (answer.input_tokens, answer.output_tokens) == (3, 1)


def test_default_wandb_model_uses_inference_pricing():
    assert config.price_for("deepseek-ai/DeepSeek-V4-Flash-0731") == (0.13, 0.28)
