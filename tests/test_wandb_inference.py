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
    assert config.price_for("Qwen/Qwen3-235B-A22B-Instruct-2507") == (0.10, 0.10)
    assert config.price_for("Qwen/Qwen3-Coder-480B-A35B-Instruct") == (1.00, 1.50)


def test_molab_wandb_uses_file_relay_without_remote_key(monkeypatch):
    from kernelevo.llm import LLMPool, WandbRelayLLM
    import kernelevo.codex_oauth as oauth

    monkeypatch.setenv("KEVO_RELAY_DIR", "/tmp/test-relay")
    monkeypatch.setenv("KEVO_WANDB_INFERENCE_RELAY", "1")
    monkeypatch.delenv("WANDB_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    pool = object.__new__(LLMPool)
    pool.cfg = {"max_llm_tokens": 1024, "wandb_inference_model": "example/model"}
    pool._instances = []
    llm = pool._make("wandb:example/model", "adapter")
    assert isinstance(llm, WandbRelayLLM)
    requests = []
    def fake_relay(request, kind, label):
        requests.append((request, kind, label))
        return {"text": "working code", "input_tokens": 10, "output_tokens": 3}
    monkeypatch.setattr(oauth, "relay_complete", fake_relay)
    answer = llm.complete([{"role": "user", "content": "write code"}])
    assert answer.text == "working code"
    assert requests[0][1] == "wandb_inference"
    assert requests[0][0]["max_tokens"] == 1024
