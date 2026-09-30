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
    assert config.price_for("deepseek-ai/DeepSeek-V4-Pro-0813") == (1.31, 3.96)
    assert config.price_for("Qwen/Qwen3-235B-A22B-Instruct-2507") == (0.10, 0.10)
    assert config.price_for("Qwen/Qwen3-Coder-480B-A35B-Instruct") == (1.00, 1.50)


def test_deepseek_uses_bounded_chat_mode_in_direct_and_relay_calls(monkeypatch):
    from kernelevo.wandb_relay import chat_options, complete_local

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="working code"))],
                usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=3))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    model = "deepseek-ai/DeepSeek-V4-Pro-0813"
    response = complete_local({"model": model, "messages": [],
                               "max_tokens": 8192, "json_mode": True})
    assert response["text"] == "working code"
    assert calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert calls[0]["response_format"] == {"type": "json_object"}
    direct = make_wandb_inference_llm(model, max_tokens=1024)
    assert direct.complete([{"role": "user", "content": "code"}]).text == "working code"
    assert calls[1]["extra_body"] == calls[0]["extra_body"]
    assert calls[1]["max_tokens"] == 1024
    assert chat_options("zai-org/GLM-5.2") == {}


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


def test_default_flash_model_also_runs_in_chat_mode():
    # With thinking on, Flash spent the whole 8,192-token budget on reasoning
    # and returned an empty adapter on every attempt.
    from kernelevo.wandb_relay import chat_options
    assert config.load("DEV")["wandb_inference_model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert chat_options("deepseek-ai/DeepSeek-V4-Flash-0731") == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_owner_scoped_relay_never_uses_the_dispatcher_key(monkeypatch):
    from kernelevo.wandb_relay import Relay, complete_local

    seen = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("WANDB_INFERENCE_API_KEY", "operator-key")
    monkeypatch.setenv("WANDB_INFERENCE_PROJECT", "operator/project")
    request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    relay = Relay({"WANDB_API_KEY": "owner-key", "WANDB_ENTITY": "owner",
                   "WANDB_PROJECT": "runs"})
    assert relay.worker(request)["text"] == "ok"
    assert seen[-1]["api_key"] == "owner-key"
    assert seen[-1]["default_headers"] == {"OpenAI-Project": "owner/runs"}

    try:
        complete_local(request, credentials={})
    except RuntimeError as e:
        assert "credential missing" in str(e)
    else:
        raise AssertionError("an owner without a key must not fall back to the operator's")

    assert Relay().worker(request)["text"] == "ok"
    assert seen[-1]["api_key"] == "operator-key"
