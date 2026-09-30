"""Serve W&B Inference on EC2 for a Molab run without sending its key to GPU code."""

import functools
import os
import re

from kernelevo.codex_oauth import Relay as FileRelay


DEEPSEEK_V4_PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro-0813"
DEEPSEEK_V4_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
CHAT_MODE_MODELS = frozenset({DEEPSEEK_V4_PRO_MODEL, DEEPSEEK_V4_FLASH_MODEL})


def chat_options(model: str) -> dict:
    """Keep DeepSeek's private reasoning from exhausting bounded code requests.

    W&B accepts vLLM chat-template options for these models. With the default
    thinking mode, real adapter prompts spent 8,192 and 16,384 output tokens
    without returning any answer; chat mode completed a coding probe in 7,047.
    Flash behaves the same: the nanoGPT adapter prompt used all 8,192 tokens on
    reasoning with thinking on, and returned a full adapter in 1,698 with it off.
    """
    if model in CHAT_MODE_MODELS:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def is_credit_error(exc: Exception) -> bool:
    """Recognize W&B quota and billing failures before the relay shortens text."""
    body = getattr(exc, "body", None)
    code = getattr(exc, "code", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict):
            code = code or detail.get("code")
    message = f"{code or ''} {exc}"
    return getattr(exc, "status_code", None) == 402 or bool(re.search(
        r"insufficient[_ ]quota|(?:insufficient|exhausted|out of).*credits|"
        r"(?:quota|billing).*(?:exceeded|limit|disabled)|"
        r"exceeded your current quota|payment required", message, re.IGNORECASE))


def complete_local(request, credentials=None):
    """Answer one relayed completion. `credentials` (WANDB_* names) replaces the
    dispatcher's own environment entirely: a public run is billed to its owner's
    key and never falls back to anything the operator configured."""
    from openai import OpenAI

    env = os.environ if credentials is None else credentials
    key = env.get("WANDB_INFERENCE_API_KEY") or env.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("W&B Inference credential missing on the dispatcher")
    base_url = os.environ.get("WANDB_INFERENCE_BASE_URL") or "https://api.inference.wandb.ai/v1"
    project = env.get("WANDB_INFERENCE_PROJECT")
    if not project and env.get("WANDB_ENTITY") and env.get("WANDB_PROJECT"):
        project = f"{env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}"
    headers = {"OpenAI-Project": project} if project else None
    client = OpenAI(base_url=base_url, api_key=key, default_headers=headers)
    params = dict(model=request["model"], messages=request["messages"],
                  max_tokens=min(int(request.get("max_tokens") or 8192), 8192))
    params.update(chat_options(request["model"]))
    if request.get("json_mode"):
        params["response_format"] = {"type": "json_object"}
    try:
        result = client.chat.completions.create(**params)
    except Exception as exc:
        if is_credit_error(exc):
            raise RuntimeError("W&B Inference credits exhausted or quota exceeded") from exc
        raise
    text = result.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(f"W&B model returned empty text ({result.choices[0].finish_reason})")
    usage = result.usage
    return {"text": text, "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0}


class Relay(FileRelay):
    label = "W&B Inference"
    worker = staticmethod(complete_local)

    def __init__(self, credentials=None):
        super().__init__()
        if credentials is not None:
            self.worker = functools.partial(complete_local, credentials=dict(credentials))
