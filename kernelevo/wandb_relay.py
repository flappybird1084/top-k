"""Serve W&B Inference on EC2 for a Molab run without sending its key to GPU code."""

import os

from kernelevo.codex_oauth import Relay as FileRelay


def complete_local(request):
    from openai import OpenAI

    key = os.environ.get("WANDB_INFERENCE_API_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("W&B Inference credential missing on the dispatcher")
    base_url = os.environ.get("WANDB_INFERENCE_BASE_URL") or "https://api.inference.wandb.ai/v1"
    project = os.environ.get("WANDB_INFERENCE_PROJECT")
    if not project and os.environ.get("WANDB_ENTITY") and os.environ.get("WANDB_PROJECT"):
        project = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
    headers = {"OpenAI-Project": project} if project else None
    client = OpenAI(base_url=base_url, api_key=key, default_headers=headers)
    params = dict(model=request["model"], messages=request["messages"],
                  max_tokens=min(int(request.get("max_tokens") or 8192), 8192))
    if request.get("json_mode"):
        params["response_format"] = {"type": "json_object"}
    result = client.chat.completions.create(**params)
    text = result.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(f"W&B model returned empty text ({result.choices[0].finish_reason})")
    usage = result.usage
    return {"text": text, "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0}


class Relay(FileRelay):
    label = "W&B Inference"
    worker = staticmethod(complete_local)
