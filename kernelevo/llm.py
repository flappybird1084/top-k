"""LLM provider interface (spec §7). One protocol, any provider behind it,
selected per role (planner/subagent/curator). StubLLM returns deterministic
fixtures with the real interface — the harness is tested with it before any
agent exists (spec §8).

Weave autopatches the anthropic/openai SDKs, so tracing needs nothing here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import config as cfgmod


@dataclass
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLM:
    provider = "base"
    supports_search = False

    def __init__(self, model: str, max_tokens: int = 8192):
        self.model = model
        self.max_tokens = max_tokens
        self._in_tokens = 0
        self._out_tokens = 0

    def _track(self, resp: Response) -> Response:
        self._in_tokens += resp.input_tokens
        self._out_tokens += resp.output_tokens
        return resp

    def usage_usd(self) -> float:
        pin, pout = cfgmod.price_for(self.model)
        return (self._in_tokens * pin + self._out_tokens * pout) / 1e6

    def complete(self, messages, *, json_mode=False, tools=None, meta=None) -> Response:
        raise NotImplementedError


def _split_system(messages):
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


class CodexOAuthLLM(BaseLLM):
    provider = "codex_oauth"

    def complete(self, messages, *, json_mode=False, tools=None, meta=None):
        from kernelevo.codex_oauth import complete
        answer=complete(dict(messages=messages,json_mode=json_mode,model=self.model))
        return self._track(Response(**answer))

    def usage_usd(self):
        # Subscription usage is counted in tokens; it is not API dollar billing.
        return 0.0


class AnthropicLLM(BaseLLM):
    provider = "anthropic"
    supports_search = True

    def __init__(self, model, max_tokens=8192):
        super().__init__(model, max_tokens)
        import anthropic
        self.client = anthropic.Anthropic()

    def complete(self, messages, *, json_mode=False, tools=None, meta=None) -> Response:
        system, msgs = _split_system(messages)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs)
        if system:
            kwargs["system"] = system
        if tools == "web_search":
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search",
                                "max_uses": 3}]
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return self._track(Response(text, resp.usage.input_tokens, resp.usage.output_tokens))


class OpenAILLM(BaseLLM):
    provider = "openai"
    # NOTE: web search on OpenAI needs the Responses API; not wired up here.
    supports_search = False

    def __init__(self, model, max_tokens=8192, base_url=None, api_key=None,
                 default_headers=None):
        super().__init__(model, max_tokens)
        from openai import OpenAI
        kw = {}
        if base_url:
            kw["base_url"] = base_url
        if api_key:
            kw["api_key"] = api_key
        if default_headers:
            kw["default_headers"] = default_headers
        self.client = OpenAI(**kw)

    def complete(self, messages, *, json_mode=False, tools=None, meta=None) -> Response:
        kwargs = dict(model=self.model, messages=messages)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return self._track(Response(
            resp.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0))


def make_wandb_inference_llm(model, max_tokens=8192):
    """W&B Inference is OpenAI-compatible (spec §7). Verified live 2026-09-12:
    base URL https://api.inference.wandb.ai/v1, plain Bearer auth with a W&B API
    key, no project header required (sent anyway for usage attribution when
    entity/project are set). GET /v1/models lists the catalog."""
    base_url = os.environ.get("WANDB_INFERENCE_BASE_URL") or "https://api.inference.wandb.ai/v1"
    api_key = os.environ.get("WANDB_INFERENCE_API_KEY") or os.environ.get("WANDB_API_KEY")
    # Billing attribution is separate from logging: WANDB_INFERENCE_PROJECT
    # ("entity/project") controls which org the inference usage bills to; it
    # must be an org with the inference gateway enabled (403 otherwise).
    headers = None
    inf_proj = os.environ.get("WANDB_INFERENCE_PROJECT")
    entity, project = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
    if inf_proj:
        headers = {"OpenAI-Project": inf_proj}
    elif entity and project:
        headers = {"OpenAI-Project": f"{entity}/{project}"}
    llm = OpenAILLM(model, max_tokens, base_url=base_url, api_key=api_key,
                    default_headers=headers)
    llm.provider = "wandb"
    return llm


# ---------------------------------------------------------------------- stub

STUB_KINDS = ["pass", "compile_fail", "mismatch", "cheat_cache"]


class StubLLM(BaseLLM):
    """Deterministic fixtures behind the real interface. The subagent stub
    cycles pass / compile-fail / mismatch / planted-cheat candidates; broken
    fixtures stay broken across repairs so repair exhaustion is exercised."""
    provider = "stub"
    _job_counter = 0

    def __init__(self, role: str):
        super().__init__(model=f"stub-{role}")
        self.role = role

    def usage_usd(self) -> float:
        return 0.0

    def complete(self, messages, *, json_mode=False, tools=None, meta=None) -> Response:
        meta = meta or {}
        if self.role == "planner":
            jobs = []
            lineages = meta["active_lineages"]
            for i in range(meta["n_jobs"]):
                kind = STUB_KINDS[StubLLM._job_counter % len(STUB_KINDS)]
                StubLLM._job_counter += 1
                jobs.append({"lineage": lineages[i % len(lineages)],
                             "strategy": f"stub:{kind}", "parent": None})
            return Response(json.dumps({"jobs": jobs}))
        if self.role == "subagent":
            from kernelevo import fixtures
            job = meta["job"]
            kind = job["strategy"].split(":", 1)[1] if job["strategy"].startswith("stub:") else "pass"
            src = fixtures.make(job["lineage"], kind)
            return Response(f"```python\n{src}\n```")
        if self.role == "researcher":
            return Response('{"brief": "stub research brief: no findings (stub mode)"}')
        if self.role == "adapter":
            # fixture adapter: wraps the bundled JEPA demo so the whole repo
            # pipeline (clone -> write -> ingest-verify -> optimize) runs
            # deterministically with no model
            return Response("```python\n"
                            "from adapters.jepa import build_model, get_dataloader, loss_fn\n"
                            "```")
        n = meta.get("n_failed", "?")
        return Response(f"stub lesson: verifier rejected {n} candidate(s) this generation; "
                        f"gates behaved as configured.")


# ------------------------------------------------------------------- factory

class LLMPool:
    """Builds and owns per-role providers; sums spend across all of them for the
    spend-cap stop condition."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._instances: list[BaseLLM] = []
        self.planner = self._make(cfg.get("planner_llm") or cfg["llm"], "planner")
        sub = cfg.get("subagent_llm") or cfg["llm"]
        subs = sub if isinstance(sub, list) else [sub]
        self.subagents = [self._make(s, "subagent") for s in subs]
        self.curator = self._make(cfg.get("curator_llm") or cfg["llm"], "curator")
        # repo comprehension is subagent-hard, so the adapter writer defaults to
        # the (first) subagent provider
        self.adapter = self._make(cfg.get("adapter_llm") or subs[0], "adapter")
        # research subagent (planner-dispatched web search)
        self.researcher = self._make(
            cfg.get("researcher_llm") or cfg.get("planner_llm") or cfg["llm"],
            "researcher")

    def _make(self, spec: str, role: str) -> BaseLLM:
        provider, _, model = spec.partition(":")
        if provider == "stub":
            llm = StubLLM(role)
        elif provider == "codex_oauth":
            llm = CodexOAuthLLM(model, self.cfg["max_llm_tokens"])
        elif provider == "anthropic":
            default = (self.cfg["anthropic_subagent_model"] if role == "subagent"
                       else self.cfg["anthropic_model"])
            llm = AnthropicLLM(model or default, self.cfg["max_llm_tokens"])
        elif provider == "openai":
            llm = OpenAILLM(model or self.cfg["openai_model"], self.cfg["max_llm_tokens"])
        elif provider == "wandb":
            llm = make_wandb_inference_llm(model or self.cfg["wandb_inference_model"],
                                           self.cfg["max_llm_tokens"])
        else:
            raise SystemExit(f"unknown LLM provider {provider!r}")
        self._instances.append(llm)
        return llm

    def subagent_for(self, job_index: int) -> BaseLLM:
        return self.subagents[job_index % len(self.subagents)]

    def total_usd(self) -> float:
        return sum(llm.usage_usd() for llm in self._instances)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"
