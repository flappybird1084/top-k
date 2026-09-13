"""Planner (spec §4.5): one LLM call per generation → JSON job list. May use the
provider's native web-search tool when it wants prior art; the orchestrator
truncates if it over-proposes."""

from __future__ import annotations

import json

from kernelevo import prompts
from kernelevo.obs import weave_op


def _parse_jobs(text: str) -> list[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner returned no JSON object")
    return json.loads(text[start:end + 1])["jobs"]


@weave_op
def plan(llm, targets: dict, summary: dict, lessons: list[str],
         n_jobs: int, generation: int, cfg: dict) -> list[dict]:
    active = [l["op"] for l in targets["lineages"] if not l.get("retired")]
    fuse_allowed = generation >= 2
    msgs = prompts.planner_prompt(targets, summary, lessons, n_jobs, generation, fuse_allowed)
    tools = ("web_search" if cfg.get("planner_web_search") and llm.supports_search else None)
    resp = llm.complete(msgs, json_mode=True, tools=tools,
                        meta={"active_lineages": active, "n_jobs": n_jobs,
                              "generation": generation})
    try:
        jobs = _parse_jobs(resp.text)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[planner] unparseable output ({e}); skipping generation")
        return []
    valid = []
    for job in jobs:
        if job.get("lineage") not in active:
            continue
        if str(job.get("strategy", "")).upper().startswith("FUSE") and not fuse_allowed:
            continue
        job.setdefault("parent", None)
        valid.append(job)
    return valid[:n_jobs]
