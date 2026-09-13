"""Planner (spec §4.5): one LLM call per generation → JSON job list. May use the
provider's native web-search tool when it wants prior art; the orchestrator
truncates if it over-proposes."""

from __future__ import annotations

import json

from kernelevo import prompts
from kernelevo.obs import weave_op


def _parse_obj(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner returned no JSON object")
    return json.loads(text[start:end + 1])


MAX_SEARCH_ROUNDS = 3


@weave_op
def plan(llm, targets: dict, summary: dict, lessons: list[str],
         n_jobs: int, generation: int, web_search: bool = False) -> list[dict]:
    # NOTE: only pass what this needs — a full cfg dict here gets logged as
    # weave inputs and its unused provider defaults (e.g. anthropic_model)
    # read as if those models were in play.
    from kernelevo import websearch
    active = [l["op"] for l in targets["lineages"] if not l.get("retired")]
    fuse_allowed = generation >= 2
    searx = web_search and websearch.available()
    msgs = prompts.planner_prompt(targets, summary, lessons, n_jobs, generation,
                                  fuse_allowed, search_enabled=searx)
    # native provider search only when there's no SearXNG (Anthropic-only)
    tools = ("web_search" if web_search and not searx and llm.supports_search
             else None)
    jobs = None
    for round_ in range(MAX_SEARCH_ROUNDS + 1):
        resp = llm.complete(msgs, json_mode=True, tools=tools,
                            meta={"active_lineages": active, "n_jobs": n_jobs,
                                  "generation": generation})
        try:
            obj = _parse_obj(resp.text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[planner] unparseable output ({e}); skipping generation")
            return []
        query = obj.get("search")
        if query and searx and round_ < MAX_SEARCH_ROUNDS:
            try:
                results = websearch.search(str(query))
            except Exception as e:  # noqa: BLE001 — instance down ≠ lost generation
                results = [{"error": f"search failed: {e}"}]
            print(f"[planner] searched: {str(query)[:80]} "
                  f"({len(results)} result(s))")
            msgs = msgs + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content":
                 "Search results:\n" + json.dumps(results, indent=1) +
                 f'\n\nContinue: another {{"search": "..."}} '
                 f'({MAX_SEARCH_ROUNDS - 1 - round_} left) or the final '
                 f'{{"jobs": [...]}}.'}]
            continue
        jobs = obj.get("jobs")
        break
    if not isinstance(jobs, list):
        print("[planner] no job list produced; skipping generation")
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
