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


MAX_RESEARCH_ROUNDS = 2


@weave_op
def plan(llm, targets: dict, summary: dict, lessons: list[str],
         n_jobs: int, generation: int, web_search: bool = False,
         researcher_llm=None) -> list[dict]:
    # NOTE: only pass what this needs — a full cfg dict here gets logged as
    # weave inputs and its unused provider defaults (e.g. anthropic_model)
    # read as if those models were in play.
    from kernelevo import researcher as researchmod
    from kernelevo import websearch
    active = [l["op"] for l in targets["lineages"] if not l.get("retired")]
    fuse_allowed = generation >= 2
    can_research = bool(web_search and websearch.available()
                        and researcher_llm is not None)
    msgs = prompts.planner_prompt(targets, summary, lessons, n_jobs, generation,
                                  fuse_allowed, search_enabled=can_research)
    jobs = None
    valid = []
    research_used = 0
    empty_retry_used = False
    for _ in range(MAX_RESEARCH_ROUNDS + 3):
        resp = llm.complete(msgs, json_mode=True,
                            meta={"active_lineages": active, "n_jobs": n_jobs,
                                  "generation": generation})
        try:
            obj = _parse_obj(resp.text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[planner] unparseable output ({e}); skipping generation")
            return []
        question = obj.get("research")
        if question and can_research and research_used < MAX_RESEARCH_ROUNDS:
            research_used += 1
            brief = researchmod.research(researcher_llm, str(question))
            print(f"[planner] research: {str(question)[:70]} -> "
                  f"{len(brief)} char brief")
            msgs = msgs + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content":
                 "Research brief:\n" + brief +
                 f'\n\nContinue: one more {{"research": "..."}} '
                 f'({MAX_RESEARCH_ROUNDS - research_used} left) or the final '
                 f'{{"jobs": [...]}}.'}]
            continue
        jobs = obj.get("jobs")
        valid = [job for job in jobs if isinstance(job, dict)
                 and job.get("lineage") in active
                 and not (str(job.get("strategy", "")).upper().startswith("FUSE")
                          and not fuse_allowed)] if isinstance(jobs, list) else []
        # An empty or invalid plan wastes a generation and retires useful
        # lineages. Ask once more with the exact allowable op names.
        if not valid and not empty_retry_used:
            empty_retry_used = True
            print("[planner] returned no valid jobs; re-prompting once")
            msgs = msgs + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content":
                 f"You proposed no valid jobs, which wastes the generation. "
                 f"Propose between 1 and {n_jobs} jobs now as "
                 f'{{"jobs": [...]}}. Allowed lineage names: {active}. '
                 + ("FUSE jobs are not allowed this generation." if not fuse_allowed
                    else "") }]
            continue
        break
    if not isinstance(jobs, list):
        print("[planner] no job list produced; skipping generation")
        return []
    for job in valid:
        job.setdefault("parent", None)
    return valid[:n_jobs]
