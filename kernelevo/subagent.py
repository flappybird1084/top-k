"""Subagent (spec §4.5): one candidate lifecycle — implement the planner's named
strategy, local repair loop with raw feedback (compiler message / numeric
mismatch verbatim, no diagnosis), through gate 2. Gates 3-4 run afterwards,
serially on the GPU, in the orchestrator."""

from __future__ import annotations

import os

from kernelevo import gates as gatesmod
from kernelevo import prompts
from kernelevo.llm import extract_code
from kernelevo.obs import current_trace_url, weave_op


@weave_op
def _run_candidate(job: dict, job_index: int, generation: int, ctx) -> dict:
    """ctx: loop.GenContext — cfg, runner, pool, lineage/parent lookups, budget checks."""
    cfg = ctx.cfg
    llm = ctx.pool.subagent_for(job_index)
    op = job["lineage"]
    lineage = ctx.lineage_info(op)
    # pre-resolved by the orchestrator — worker threads must not touch SQLite
    parent_source = job.get("_parent_source")
    parent_lat = job.get("_parent_latency")
    messages = prompts.subagent_prompt(job, lineage, parent_source, parent_lat,
                                       ctx.lessons)
    result = dict(
        lineage=op, strategy=job["strategy"], parent=job.get("parent"),
        parents=job.get("parents"), model_name=llm.model,
        source_kind="fusion" if str(job["strategy"]).upper().startswith("FUSE") else "mutation",
        gate_reached=0, compile_ok=False, correct_ok=False, repairs_used=0,
        code_path=None, source_hash=None, flags=None, failure_note=None,
        weave_trace_url=current_trace_url(),
    )

    for attempt in range(cfg["max_repairs"] + 1):
        why = ctx.budget_exceeded()
        if why:
            result["failure_note"] = f"[infra] aborted before attempt {attempt}: {why}"
            return result
        resp = llm.complete(messages, meta={"job": job, "attempt": attempt})
        src = extract_code(resp.text)
        name = f"gen{generation}_{op}_{job_index}_a{attempt}.py"
        path = os.path.join(ctx.candidates_dir, name)
        with open(path, "w") as f:
            f.write(src)
        result.update(code_path=path, source_hash=gatesmod.source_hash(src),
                      repairs_used=attempt, flags=",".join(gatesmod.scan_flags(src)) or None)

        ok, msg = ctx.runner.compile(path, op)
        if not ok:
            result.update(compile_ok=False, failure_note=msg[:2000])
            if attempt < cfg["max_repairs"]:
                messages = messages + [{"role": "assistant", "content": resp.text},
                                       prompts.repair_message("compile", msg)]
                continue
            return result
        result.update(compile_ok=True, gate_reached=1)

        v = ctx.runner.verify(path, op, upto=2, incumbents=ctx.incumbents_snapshot())
        if not v.get("correct_ok"):
            note = v.get("failure_note") or "unknown mismatch"
            result["failure_note"] = note[:2000]
            if attempt < cfg["max_repairs"]:
                messages = messages + [{"role": "assistant", "content": resp.text},
                                       prompts.repair_message("mismatch", note)]
                continue
            return result

        result.update(correct_ok=True, gate_reached=2, failure_note=None)
        return result
    return result


def run_candidate(job, job_index, generation, ctx):
    import json,time
    event=dict(id=f'{generation}-{job_index}',kernel=job['lineage'],strategy=job['strategy'],started_at=time.time(),stage='Generating and checking correctness')
    print('[evaluation] '+json.dumps(event),flush=True)
    try:return _run_candidate(job,job_index,generation,ctx)
    finally:print('[evaluation] '+json.dumps({'id':event['id'],'finished':True}),flush=True)
