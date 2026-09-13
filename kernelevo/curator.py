"""Curator (spec §4.5): one call per generation; distills that generation's
acceptances and failures into 2-5 lesson lines for the next generation's
prompts. This is the learning mechanism."""

from __future__ import annotations

from kernelevo import prompts
from kernelevo.obs import weave_op


@weave_op
def curate(llm, gen_digest: dict) -> list[str]:
    resp = llm.complete(prompts.curator_prompt(gen_digest),
                        meta={"n_failed": gen_digest.get("n_failed")})
    lines = [ln.strip("-• \t") for ln in resp.text.splitlines() if ln.strip()]
    return lines[:5]
