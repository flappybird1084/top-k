"""Recipe-golf: staged evolution over TRAINING RECIPES instead of kernels.

A recipe candidate is one self-contained python file exposing:

    def build_model() -> torch.nn.Module     # required — the architecture
    def make_optimizer(model) -> Optimizer   # required — optimizer + hparams
    def lr_schedule(step: int) -> float      # optional — multiplier on base lr
    TRAIN_HINTS = {"grad_clip": 1.0}         # optional

Everything gameable is HARNESS-OWNED and identical for every candidate: the
training data and its order, the loss function (the base adapter's), the
held-out validation batches and eval code, seeds, the wall-clock train budget,
and a parameter cap. The signal is mean held-out val loss after training for
the phase's budget. Hyperparam-phase candidates are architecture-locked to
their parent via a parameter-shape fingerprint.
"""

from __future__ import annotations

import hashlib
import json

from kernelevo.obs import weave_op

RECIPE_CONTRACT = """\
Output exactly one Python code block: a single self-contained recipe file.

Required functions:
- build_model() -> torch.nn.Module   # returns the model on CPU
- make_optimizer(model) -> torch.optim.Optimizer
Optional:
- lr_schedule(step: int) -> float    # multiplier applied to each group's base lr
- TRAIN_HINTS = {"grad_clip": <float>}   # gradient clipping max-norm

Hard rules (violations are rejected mechanically):
- The harness owns training data, loss function, validation data, evaluation
  code, seeds, and the wall-clock training budget. Your model must remain
  call-compatible with the base loss function shown below.
- Parameter budget: total parameters must not exceed the stated cap.
- No file I/O, no downloads, no threads, no changing global torch state.
- The objective is HELD-OUT VALIDATION LOSS after a fixed wall-clock training
  budget on one GPU — throughput and convergence both matter.
"""

ARCH_RULES = """\
This is an ARCHITECTURE phase: you may redesign the model architecture and the
optimizer freely (within the parameter cap and call-compatibility), and set
any hyperparameters. Implement exactly the strategy you were given.
STRONGLY prefer importing the repo's model classes and modifying only what the
strategy names (subclass, patch modules, adjust config) over reimplementing
the model from scratch — a rebuild silently loses unstated details (positional
encodings, norm placement, init, auxiliary pathways) and reliably scores worse.
"""

HP_RULES = """\
This is a HYPERPARAMETER phase: you MUST keep the parent recipe's architecture
byte-identical (same parameter names and shapes — this is verified by
fingerprint and mismatches are rejected). Change ONLY: the optimizer and its
hyperparameters, lr_schedule, and TRAIN_HINTS. Start from the parent file and
edit those parts.
"""


def baseline_recipe_source(adapter_name: str) -> str:
    """A valid recipe file that exactly clones the baseline (base adapter model
    + harness AdamW). Used as the parents' seed text and the stub fixture."""
    return f'''\
import torch
from kernelevo.ingest import load_adapter

_base, _ = load_adapter({adapter_name!r})


def build_model():
    return _base.build_model()


def make_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
'''


def arch_fingerprint(model) -> str:
    sig = sorted((n, tuple(p.shape)) for n, p in model.named_parameters())
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ planner

def recipe_planner_prompt(phase, base_summary, outcomes, lessons, n_jobs, parents):
    rules = ("propose ARCHITECTURE/optimizer modification strategies"
             if phase["kind"] == "architecture" else
             "propose HYPERPARAMETER-tuning strategies (architecture is frozen)")
    parent_txt = ("\n## Parents (id: summary)\n" + "\n".join(
        f"- {p['id']}: val_loss={p.get('val_loss')} — {str(p.get('strategy'))[:100]}"
        for p in parents) if parents else "")
    return [
        {"role": "system", "content":
         "You are the planner of a staged evolutionary search over TRAINING "
         "RECIPES. Each generation you name specific, plain-language strategies; "
         "parallel subagents implement exactly what you name; a deterministic "
         "harness trains each candidate for a fixed wall-clock budget and ranks "
         "by held-out validation loss. Diversity across jobs comes from you. "
         "Entries noted [infra] were harness failures — draw nothing from them. "
         "Respond with JSON only."},
        {"role": "user", "content":
         f"Phase: {phase['kind']} (train budget {phase['train_seconds']}s per "
         f"candidate). Propose at most {n_jobs} jobs — {rules}.\n\n"
         f"## Baseline / target\n{json.dumps(base_summary, indent=1)}\n"
         f"{parent_txt}\n"
         f"## Previous outcomes this run\n{json.dumps(outcomes, indent=1)}\n\n"
         f"## Lessons\n" + ("\n".join(f"- {t}" for t in lessons) or "(none)") + "\n\n"
         'Respond with JSON: {"jobs": [{"strategy": <specific plain-language '
         'strategy>, "parent": <parent candidate id or null>}]}'},
    ]


# ----------------------------------------------------------------- subagent

def recipe_subagent_prompt(phase, strategy, base_source, parent_source,
                           loss_source, param_cap, lessons):
    rules = ARCH_RULES if phase["kind"] == "architecture" else HP_RULES
    current = parent_source or base_source
    return [
        {"role": "system", "content":
         "You write one training-recipe file implementing exactly the strategy "
         "you are given; you do not free-associate. " + RECIPE_CONTRACT},
        {"role": "user", "content":
         rules +
         f"\nParameter cap: {param_cap:,} total parameters.\n"
         f"Held-out signal: mean val loss after {phase['train_seconds']}s of "
         f"training (harness-owned).\n\n"
         f"## Strategy to implement\n{strategy}\n\n"
         f"## Current recipe / architecture (your starting point)\n"
         f"```python\n{current[:14000]}\n```\n\n"
         f"## Harness-owned loss function (your model must stay compatible)\n"
         f"```python\n{loss_source[:3000]}\n```\n\n"
         f"## Lessons\n" + ("\n".join(f"- {t}" for t in lessons) or "(none)")},
    ]


def repair_prompt(note):
    return {"role": "user", "content":
            f"The recipe failed the harness's load/sanity check:\n\n{note}\n\n"
            "Fix it. Keep the same strategy. Respond with exactly one Python "
            "code block containing the full corrected file."}


@weave_op
def author_recipe(llm, phase, job, base_source, parent_source, loss_source,
                  param_cap, lessons, max_repairs, save_fn, check_fn,
                  fallback_source=None):
    """One recipe-candidate lifecycle: write file, cheap load-check, repair on
    raw feedback. Heavy (budgeted-train) evaluation happens later, serially."""
    from kernelevo.llm import extract_code
    msgs = recipe_subagent_prompt(phase, job["strategy"], base_source,
                                  parent_source, loss_source, param_cap, lessons)
    result = dict(strategy=job["strategy"], parent=job.get("parent"),
                  model_name=llm.model, code_path=None, repairs_used=0,
                  load_ok=False, failure_note=None)
    for attempt in range(max_repairs + 1):
        resp = llm.complete(msgs, meta={"recipe_phase": phase["kind"], "job": job,
                                        "fallback_source": fallback_source or base_source,
                                        "attempt": attempt})
        path = save_fn(extract_code(resp.text), attempt)
        result.update(code_path=path, repairs_used=attempt)
        ok, note = check_fn(path)
        if ok:
            result.update(load_ok=True, failure_note=None)
            return result
        result["failure_note"] = note[:2000]
        if attempt < max_repairs:
            msgs = msgs + [{"role": "assistant", "content": resp.text},
                           repair_prompt(note[:3000])]
    return result


def is_better_final(loss, accepted, winner):
    """Finals compare accepted candidates at the same finals budget."""
    return bool(accepted) and (winner is None or loss < winner['final_val_loss'])
