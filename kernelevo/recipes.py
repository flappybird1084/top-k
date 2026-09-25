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
If the strategy asks to ADD something the model report shows is ALREADY
PRESENT, do not invent a substitute change to justify the label: state the
fact in your file's docstring and make no change beyond the strategy's other
components — an honestly-measured no-op teaches the planner more than a
smuggled unrelated edit. Do not bundle extras the strategy did not name
(learning-rate schedules, precision changes, size changes); the harness logs
a mechanical diff of every candidate, so undeclared changes are visible.
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
    sig = {
        "parameters": sorted((n, tuple(p.shape)) for n, p in model.named_parameters()),
        "modules": [(n, type(m).__module__, type(m).__qualname__)
                    for n, m in model.named_modules()],
    }
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:16]


def module_inventory(model) -> dict:
    """Counter of module class names — the mechanical 'what is this model made
    of' record used for the label-vs-diff line."""
    from collections import Counter
    return dict(Counter(type(m).__name__ for m in model.modules()))


def inventory_diff(base: dict, cand: dict) -> str:
    """Human-readable module-inventory delta ('' when structurally identical)."""
    if not base or not cand:
        return ""
    parts = []
    for name in sorted(set(base) | set(cand)):
        d = cand.get(name, 0) - base.get(name, 0)
        if d:
            parts.append(f"{'+' if d > 0 else ''}{d} {name}")
    return ", ".join(parts)


def model_report(model, max_chars: int = 9000) -> str:
    """Ground-truth report of the ACTUAL instantiated baseline model, given to
    the planner and subagents so strategies are proposed against reality
    instead of a guessed-at architecture (run ef48abdf spent two generations
    'introducing' RMSNorm/SwiGLU/RoPE/Muon that the repo already shipped).

    Contents: parameter count, module-class inventory, and the source of each
    distinct module class (via inspect), capped."""
    import inspect
    n_params = sum(p.numel() for p in model.parameters())
    inv = module_inventory(model)
    lines = [f"total parameters: {n_params:,}",
             "module classes (name: count): " +
             ", ".join(f"{k}: {v}" for k, v in sorted(inv.items()))]
    seen, budget = set(), max_chars - sum(len(l) for l in lines)
    for m in model.modules():
        cls = type(m)
        if cls.__name__ in seen or cls.__module__.startswith("torch."):
            continue
        seen.add(cls.__name__)
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        chunk = f"\n### class {cls.__name__}\n```python\n{src[:2500]}\n```"
        if budget - len(chunk) < 0:
            lines.append("\n(further class sources omitted for length)")
            break
        lines.append(chunk)
        budget -= len(chunk)
    return "\n".join(lines)


# ------------------------------------------------------------------ planner

def recipe_planner_prompt(phase, base_summary, outcomes, lessons, n_jobs, parents,
                          research_enabled=False, model_report=None):
    rules = ("propose ARCHITECTURE/optimizer modification strategies"
             if phase["kind"] == "architecture" else
             "propose HYPERPARAMETER-tuning strategies (architecture is frozen)")
    researchline = (
        'A research subagent is available: to have it investigate prior art '
        '(e.g. what worked in public training-speedrun efforts, papers, repos) '
        'before you plan, respond with ONLY {"research": "<question>"} and its '
        'brief will be returned to you (up to 2 dispatches). Optional — respond '
        'with jobs directly if you do not need it.\n' if research_enabled else "")
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
         f"## Baseline / target\n{json.dumps(base_summary, indent=1)}\n" +
         (f"\n## Baseline model — ground truth (instantiated structure and "
          f"class sources)\nVerify every strategy against this: proposing to "
          f"ADD a feature listed below wastes the slot on a no-op.\n"
          f"{model_report}\n" if model_report else "") +
         f"{parent_txt}\n"
         f"## Previous outcomes this run\n{json.dumps(outcomes, indent=1)}\n"
         "(strategies that FAILED TO AUTHOR were never evaluated — their idea "
         "is untested; re-proposing them in a simpler form is often worth a "
         "slot)\n\n"
         f"## Lessons\n" + ("\n".join(f"- {t}" for t in lessons) or "(none)") + "\n\n"
         + researchline +
         'Respond with JSON: {"jobs": [{"strategy": <specific plain-language '
         'strategy>, "parent": <parent candidate id or null>}]}'},
    ]


# ----------------------------------------------------------------- subagent

def recipe_subagent_prompt(phase, strategy, base_source, parent_source,
                           loss_source, param_cap, lessons, model_report=None):
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
         f"## Strategy to implement\n{strategy}\n\n" +
         (f"## Baseline model — ground truth (instantiated structure and "
          f"class sources)\n{model_report}\n\n" if model_report else "") +
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
                  fallback_source=None, model_report=None):
    """One recipe-candidate lifecycle: write file, cheap load-check, repair on
    raw feedback. Heavy (budgeted-train) evaluation happens later, serially."""
    from kernelevo.llm import extract_code
    msgs = recipe_subagent_prompt(phase, job["strategy"], base_source,
                                  parent_source, loss_source, param_cap, lessons,
                                  model_report=model_report)
    result = dict(strategy=job["strategy"], parent=job.get("parent"),
                  model_name=llm.model, code_path=None, repairs_used=0,
                  load_ok=False, failure_note=None, tokens_in=0, tokens_out=0)
    for attempt in range(max_repairs + 1):
        resp = llm.complete(msgs, meta={"recipe_phase": phase["kind"], "job": job,
                                        "fallback_source": fallback_source or base_source,
                                        "attempt": attempt})
        # summed from each response (exact per-candidate; pool-level deltas
        # would race across the parallel authoring threads)
        result["tokens_in"] += resp.input_tokens
        result["tokens_out"] += resp.output_tokens
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


def accepts_architecture_final(loss, baseline_loss, margin, arch_fp, baseline_arch_fp):
    """A recipe final must improve loss and retain a changed model structure."""
    return bool(arch_fp and baseline_arch_fp and arch_fp != baseline_arch_fp and
                loss < baseline_loss * (1 - margin))
