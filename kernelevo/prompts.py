"""Prompt builders for planner, subagent, curator — plus the condensed Triton
language reference injected into every subagent prompt (spec §4.4)."""

from __future__ import annotations

import json

TRITON_REF = """\
# Triton quick reference (the subset in use)
import triton
import triton.language as tl

@triton.jit                       # kernels are launched as k[grid](args, META=...)
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)               # block index; also axis=1,2
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # tl.arange end must be power of 2
    mask = offs < n                           # guard every ragged load/store
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x * 2, mask=mask)

# launch: grid = (triton.cdiv(n, BLOCK),); k[grid](x, y, n, BLOCK=1024)
# tensors are passed as pointers; strides via t.stride(i); pass shapes explicitly.

# reductions (within a block):
#   tl.sum(x, axis=0), tl.max, tl.min — reduce over a block dimension.
#   row-wise ops: load a whole row with offs = row * stride + tl.arange(0, BLOCK_N)
# matmul: acc = tl.zeros((BM, BN), dtype=tl.float32)
#   loop over K in steps of BK: a = tl.load(...); b = tl.load(...);
#   acc += tl.dot(a, b)          # operands (BM,BK)x(BK,BN), dims >= 16
# block pointers:
#   p = tl.make_block_ptr(base, shape=(M,N), strides=(sm,sn), offsets=(m0,n0),
#                         block_shape=(BM,BN), order=(1,0))
#   x = tl.load(p, boundary_check=(0,1)); p = tl.advance(p, (0, BK))
# math: tl.exp, tl.rsqrt, tl.sqrt, tl.math.tanh, tl.where(cond, a, b), x.to(tl.float32)
# atomics: tl.atomic_add(ptr + offs, x, mask=mask)
# autotune:
#   @triton.autotune(configs=[triton.Config({'BLOCK': 512}, num_warps=4), ...],
#                    key=['n'])
#   @triton.jit
#   def k(...): ...
# constraints: BLOCK sizes are tl.constexpr and powers of 2; accumulate matmuls
# in fp32; loads/stores must be masked at boundaries; no python control flow on
# tensor values inside the kernel.
"""

OUTPUT_CONTRACT = """\
Output contract — respond with exactly one Python code block:
- One self-contained file. It must expose `kernel(*args)` matching the reference
  signature above. Optionally expose `autotune_configs`.
- For differentiable ops, autograd through kernel() must work: implement a
  torch.autograd.Function whose forward and backward launch your Triton kernels,
  and have kernel() call .apply(...). Gradients are checked against eager.
- Exact output shape and dtype are asserted. Inputs are fresh every call; one
  verification shape was never observed at profile time — do not hardcode sizes.
- No file I/O, no torch.compile, no caching results across calls.
"""


def planner_prompt(targets, summary, lessons, n_jobs, generation, fuse_allowed,
                   search_enabled=False):
    fuse = ("You MAY propose FUSE jobs that combine two or more accepted candidates "
            "(strategy starts with 'FUSE:', field 'parents' lists their ids). Never "
            "required; unfused variants stay in the population.\n"
            if fuse_allowed else
            "FUSE jobs are not allowed yet (first allowed at generation 2).\n")
    searchline = (
        'A research subagent is available: to have it investigate prior art or '
        'documentation before you plan, respond with ONLY '
        '{"research": "<question for the researcher>"} and its brief will be '
        'returned to you (up to 2 dispatches). Optional — respond with jobs '
        'directly if you do not need it.\n' if search_enabled else "")
    return [
        {"role": "system", "content":
         "You are the planner in an evolutionary search over Triton kernels. Each "
         "generation you propose implementation strategies; parallel subagents "
         "implement exactly what you name; a deterministic verifier accepts or "
         "rejects. Strategies must be plain-language and SPECIFIC — diversity "
         "across jobs comes from you. Past outcomes whose note starts with "
         "[infra] were harness "
         "failures — they say nothing about the strategy, so do not design "
         "around them. Respond with JSON only."},
        {"role": "user", "content":
         f"Generation {generation}. Propose at most {n_jobs} jobs.\n\n"
         f"## Targets (profiled hot ops)\n{json.dumps(targets_brief(targets), indent=1)}\n\n"
         f"## Archive summary (incumbents, last generations, parents)\n"
         f"{json.dumps(summary, indent=1)}\n\n"
         f"## Lessons from previous generations\n" + ("\n".join(f"- {t}" for t in lessons) or "(none)") +
         "\n\n" + fuse + searchline +
         'Respond with JSON: {"jobs": [{"lineage": <op name>, "strategy": <specific '
         'plain-language strategy>, "parent": <candidate id or null>'
         ', "parents": [<ids>]  // FUSE only\n}]}\n'
         "Prefer lineages with the largest % of step time and strategies that the "
         "lessons and past outcomes suggest are untried or promising."},
    ]


def targets_brief(targets):
    return {l["op"]: {"pct_step_time": l["pct_step_time"],
                      "shapes": l["shapes"][:2],
                      "signature": l["signature"]}
            for l in targets["lineages"]}


def subagent_prompt(job, lineage, parent_source, parent_latency_us, lessons):
    if parent_source:
        lat = f"latency {parent_latency_us:.1f}us" if parent_latency_us else "latency unknown"
        parent = f"## Parent kernel ({lat})\n```python\n{parent_source}\n```\n"
    else:
        parent = "## No parent kernel — start from the strategy and the reference.\n"
    return [
        {"role": "system", "content":
         "You write a single Triton GPU kernel implementing the exact strategy you "
         "are given. You do not free-associate or pick a different approach. "
         "Correctness against eager PyTorch (outputs AND gradients) is verified by "
         "an external deterministic harness; then it must beat the incumbent's "
         "latency. " + OUTPUT_CONTRACT},
        {"role": "user", "content":
         f"## Op: {job['lineage']}\n"
         f"Reference signature:\n{lineage['signature']}\n\n"
         f"Observed shapes/dtypes (verification uses these + one unseen):\n"
         f"{json.dumps(lineage['shapes'], indent=1)}\n\n"
         f"## Strategy to implement\n{job['strategy']}\n\n"
         f"{parent}\n"
         f"## Lessons from previous generations\n"
         + ("\n".join(f"- {t}" for t in lessons) or "(none)") + "\n\n"
         + TRITON_REF},
    ]


def repair_message(kind: str, detail: str) -> dict:
    if kind == "compile":
        body = f"The kernel failed to compile/run. Compiler output:\n\n{detail}\n"
    else:
        body = f"The kernel compiled but does not match eager numerically:\n\n{detail}\n"
    return {"role": "user", "content":
            body + "\nFix the kernel. Keep the same strategy. Respond with exactly "
                   "one Python code block containing the full corrected file."}


ADAPTER_CONTRACT = """\
Write ONE self-contained Python file (respond with exactly one code block) that
exposes the harness contract for the repo shown below:

  def build_model() -> torch.nn.Module      # the repo's stated/primary model
  def get_dataloader(split: str)            # "train" | "val" -> iterable of batches
  def loss_fn(model, batch) -> torch.Tensor # scalar, finite, requires grad

Hard requirements:
- Importing the file must be cheap: no training, no downloads at import time.
- build_model() returns the model on CPU; the harness moves it to the device.
- Deterministic: seed everything; get_dataloader must yield the SAME batches in
  the SAME order every time it is called. Constant shapes across steps strongly
  preferred (pad/crop if needed). The harness consumes ~60 consecutive steps per
  measurement — yield at least 100 batches (it cycles the loader if exhausted).
- Use the repo's real data pipeline if it runs offline with no credentials or
  large downloads; otherwise generate synthetic data matching the real batch
  spec (shapes, dtypes, value ranges) with a fixed seed, and say so in a comment.
- Pick a batch size that comfortably fits one GPU — but err LARGE: a training
  step should take at least ~20-50ms on a modern GPU, or the harness's timing
  gates have poor signal-to-noise and utilization looks idle. Unless the user's
  comments say otherwise, prefer the repo's real config scale over toy sizes.
- The repo is already on sys.path (the harness prepends that) — import its
  modules directly; do not copy model code unless imports are impossible.
- After constructing the model in build_model(), you may call
  `from kernelevo import patch; patch.fuse_mlp_blocks(model)` to expose fused
  Linear+GELU(tanh) blocks to the optimizer. F.layer_norm and F.rms_norm calls
  are intercepted automatically — do not rewrite norms.
- IMPORTANT: to give the optimizer targets, route the model's MLP epilogues
  through the op registry where a matching op exists: `from kernelevo import ops`
  then ops.gelu_mlp(x, w, b) for gelu(linear(x)), ops.relu2_mlp(x, w, b) for
  relu(linear(x))**2 (nanochat-style), or ops.swiglu_mlp(x, w1, w2) for
  silu(x@w1.T)*(x@w2.T) (Llama/SwiGLU-style; the down-projection stays
  separate). Patch the model's block forward (e.g. monkeypatch the MLP module
  class) rather than copying the whole model. Hand-rolled norm modules (a
  custom RMSNorm class) are NOT intercepted automatically — route them through
  ops.rms_norm(x, weight, eps) when the weight is a flat [D] tensor; leave
  norms with oddly-shaped weights native.
- DTYPES: op call sites require x, w, b in the SAME dtype. Repos that cast
  activations to bf16 mid-forward (nanochat does) while keeping fp32 weights
  will crash with "expected mat1 and mat2 to have the same dtype". Fix it
  structurally: convert the WHOLE model to one dtype in build_model() (e.g.
  model.bfloat16(), or strip the repo's internal .bfloat16()/autocast casts and
  stay fp32) — do not sprinkle .to(x.dtype) casts at call sites.
- If the model has an EMA/momentum-updated twin, apply it in an optional
  `model.post_optimizer_step()` method using
  `from kernelevo import ops; ops.ema_update(target_p.data, online_p.data, m)`.
"""


def adapter_writer_prompt(survey: str, comments: str, device: str):
    return [
        {"role": "system", "content":
         "You are the adapter-writing agent of a kernel-evolution harness. Your "
         "output is verified by running one real training step (model build, one "
         "batch, loss, gradient check); raw tracebacks come back to you until it "
         "passes or attempts run out. " + ADAPTER_CONTRACT},
        {"role": "user", "content":
         f"Target device (after harness moves the model): {device}\n\n"
         f"## User comments / guidance\n{comments or '(none)'}\n\n"
         f"## Repository survey\n{survey}"},
    ]


def curator_prompt(gen_digest):
    return [
        {"role": "system", "content":
         "You are the curator of an evolutionary kernel search. From this "
         "generation's outcomes, write 2-5 short standalone lessons (one line "
         "each) that will be injected into the next generation's planner and "
         "subagent prompts. Draw lessons from failures as well as acceptances: "
         "which strategies worked, which failure modes recur, what to avoid. "
         "Candidates whose failure_note starts with [infra] were killed by "
         "harness/environment failures — their outcome says NOTHING about their "
         "strategy; draw no kernel lessons from them (at most note the "
         "generation was lost to infrastructure). "
         "Respond with only the lesson lines, one per line, no numbering."},
        {"role": "user", "content": json.dumps(gen_digest, indent=1)},
    ]
