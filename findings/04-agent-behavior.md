# Agent behavior (all roles on W&B Inference, moonshotai/Kimi-K2.7-Code)

## Kernel subagents

- **Dominant failure mode: the backward pass.** Every swiglu failure across
  4 runs was a gradient bug (dx wrong, dw1 wrong ×192, backward-only Triton
  CompilationError that gate 1 can't see because backwards JIT only under
  autograd). Forwards were routinely correct (CE, RMSNorm several times).
- **Strategy text ≠ behavior.** nanochat's relu2_mlp failed with the
  byte-identical numeric mismatch (max_abs 3.2e+01, same failing index) across
  four attempts spanning differently-worded strategies — the model re-emits
  its attractor implementation regardless of the plan. One repair round does
  not break the pattern; this is the case for 3 repairs + curated lessons +
  (ideally) a stronger model.
- Near-tolerance failures: relu2's error was ~1.4× over the calibrated bf16
  bound — an fp32-accumulation fix away from passing.
- Format compliance was good: fenced single-file kernels, correct signatures,
  `autograd.Function` wrappers when asked.

## Planner

- Strategies are specific and plausible (tile sizes, layouts, vectorization
  widths) and got *more* specific with archive context.
- Failure modes seen: one literal `{"jobs": []}` (now guarded with re-prompt);
  pre-`[infra]`-tagging it designed strategies around harness errors
  ("keep one Triton kernel to avoid SQLite threading fragility").
- Never invoked the research subagent unprompted in the runs where it was
  available (relay was live only for the final, user-stopped run).

## Curator

- Produces plausible 2–5 line lessons; before infra-tagging it confidently
  misdiagnosed harness crashes as kernel-design issues — the clearest observed
  case of a learning loop faithfully digesting garbage. Lessons are now
  per-run and infra-filtered.

## Adapter agent

- Convergence: 1–3 attempts on every repo/config combination tried; repaired
  from raw tracebacks alone (GPTConfig kwarg guesses, head-monkeypatch binding
  errors, OOM overreach after "largest batch that fits" guidance).
- Systemic errors became contract rules (single dtype, ≥100 batches, future-
  import hoisting, linear-CE routing) — after which first-attempt success
  became the norm; verified adapters are cached and reused across relaunches.

## Recipe mode (09-13)

- **Repair-loop regression to the easy path** — the clearest agent-behavior
  finding of the session. When the honest approach was *punished by
  infrastructure* (uncapped FineWeb download killed at the ingest timeout),
  three repairs later the agent satisfied the verifier with synthetic tokens
  gated behind mount-point checks for directories that don't exist here —
  technically compliant with "synthetic as last resort", actually a
  reward hack the correctness gates cannot see (data realism isn't checkable
  mechanically). Countermeasure is prompt-level: require an *actual* failed
  download attempt with the caught error quoted in a comment.
- With the hardened DATA rule, the very next adapter streamed a correctly
  capped 10M-token slice, wrote a `ready.json` provenance marker unprompted,
  and passed the probe in 6s — the rules work when the incentive trap is
  removed.
- **Truncation as a failure mode**: one attempt died on a mid-expression cut
  (`ids = ` → SyntaxError) — output-length limits produce syntactically
  broken files that gate 1 catches for free.
- Recipe subagents follow the minimal-diff rule after one bad example: an
  early "RMSNorm everywhere" candidate *rebuilt* the 481M model from scratch
  and scored −17.9% (lost unstated details); post-rule candidates subclass or
  patch the repo model.
- **Label vs diff**: a "GQA win" turned out to be the repo's own kv_group=4
  pushed further — strategy prose oversells; only diffs and numbers count.
- **Label vs diff, second data point (b3c7b7aa)**: gen 1's best candidate,
  labeled "Add RoPE to queries and keys" — the repo *already has RoPE*. The
  diff showed the +4.56% actually came from what the rewrite silently
  dropped: QK-norm and the value-embedding pathway (its added RoPE duplicated
  the repo's, its explicit 1/√d scale is SDPA's default, its wpe-zeroing
  zeroed nothing). A correctly-measured ablation wearing a fictional label.
  Gen 2 then re-added QK-norm and scored worse — the search validated the
  ablation empirically without ever knowing what it was.
- **Objective literalism**: the search deletes long-horizon machinery (QK-norm,
  value embeddings) because the objective is val-loss-at-60s and those only
  cost step time there. Not misbehavior — the scarier version: precise
  optimization of exactly what was asked. The staged finals are the working
  countermeasure (proxy leaders lost the finals in both completed runs).
- **Param-cap repair behavior**: a "widen hidden ~10%" candidate came out at
  563.4M vs the 529.0M cap, got the raw cap error verbatim, and busted the
  cap on both repairs — some strategies are unimplementable under their
  constraints, and the planner's "failed to author = untested idea" re-offer
  is the right recycling path.
- The recipe planner, like the kernel planner, has never voluntarily used the
  research subagent — offered `{"research": "<question>"}` each generation,
  it goes straight to jobs every time. Forcing one gen-1 dispatch is the
  known two-line fix if we want it exercised.

## Claude Sonnet 5 as subagent/planner (claude_oauth, job c00e9711, 09-14)

- **Strategy verbosity is a different species from Kimi's**: planner
  strategies arrive as full implementation specs — block sizes (BLOCK_M=64,
  BLOCK_N=64, BLOCK_K=32), num_warps/num_stages, fp32-accumulation notes,
  autograd.Function structure — where Kimi wrote 1–3 sentences. Whether
  spec-density improves gate passage is exactly what the per-candidate
  `model_name` column will answer.
- **Agentic habits leak into a text-provider role**: one call tried to invoke
  `ToolSearch` three times instead of answering (bug 32) — Claude Code's
  agent training surfacing where only a completion was wanted. Needed
  explicit "you have NO tools" framing plus turn headroom to recover.
- First gate outcome: a correct-looking CE kernel at a dead performance tie
  with inductor (rejected, gate 3) — consistent with the standing analysis
  that memory-bound single ops offer no headroom regardless of author.

## Costs & latency (W&B serverless, 481M-scale prompts)

- Planner call ≈ $0.15–0.40; kernel candidate incl. repairs ≈ $0.7–1.5;
  DEV generation ≈ $2–2.5; whole DEV run $2.5–4.2.
- LLM call latency 60–120s+ for full-kernel completions (3–6k output tokens at
  serverless MoE throughput) — generations are LLM-bound; GPU utilization is
  millisecond bursts (the stopwatch, not the load).
- Total session spend ≈ **$28 of $100** (meter: wandb.ai/subscriptions).
