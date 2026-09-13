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
- The recipe planner, like the kernel planner, has never voluntarily used the
  research subagent — offered `{"research": "<question>"}` each generation,
  it goes straight to jobs every time. Forcing one gen-1 dispatch is the
  known two-line fix if we want it exercised.

## Costs & latency (W&B serverless, 481M-scale prompts)

- Planner call ≈ $0.15–0.40; kernel candidate incl. repairs ≈ $0.7–1.5;
  DEV generation ≈ $2–2.5; whole DEV run $2.5–4.2.
- LLM call latency 60–120s+ for full-kernel completions (3–6k output tokens at
  serverless MoE throughput) — generations are LLM-bound; GPU utilization is
  millisecond bursts (the stopwatch, not the load).
- Total session spend ≈ **$28 of $100** (meter: wandb.ai/subscriptions).
