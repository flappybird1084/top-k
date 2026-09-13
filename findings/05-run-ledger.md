# Run ledger (chronological)

All jobs dispatched to the same molab notebook (RTX PRO 6000 Blackwell,
sb-d09d5c025a44d191.sb.molab.run) with Kimi-K2.7-Code on W&B Inference for
all roles. Job artifacts (kernels, archives, logs) live under `jobs/<id>/`
locally; W&B runs + Weave traces at
https://wandb.ai/rianbutala-ucla/kernel-evolution.

## Job 29b07762 — karpathy/nanochat, DEV
Repeatedly relaunched while hardening the harness (molab auth/UA, marimo
variable semantics, stale-exit poller, SQLite threading ×2, dataloader
cycling, wallclock split — see 02-harness-bugs.md). Registry grew rms_norm +
relu2_mlp for it. Final successful run:
- **rms_norm ACCEPTED gen 1: step 8.95 → 8.58ms (−4.1% vs torch.compile)**,
  MFU 0.037 — the system's first verified win.
- gen 2 rms_norm passed gate 3, **rejected at gate 4** (8.66 vs 8.89ms =
  2.6%, inside the 3% calibrated margin).
- relu2_mlp: identical gate-2 mismatch across 4 attempts (bf16 accumulation,
  ~1.4× over tolerance) — retired.
- Spend across all attempts ≈ $8.

## Job 32070283 — xerneas3318/modern-lm, DEV (batch 12)
First big-model target. Adapter OOM'd once then settled at 481M/batch 12,
step ~457.5ms; lineages swiglu 19.6% / rms 1.8% / CE 1.6%. Exposed the
orchestrator memory-retention bug (94.8GB held at self-test) and the
compile-cache poisoning bug. After fixes: swiglu failed gate 2 (grads),
CE aborted by the then-$3 DEV cap. No acceptances. Spend ≈ $7 across attempts.

## Job c419dea5 — modern-lm, DEV (batch 16, user-submitted via web UI)
Step 617.0ms; swiglu 21.7% / CE 2.2% / rms 2.0%. gen 1: swiglu backward-only
Triton CompilationError; **cross_entropy passed full correctness** (first CE
to do so) then failed gate 3 at 21,483µs vs 7,781µs. gen 2 aborted by the $3
cap ($4.17 total). Led to: $10 DEV cap, [infra] tagging of cap aborts.

## Job 7b4bf1b9 — modern-lm, DEV (all prior fixes live)
gen 1: planner returned a literal empty job list — a wasted generation that
also burned a barren tick per lineage (guard added afterwards). gen 2: swiglu
grad mismatch; rms correct but 608 vs 336µs at gate 3. All lineages retired;
first run to print the `[result]` final-performance block. $2.50.

## Job bbcb0e14 — modern-lm, RUN profile ($45 cap) — stopped by user
The first funded search: 8 candidates/gen × 3 repairs, research relay live,
comments steering the head+loss through the new fused boundary.
Got through: adapter routed **linear_cross_entropy** (attempt 3; attempt 2 was
a head-monkeypatch binding bug), profile step 583.5ms with lineages
**swiglu 22.9% + linear_cross_entropy 5.4%** (RUN's 5% floor filtered the
small norms — all 8 candidates would have concentrated on the two real
targets). **Stopped by user during calibration, before generation 1.** ≈$1.
The configuration remains ready to relaunch as-is.

## Aggregate
- Credit spent ≈ $28 of $100 (authoritative meter: wandb.ai/subscriptions).
- Accepted kernels: 1 (nanochat rms_norm). Correct-but-slower: 2 (CE, rms at
  481M). Cheats rejected: every calibration, 9/9 runs.
- The RUN-scale search — the configuration designed to find acceptances —
  has never completed a single generation.
