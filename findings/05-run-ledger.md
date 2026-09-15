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

# Recipe-golf era (2026-09-13; sandboxes rotate — sb-1b36…, sb-5141…)

All recipe runs target xerneas3318/modern-lm (481M) with the DEV schedule:
2 architecture gens × 8 candidates × 60s + 1 hyperparam gen × 8 × 120s +
top-2 finals × 300s; signal = held-out val loss; parent pool 4, cap ratio 1.1.

## Jobs 006e2ef1 / fbec6d72 — recipe shakedowns
First recipe-mode runs; surfaced the grad-enabled-eval and EMA-hook bugs and
the original synthetic-token entropy-floor incident that created the standing
DATA rule (bugs 18–20).

## Job a555f12c — killed by molab (HTTP 410) mid-generation-2
Sandbox terminated under the run; winning recipe sources existed only remotely
and were lost. → continuous mid-run source/archive sync (bug 21).

## Job 75890bd0 — the first complete recipe run: **−7.21% val loss**
Real FineWeb data. Winner `g3_hyperparam_4_a0.py`: cyclic sawtooth lr schedule
on a parallel-block + reduced-KV-projection architecture; lineage baseline →
#8 → #16 → #24 → #29. Finals upset: the best *proxy* candidate (#27, +15.8%
at 120s) lost at the 300s budget — measured horizon compression, proxy ≈+15%
→ finals +7.21%, and the reason finals re-train top-k instead of trusting
proxy rank.

## Job a6f5541e — cap-truncated; wrong winner announced then corrected
Web form had no spend-cap field → ran at the $10 DEV default and was
cap-cancelled after generation 1. Its finals comparison used the previous
winner's proxy loss (bug 23): announced GQA+FFN +1.91%, true winner
positional-encoding **+2.29%**. Also the origin of the label-vs-diff lesson:
the "GQA win" was pushing the repo's existing kv_group=4 further — trust
numbers and diffs, never strategy prose.

## Job d4f31b59 — the noise run (killed deliberately)
Attempt 1 honestly streamed FineWeb uncapped → killed at the 900s ingest
timeout; attempts 2–3 OOM / `Block.forward() missing 've'`; attempt 4
"verified" on synthetic uniform tokens behind fictional mount-point checks.
Baselines 10.8750 / 10.8750 / 10.8438 = entropy floor in bf16 ulps — the
search signal was pure noise. Killed at generation 1, ≈$1.77. → bug 25's
DATA-rule hardening + 1800s timeout.

## Job b3c7b7aa — the recovery run: **−4.85% val loss, second finals upset**
Same specs, fixed prompts. Attempt 1: truncated file (`ids = ` SyntaxError).
Attempt 2: adapter *correct* — streamed a capped 10M-token FineWeb-Edu slice,
whole probe passes in 6s — but hung at interpreter exit (bug 26), manually
reaped. Attempt 3: verified in seconds on the warm cache. Real-data baselines:
**60s → 6.6137, 120s → 6.3979, 300s → 5.5222** (vs the noise run's flat
10.87).
- **Gen 1 (arch, 60s)**: 5/8 accepted — best +4.56% ("Add RoPE", actually a
  QK-norm + value-embedding *ablation*, see 04); SwiGLU/RMSNorm +3.43%; MQA
  +2.81%; +2 layers +2.45%; sliding-window +1.53%. Lion evaluated but worse
  (7.04). One candidate burned both repairs on the param cap (563.4M vs
  529.0M) and was dropped pre-GPU — the cap gate costing seconds, not a slot.
- **Gen 2 (recombination)**: both crossovers of gen 1's top two ideas took
  the lead — lean-attention × RMSNorm-everywhere **6.2350 (+5.73%)** and the
  mirror-image 6.2596 (+5.35%). Re-adding QK-norm to the ablation parent
  scored *worse* (6.4041) — the search empirically confirmed the ablation.
- **Gen 3 (hyperparam, 120s)**: 7/8 candidates destroyed by a partial
  hot-patch (bug 29, `[infra]`-tagged, no false lessons). The lone survivor —
  lr bracketed *upward* on the gen-2 leader — posted the run's best proxy:
  **5.8557 (+8.48% at 120s)**.
- **Finals (300s)**: hot-LR compressed to +4.64% (5.2662) and **lost** to the
  gen-2 architecture crossover: winner **5.2546 (+4.85%**, ≈23% perplexity
  reduction at equal wall-clock**)**. Second consecutive finals upset of the
  proxy leader. Total LLM spend **$24.39**; exit 0, 107 artifacts synced.

## Job c00e9711 — first claude_oauth run (kernel mode, in flight 09-14)
modern-lm DEV, `--llm claude_oauth:sonnet` (Claude Sonnet 5 via the local
Claude subscription, relay-transported — spend $0.00 by construction). New
provider verified end to end in production: relay round-trips, Weave-traced
completions, calibration cheats rejected. Gen 1: cross_entropy kernel reached
gate 3 and was correctly rejected at a dead tie (7706.9 vs 7704.2µs); one
swiglu slot lost to bug 32 (ToolSearch under --max-turns 1), fixed on branch
`rian-claude-agent-sdk`. Profile note: DEV threshold admitted sub-5% lineages
(CE 1.55%, rms 0.96%) — swiglu (29.4%) is the real target.

## Aggregate
- Credit spent ≈ $70–75 of $100 (authoritative meter: wandb.ai/subscriptions);
  claude_oauth runs add $0 (subscription).
- Kernel mode — accepted kernels: 1 (nanochat rms_norm). Correct-but-slower:
  3 (CE ×2, rms at 481M). Cheats rejected: every calibration. The RUN-scale
  kernel search has never completed a generation.
- Recipe mode — completed searches: **2** (75890bd0 −7.21%; b3c7b7aa −4.85%,
  hobbled by the gen-3 wipeout); one cap-truncated (+2.29% at gen 1); one
  noise run caught and killed. **Both completed searches saw the proxy leader
  lose the matched-budget finals** — horizon compression is now a replicated
  observation, not an anecdote.
