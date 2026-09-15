# Verified results and measurements

All numbers measured on molab, NVIDIA RTX PRO 6000 Blackwell Server Edition
(97.9GB), CUDA 13.2 driver, torch 2.x/py3.13 in the notebook venv. Incumbents
are per-op `torch.compile(dynamic=False)` re-measured in-session; acceptance
margins calibrated per run from measured noise (typically 3% in-model).

## The accepted kernel (nanochat)

- Target: karpathy/nanochat via agent-written adapter (11.5M params, batch 4).
- Lineages found: `relu2_mlp` (2.9–3.0%), `rms_norm` (2.3–2.4%).
- **Generation 1: `rms_norm` ACCEPTED** — "row-parallel Triton RMSNorm
  autograd.Function" (Kimi-K2.7-Code). Training step **8.95ms → 8.58ms**
  (−4.1% vs torch.compile incumbents; eager reference 8.73ms → −1.7% vs eager),
  MFU 0.037. Source: `kernels/accepted_rms_norm_nanochat.py`.
- Win mechanism: fused forward+backward pair beating inductor's autograd-
  generated backward; the −4.1% step win exceeds the op's ~2.3% GPU-time share
  because the 8ms toy step is kernel-launch-bound, so replacing several
  launches with one saves wall time beyond pure GPU time.
- **Generation 2: correct rejection** — a multi-row-packed RMSNorm passed
  gate 3 (faster in isolation than the accepted kernel) but measured 8.66ms
  in-model vs re-measured incumbent 8.89ms: a 2.6% delta *inside* the 3%
  calibrated noise margin → rejected. The verifier declining a flattering
  number is the system's core trust property working.

## Correct-but-slower kernels (the honest negatives)

- `cross_entropy` (modern-lm, 481M): Kimi's kernel passed full correctness
  (outputs + gradients) — the first CE candidate to do so — then failed gate 3:
  **21,483µs vs inductor's 7,781µs** (2.8× slower). Source:
  `kernels/correct_but_slow_cross_entropy.py`.
- `rms_norm` forward+backward (modern-lm): correct, **608µs vs 336µs** — 1.8×
  slower than inductor in isolation.
- Interpretation: correctness is reachable; beating an autotuned compiler on
  memory-bound ops requires tuning quality that one-shot + one repair rarely
  produces.

## Baselines and profiles

| model | params | batch | step (eager impls) | addressable lineages |
|---|---|---|---|---|
| nanochat (agent adapter) | 11.5M | 4 | 8.24–8.87ms | relu2 3.0% + rms 2.3% ≈ 5% |
| modern-lm (batch 12) | 480.9M | 12 | 457.2–457.8ms | swiglu 19.6% + rms 1.8% + CE 1.6% ≈ 23% |
| modern-lm (batch 16) | 480.9M | 16 | 583.5–617.0ms | swiglu 21.7–22.9% + linear_CE 5.4% ≈ 28% |

- Reproducibility: repeated profiles of the same config agree to <0.1%
  (457.79 / 457.51 / 457.19ms across three runs).
- Toy-scale inversion: on nanochat, per-op torch.compile incumbents (8.95ms)
  measured *no better than eager* (8.73–8.87ms) — per-op compiled-callable
  guard overhead × 73 norm calls cancels the kernel wins on an 8ms step. At
  481M/600ms scale this inversion disappears.
- `linear_cross_entropy` routing (RUN job, stopped early): adapter successfully
  rewired head+loss through the fused boundary; the lineage profiled at
  **5.38%** of step vs 1.56% for bare CE — 3.5× the attack surface, plus the
  un-materialized-logits saving (~3.2GB bf16 per step at 32k tokens × 50304
  vocab) that torch.compile structurally never captures.

## Recipe-golf results (staged evolution over training recipes, 09-13)

- **Completed search (job 75890bd0): −7.21% held-out val loss at the 300s
  finals budget** vs the same-budget baseline (base model + harness AdamW).
  Winner: cyclic sawtooth lr schedule on a parallel-block +
  reduced-KV-projection architecture, lineage baseline → #8 → #16 → #24 → #29
  across 2 arch gens + 1 hyperparam gen.
- **Horizon compression measured**: proxy-budget wins ≈ +15% shrank to +7.21%
  at 5-min finals; the best 120s-proxy candidate (+15.8%) *lost* the finals —
  why top-k re-trains at the target budget instead of trusting proxy rank.
- Cap-truncated run (a6f5541e): +2.29% after one generation (positional-
  encoding variant; corrected from a mis-announced +1.91% — see bug 23).
- **Data-realism control**: with synthetic uniform tokens the pipeline
  verifies but every val loss pins at the entropy floor (10.8438–10.8750,
  bf16-quantized ln 50304). With real FineWeb-Edu (run b3c7b7aa, capped 10M
  tokens): 60s → 6.6137, 120s → 6.3979, 300s → 5.5222 (191/384/961 steps).
  The floor signature is now the documented tripwire for fake data.
- **Second completed search (job b3c7b7aa): −4.85% val loss at 300s**
  (5.5222 → 5.2546, ≈23% perplexity at equal wall-clock) for $24.39 LLM
  spend, despite losing 7/8 hyperparam candidates to bug 29. Winner: the
  gen-2 crossover of gen 1's two best ideas — lean attention (QK-norm +
  value-embedding ablation) × RMSNorm-everywhere.
- **Finals upset replicated**: the best proxy candidate (hot-LR, +8.48% at
  120s) compressed to +4.64% at 300s and lost to the architecture crossover
  (+4.85%) — the second consecutive run where matched-budget finals
  overturned the screening leader. Horizon compression is a replicated
  property of short-budget screening, not a one-off.
- **Horizon dependence, measured from the inside**: the winning lineage's
  founding "win" (+4.56% at 60s) came from *removing* the repo's QK-norm and
  value-embedding pathway — long-horizon stabilizers that only cost step
  time at 191 steps. Gen 2 re-added QK-norm and scored worse (6.4041 vs
  parent 6.3122): the search ran the ablation in both directions and agreed.
  Recipe wins are claims about the evaluated horizon, nothing longer.

## Claude provider first light (job c00e9711, kernel mode, 09-14)

- `claude_oauth:sonnet` (Claude Sonnet 5 via subscription OAuth + relay, $0
  metered spend) ran the kernel search end to end: calibration cheats
  rejected, Weave-traced completions, relay round-trips stable.
- Gate-3 discipline on a dead tie: Sonnet's streaming-logsumexp
  cross_entropy kernel measured **7706.9µs vs the inductor incumbent's
  7704.2µs** — functionally identical performance, correctly rejected
  against the 3% margin.

## Verifier integrity (every run)

- Both planted cheats (output-caching, shape-hardcoded) rejected at gate 2 at
  every calibration — the run aborts if they ever pass. One real incident:
  a *transient OOM* cached as the cheat's compile result made the self-test
  fail closed (abort), never open.
- Fresh-input trials caught the caching cheat on trial 2 each time; the unseen
  shape caught the hardcoded cheat each time.
