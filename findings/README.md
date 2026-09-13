# Findings — kernel evolution, 2026-09-12/13 session

Everything learned building and running the system end to end: 5 jobs, 2 target
repos, ~9 remote GPU runs on a molab RTX PRO 6000 Blackwell, ~$28 of the $100
W&B credit spent (rest intact).

| file | contents |
|---|---|
| [01-results.md](01-results.md) | Verified performance results and all key measurements |
| [02-harness-bugs.md](02-harness-bugs.md) | Every bug found in production, root cause → fix |
| [03-performance-analysis.md](03-performance-analysis.md) | Why inductor is hard to beat, where wins actually live |
| [04-agent-behavior.md](04-agent-behavior.md) | How the LLM agents behaved: failure modes, costs, latencies |
| [05-run-ledger.md](05-run-ledger.md) | Chronological run log with configs, outcomes, spend |
| [kernels/](kernels/) | The two most important agent-written kernels (verbatim) |

**Headline facts:**
- One agent-written Triton kernel accepted through all four gates: RMSNorm on
  karpathy/nanochat, training step **8.95 → 8.58 ms (−4.1% vs torch.compile,
  −1.7% vs eager)**, gradients verified.
- The verifier rejected a *faster-in-microbenchmark* follow-up because its
  in-model win (2.6%) fell inside the calibrated 3% noise margin — and rejected
  both planted cheating kernels at every single startup.
- On a 481M-param stranger repo (xerneas3318/modern-lm) the platform
  self-assembled everything — adapter, op routing (73 norms + 24 SwiGLUs
  matched behaviorally), the fused `linear_cross_entropy` boundary (5.4% of
  step) — but no kernel was accepted at DEV search scale; the funded RUN search
  was stopped by user at calibration.

Live dashboards: W&B runs/artifacts and Weave traces at
https://wandb.ai/rianbutala-ucla/kernel-evolution (+ `/weave`).
