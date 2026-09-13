# Performance analysis: why inductor mostly wins, and where it can't

## The incumbent is a Triton-writing compiler

`torch.compile` per-op generates shape-specialized Triton and autotunes block
sizes/warps over a config sweep. On memory-bound ops (norms, softmax/CE,
elementwise chains) its output sits at ~85–95% of DRAM bandwidth — the ceiling
is physics, not code quality. Measured consequences:

- Kimi's correct RMSNorm: 608µs vs inductor 336µs (untuned-but-correct ≈ 2×).
- Kimi's correct cross-entropy: 21.5ms vs 7.8ms isolation.
- Our one win came from the compiler's soft flank (see below), not from
  out-tuning it.

## Amdahl at training scale (measured)

A transformer training step ≈ 65–75% GEMMs+attention (cuBLAS / flash-attention
— already at hardware peak *in eager*), ~23–28% fusable glue, remainder
optimizer/data. Our profiles: nanochat addressable ≈ 5%, modern-lm ≈ 23–28%.
This is why whole-model torch.compile only yields ~20–40% end-to-end, and it
bounds ANY per-op optimizer, including ours: whole-step wins from like-for-like
ops are structurally single-digit.

## Where wins actually live (ranked, with evidence)

1. **Boundary-moving fusions the compiler never performs.** Head-matmul+loss
   (`linear_cross_entropy`): at 32k tokens × 50,304 vocab, ~3.2GB of bf16
   logits materialize fwd + grads bwd, every step; a chunked/online-logsumexp
   kernel deletes that traffic. Liger-Kernel's public numbers on this exact op
   are double-digit. Our adapter agent successfully routed modern-lm's
   head+loss through this boundary (5.38% profiled share vs 1.56% bare-CE) —
   the funded search on it was stopped before generation 1.
2. **Backward passes.** Inductor's autograd-generated backwards (atomic
   weight-grad reductions, extra passes) are its weakest output. Our accepted
   nanochat kernel won *because* a hand-fused fwd+bwd pair replaced them.
3. **Multi-GEMM sharing.** modern-lm's SwiGLU: inductor runs 2 GEMMs + a
   pointwise kernel, writing/re-reading ~1GB of [32k, 4096] intermediates per
   layer-pair; a fused dual-GEMM with the gate in-register deletes that — IF
   the generated GEMM tiles reach ~cuBLAS throughput (the risk that ate every
   attempt so far, via broken backwards).
4. **Launch-overhead regimes.** On the 8ms nanochat step, per-op compile
   wrappers' guard overhead × 73 calls made torch.compile incumbents *no
   faster than eager* (8.95 vs 8.73ms) — and made our single-launch kernel's
   step win (−4.1%) exceed its GPU-time share (2.3%). Small-model/decode
   regimes reward kernel-count reduction beyond FLOP math.

## Training vs inference asymmetry

Dropping the backward (inference mode) raises agent correctness rates (every
observed swiglu/rms failure was in the backward) but simultaneously removes
inductor's weak flank and the loss-fusion jackpot — forward-only inductor is
inductor at its strongest. The famous handwritten inference wins (paged/flash
attention, decode megakernels, quantized formats) all live in machinery this
harness deliberately doesn't model (attention, KV cache, decode loop). Within
scope, inference-mode wins project to ~1–2% whole-step; training holds the
bigger single number via linear-CE. An inference "objective" exists in
bench.py as a wired-off capability.

## Trust machinery results

- Calibrated margins mattered: run-to-run incumbent step re-measurements moved
  by up to ±2.5% (8.89 vs 8.95ms) — exactly the noise the 3% margin absorbs,
  and exactly why a 2.6% "win" was correctly refused.
- The two-baseline display (eager AND torch.compile) exists because the eager
  comparison alone flatters every kernel project; ours is the harder claim.
