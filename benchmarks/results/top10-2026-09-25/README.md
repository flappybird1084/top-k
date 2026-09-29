# Measured project improvements

[W&B experiment](https://wandb.ai/rianbutala-ucla/kernel-evolution/runs/4na2al2l) · [Full numerical evidence](summary.json) · [Candidate gates](evidence.json)

Ten pinned repositories were tested sequentially on a Molab RTX PRO 6000 using
W&B Inference `deepseek-ai/DeepSeek-V4-Pro-0813`. This table highlights accepted
improvements. All attempts remain in the evidence files.

| Project | Kernel step time | Architecture validation loss |
|---|---|---|
| timm | **16.8612 → 15.2596 ms · 9.499% lower** | **7.30554 → 6.97424 · 4.535% lower** |
| nanochat | **48.3519 → 47.5122 ms · 1.736% lower** | — |
| Diffusers | — | **0.56563 → 0.49026 · 13.324% lower** |
| Transformers | — | 0.00007215 → 0.00002586; near-zero loss |
| LitGPT | — | 4.3481e-8 → 0; loss-floor effect |

Kernels compare full training-step time against compiled incumbents. Architecture
compares held-out loss at the same 120-second budget and requires structural
change. All batches here are deterministic and synthetic: these results do not
establish downstream task quality. Transformers and LitGPT are near the synthetic
loss floor; their relative reductions are not quality gains. A dash means no
accepted improvement in that domain.

This benchmark used recipe v1 acceptance. Later protocols are recorded separately.
