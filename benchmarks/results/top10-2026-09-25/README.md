# Top-K ten-repository Molab benchmark (2026-09-25)

[W&B summary and evidence artifact](https://wandb.ai/rianbutala-ucla/kernel-evolution/runs/4na2al2l) · [Machine-readable summary](summary.json) · [Candidate gate evidence](evidence.json)

Ten repositories were pinned to commits and attempted sequentially on a Molab NVIDIA RTX PRO 6000 Blackwell GPU. The system used W&B Inference model `deepseek-ai/DeepSeek-V4-Pro-0813` in `chat-no-thinking` mode. Training batches were deterministic and synthetic, generated in memory; these results do not establish downstream task quality on real datasets.

Kernel results compare the full training-step time, including host-to-GPU transfer, against a `torch.compile` incumbent. Architecture results compare held-out validation loss after the same 120-second training budget and require a verified structural model change. Each row's pinned commit, attempt status, numerical result, and candidate gates are in the JSON files. A failed row has **no comparable measurement**, regardless of intermediate candidates.

| # | Repository | Kernel result | Architecture result |
|---:|---|---|---|
| 1 | Ultralytics | Failed before comparison | Failed before comparison |
| 2 | timm | **16.8612 → 15.2596 ms**, 9.499% faster than compiled baseline | **7.30554 → 6.97424 loss**, 4.535% lower |
| 3 | nanochat | **48.3519 → 47.5122 ms**, 1.736% faster than compiled baseline | Failed before comparison |
| 4 | CleanRL | Failed before comparison | Failed before comparison |
| 5 | torchvision | Failed before comparison | 2.57138 → 2.75696 loss; structural candidate was worse |
| 6 | Transformers | Measured; no accepted kernel | 0.00007215 → 0.00002586 loss; 64.154% relative reduction near the loss floor |
| 7 | Diffusers | Failed before comparison | **0.56563 → 0.49026 loss**, 13.324% lower |
| 8 | Stable-Baselines3 | Failed before comparison | -0.66606 → -7.44600 PPO surrogate objective; not a validated RL performance gain |
| 9 | LitGPT | Failed before comparison after notebook disconnection | 4.3481e-8 → 0 loss; synthetic-task floor effect |
| 10 | Detectron2 | Failed before comparison | Failed before comparison |

**Totals:** kernel: 3 measured, 2 accepted improvements, 7 failed before a completed comparison. Architecture: 6 measured, 4 accepted positive-loss gate improvements, 4 failed before comparison. The PPO row also passed the structural gate but has a negative baseline, so no percentage gain or validated RL improvement is claimed.

The two accepted kernel candidates beat the compiled incumbents. Eager PyTorch timings are not included in the published evidence, so no speedup over eager is claimed. Transformers and LitGPT architecture percentages are dominated by near-zero synthetic validation losses. LitGPT's kernel retry ended in an infrastructure failure, and the dispatcher stopped. W&B Inference credit exhaustion was not observed.

Architecture rows here all used the original recipe v1 acceptance protocol. A later correction to nonpositive-baseline acceptance and percentage logging is versioned separately and was not mixed into this report.
