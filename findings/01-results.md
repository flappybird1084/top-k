# Recipe and kernel case studies

## modern-lm: 4.85% lower validation loss

At the same 300-second training budget on FineWeb-Edu, validation loss fell from
**5.5222 to 5.2546**. Baseline and winner both had **481M parameters**. The winning
recipe combined lean attention with RMSNorm changes. The result applies to this
training horizon. Run: `b3c7b7aa`; reported LLM spend: $24.39.

A second completed recipe search achieved **7.21% lower validation loss**
(5.848 → 5.426) at 300 seconds with a **298M-parameter winner** against a 481M
baseline. Its parallel attention/MLP block, reduced KV projections, and cyclic
learning rate traded model size for more training steps. Run: `75890bd0`.

## nanochat: a verified Triton kernel

A row-parallel RMSNorm forward/backward kernel reduced the measured training
step from **8.95 to 8.58 ms**: **4.1% lower** than the compiled incumbent and
**1.7% lower** than the measured eager baseline. This early run used an 11.5M
parameter adaptation, batch size 4, on an RTX PRO 6000.

[Accepted kernel source](kernels/accepted_rms_norm_nanochat.py).
This is a separate workload from the later ten-project benchmark.

## More projects

The [multi-project results](../benchmarks/results/top10-2026-09-25/README.md)
include accepted timm, Diffusers, and nanochat comparisons, with numerical
baselines, pinned repositories, and candidate gate evidence.
