# Successful experiments

Explore the interactive results at [top-k.dev](https://top-k.dev).

| Experiment | Result | Evidence |
|---|---|---|
| timm | 9.499% lower step time; 4.535% lower validation loss | [Benchmark](../benchmarks/results/top10-2026-09-25/README.md) |
| Diffusers | 13.324% lower validation loss at 120 s | [Benchmark](../benchmarks/results/top10-2026-09-25/README.md) |
| nanochat | 1.736% lower step time | [Benchmark](../benchmarks/results/top10-2026-09-25/README.md) |
| modern-lm | 4.85% lower validation loss at 300 s; 481M parameters | [Case study](01-results.md) |

Kernel comparisons use compiled incumbents. Architecture comparisons use equal
training time. Benchmark workloads use synthetic batches; modern-lm uses
FineWeb-Edu. This is a selection of accepted results, not a success-rate claim.
Full attempt outcomes and candidate gates remain in the benchmark JSON.
