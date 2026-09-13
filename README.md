# Kernel Evolution

An evolutionary Triton search with a deterministic verifier, SQLite archive,
Codex OAuth model calls, and optional W&B/Weave observation. The current GPU-tested
slice runs JEPA-style EMA replacement and fusion across independent EMA call sites
through all four gates. It is not yet the
entire Design Spec v3.

The first Sol pilot is complete: five evaluated generations, 20 candidates,
$2.7142 in token-equivalent cost, and no confirmed speedup after a baseline audit.
See [PILOT_RESULTS.md](PILOT_RESULTS.md) for preserved evidence and the corrections.
The subsequent handwritten fusion fixture reduced step time by 14.2% against a
fresh per-region Inductor baseline; see [FUSION_RESULTS.md](FUSION_RESULTS.md).
A subsequent whole-step compiler baseline was substantially faster, so the
fixture result is integration evidence, not a confirmed compiler improvement.

The first live pilot uses GPT-5.6 Sol, four candidates per generation, at most five
generations, and a $100 API-equivalent ceiling. The ceiling is a guardrail, not a
spending target. Costs are estimates from reported tokens at standard published
rates, not subscription charges. Unknown usage retains its reservation rather
than being recorded as free.

```bash
# In the Linux CUDA environment, from the repository directory:
python search.py --profile RUN --adapter adapters.jepa --lineage ema_update \
  --run-dir runs/new-experiment --prepare-only

# After reviewing preparation results and passing the stub checks:
python search.py --profile RUN --adapter adapters.jepa --lineage ema_update \
  --run-dir runs/new-experiment

# A standalone fixture search performs real GPU checks with no LLM calls:
python search.py --profile DEV --llm stub --adapter adapters.jepa \
  --lineage ema_update --run-dir runs/stub-check

python -m unittest discover -s tests -v
```

Use a new run directory for a new experiment. An existing prepared directory
reuses its measured calibration and seed sources; completed generations are not
rerun. CPU execution cannot produce accepted GPU candidates. Model parameters,
optimizer state, RNG, and the fixed batch are restored for paired step timing.
Data loading and correctness checks are outside timed regions.
Functional discovery is enabled by default. It recognizes supported LayerNorm
and Linear→GELU patterns in FX graphs and verifies loss, all parameter gradients,
tensor state, and scalar module state across three restored RNG states before
using a rewrite. It preserves the adapter's model object and parameter identities.
Unsupported traces and failed verification retain the existing module path, with
the reason recorded in `profile.json`.
Prepared directories with an older benchmark protocol are rejected. The original
`sol-pilot` directory is historical evidence and cannot be resumed with the
corrected direct-Inductor baseline.

The four gates compile/launch the candidate, compare numerical results on fresh
observed and unseen shapes, measure isolated speed, and measure complete
forward/loss/backward/AdamW/post-step runtime. Normalization backward outputs are
checked directly against autograd's input/weight/bias gradients. EMA runs after
the optimizer under no-grad. Microbenchmarks do not evaluate a validation dataset
or perform unrelated whole-model backpropagation.

Performance timing uses balanced baseline/candidate blocks. Every block must
clear the calibrated margin. A borderline case receives at most one additional
four-block measurement; uncertain results are archived as inconclusive. Each
candidate's GPU work runs in a disposable process. A timeout kills its process
group; persistent device failure may require stopping the environment.

CPU workers capture modal Triton launches using fake tensors, compile for the
recorded GPU target without initializing CUDA, and cache source/shape/target
artifacts before joining the GPU queue. Unsupported host scaffolding is deferred
to the GPU worker. Autotuning measurements and additional autograd compilation
remain on the GPU. Repair calls release the GPU lock. `compile_workers` and
`llm_concurrency` bound the workers; `subagent_models` optionally alternates models,
and each repair retains its candidate's original model.

From generation 2, a discovered `fused_ema_update` target may fuse all independent
EMA call sites using one or more accepted parent kernels. Its parent EMA lineage
and unfused variants remain available. Fusion tracing rejects aliases, dependencies,
and hooks with other state mutations. The complete tuple of EMA updates is the
unit of isolated comparison; gate 4 compares the original unfused hook with the
fused replacement. Normal timing excludes profiler labels.

Demo data is publicly available:

- JEPA-style ViT: 5,922,816 total parameters, CIFAR-10 resized to 64px, fixed 4x4
  target block on an 8x8 patch grid, context encoder, frozen EMA teacher, predictor.
  This is a small workload inspired by JEPA, not a reproduction of pretrained
  I-JEPA results. [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html)
- Decoder: 21,589,632 parameters, twelve layers, width 384, sequence length 512, byte vocabulary, tied
  embeddings, LayerNorm, GELU MLPs. Data is the first fixed 4,096 TinyStories train
  stories from revision `f54c09f`, cached with a SHA-256 checksum. No pretrained
  weights or dataset token is required. [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

`archive.sqlite` is authoritative. `prepared.json`, `targets.json`, seed sources,
candidate/repair sources, worker outputs, and logs live under the run directory.
W&B writes happen after archive commits and do not determine acceptance. The
marimo notebook is a three-cell read-only view of this state.

From the local development machine, `scripts/molab.py` executes scratchpad code
through the installed marimo-pair skill, `scripts/sync_molab.py` uploads source
files, and `scripts/pull_molab.py sol-pilot` retrieves a consistent archive backup
and candidate sources. Connection credentials live in a private file outside the
repository. W&B and Codex credentials also stay outside source and notebook cells.

Remaining v3 work is explicit:

- Extraction recognizes standard one-dimensional `nn.LayerNorm`, module and
  functional Linear/GELU patterns, functional LayerNorm with a weight, and explicit
  composite regions. General AOT graph discovery and RMSNorm backward are still
  incomplete; unsupported patterns remain eager and are not proposed for replacement.
- Isolated Inductor seeds call captured backend entries directly. Gate 4 uses
  whole-step `torch.compile` for both baseline and candidate integration, checking
  loss, parameter gradients, model state, and AdamW state before timing. Profiling
  still ranks regions in eager execution; mapping costs from whole-step compiled
  regions back to original operations remains incomplete.
- Fusion currently covers independent EMA call sites. Arbitrary cross-operation
  graph fusion and selecting arbitrary subsets of sites are not yet implemented;
  proposals without an executable contract fail closed.
- The active provider is Codex OAuth. Other provider adapters, retrieved-source
  candidate ingestion, and native-search result caching still need completion.
  `providers.py` contains API adapter prototypes that are not connected to the
  live search.
- W&B/Weave mirroring exists, but the full requested panel set, complete provider
  child-span coverage, and all trace URL fields need further integration.

Sol token rates used by this pilot are $4/M input, $0.40/M cached input, and $20/M
output, retrieved 2026-09-12. Long-context multipliers are represented in the
accounting module. [Official pricing](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
