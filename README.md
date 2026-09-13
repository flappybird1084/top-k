# Kernel Evolution

An evolutionary Triton search with a deterministic verifier, SQLite archive,
Codex OAuth model calls, and optional W&B/Weave observation. The current GPU-tested
slice runs JEPA-style EMA replacement through all four gates. It is not yet the
entire Design Spec v3.

The first Sol pilot is complete: five evaluated generations, 20 candidates,
$2.7142 in token-equivalent cost, and no confirmed speedup after a baseline audit.
See [PILOT_RESULTS.md](PILOT_RESULTS.md) for preserved evidence and the corrections.

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

- Extraction currently recognizes standard one-dimensional `nn.LayerNorm`,
  adjacent Linear/GELU modules in Sequential, and explicit composite regions.
  It does not yet discover arbitrary functional operator regions in an AOT graph.
- Inductor seeds call captured backend entries directly after extraction, without
  the outer Dynamo wrapper. They are compiled per region; this is not a whole-step Inductor
  baseline. Report speedups against the named baseline and include native eager
  step time when interpreting results.
- Cross-region fusion and CPU compilation overlapping GPU verification are not
  implemented. Cross-region fusion proposals fail closed.
- The active provider is Codex OAuth. Other provider adapters, retrieved-source
  candidate ingestion, and native-search result caching still need completion.
  `providers.py` contains API adapter prototypes and `fusion.py` an experimental
  EMA fusion contract; neither is connected to the live search.
- W&B/Weave mirroring exists, but the full requested panel set, complete provider
  child-span coverage, and all trace URL fields need further integration.

Sol token rates used by this pilot are $4/M input, $0.40/M cached input, and $20/M
output, retrieved 2026-09-12. Long-context multipliers are represented in the
accounting module. [Official pricing](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
