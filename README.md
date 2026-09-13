# Kernel Evolution

An evolutionary Triton agent platform with repository/prompt intake, an
agent-generated adapter, compiled-target discovery, parallel kernel-writing
agents, a deterministic verifier, SQLite archive, and linked Weave traces.
RUN uses GPT-6 Astra through Codex OAuth for adapter, planner, kernel agents,
and curator. The platform never supplies handwritten optimization candidates
to a live search. Supported operation contracts remain narrower than arbitrary
PyTorch; limitations are listed below.

The first Sol pilot is complete: five evaluated generations, 20 candidates,
$2.7142 in token-equivalent cost, and no confirmed speedup after a baseline audit.
See [PILOT_RESULTS.md](PILOT_RESULTS.md) for preserved evidence and the corrections.
Earlier handwritten experiments are documented separately in
[FUSION_RESULTS.md](FUSION_RESULTS.md); they are not agent search results.

Runs use four candidates per generation, at most five generations, and a
$100 API-equivalent ceiling. The ceiling is a guardrail, not a
spending target. Costs are estimates from reported tokens at standard published
rates, not subscription charges. Unknown usage retains its reservation rather
than being recorded as free.

```bash
# In the Linux CUDA environment, from the repository directory:
python search.py --profile RUN --model gpt-6-astra --repo /path/to/repository \
  --prompt 'Optimize the existing training model while preserving its data and loss semantics' \
  --run-dir runs/new-experiment

# A supplied adapter can be a module or a Python file:
python search.py --profile RUN --adapter adapters.language --run-dir runs/language

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
Repository source and generated adapter hashes are pinned at intake. The adapter
agent sees model/training source, excluding secrets, datasets, fixtures, and
candidate kernels. It reuses repository model/data/loss code; failed ingestion
returns raw feedback for bounded repairs. Finite loss and gradients establish
executability, not a proof of arbitrary repository semantic equivalence.
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
Weave export uses a durable SQLite outbox. A live run verifies server read-back
before making model calls. Explicit spans link run → generation → candidate →
agent calls/repairs and GPU gates. Prompts, responses, model names, token counts,
source provenance, gate results, and real trace URLs are recorded. Temporary
network failures retain unsent spans for replay.

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
- Compiled target discovery maps observed CUDA kernel names to Inductor source
  metadata. Associated fused-region cost can be shared and is only partial
  attribution; unknown/external kernels remain explicitly unmapped. It never
  treats eager timing as a compiled cost or promised speedup. No target over the
  configured cutoff yields a clean `no_targets` result, not fabricated jobs.
- Fusion currently covers independent EMA call sites. Arbitrary cross-operation
  graph fusion and selecting arbitrary subsets of sites are not yet implemented;
  proposals without an executable contract fail closed.
- The active provider is Codex OAuth. Planner-requested native research is cached
  by query and may seed retrieved-source jobs through the same gates. Alternative
  API provider prototypes in `providers.py` are not connected to the live CLI.
- W&B's full requested panel set is not yet configured. Individual agent and
  gate traces are explicitly exported to Weave, independent of SDK autopatching.

Sol token rates used by this pilot are $4/M input, $0.40/M cached input, and $20/M
output, retrieved 2026-09-12. Long-context multipliers are represented in the
accounting module. [Official pricing](https://developers.openai.com/api/docs/models/gpt-5.6-sol)

Whole-model agent search is available with `--search-scope whole_model`. Candidates expose
`install(model, optimizer)` and may replace instance-local computation throughout the model;
the harness retains the adapter, loss, data, training order, and full-step compilation.
This scope exposes the complete GPU profile, including external GEMMs, without the operator
allowlist. Each candidate must launch its own Triton kernel during backward. It is checked
against original eager on three real batches plus an unseen batch size, including two
sequential optimizer updates and separate parameter-update comparisons. Since the candidate
unit is the whole model, it proceeds from correctness directly to paired whole-step timing;
there is no isolated-op timing surrogate. Whole-model installations are complete programs,
so accepted parents can be extended or combined in later generations.

`--no-spend-cap` disables the dollar stop while preserving token/cost accounting. RUN still
has a hard maximum of five generations. Candidate optimization code is produced by the
configured platform model, not by the platform implementation.
