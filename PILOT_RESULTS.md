# Sol pilot results

The first live pilot completed **five evaluated generations and 20 candidates**
on molab. It stopped at `max_generations`, without spending the $100 ceiling.
**No speedup remains confirmed after the baseline audit.**

[W&B run](https://wandb.ai/stephenslee0127-acme/kernel-evolution/runs/s9ow4mff)
contains the final `audit/*` summary, audited candidate table, lessons, and audit
artifact. Earlier time-series entries remain visible as historical evidence.
The authoritative archive is `runs/sol-pilot/archive.sqlite`; a consistent copy
and the generated candidate sources have been retrieved to the local workspace.

| Usage | Amount |
| --- | ---: |
| Model | GPT-5.6 Sol, Codex OAuth |
| LLM calls, including connection check and recovery | 33 |
| Input tokens, including cached input | 480,552 |
| Cached input tokens | 50,688 |
| Output tokens | 48,725 |
| API-equivalent token cost | **$2.7142312** |
| Planner | $0.7105400 |
| Implementer, including connection check | $1.7653144 |
| Curator | $0.2383768 |

Dollar amounts estimate token cost at the configured published API rates. They
cover calls issued by the search harness. Interactive implementation work in this
Codex task and GPU rental costs are excluded; this is not an OAuth subscription invoice.

The workload was a 5,922,816-parameter JEPA-style model, batch 16, CIFAR-10 resized
to 64px, FP32 on an RTX PRO 6000 Blackwell Server Edition. Profiling selected EMA
updates; normalization backward fell below the 5% eligibility threshold. The
five-minute calibration raised the acceptance margins to **10.31% isolated** and
**6.06% whole-step**, while FP32 numerical tolerances stayed unchanged.

All 20 live candidates compiled and passed numerical checks. Seventeen stopped
at gate 3; three reached whole-step timing. One initially passed gate 4.
The zero-cost fixture run separately exercised a valid kernel, a compiler
failure, a numerical mismatch, and a cached-output cheat. Both calibration
cheats were rejected at gate 2.

The planner exposed an orchestration bug: multiple parents for the same
operation were incorrectly treated as cross-region fusion. Three empty planning
attempts were preserved in `generation_attempts`; their token costs remain
included. After fixing the validator, the saved plan was reused and the same
W&B run resumed. There were five evaluated generations, not eight.

The accepted candidate was `cand_01_03_ccf10e`, using
`target + 0.004 * (source - target)` and size-dependent Triton launch choices.
The original baseline timed the outer Dynamo wrapper around an Inductor seed.
That wrapper inflated the comparison. An independent audit called the captured
Inductor backend directly and preserved autograd and argument ordering.

| Audited comparison | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| Isolated EMA, shape-weighted median | 32.815 µs | 27.204 µs | Inconclusive across paired blocks |
| Full training step | 14.663 ms | 14.364 ms | Inconclusive across paired blocks |

The roughly 2% whole-step median difference did not consistently clear the
6.06% margin. Its acceptance was revoked. The original candidate row is preserved
in `candidate_audits`, along with all audit measurements, and the incumbent was
returned to the seed. The previously uploaded kernel artifact is marked with
`acceptance_invalidated` metadata. This is not a claim that the candidate is
slower; it is a failure to establish a robust improvement.

The code now extracts direct Inductor entries and rejects prepared runs from an
older benchmark protocol. A new experiment must recalibrate in a new directory.
No second paid search was started. The completed-run heartbeat is paused.

All 16 tests passed in the molab environment. GPU regression checks cover direct-entry argument ordering, unseen shapes,
autograd, and absence of a Dynamo re-entry during repeated calls. The second demo
adapter also completed a real forward/loss/backward/AdamW step on the pinned
TinyStories data: 21,589,632 parameters, sequence 512, batch 16. Its tied embedding
initialization was corrected after the smoke test exposed saturated initial
logits; initial loss is now approximately 5.54.

This is a tested pilot, not full v3 completion. General graph discovery,
cross-region fusion, overlapping CPU compilation, retrieved-seed ingestion,
and complete provider/telemetry integration remain listed in README.md.
