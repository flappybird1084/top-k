# Astra platform pilot

The autonomous platform completed five generations and 20 Astra-authored kernel candidates. All 20 ultimately passed compilation and correctness; **none established an accepted whole-model speedup**. The incumbent remains unchanged.

Run: `astra-platform-lm-002`, Molab RTX PRO 6000 Blackwell Server Edition, source commit `b07c16c` on `andre-branch`. Input was a repository and a prompt; an Astra adapter agent wrapped the existing 21.6M-parameter language model. The adapter, planner, kernel subagents, and curator all used `gpt-6-astra` through Codex OAuth. No candidate kernels or optimization strategies were manually authored for this run.

[W&B run](https://wandb.ai/stephenslee0127-acme/kernel-evolution/runs/1ij12pf9) · [Weave run tree](https://wandb.ai/stephenslee0127-acme/kernel-evolution/r/call/a7c02e80-6b8a-459f-95aa-ae9bf6b8a440)

| Result | Count |
| --- | ---: |
| Generations completed | 5 |
| Candidate lifecycles | 20 |
| Passed compilation and correctness after permitted repairs | 20 |
| Gate 3 slower | 11 |
| Gate 3 inconclusive | 4 |
| Gate 4 inconclusive | 5 |
| Accepted | 0 |
| Completed LLM calls, all linked in Weave | 33 |
| Finished Weave spans, confirmed exported | 168 |
| Trace export errors | 0 |

The loop advanced through planner proposals, parallel candidate agents, raw-feedback repair, deterministic GPU gates, archive writes, and curator lessons without manual strategy changes. There were 22 subagent calls for 20 candidates, five planner calls, five curator calls, and one adapter call. The run stopped automatically with `max_generations`; its heartbeat monitor was then paused.

The user approved a **2% target-selection cutoff** for this pilot because neither demo had an eligible target above 5%. The selected language-model LayerNorm backward region accounted for approximately 4.85% of compiled step time. This is an associated fused-region upper bound, not exclusive operator cost. Correctness checks and calibrated acceptance margins were unchanged: approximately 5.28% in isolation and 1% in-model. Five candidates reached whole-model timing; all were inconclusive. Isolated improvements are not reported as accepted training speedups. Microchecks measure correctness and latency, not BPD or validation-loss convergence.

| Model usage | Estimated USD |
| --- | ---: |
| Current adapter | 0.222600 |
| Planners | 5.844792 |
| Kernel subagents, including repairs | 4.991680 |
| Curators | 2.724370 |
| Current run total | 13.783442 |
| Two previous no-target intake attempts | 0.457680 |
| Combined Astra pilot | **14.241122** |

These are token-based API-equivalent estimates, not an invoice. They exclude GPU hosting and interactive platform development. The combined authorized ceiling was $100; no additional run was started.

After the search completed, all **59 platform tests passed on Molab** in 15.561 seconds. Final evidence lives in `runs/astra-platform-lm-002/archive.sqlite`, `result.json`, generated candidate sources, input provenance, and the linked Weave tree. Source and report changes were committed incrementally.

This verifies an actual autonomous Astra agent loop, not full coverage of every v3 extension. Discovery supports the implemented operator patterns rather than arbitrary PyTorch graphs; general RMSNorm discovery and arbitrary cross-operator fusion remain incomplete. Compiled-region attribution is partial, and the full custom W&B panel set is not configured. The pilot produced no performance improvement and should not be presented as one.
