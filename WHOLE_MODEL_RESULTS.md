# Whole-model Astra run: interrupted

**No accepted speedup.** Molab returned HTTP410 `sandbox terminated` on the status check at 2026-09-13 05:02 UTC. W&B marks the run crashed. The cause of sandbox termination is not established.

[W&B run](https://wandb.ai/stephenslee0127-acme/kernel-evolution/runs/ox8clf2p) · [Weave run tree](https://wandb.ai/stephenslee0127-acme/kernel-evolution/r/call/28d6042f-4244-4b1f-9abd-f7437979e928)

The run reached generation2 of the authorized maximum5. Uploaded traces contain11 completed candidate lifecycles:6 passed full-model correctness and all6 reached full-step timing; none cleared acceptance. Generation1 also includes1 no-launch rejection,1 numerical rejection, and3 timeouts. Two further generation2 candidate lifecycles were unfinished in the recovered traces. These are not five completed generations.

All proposals and candidate sources came from platform Astra calls. The interface exposed the complete model/profile and required a candidate-owned Triton launch during backward. Original FP32 training semantics, data, loss, named parameters, and AdamW hyperparameters remained fixed. Whole-model candidates were compared directly in full-step timing, without an isolated-op timing surrogate.

| Candidate | Generation | Incumbent ms | Candidate ms | Outcome |
| --- | ---: | ---: | ---: | --- |
| cand_01_00_57a6a4 | 1 | 33.146 | 33.225 | gate4_slower |
| cand_01_01_cc5935 | 1 | 33.557 | 33.345 | gate4_inconclusive |
| cand_01_02_597f2b | 1 | 33.285 | 49.955 | gate4_slower |
| cand_02_00_b88fcc | 2 | 33.519 | 33.136 | gate4_inconclusive |
| cand_02_01_f16d84 | 2 | 33.260 | 33.267 | gate4_slower |
| cand_02_02_b9940c | 2 | 33.541 | 33.538 | gate4_slower |

Two platform guard false positives affected the first two candidates: lazy PyTorch compiler initialization modifies shared internal methods. They were reproduced with a no-op installation and fixed by warming/restoring the original compiled model before guard snapshots. No numerical or performance thresholds were changed to resolve them. Eight no-op GPU correctness checks and eight guard tests passed before resuming. The initial complete platform suite passed68 tests. Fix commits include `d17cb16` and `fcbce37`; per-result verifier hashes and platform patch events identify the change.

Recovered locally:157 Weave spans (149 finished),27 verbatim agent-source responses, and completed candidate results. Ten final candidate source hashes match recovered model responses; the eleventh completed lifecycle timed out before producing source. All six correct candidates have matching recovered source responses. Data is under `runs/astra-whole-model-001/`, including `weave_recovery.json`, `recovered_results.json`, and `recovered_agent_sources/`. The local SQLite snapshot predates the crash and must not be presented as a final archive. No recovered source was edited or executed during recovery.

The original GPU endpoint is unavailable. A fresh Molab pairing URL/token or access to the restored notebook is required to continue. Preserve the five-generation allowance when recovering; do not silently restart it. The monitor is paused until a working runtime is available.
