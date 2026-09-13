# Top-Kernel — Stage Pitch (~2:40)

Point any GitHub repo at us. An evolutionary swarm of LLM agents rewrites its
training recipe — architecture, optimizer, LR schedule — and a deterministic
harness that no model can sweet-talk proves every claim on held-out data.
Same GPU, same wall-clock, better model.

---

## Slide 1 — Cold open (0:00–0:20)

**On screen:** two loss curves, same 5 GPU-minutes. Baseline 5.52, evolved lower.

> Everyone optimizes models. Almost nobody optimizes the *recipe* — the
> architecture tweaks, the optimizer, the schedule — because searching that
> space costs a grad student a month. We built a swarm of LLM agents that does
> it in 42 minutes, on any repo you hand it, and a verifier that makes it
> impossible for them to lie about the results.

**Beat:** say "impossible to lie" slowly. That's the thesis.

## Slide 2 — What it does (0:20–0:50)

**On screen:** pipeline diagram: repo URL → adapter agent → profile →
generations (planner → 8 parallel subagents → gates) → archive → curator.

> You give it a git URL. An agent reads the repo, finds the *real* data
> pipeline, and writes the adapter itself — debugging its own crashes from raw
> tracebacks, no human in the loop. Then staged evolution: architecture
> generations first, then hyperparameters with the architecture frozen by
> parameter fingerprint, then finals at a longer horizon. A planner LLM names
> strategies, eight subagents implement them in parallel, a curator distills
> each generation into lessons for the next. Kimi-K2 on W&B Inference, traced
> end-to-end in Weave, every candidate its own W&B run.

## Slide 3 — The verifier is the product (0:50–1:25)

**On screen:** the four gates + "cheat self-test: 9/9 rejected".

> Here's the part we're proudest of: correctness is a constraint, not an
> objective, and **no model ever grades its own output**. Every candidate runs
> the same wall-clock budget with harness-owned data, loss, seeds, and eval.
> The signal is held-out validation loss — nothing else counts.
>
> And we test the tester. Every run starts by injecting two deliberately
> cheating kernels — one caches outputs, one hardcodes shapes. If the ladder
> passes either, the run aborts. Nine calibrations, nine rejections. When an
> agent handed us random noise disguised as a dataset, the harness caught it
> the only way that matters: val loss pinned at the entropy floor, 10.875,
> flat forever. Real data reads 6.6 after sixty seconds. You cannot fake
> learning against held-out data you don't control.

## Slide 4 — Results (1:25–1:55)

**On screen:** three numbers, big: **+7.2%** · **−4.1%** · **9/9**.

> On a 481M-param modern transformer LM training on FineWeb-Edu: the evolved
> recipe — a cyclic sawtooth LR schedule on a parallel-block, reduced-KV
> architecture the agents designed — beats the repo's own recipe by **7.2%
> validation loss at equal wall-clock**. Free win, same GPU bill.
>
> Kernel mode is the same machine pointed lower: an evolved Triton RMSNorm
> beat `torch.compile` by **4.1%** end-to-end step time — verified against a
> re-measured incumbent, outputs *and gradients* matched to calibrated
> tolerances, not stored numbers.

## Slide 5 — It heals itself (1:55–2:20)

**On screen:** the adapter self-repair panel: attempt 1 FAILED (timeout) →
attempt 2 FAILED (OOM) → attempt 3 FAILED (TypeError) → attempt 4 VERIFIED.

> None of today's runs went smoothly — and that's the demo. Adapters crashed,
> downloads hung, a worker finished its job and then wedged at exit. The
> harness fed every raw failure back, repaired, and finished anyway. The UI
> shows the whole fail-and-recover trail, live. This is what unattended
> actually means.

## Slide 6 — Close (2:20–2:40)

> Compilers optimize what your code *is*. We evolve what your training *does*
> — and prove it with a judge no one can bribe. Point it at your repo tonight;
> read the lineage tree over coffee.

---

## Pre-stage checklist

1. `web.py` running, job page open on the lineage tree + self-repair panel
   (restart `web.py` first — panels shipped today).
2. W&B workspace filtered to the run group; per-candidate runs sorted by
   `val_loss`; the `adapter_self_repair` table pinned.
3. Weave trace of one planner → subagent → gate chain pre-loaded.
4. `findings/01-results.md` open in a tab for exact numbers under questioning.
5. Backup: screenshots of the +7.2% finals table and the cheat-rejection log
   line, in case the venue Wi-Fi dies.
6. If a run is live, know its current generation and spend before walking up.

## Strategic context

This is a W&B-sponsored event: lean into the native stack — W&B Inference
(Kimi-K2) powers every role, Weave traces every agent call, per-candidate
W&B runs make the evolution *visible*. The verifier-integrity story
(cheat self-test, held-out-only signal, entropy-floor catch) is the
differentiator; every other agent demo asserts, ours proves. Recipe golf is
the headline; kernel mode is the depth answer, not the lead.

## Q&A prep

**"Why not just torch.compile?"**
We use it — it's the *baseline* our kernels must beat, and beat by 4.1% on
RMSNorm. But no compiler will add four layers, swap your optimizer to a
sawtooth schedule, or redesign your KV projections. Recipe space is invisible
to compilers; that's why it's our headline.

**"How do you know the LLM isn't reward-hacking?"**
It's tried. Defenses: fresh seeded inputs per call, gradients checked, shapes
it hasn't seen, tolerances from a per-GPU noise calibration, incumbents
re-timed in-session, and two planted cheats that must be rejected or the run
aborts. The verifier is pure code; agents never touch data, loss, seeds, or
eval.

**"Is +7.2% actually significant?"**
The acceptance margin (0.3%) comes from measured run-to-run noise; 7.2% is
~24× that. And the finals stage exists precisely because short-horizon wins
compress: our best 60-second candidate (+15.8%) *lost* the 300-second final
to a steadier lineage. The harness catches its own proxy overfitting.

**"What happens with a garbage repo or garbage data?"**
Live demo answer: an agent once invented mount points and fell back to random
tokens. Val loss froze at the entropy floor and the run was worthless *and
obviously* worthless — 10.875 vs 6.6. We added download caps and
cache-verification rules to the adapter contract the same hour. The failure
mode is loud, not silent.

**"Why single GPU? Does this scale?"**
Deliberate scope: one GPU, one full training step (forward, loss, backward,
optimizer), so the signal is clean and every claim reproducible. The search
parallelism is in the agents, not the hardware — eight candidates author
concurrently and evaluate in a pipeline. Multi-GPU changes the engineering,
not the method.

## Honest caveats (not in the spoken pitch)

- Recipe wins are horizon-dependent: +7.2% is at the 300s finals budget on a
  481M model; we have not yet run multi-hour horizons.
- The kernel-mode win (−4.1% vs compile, −1.7% vs eager) is one op on one
  model; inductor sits near the memory-bandwidth ceiling on most single ops,
  and we say so — structural headroom is in boundary fusions
  (`linear_cross_entropy`) and recipe space, which is why we pivoted there.
- Wall-clock-budgeted training carries ±1-step noise in val loss; the
  acceptance margin absorbs it but exact numbers wiggle across reruns.
- ~$50 of the $100 inference credit spent across all runs, including the
  failures we show off.
