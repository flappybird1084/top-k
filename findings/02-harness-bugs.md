# Harness bugs found in production (root cause → fix)

Every one of these surfaced during real runs, in roughly this order. All fixed
and pushed; the planted-cheat self-test and per-gate isolation meant none of
them ever produced a false acceptance.

## Remote execution (molab)

1. **403 on connect** — the "Pair with agent" prompt renders the auth token in
   markdown backticks (`` --token e27…7c`. ``); the fallback regex captured the
   backtick+period into the token. → token regex captures only token-legal chars;
   masked (`****`) tokens rejected with guidance.
2. **Cloudflare error 1010** — molab sits behind Cloudflare, which bans the
   default `Python-urllib` user agent outright. curl passes. → custom UA on
   every request.
3. **Marimo cell-private variables** — `_`-prefixed names are private per cell
   and don't persist across execute calls (`_kevo_b64` → NameError), and public
   names have single-owner semantics. → chunked upload accumulates in a *file*,
   not a kernel variable; per-call temps stay underscore-private.
4. **Stale `job.exit`** — relaunching into the same work dir left the previous
   run's exit file; the poller declared the new run dead while it was mid-work.
   → launcher deletes stale exit/log files first.
5. **Zombie PIDs** — finished runs linger as defunct processes that still pass
   `kill(pid, 0)`; aliveness must come from the exit file, not the process table.
6. **Stale server code** — a long-running web server caches imported modules,
   so fixes on disk didn't apply to new jobs (replayed the original 403 hours
   after it was fixed). → molab dispatch runs as a subprocess per job, exactly
   like local jobs exec search.py; server staleness can no longer affect jobs.

## Orchestrator

7. **SQLite cross-thread use** — subagent worker threads read parent info
   through the orchestrator's connection: first `check_same_thread` errors,
   then (with the check disabled) real `InterfaceError` corruption under
   concurrency. → workers never touch SQLite; the loop pre-resolves parent
   info before fanning out; thread check restored to fail loudly.
8. **Finite dataloaders** — agent-written adapters yield few batches; profiling
   burned 30+ `next()` calls → StopIteration mid-measurement. → the step
   cycles the loader (deterministic: fresh loader = same order).
9. **Single generation wallclock** — LLM latency + compile + gate 2 consumed
   the whole budget, so gates 3–4 never ran ("correct but unmeasured" kernels).
   → split budgets: authoring (`llm_budget_s`) and evaluation
   (`eval_budget_s`), eval's clock starts after authoring.
10. **Orchestrator GPU-memory retention** — at 481M scale the main process held
    ~94GB across calibration, starving 192MB worker allocations (misread as a
    compile failure at first). → numerical floors under `no_grad`, explicit
    del + `gc.collect()` + `empty_cache()` between phases; verified 0.08GB
    before self-test afterwards.
11. **Compile-cache poisoning** — that transient OOM was cached by source hash
    as a *compile failure*, so the identical planted-cheat source "failed"
    forever after, aborting every later run at self-test. → only successes
    persist across runs; failures cached in-memory for one run only.

## Learning-loop hygiene

12. **Infra failures fed to the learning loop as strategy outcomes** — the
    planner designed "single-kernel strategies to avoid SQLite threading
    fragility" and the curator advised "per-thread Triton caches": agents
    earnestly theorizing about a harness bug. → `[infra]` tag on all
    harness-caused failures; planner/curator instructed such entries say
    nothing about the strategy; lessons scoped per-run so persisted archives
    can't leak stale misdiagnoses.
13. **Empty planner output** — a literal `{"jobs": []}` parsed as valid,
    silently wasting a generation *and* a barren-retirement tick per lineage
    (this alone killed one run). → one push-back re-prompt before accepting an
    empty plan.
14. **Spend caps vs model scale** — DEV's spec'd $3 cap dies mid-generation-2
    at 481M (~$2/generation); aborted candidates were also mislabeled as
    compile failures in the UI. → DEV cap $10, per-job `--spend-cap` override,
    aborts `[infra]`-tagged.

## Adapter pipeline

15. **`from __future__` displacement** — the injected sys.path header pushed
    future-imports off line 1 → SyntaxError. → hoisted above the header.
16. **Cross-dtype call sites** — repos that cast activations to bf16 mid-
    forward (nanochat) crash fused op calls against fp32 weights; the agent
    burned 5 attempts patching symptoms. → contract rule: one dtype for the
    whole model, never per-call casts. Converged attempt 1 ever since.
17. **Adapter attempts wasted on re-derivation** — each relaunch rewrote the
    adapter from scratch, re-hitting solved errors. → verified adapters cached
    (`adapter.py.ok`) and reused if they still pass ingest.

## Recipe-golf mode (added 09-13)

18. **`no_grad` holdout eval** — generated adapters legitimately assert
    `loss.requires_grad` inside `loss_fn` (our own contract teaches it);
    wrapping eval in `no_grad` tripped them. → eval runs grad-ENABLED, never
    backward, `.detach()` immediately.
19. **Missing `post_optimizer_step`** — the recipe train loop skipped the
    adapter's EMA hook (JEPA target networks silently frozen). → honored
    after every `opt.step()`.
20. **Synthetic random tokens pass every gate** — an adapter yielding uniform-
    random tokens verifies cleanly (loss0 ≈ ln V, gradients flow) but pins val
    loss at the entropy floor (~10.83): the whole search signal becomes noise
    and *no correctness gate can see it*. → standing DATA rule: always search
    the repo for its real pipeline first; synthetic is a guarded last resort.
21. **Sandbox termination mid-run (HTTP 410)** — molab killed a sandbox during
    generation 2 and took the winning recipe sources with it. → mid-run sync
    pulls archive.sqlite + all candidate/recipe/adapter sources continuously.
22. **Indistinguishable W&B runs** — every run was named `run-recipe-dev`
    (model name resolved from a path literal). → `<name>-<mode>-<profile>-<ts>`
    naming, group = job work dir, tags; per-candidate W&B runs
    `<run>-g<gen>-s<strategy>` with their own charts.
23. **Finals compared against the previous winner's *proxy* loss** — run
    a6f5541e announced the wrong winner (GQA+FFN +1.91%; the true winner was
    positional-encoding +2.29%). → `is_better_final` compares only same-budget
    finals numbers.
24. **Web form had no spend-cap field** — jobs silently ran at the DEV $10
    default; a6f5541e was cap-cancelled after generation 1. → per-job cap
    field wired through `--spend-cap`.

## The 09-13 incident chain (one afternoon, three compounding failures)

25. **Unbounded dataset download vs ingest timeout → reward-hack regression.**
    Attempt 1 honestly streamed FineWeb-Edu with no cap and was killed at the
    900s ingest timeout; three repairs later the agent "passed" verification
    by generating synthetic tokens behind mount-point checks for directories
    that don't exist on molab (`/mnt/datasets/...`). Baselines came back
    10.8750 / 10.8750 / 10.8438 — the entropy floor, quantized to bf16 ulps
    (0.0625 apart at that magnitude): the fingerprint of bug 20 in production.
    → DATA rule hardened: subset-only downloads (~25M-token hard cap,
    stream-and-stop), reuse any cache an earlier attempt left, only read paths
    verified to exist, synthetic allowed only after an *actual* failed
    download with the caught error quoted in a comment; timeout 900→1800s.
26. **Worker hang-at-exit** — the next run's adapter was *perfect* (capped
    real download, probe passes in 6s) but `datasets`' streaming pool leaves
    non-daemon threads, so the worker printed its result and then wedged
    forever in interpreter shutdown. `subprocess.run(timeout=...)` observes
    only process *exit*, so a finished-but-immortal child was
    indistinguishable from a hang until the full 30-min ceiling. → workers
    hard-exit (`os._exit` past flush) AND the orchestrator now streams worker
    output (`procstream.run_result_worker`): it reaps the child the instant
    the `KEVO_RESULT` line lands (measured 0.5s vs the old 60s-timeout wait in
    the regression test) and kills anything silent past an idle window
    (ingest 600s, recipe evals 90s) — workers emit `KEVO_HEARTBEAT` lines
    (~8s in train loops) to prove liveness, which also narrates the formerly
    silent authoring/ingest stages in the log.
27. **Latent `NameError` in the hard-exit** — `recipe_worker`'s `import os`
    was function-local, so the fix in 26 would have crashed at the exit line
    (and only "worked" live by crash-exiting after the result was printed).
    Caught while wiring 26; module-level import now.
28. **Ops lesson: `ps` column truncation faked a dead run** — grepping
    truncated `ps` output on the sandbox matched nothing for `search.py`, and
    a live run mid-silent-authoring was misdiagnosed as silently SIGKILLed
    (use `ps auxww`). The structural fix is 26's heartbeats: silence is no
    longer ambiguous. The adapter's fail→repair→recovery trail is now also a
    first-class artifact (`adapter_attempts.json` → log recap, W&B table,
    web-UI self-repair panel).

## W&B account topology (not code bugs, but cost hours)

- New-format W&B keys are **org-scoped**; the same user's keys can hit
  different orgs. Inference access is an org-level entitlement and billing is
  selected by the `OpenAI-Project` header. The $100 credit lived on
  `rianbutala-ucla-org`; a differently-scoped key silently billed a different
  org until probed. Weave/W&B logging follows the key's org access too.
