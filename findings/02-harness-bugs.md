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

## W&B account topology (not code bugs, but cost hours)

- New-format W&B keys are **org-scoped**; the same user's keys can hit
  different orgs. Inference access is an org-level entitlement and billing is
  selected by the `OpenAI-Project` header. The $100 credit lived on
  `rianbutala-ucla-org`; a differently-scoped key silently billed a different
  org until probed. Weave/W&B logging follows the key's org access too.
