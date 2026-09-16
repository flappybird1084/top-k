# Independent audit — claims vs. evidence, and defects in the perimeter

Audit date 2026-09-15, against `rian-claude-agent-sdk`. Three independent passes:
one reconciling the README's headline claims against `jobs/*/run/archive.sqlite`,
the run logs, and the checked-in kernel artifacts; one reading the agent, UI, and
process-plumbing layers for correctness/liveness/security; one reading the
verifier ladder and op routing directly. Findings below were then spot-checked
against the raw sources.

**Verification legend** — every entry carries one:
`[arch]` verified against a run archive or log · `[read]` verified by reading the
code · `[run]` reproduced by executing the documented command · `[audit]`
single-audit report, not independently reproduced.

Scope note: this audits the *fixed* tree, not a running deployment. The judges
path is dormant unless `JUDGES_*` env is set; the relay/sandbox findings apply
there and are inert in ordinary single-user local use.

---

## Part 1 — Claims

### 1a. Reconciles to the digit (nothing is fabricated)

1. **nanochat kernel, `29b07762`** `[arch]` — candidate 22, `gate_reached=4`,
   `step_time_ms=8.5846`, `incumbent_step_time_ms=8.9507`, `mfu=0.03702`,
   `model_name='moonshotai/Kimi-K2.7-Code'`. The published artifact
   `findings/kernels/accepted_rms_norm_nanochat.py` is byte-identical to the
   archived candidate, and contains genuine Triton fwd + bwd kernels in a
   `torch.autograd.Function` with a real `def backward`. Counting is honest:
   **1 of 16** mutation candidates accepted; the other 8 `accepted` rows in that
   archive are `source_kind='inductor'` gen-0 seeds.
2. **Recipe run `75890bd0`** `[arch]` — the lineage
   `baseline → #8 → #16 → #24 → #29` checks out by `parent_id`; the 300s-vs-300s
   budget is genuinely matched and the baseline was re-measured in-session. The
   **horizon-compression story verifies exactly**: #27's 120s proxy was −15.8%
   vs the 120s baseline, and it then *lost* the finals to #29. That is the staged
   design catching horizon overfitting in the wild, not a claim about it.
3. **Planted cheats** `[arch]` — 9 calibration rows across 6 jobs, all 9 with
   `cheat_cache.correct_ok=0` **and** `cheat_hardcode.correct_ok=0`, with
   distinct failure signatures (the caching cheat dies at
   `max_abs_err=6.016e+03 … at index (2, 308, 804)` on trial 2; the hardcode
   cheat raises `AssertionError: unsupported shape` on the unseen shape). The
   mechanism is real code in `calibrate.py` and `raise SystemExit`s if either
   passes.
4. **Smaller numbers** `[arch]` — CE `21482.6` vs `7781.3` µs, rms `608.4` vs
   `336.3`, +2.29% (5.8433 → 5.7095) all check out.

### 1b. Inflated by omission (the arithmetic is right; the win statements strip the context)

5. **The flagship −7.21% is a 481M-vs-298M comparison at matched wall-clock**
   `[arch]`. Baseline `480,881,212` params → `val_loss 5.8477` at 300s; winner
   `298,391,326` params → `5.4258`. The winner is **38.0% smaller**; its
   sibling finals candidate (#28) is `303M`/`5.5781`, so *both* finals
   candidates were smaller models. The param cap is a ceiling only, shrinking is
   legal, and a smaller model takes more optimizer steps in the same 300s. The
   parameter count of the winner appears nowhere in `README.md` or `findings/`.
   This is the largest integrity gap in the doc set.
6. **The kernel win is quoted against the weaker of two available baselines**
   `[arch]`. `targets.json` records the target's eager step as **8.732 ms**; the
   `torch.compile` incumbent measured **8.95 ms** — inductor was *2.5% slower
   than eager* on this launch-bound step. So "−4.1% vs torch.compile" is
   "**−1.7% vs eager**". `findings/README.md` discloses both; the top-level
   `README.md` quotes only the flattering one.
7. **The −4.85% headline has no write-up at all** `[arch]`.
   `grep -rn "4\.85\|5\.2546" findings/` returns nothing; `01-results.md` reports
   only that run's baselines and `05-run-ledger.md` calls it "in flight". The
   top-level README nevertheless carries it as a headline result.
   **Resolved 2026-09-15**: the write-up existed on `main` (findings commit
   `2ba324e`) and postdated this branch's fork point — the audit grep ran
   against the branch. Merged in; `01-results.md` and `05-run-ledger.md` now
   carry the full b3c7b7aa account. Bonus correction found while closing this:
   the archive puts the b3c7b7aa finals winner at **480,881,212 params — the
   same count as the baseline** (like-for-like), and the README cell that
   briefly said "382M" (run `ef48abdf`'s winner, a conflation) is fixed.
8. **…and it conceals a 7-of-8 infra collapse in the phase meant to refine it**
   `[arch]`. The winner of `b3c7b7aa` is an *architecture*-phase candidate; the
   gen-3 hyperparam phase lost 7 of 8 candidates to
   `ModuleNotFoundError: No module named 'kernelevo.procstream'` (a deployment
   failure of bug 26's own fix, visible only in `candidates.failure_note`). The
   surviving candidate lost the finals, so that phase contributed zero.
   `log.txt` contains **no trace** — the run reads as a clean success.
9. **The cheat boast is scoped to a mode that has no calibrations** `[arch]`.
   All 10 recipe archives have **zero** calibration rows; recipe gates are
   load/param-cap/arch-lock/sanity, with no correctness gate analogous to gate 2.
   "Every calibration" is true and kernel-mode-only, and the ledger's "9/9 runs"
   conflates 9 rows across 6 jobs with runs.
10. **`cheats_rejected` carries no information** `[read]` — hardcoded to `1` at
    `kernelevo/loop.py:144`. It is true by construction (the run would have
    aborted), but the column is not evidence; `detail_json` is.
11. **Recipe "accepted" is not a quality signal** `[arch]` — it means *loaded,
    passed the caps, trained, finite val loss* (75890bd0: 22/29; b3c7b7aa:
    17/29). The doc set never spells this out.
12. **Both recipe wins are n=1**, `[arch]` selected *and* reported on the same
    fixed 8 eval batches (`eval_batches: 8, seed: 1234` across all candidates and
    the baseline). No test set, no seed replication, no multiple-comparisons
    discussion — each headline is the max over a 29-candidate search.
13. **Cost is contradictory and understated 1.5–2.5×** `[arch]`.
    `04-agent-behavior.md` says "≈$28 of $100"; `findings/README.md` and
    `05-run-ledger.md` say "≈$45–50"; summing the authoritative
    `[loop] stopped: … total LLM spend $X` lines gives **≈$65.4**, and the
    `claude_oauth` jobs log `$0.00` (flat-rate), so real cost is higher still.

### 1c. Contradictions inside the doc set

14. `03-performance-analysis.md` argues at length that showing both baselines
    means "ours is the harder claim" — in the same document that establishes the
    `torch.compile` comparison is the *easier* one on an 8ms launch-bound step.
15. `05-run-ledger.md` lists **1** completed recipe search; the archives hold
    **2**. The doc set does not contain its own second headline.
    **Resolved 2026-09-15**: same root cause as 7 — the updated ledger lived on
    `main` past the branch fork point. Post-merge, `05-run-ledger.md` records
    both completed recipe searches (75890bd0 and b3c7b7aa) plus the later runs.
16. `01-results.md` says "typically 3% in-model"; actual `gate4_margin` values
    are 0.010–0.060. CLAUDE.md's "10 warmup + 30 timed steps" is the RUN default;
    every cited run used `gate4_steps=10`, and "median-of-50" (gate 3) was 20 in
    DEV. The docs describe a profile the cited runs didn't use.

### 1d. Stated invariants that don't hold

17. **"Every stop writes `stop_reason`" fails for every recipe archive**
    `[arch]` — all 10 have an empty `generations.stop_reason` on the final
    generation (4 also have `n_candidates = NULL`). Kernel runs do write it.
    CLAUDE.md calls SQLite "the authoritative local state."
18. **A stated hard scope boundary has drifted in code** `[read]`.
    `cfg["objective"]` exists (`config.py:41`) and `bench.make_step` implements a
    forward-only `no_grad` path (`bench.py:63-67`), while CLAUDE.md lists
    "no forward-only/inference mode" as a boundary not to relax. It is currently
    **inert** — nothing reads `cfg["objective"]` — but the path exists.

---

## Part 2 — Defects

### 2a. Security / isolation (judges deployment only)

19. **The LLM relay directory lives inside the untrusted runner's writable tree**
    `[read]`. `molab.py:334` sets `KEVO_RELAY_DIR=work + "/run/search_relay"`;
    `judges_sandbox.py:15-18` chowns that entire tree to the per-run untrusted
    uid; the dispatcher answers LLM prompts by writing `*.res.json` into it
    (`codex_oauth.py:91-94`). Untrusted repo/candidate code can therefore read
    every pending `*.req.json` (full harness prompts, repo source) and write the
    matching response first. The dispatcher *polls*, so this is not a race to
    win. → move the relay outside `work`, or make it root-owned and write-only
    for the runner.
20. **`judges_sandbox.py` names a sandbox it does not run** `[read]`.
    `command()` — bubblewrap with `--unshare-user/pid/ipc/uts`, `--die-with-parent`,
    `--cap-drop ALL`, ro-bind allow-list — has exactly one occurrence in the
    codebase: its own `def`. The consumer (`molab.py:344-346`) pulls
    `user_command` by name. What actually executes is `setpriv` alone: no
    filesystem namespace, no network namespace, no seccomp, no read-only root.
    Credit: `user_command`'s docstring is honest ("not a replacement for a VM
    security boundary") and it **fails closed** if `setpriv` is missing or the
    uid is < 200000. The code doesn't lie; the filename does. → either wire
    `command()` in or rename the module.
21. **The Origin check is CSRF protection, not authentication** `[read]` —
    `judges_server.py:40` compares one request header, satisfiable by any
    script. The module docstring's "Restricted public gateway" oversells it. The
    real controls are good and are enforced: signed timed sessions, ownership
    checked against the *signed* visitor id, jid shape validated by
    `re.fullmatch` **before** any filesystem access, Idempotency-Key re-derived
    as `sha256(visitor + key)`, method allow-list, `MAX_CONTENT_LENGTH`.
22. **Hardcoded deployment origin** `[read]` — `judges_server.py:27` pins
    `https://top-kernel-demo.andre520395.chatgpt.site` with no env override; a
    redeploy 403s everything until someone edits source.
23. **A live molab token sits in plaintext in `jobs/*/job.json`** `[read]`.
    Not leaked — `jobs/` is gitignored and 0 files are tracked — but rotate it,
    and note the same pasted token is deliberately `pop`ped by
    `ui_server.py:347` and never exposed by `snapshot()`. No committed
    credentials: `.env` is gitignored, `.env.example` is empty-valued.

### 2b. Liveness and correctness

24. **The codex relay permanently wedges** `[audit]` (code `[read]`) —
    `codex_oauth.py:83-97`: `del self.pending[rid]` sits inside the success
    branch and nothing cleans up on failure, while `:84` skips submission once
    `len(self.pending) >= 8`. After ~8 transient delivery errors the provider is
    dead for the life of the process — no timeout, no expiry, no retry cap, no
    log line — and sandbox-side `complete()` then blocks 15 min per call before
    raising. The audit reproduced saturation at 8 with requests 9–10 never
    submitted. The sibling `claude_oauth.RETRY_DELAY_S` machinery shows transient
    failures were thought about on the CLI side and not the relay side. → always
    pop on all paths; log failures; bound the wait.
25. **`judges_pool._worker` dies permanently on one bad job** `[read]` — the
    `except Exception` at `:61-64` calls `self.load_job(jid)` *inside the
    handler*, and `load_job` is the first thing in the `try` (`:50`). A
    missing/corrupt `job.json` raises inside the handler, escapes `_worker`, and
    ends `while True`: that worker thread is gone, its queue never drains,
    later jobs sit at "Queued for GPU N" forever, nothing is logged, no
    supervisor. → read the job once outside the try; never call `load_job` in
    the handler.
26. **`check_login()` does not check login** `[read]` — `claude_oauth.py:28-31`
    only calls `shutil.which('claude')`. Codex's counterpart runs
    `codex login status` and rejects API-key auth, and a test covers it.
    `ui_server.py:141-143` uses it as configuration validation, so an
    installed-but-signed-out machine passes validation, queues the job, and
    fails at the first LLM call. → mirror the codex check.
27. **`extract_code` picks the longest fenced block** `[read]` —
    `llm.py:269-273` is `max(blocks, key=len)`. A longer "for reference, the
    original was…" block returns the *reference*, not the answer, in a repair
    loop whose prompt says "respond with the full corrected file". Also the
    regex requires a `\n` after the fence, so a single-line fenced answer matches
    nothing and is returned **with the fences still attached**. On the critical
    path of every candidate. → prefer the last block; require an explicit
    language tag; strip fences on the fallback path.
28. **The curator validates nothing** `[read]` — `curator.py:12-16` documents
    "2–5 lesson lines" and enforces no floor and no shape. Whatever the curator
    emits is injected verbatim into every subsequent generation's planner **and**
    subagent prompts.
29. **`procstream` reap-path truncation (latent)** `[audit]` — on the result-line
    reap path the parent breaks without joining its reader threads
    (`procstream.py:113-115`) then reads the sinks (`:133-134`); the audit
    measured 383 of 5000 trailing lines retained. **Harmless today**: every
    caller (`recipe_loop.py:58-62`, `gates.py:76-80`,
    `adapter_writer.py:101-110`) reads `r.result` on that path and touches
    `r.stdout` only on crash/timeout paths. It becomes a wrong-answer bug the day
    someone logs stdout on success.
30. **Leaked observer processes** `[read]` — `judges_pool.py:59` `Popen`s
    `judges_observer.py` without wait, timeout, or kill, and the observer is
    itself an unbounded poll loop; one leaked process per hung job. Its path is
    also rebuilt as `Path(__file__).parents[1]/'jobs'/jid` instead of
    `web.JOBS_DIR`, so it breaks when `JOBS_DIR` is overridden. `status()` reads
    `self.active` from another thread without the lock.

### 2c. Measurement

31. **Gate 3's inputs are frozen across iterations** `[read]` —
    `bench.time_op` loops `fn(*args)` over one fixed args list (`bench.py:29-43`),
    and `go = torch.randn_like(out)` is computed once. Gate 2's caching defense
    is *fresh seeds per trial*, which kills the naive replay-always cheat but not
    a cache keyed on input identity or content: a keyed memo passes gate 2
    cleanly and wins gate 3 for free. The threat model contains only the naive
    version. `scan_flags` would note a module-level cache, but flags are advisory
    and never auto-reject (`gates.py:32-34`). → rotate inputs between iterations
    (or vary the seed per iteration) in gate 3.
32. **Gate 4 — the gate that produces the objective — is the least rigorous**
    `[read]`. `verify_worker.py:183-184` is a single-shot A/B: one in-model run
    for the incumbent, a separate one for the candidate, no interleaving, no
    repetition. Its margin isn't measured either: `gate4_margin =
    max(default, lat_spread / 2)` reuses *half the isolation spread*, and the
    comment at `calibrate.py:77-80` admits it ("re-measuring is expensive; scale
    conservatively"). Every other gate re-measures; this one extrapolates, at a
    1% default margin.
33. **Routing is an unverified one-way door** `[read]`. `auto_route` replaces
    `mod.forward` (`patch.py:151,158,203`) and nothing ever compares the routed
    model against the unrouted one — `loss0` is recorded (`ingest.py:56`) but
    never checked against a reference. The probe tolerance is a hardcoded
    constant (`patch.py:102-105`: 2e-2 rtol bf16), the one place that violates
    the codebase's own "tolerances come from `calibrate()`, not constants"
    principle. A misroute silently changes the model's math and no gate can see
    it, because the incumbent runs on the changed model too. Mitigating and
    important: `auto_route` runs in *every* process that builds the model
    (`ingest`, `profiling`, `verify_worker`), so incumbent-vs-candidate stays
    apples-to-apples — the relative win holds, and the absolute claim "we sped up
    this repo's model" carries an asterisk. → add a routing-fidelity check
    (routed vs unrouted loss on one real batch) at ingest.

### 2d. Tests and dead weight

34. **The documented test command runs half the suite and reports green**
    `[run]`. `python -m unittest discover -s tests -q` → **Ran 22 tests, OK**;
    `python -m pytest tests -q` → **44 passed**. Seven files are pytest-style
    module functions taking `tmp_path`/`monkeypatch`, which the unittest loader
    skips **silently**. No `conftest.py`, no `pytest.ini`, no CI config, and
    CLAUDE.md says both "no test framework is set up" and "`tests/` (pytest
    suite)". → pick one runner and fix the documented command and CLAUDE.md.
35. **The suite is thorough where it exists and absent where it matters**
    `[audit]`. Assertions are real (idempotency keys, queue-put-once, secret
    masking, Origin/CSRF rejection, URL allow-lists dropping
    `javascript:alert(1)`, archive→snapshot shaping, token-column migration,
    precision casting, relay round-trips against a fake `claude`/`codex` binary).
    But **zero** tests import `planner`, `subagent`, `curator`, `prompts`,
    `researcher`, `procstream`, `ingest`, or any of the judges stack — the whole
    agent loop and the whole public deployment are untested.
36. **`web.py` carries ~400 dead lines (49% of the file)** `[audit]` —
    `_job_evolution`, `_lineage_tree`, `_job_files`, `_fragments`,
    `_adapter_trail` plus their templates duplicate `ui_server.snapshot`'s
    archive→view shaping, and **have already diverged**: `web.py` keys baselines
    off `generation == 0` (`:606`, `:704`) while `ui_server` splits on
    `row.get('phase')` (`:219-220`), and `ui_server` *infers* `mode='recipe'`
    from data (`:221`) where `web.py` trusts `job['mode']`. Two independently
    maintained derivations of the same thing.
37. **Stale comment contradicting the code below it** `[read]` —
    `claude_oauth.py:47-52` says "`--max-turns 1` plus the tool denylist keeps
    print mode from acting like an agent"; `:69` passes `--max-turns 4`, and a
    later comment (`:65-68`) explains the change. The provider's isolation is
    also prompt-level: there is no `--no-tools`, and the denylist covers only
    *known* tool names.

---

## Part 3 — What holds up, and why this is worth fixing rather than rewriting

Entirely real, and not the kind of thing a plausible-looking generator produces:

- **`procstream.py`** — both docstring guarantees implemented: the result-line
  reap genuinely fires (verified against a worker still sleeping), and the kill
  is a whole-group SIGKILL via `start_new_session=True` + `os.killpg`, with a
  `proc.kill()` fallback and bounded wait. `on_line` failures can't kill a reader
  thread.
- **`dispatch_once.py`** — 18 lines, `fcntl.flock`, atomic `.tmp` + `replace`, and
  a `dispatch-started` sentinel that *raises* rather than relaunching an unknown
  prior attempt.
- **`ui_server.py`** — `snapshot()` reads through a read-only SQLite URI and
  dedupes evaluation events against archived rows (naming the phantom-bubble bug
  it fixed); `redact()` strips ANSI and `api_key|token|password|secret`, `Bearer`,
  `sk-…` from LLM-derived text before it reaches the browser; `repo_url`/`data_url`
  are real allow-lists; `web.py:781-789` does realpath containment, not the naive
  prefix check. Autoescaping was checked specifically because the templates
  interpolate LLM-authored `strategy`/`failure_note`/kernel source — all four
  escape, and `|safe` is used only on server-built fragments.
- **The verifier core** — gate 2 checks gradients via `autograd.grad` with a
  random `grad_output`, rejects a candidate whose output doesn't require grad
  (telling it to wrap in a `torch.autograd.Function`), detects illegal in-place
  input mutation, and covers an unseen shape with fresh seeds per trial.
  Calibration only ever *tightens* (`max()` at `calibrate.py:75,79,98,99`), and
  gate 3 times the incumbent two lines before the candidate — "never compared to
  a stored number" is literally true.
- **Behavioral routing** (`patch.py:96-222`) — probes modules with seeded inputs
  against eager references, disambiguating rms_norm/layer_norm (bias × eps
  candidates) and swiglu/geglu/gate-order (all variants tried), with cheap
  structural guards first and unmatched modules left untouched.
- **The prompts are engineered from observed failures** — the subagent is
  forbidden to free-associate (diversity is the planner's job), the planner is
  told `[infra]` notes are harness failures that must not shape strategy, repair
  says "keep the same strategy", and the output contract *discloses* the
  anti-cheat mechanisms rather than laying traps.
- **02 and 04 are majority failure report, not victory lap** — 28 numbered bugs
  with root cause → fix, including an agent reward-hack no correctness gate could
  see, plus "strategy text ≠ behavior" and "the curator faithfully digesting
  garbage".

The through-line across all three passes is the same one: **the artifact is more
honest than its presentation.** A verifier that genuinely aborts on a passing
cheat, next to a README that quotes the easier of two baselines. A cheat
self-test that really runs, next to a boast scoped to "EVERY calibration" in a
mode that has none. A suite of real assertions behind a command that runs half of
them and says OK. A fail-closed isolation instinct filed under a name describing
isolation it never reaches.

The centre of this system is real engineering with a specific fingerprint of
operational debugging. The perimeter is where it would hurt you, and the
documentation inflates the perimeter specifically.

---

## Priority

| # | Item | Why first |
|---|---|---|
| 19 | Relay dir outside the untrusted uid tree | Only entry that lets untrusted code steer the harness; judges path only |
| 24 | Codex relay permanent wedge | Reproduced; silently kills the provider mid-run |
| 25 | `judges_pool._worker` self-killing handler | One bad file; silent permanent queue death |
| 27 | `extract_code` block selection | Five lines; on the critical path of every candidate |
| 34 | One authoritative test runner | The repo's thesis is verifiable claims |
| 31 | Gate 3 frozen inputs | The one gap in the reward-hack threat model |
| 5 | Disclose the winner's parameter count | A README edit, not engineering |
| 17 | `stop_reason` for recipe runs | CLAUDE.md calls SQLite authoritative |
| 33 | Routing-fidelity check | Turns an asterisk into a measurement |
| 32 | Gate 4 interleaving / real margin | The gate that produces the objective |

## Remediation (2026-09-15, same day, branch `rian-claude-agent-sdk`)

All ten priority items fixed, plus 22/26/37; regression tests in
`tests/test_audit_fixes.py`; suite 53/53 under the now-authoritative runner.

| # | Fix |
|---|---|
| 19 | Judge-run relay dir moved outside the runner-owned tree (`<work>_relay`, root-owned mode 0733: runner can create requests and read a response by exact name, cannot list); threaded through dispatch/poll/sandbox/Relay |
| 24 | `Relay.service` releases the slot on every path: delivery-failure retry cap (3), worker-exception error answers, 900s expiry sweep, saturation + failure log lines |
| 25 | `judges_pool._worker` survives a corrupt job (no `load_job` in the handler; traceback printed; thread and queue live on); `status()` under the lock; observer path uses `web.JOBS_DIR` |
| 27 | `extract_code`: last python-tagged block preferred, then last fenced block, single-line fences handled, stray fences stripped on the raw fallback |
| 34 | `pytest.ini` makes pytest the one runner; `UI.md`'s unittest command replaced (it silently skipped 7 files) |
| 31 | Gate 3 rotates input (and grad-output) content between timed iterations, outside the event window, identically for incumbent and candidate — content-keyed memoization now fails gate 3 |
| 5/6/9 | README results table discloses winner param counts (75890bd0: 298M vs 481M baseline, framed as a recipe win at this horizon; b3c7b7aa: 481M, like-for-like — an earlier "382M" in this cell conflated run ef48abdf and is corrected), quotes the kernel win against both baselines (−4.1% vs compile, −1.7% vs eager), and scopes the cheat self-test to kernel-mode calibrations |
| 17 | Recipe loop writes `stop_reason` (`recipe_complete` / `deadline_or_spend_cap`); a crashed generation closes its row with `crashed: <err>` instead of leaving NULLs; verified via stub pipeline |
| 33 | Ingest computes loss on the same batch before and after `auto_route` (grad enabled, seeded, detached) and aborts on drift > max(2% rel, 1e-3) — misroutes can no longer silently change the model's math |
| 32 | Gate 4 interleaves `gate4_reps` (default 3) incumbent/candidate measurements and compares medians |
| 22 | `JUDGES_ORIGIN` env overrides the hardcoded deployment origin |
| 26 | `claude_oauth.check_login` verifies sign-in with one minimal haiku call instead of only `which('claude')` |
| 37 | Stale `--max-turns 1` comment corrected; prompt-level isolation limits stated in the comment |

7 and 15 resolved by merging `main`'s findings commit (`2ba324e`), which
postdated the branch fork point — see the annotated entries above.

Still open from this audit: 8 (b3c7b7aa gen-3 collapse not in `log.txt` —
recorded here and in 02), 12 (n=1 / no seed replication), 13 (cost
reconciliation), 20 (`command()` bwrap unwired), 28 (curator output
unvalidated), 29 (procstream reap-path truncation, latent), 30 (observer
lifecycle beyond the path fix), 35 (agent-loop test coverage), 36 (web.py dead
lines), 18 (inert inference path vs stated boundary).

## Not audited

Whether the −7.21% would survive a test set or seed replication (see 12); the
molab remote dispatcher beyond the relay findings; `ops.py`'s per-op perturb
implementations beyond spot checks; the `.cjs` replay harnesses (run by neither
runner). The judges findings are static-analysis only — no deployment was
exercised.
