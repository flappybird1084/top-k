# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Implemented per **Kernel Evolution — Design Spec v3**; the spec still governs the invariants below. Layout: `search.py` (entry), `web.py` (barebones Flask frontend, :8420 — still the internal job engine), `ui/` + `ui_server.py`/`ui_asgi.py` (the real frontend: `uvicorn ui_asgi:app --port 8767` — repo intake with dataset discovery, per-run Settings dialog, diagram + generation panels; wraps web.py's job store/worker), `judges_server.py` + `kernelevo/judges_{pool,sandbox}.py` (time-limited public judging deployment; dormant unless `JUDGES_*` env is set), `config.py` (DEV/RUN profiles), `kernelevo/` (harness: ops registry, archive, gates + workers, LLM providers, loops, repo/adapter pipeline, molab dispatch), `adapters/` (JEPA + LM demo models), `notebooks/` (molab launcher + sqlite viewer), `tests/` (pytest suite), `findings/` (run ledger, harness-bug log, analyses). Secrets go in `.env` (committed with empty values — never commit filled keys).

Beyond the spec, the §11 "adapter-writing agent" extension is now in scope and implemented: given a git repo, `kernelevo/adapter_writer.py` surveys it, an LLM writes `adapter.py` for the repo's stated model, and `kernelevo/ingest_worker.py` verifies it in a subprocess with raw-traceback repair up to `max_debug_turns` — only then does optimization start. Arbitrary repo models are routed through the op registry by `kernelevo/patch.py`: a global `F.layer_norm` interception (the op's eager reference keeps the original captured at import — never break that, it prevents recursion) and an opt-in Linear→GELU(tanh) fusion pass (exact GELU is deliberately not fused — it would change the repo model's math).

## What this system is

Given any PyTorch model behind a small adapter contract, the system profiles one training step, picks the hot ops, and runs an evolutionary search over **Triton kernels** for those ops. Each generation: a planner LLM proposes named strategies → parallel subagent LLMs implement them → every candidate passes a **deterministic verifier ladder** → survivors enter a SQLite archive and seed the next generation. A curator LLM distills each generation's results into "lessons" injected into the next generation's prompts.

- **Objective:** minimize training step time (equivalently maximize MFU) for the fixed model and batch.
- **Correctness is a constraint, not an objective** — enforced by the verifier ladder, which is pure code. No model ever grades its own output.

## Hard scope boundaries (do not relax)

- Single GPU, single node. One full training step always: forward → loss → backward → optimizer. No forward-only/inference mode.
- Triton kernels only, and only for ops where inductor emits Triton. Attention (`F.scaled_dot_product_attention`), conv/patchify, and embedding lookups are explicitly out of scope.
- Loss is checked, never optimized.

## Architecture

```
search.py                     owns the GPU; runs the loop unattended
  calibrate()                 noise floor, tolerances, cheat self-test, flops_per_sample
  ingest() / profile()        adapter → targets.json (hot ops ∩ allowed ops, ≥5% step time)
  generation loop             planner → subagents → gates → archive → select → curator → stop checks
  archive.sqlite              authoritative local state
W&B / Weave                   one-way mirror only
```

Key invariants that hold the design together:

- **`archive.sqlite` is the only state the loop reads.** W&B/Weave is a fire-and-forget mirror; the loop never reads from it. Write SQLite first, then mirror.
- **The verifier ladder is external and deterministic** (gates ordered cheapest-first):
  1. Compiles (in a CPU process pool, off the GPU critical path; artifacts cached by source hash).
  2. Numerically matches eager — outputs **and gradients** — on top-3 observed shapes plus one unseen shape, fresh seeded inputs per call. Tolerances come from `calibrate()`, not constants.
  3. Faster in isolation vs the incumbent, **re-measured in-session** (never compared to a stored number), margin from `calibrate()`.
  4. Faster in-model — interleaved incumbent/candidate repetitions (`gate4_reps`, default 3), medians compared; warmup/steps come from the config profile (DEV and RUN differ). This gate produces the objective: `step_time_ms`, `samples_per_s`, `mfu`. Gate 3 rotates input content between timed iterations (outside the CUDA-event window, identically for both sides) so content-keyed memoization cannot pass.
- **Reward-hack defenses are a tested feature:** `calibrate()` runs two planted cheating kernels (output-caching, shape-hardcoded) through the ladder and aborts the run if either passes gate 2. Keep those cheats working as the regression test for the verifier.
- **Repair loop gives raw feedback, no diagnosis:** compiler message or numeric mismatch verbatim, up to `max_repairs` (default 3). A slower-but-correct kernel gets zero retries — it's archived with its strategy for the curator. Diagnosis on performance failure is a defined extension hook, not current behavior.
- **Adapter contract** (`adapter.py`): `build_model()`, `get_dataloader(split)`, `loss_fn(model, batch)`. The harness owns optimizer (fixed AdamW), seeding, step loop, profiling, calibration, verification — **no per-model code in the harness**.
- **LLM provider interface** is a `Protocol` (`complete`, `usage_usd`) with per-role config (`planner_llm`, `subagent_llm`, `curator_llm`, `adapter_llm`, `researcher_llm`); Anthropic/OpenAI/W&B-Inference/Stub implementations plus two subscription-OAuth providers that need no API key: `codex_oauth` (Codex CLI, ChatGPT login) and `claude_oauth` (`claude -p` headless — the engine the Agent SDK wraps; `--system-prompt` replaces the agent persona, json_mode is prompt-enforced with fence-stripping, transient failures retry once after 30s). On molab, OAuth completions ride the `KEVO_RELAY_DIR` file relay serviced by the dispatcher on the operator's machine — keys/logins never enter the sandbox; both report `usage_usd()=0`, so the spend cap does not bind on them. Web search is a SearXNG-backed research subagent the planner can dispatch (`{"research": …}`), relayed the same way from molab.
- **Workers are reaped on their result line, not process exit** (`kernelevo/procstream.py`): the orchestrator acts on `KEVO_RESULT {json}` the instant it is printed and kills the child (a worker that finished but wedged at interpreter shutdown once froze a run for 25 min), and an idle timeout kills workers silent past their window — long stages emit `KEVO_HEARTBEAT` lines to stay alive (and surface as progress in the log). Token usage is accounted per generation and per candidate (`tokens_in`/`tokens_out` columns on both tables). The adapter-writer records its fail→repair→verified trail in `adapter_attempts.json` (synced mid-run; shown in both UIs and mirrored to W&B).
- **Selection per lineage:** rank accepted candidates by gate-4 step time, keep top 40–50% as parents, always keep the incumbent, and keep one candidate per distinct strategy that passed gate 2 even if slow (fusion draws on strategy diversity). `FUSE` jobs are allowed from generation 2.
- **Failure handling:** per-lineage retirement after `retire_after` barren generations (budget redistributes); systemic halt if nothing passes gate 1 for `systemic_halt_after` generations (signals broken environment, not hard search); generation wallclock kills hung GPU subprocesses and resets the CUDA context. Every stop writes `stop_reason`.

The archive schema (`models`, `calibration`, `lineages`, `candidates`, `generations`, `lessons`) and the planner/subagent/curator prompt contents are specified in the design spec — consult it before changing either.

## Recipe-golf mode (merged to `main`)

`--mode recipe` (or the web form's mode select) switches from kernel evolution
to staged evolution over **training recipes**: architecture(+optimizer)
generations first, then hyperparameter generations (architecture locked to the
parent by parameter-shape fingerprint), then finals. Signal = held-out
validation loss after wall-clock-budgeted training. The harness owns
everything gameable: data, the base adapter's loss, val batches, eval code,
seeds, budgets, and a parameter cap. Candidates are single files exposing
`build_model()` / `make_optimizer(model)` / optional `lr_schedule(step)` /
`TRAIN_HINTS`. Key modules: `kernelevo/recipes.py` (contract + prompts +
authoring), `kernelevo/recipe_worker.py` (subprocess: budgeted train + holdout
eval; gates load/param_cap/arch_lock/sanity), `kernelevo/recipe_loop.py`
(phases → finals). Config: `cfg["recipe"]` (see `RECIPE` in config.py; all
knobs exposed in the web form and via `--recipe-json`). DEV schedule ≈ 42 min
GPU: 2×8×60s arch + 1×8×120s hyperparam + 2×300s finals. Wall-clock budgets
introduce ±1-step noise in val loss — `loss_margin_rel` absorbs it.
Audit-driven invariants (run ef48abdf): planner/subagents receive a
ground-truth `model_report` (instantiated module inventory + class sources) so
strategies can't "introduce" features the repo already ships; the harness owns
precision (`cfg["recipe"]["precision"]`, default bf16, cast uniformly onto
baseline AND candidates so throughput levers are equal); every evaluated
candidate gets a mechanical `[recipe-diff]` line (params Δ, module-inventory
Δ, schedule/hints, steps) that also feeds the planner's outcomes — trust the
diff, never the strategy prose.

## Running and testing

**Use uv for all Python tooling in this project** (user requirement): install dependencies with `uv pip install -r requirements.txt` / `uv pip install <pkg>` — never bare `pip` — and invoke Python as `uv run python …` (or the project venv's `python`), not `python3`. The molab remote dispatcher also prefers `uv pip` on the notebook, falling back to pip only if uv is absent.

- Full run: `python search.py --adapter adapters/jepa.py` (or `adapters/lm.py`). Profile from `--profile DEV|RUN` or `KERNELEVO_PROFILE`; default DEV. Output lands in `runs/<name>-<profile>-<ts>/` (also symlinked as `runs/latest`).
- Repo pipeline: `python search.py --repo <git-url-or-path> --comments "…" --max-debug-turns 5` — clones, writes + ingest-verifies an adapter, then optimizes. This executes repo and LLM-written code; GPU box only.
- Web frontends: `uv run uvicorn ui_asgi:app --host 127.0.0.1 --port 8767` is the real UI (repo URL + mode upfront; everything else — LLM provider, profile, spend cap, pasted molab pair prompt, recipe schedule — behind the Settings dialog, with `KEVO_UI_*` env / `KEVO_MOLAB_CONNECTION_FILE` as server-side fallbacks; a dataset-discovery step confirms the data link before queueing). `python web.py` → :8420 is the barebones form; both share the same `jobs/<id>/` store. Jobs run serially in a worker thread: "local" shells out to `search.py`; "molab remote" (`kernelevo/molab.py`) drives the notebook's marimo kernel HTTP API (`GET /api/sessions`, `POST /api/kernel/execute` with Bearer token + `Marimo-Session-Id`, SSE response) — it uploads the project as a base64 tarball, installs deps, launches `search.py` detached with log/exit files, polls the log back, services search/OAuth relay requests, and syncs the archive + candidate sources mid-run. Connection details come from the notebook's "Pair with agent" prompt (URL + token; the token shown on screen is masked — only the copied prompt has the real one). Session ids go stale on browser reconnect, so the session is re-resolved every call and the notebook tab must stay open. Restarting a server kills the dispatcher subprocesses of jobs it launched — don't restart while its run is live.
- **Harness smoke test — no model or API key needed**: `python search.py --adapter adapters/jepa.py --llm stub`. The stub planner/subagent/curator return deterministic fixtures (passing kernel, compile failure, mismatch, planted cheat — see `kernelevo/fixtures.py`); gates, archive, and calibration run for real. This is how gate bugs are reproduced.
- `--lineage=<name>` restricts a run to one lineage; `--llm provider[:model]` overrides all roles (e.g. `--llm claude_oauth:sonnet` — sonnet/opus/haiku aliases or full model ids).
- The loop requires a CUDA GPU (it runs on molab via `notebooks/molab_run.py`). On a CPU box you can still test most of the harness: gate 2 logic, the compile/verify subprocess ladder, adapters, archive, stub LLM, and prompts all work with `cfg["device"]="cpu"` — see the invariant that gates 3–4 (CUDA-event timing) and profiling are the only CUDA-hard parts.
- Viewer: `marimo run notebooks/viewer.py -- --archive runs/latest/archive.sqlite` (three cells, reads sqlite, no state).
- Tests: `uv run python -m pytest tests/ -q` (~44 tests: gates/architecture, claude/codex OAuth providers, recipe-audit helpers, UI settings, discovery, dispatch). Run it after any harness or ui_server change. Deeper verification is still the stub-mode run plus targeted `uv run python - <<EOF` scripts against modules.
