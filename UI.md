# UI on Rian's branch

Worktree: `codex/rian-ui`, merged with `origin/rian` at `c1471b5` and architecture pivot `origin/rian-recipe-golf` at `fb5f7cd`.

The frontend now uses Rian's existing Flask job queue, `kernelevo.molab_dispatch`, and integer-ID archive schema. The previous platform backend is not included. Optimization and verification remain in `kernelevo`.

## Start

```sh
uv venv
uv pip install -r requirements-ui.txt
.venv/bin/uvicorn ui_asgi:app --host 127.0.0.1 --port 8767
```

For local GPU execution, also install `requirements.txt`. Molab dispatch installs its remote engine dependencies itself. This development server is intended for loopback access, not public hosting.

Configuration is server-side:

- `KEVO_UI_TARGET`: `molab` (default) or `local`.
- `KEVO_MOLAB_CONNECTION_FILE`: JSON with `url` and `token`; defaults to `~/.local/state/kernel-evolution/molab.json`. Secrets stay out of frontend responses and git.
- `KEVO_UI_PROFILE`: `RUN` (default) or `DEV`.
- `KEVO_UI_LLM`: optional existing Rian provider override.
- Existing `.env` settings configure model credentials and W&B. Install the optional W&B dependency in `requirements-ui.txt` to read run summaries.

Repo submission creates a draft; submitting the data link queues exactly one job. Rian's adapter agent then checks the repo and selected data together. This UI does not claim that a syntactically valid data URL has already passed sample verification. GitHub `/tree/branch` URLs are cloned using the selected branch, including slash-separated names.

The dashboard reads the latest model's candidates from `jobs/<id>/run/archive.sqlite`. Rian's gate field means **highest gate passed**. W&B/Weave URLs are retained from process logs, independently of the bounded terminal tail. New subagent lifecycle log events populate running evaluations while generation results are pending. Molab archive snapshots sync every ten seconds; the browser refreshes every three seconds. GPU readings are node-wide. The embedded marimo notebook reads the same archive adapter every five seconds.

Run cancellation is not exposed because this branch's dispatcher does not yet offer a safe remote cancellation contract. BPD is not computed by this adaptation; the tabbed results page uses validation loss for recipes and step time for kernels; it does not relabel validation loss as BPD. Illustrative playback is available only with `?demo=1`.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -q
```

Tests cover Rian's actual SQLite schema, branch URLs, idempotent submission, data handoff, secret omission, and cross-origin rejection. No GPU or paid model run is launched by these tests.

Codex OAuth: install Codex CLI on the UI server and run `codex login` with ChatGPT. Set `KEVO_UI_LLM=codex_oauth` in the server environment or ignored `.env`. The GPU sends completion requests through the existing authenticated Molab relay; the dispatcher invokes Codex locally. OAuth credentials are never uploaded to Molab. Keep this server running throughout the job. Requests time out after 5 minutes locally and 15 minutes in the relay; failures propagate to the engine. This is a local, single-user connection, not multi-user OAuth login.

Use `KEVO_UI_PROFILE=DEV` and `KEVO_UI_MAX_GENERATIONS=2` for the walkthrough. OAuth consumes the signed-in account’s subscription usage; the engine’s API-dollar estimate is zero for this provider and is not a subscription usage limit. Generation and wall-clock limits still apply. W&B authentication remains separate.

Architecture + Kernels UI: `/workspace.html?demo=1` is a local, illustrative playback (no training or observability writes). Both workspace and results pages have keyboard-accessible tabs. Architecture plots compare `val_loss` only against baseline rows with the same `train_secs`; the budget selector follows the latest evaluation until selected manually. Kernel metrics remain milliseconds and verified speedup. Architecture defaults to recipe mode at intake; the user can choose Kernels.

Live dual-run view: pass `architecture_run=<recipe-job-id>&kernel_run=<kernel-job-id>` with optional `view=architecture|kernel`. Each tab fetches its own archive, W&B summary, Weave traces, and notebook. A single run only populates its actual mode; the other view stays empty. The two engines run independently; this UI does not imply architecture winners have already undergone kernel search.

Recipe evaluations now emit running events, store available Weave evaluation URLs, and log candidate loss, matched-budget baseline, training budget, parameters, phase, and acceptance to W&B. Missing service credentials leave honest empty states. The finals selection fix compares accepted final losses, not earlier-round parent losses.

Review note: upstream recipe finalist/parent selection still ranks raw losses across phases with different time budgets; those rankings are not normalized. The UI deliberately groups plotted comparisons by equal budget. The hyperparameter guard checks parameter names/shapes, not full model semantic equivalence. These are engine limitations, separate from the tabbed display.
