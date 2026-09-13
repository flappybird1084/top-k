# UI on Rian's branch

Worktree: `codex/rian-ui`, based on `origin/rian` at `a8635d5`.

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

Run cancellation is not exposed because this branch's dispatcher does not yet offer a safe remote cancellation contract. BPD is not computed by this adaptation; the results page only displays `result.json`'s `bpd_comparison.baseline` and `.kernel` when an evaluator records them. No synthetic replay is enabled.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -q
```

Tests cover Rian's actual SQLite schema, branch URLs, idempotent submission, data handoff, secret omission, and cross-origin rejection. No GPU or paid model run is launched by these tests.

Codex OAuth: install Codex CLI on the UI server and run `codex login` with ChatGPT. Set `KEVO_UI_LLM=codex_oauth` in the server environment or ignored `.env`. The GPU sends completion requests through the existing authenticated Molab relay; the dispatcher invokes Codex locally. OAuth credentials are never uploaded to Molab. Keep this server running throughout the job. Requests time out after 5 minutes locally and 15 minutes in the relay; failures propagate to the engine. This is a local, single-user connection, not multi-user OAuth login.

Use `KEVO_UI_PROFILE=DEV` and `KEVO_UI_MAX_GENERATIONS=2` for the walkthrough. OAuth consumes the signed-in account’s subscription usage; the engine’s API-dollar estimate is zero for this provider and is not a subscription usage limit. Generation and wall-clock limits still apply. W&B authentication remains separate.
