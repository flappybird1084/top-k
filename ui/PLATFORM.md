# Top-Kernel platform

The frontend now uses the real API. It does not synthesize repository activity, validate data with timers, or navigate into a fake GPU run. The earlier standalone `serve_run.py` remains a read-only diagnostic tool; use the ASGI server below for the complete application.

## Start

Install `platform_backend/requirements.txt` in a virtual environment. From this workspace:

```
python ui/platform_backend/deploy_worker.py
uvicorn asgi:app --app-dir ui/platform_backend --host 127.0.0.1 --port 8766 --workers 1
```

The existing marimo connection is read from `~/.local/state/kernel-evolution/molab.json` by `top-k/scripts/molab.py`. Tokens stay in that private file and are not copied into frontend assets, query strings, or job records. Deployment copies only the new worker helpers into `/marimo/top-k-platform`; it does not overwrite the GPU engine or notebook. Engine dependencies and W&B/Codex credentials must already be configured in that runtime.

## Actual flow

1. Submit a public GitHub repo. An idempotent request creates a durable job ID. The GPU-side worker clones a bounded shallow checkout and records its commit, then statically inspects source and loader references without executing repository code.
2. A single discoverable Hugging Face dataset is checked automatically. Otherwise the data popup asks for a Hugging Face dataset or GitHub data repo. It verifies accessible metadata, a training split/sample (HF), or an actual supported file (GitHub). Gated/private data needs a separately designed per-user credential flow and is rejected explicitly. Generic arbitrary data URLs are intentionally unsupported.
3. The job acquires the single-GPU queue lock. The existing adapter agent gets the exact dataset URL, revision, split, and schema. Real ingestion checks the model/data contract. When the engine emits `input_ready`, the browser opens the run page; failed preparation shows a real failure. No fallback to fabricated data or metrics is implemented.
4. A background synchronizer reads the authoritative remote archive every five seconds; the browser polls the local authenticated API. Runtime reconnects, process restarts, terminal states, and cancellation are handled. An interrupted network retains the last snapshot with a stale-connection notice.
5. The run page embeds a separate, read-only marimo notebook on the same origin. It reads the synchronized run data and refreshes every five seconds. This is a real marimo app, not a proxy around molab's CSP, and does not expose the original editor or execute training in the web process.

## Deployment scope and security

This is an authenticated **single-owner workspace**, not a public multi-tenant service for arbitrary users. Submitted model code executes with that workspace's GPU permissions. Before allowing unrelated tenants, provision isolated per-tenant workers, scoped secrets, and container/resource/network policies; this implementation must not be advertised as a multi-tenant sandbox.

Set a strong `TOPK_ACCESS_TOKEN` and `TOPK_PUBLIC_ORIGIN=https://your-host` for a hosted deployment behind HTTPS. Login creates a signed, expiring HttpOnly SameSite cookie. HTTP APIs and notebook HTTP/WebSocket access check it; state-changing requests validate Origin. Without a token only loopback clients are accepted. Backend/state files are excluded from the static allowlist. Use one Uvicorn process because marimo sessions and the remote control queue are process-local. A reverse proxy must preserve WebSocket upgrades. Do not deploy publicly with missing authentication.

Defaults: `TOPK_RUN_SPEND_CAP=3` dollars API-equivalent estimate per run; `TOPK_DAILY_SPEND_CAP=30` in reserved daily run budgets; three unfinished jobs maximum; one GPU job; three-hour run deadline. Budget reservations are conservative and not refunds of measured spend. User submissions launch actual potentially billable work within these configured limits; starting the web server itself does not launch a run.

## Integrations

W&B and Weave URLs are read from each run's archive. The Weave pane displays recorded calls with links to the official trace detail; no iframe support is falsely assumed. W&B's run link works with the user's W&B sign-in. An actual W&B report embed still requires an explicit public report URL: the platform never publishes private experiment data automatically. To connect an approved public report, POST `{"wandb_report_url":"https://wandb.ai/…/reports/…"}` to `/api/runs/ID/integrations` from the authenticated origin. The override persists across remote updates. The URL is validated and no report permissions are changed.

The new `/notebook/` embed is served by marimo under this application's authentication. It is a viewer of the actual GPU archive, not the original molab editor. The editor remains restricted by molab's own frame policy.

## Validation

```
python -m unittest discover -s ui/platform_tests -v
```

Tests cover source URL validation, data sample validation, access control, cross-origin rejection, idempotency, and data-state transitions. Local notebook execution and HTTP serving are also checked during implementation. A public Hello-World repository is used for read-only remote intake smoke testing; a real TinyStories sample is checked without launching GPU training. Full billed optimization is not silently launched as a UI test.

Repository input accepts GitHub repository URLs and `/tree/BRANCH` links, including slash-separated branch names. The worker resolves the selected ref through GitHub, shallow-fetches its immutable commit, and records both branch and commit in run state. Folder links are not repository branch selections.

Runtime panel: authenticated GET `/api/runs/:id/runtime` reads `nvidia-smi` utilization and memory plus the last 100 console lines (24 KB maximum) from that run. GPU usage is node-wide and sampled every five seconds while the page is visible. Console output is read-only and credential patterns are redacted. This follows the subprocess-output approach in `rian:notebooks/molab_run.py`; it does not expose an interactive shell.
