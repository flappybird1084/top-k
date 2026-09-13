# Run workspace

`workspace.html` is a read-only, empty-first UI. It never generates candidates, numbers, model choices, or artificial progress. `front.js` remains an onboarding prototype: it does not verify repositories or data or start GPU jobs. Its transition intentionally opens an unconnected workspace until a real backend returns a run ID.

The production backend should return a run ID after onboarding and navigate to `workspace.html?run=ID`. The UI polls same-origin `GET /api/runs/ID` every three seconds; authorize this endpoint per run on the server. No API keys belong in browser config or URLs. `serve_run.py` is a localhost development bridge to an explicitly selected SQLite archive, not a public production server.

Example: `python3 ui/serve_run.py --archive /absolute/path/archive.sqlite --repo https://github.com/team/model`. Open `http://127.0.0.1:8766/workspace.html?run=current`. Omitting the archive provides an empty run; no other repository's archive is selected automatically.

Snapshot contract:
```json
{
  "repo": "https://github.com/team/model",
  "data": "https://huggingface.co/datasets/team/data",
  "status": "running",
  "generation": 1,
  "baseline_ms": null,
  "candidates": [],
  "traces": [],
  "integrations": {
    "wandb_url": null,
    "wandb_embed_url": null,
    "weave_url": null,
    "weave_embed_url": null
  }
}
```
Candidate and trace fields match the selected fields in `serve_run.py`. Supply `baseline_ms` only from a calibrated measurement. Accepted candidate measurements populate the chart; rejected candidates never become chart wins. Trace inputs/outputs are deliberately omitted from the local bridge. A public backend must additionally sanitize exception text and enforce authorization.

W&B reports: supply the official report Share → Copy embed code iframe URL with `--wandb-report-url`. Only public reports support documented embedding; do not publish private experiment data automatically. https://docs.wandb.ai/models/reports/embed-reports

Weave: the default integrated trace list reads actual archive trace records and opens their official Weave URLs. A generally supported authenticated Weave iframe API has not been established. Only supply `--weave-embed-url` if you have verified that deployment permits embedding; the external link remains available. Never proxy around frame restrictions. Both embedding fields accept HTTPS wandb.ai URLs only.

Weave logo: https://site.wandb.ai/wp-content/uploads/2025/10/Weave_logo_25.svg (official W&B site). W&B logo is user supplied.

marimo: supply a trusted, backend-configured `integrations.marimo_url` (local bridge: `--marimo-url`). A full-width notebook frame appears below W&B and traces only when configured. Use a hosted read-only marimo app backed by the GPU runtime for live run data; a WASM playground is a separate client-side runtime. Never accept arbitrary notebook URLs from untrusted run payloads in production; authorize the notebook endpoint per user/run.

Current connection (wired September 12, 2026): `astra-platform-lm-001`, repo `https://github.com/flappybird1084/top-k`, dataset `https://huggingface.co/datasets/roneneldan/TinyStories`, W&B run `5h5buppj` in `stephenslee0127-acme/kernel-evolution`. The authoritative archive was pulled from marimo; the run and W&B state are finished with `no_targets` and no candidate measurements. The platform renders this terminal outcome rather than waiting indefinitely.

The existing molab runtime's frame-ancestors policy excludes this platform. `marimo_url` therefore supplies only an external link; `marimo_embed_url` is separate and must only be set for a deployment that explicitly permits this platform's origin. Do not remove or proxy around the host policy. No public W&B report has been created, so W&B is connected as a run link and the archive-backed trace panel is populated. Actual W&B iframe publication remains pending an approved public report.

To refresh a copied archive: `python3 top-k/scripts/pull_molab.py astra-platform-lm-001`. The UI polls the local archive every three seconds. This is not continuous synchronization with molab; an active remote run needs a scheduled sync or a deployed API beside its archive. No new training jobs are launched by this integration.
