# Twenty-five-repository benchmark

`repos.json` is the user's ordered set of 25 public training repositories,
with a pinned source commit for each one.
Inclusion is not a claim that this harness can ingest or improve every model.

Run on a Linux CUDA host with the project's dependencies installed via `uv`,
the repository's data access, and W&B Inference credits. Set `WANDB_API_KEY`,
`WANDB_ENTITY`, and `WANDB_PROJECT` in the environment. W&B documents the
OpenAI-compatible endpoint, project attribution, available models, and
per-project/user concurrency limits at:

- https://docs.wandb.ai/inference/api-reference
- https://docs.wandb.ai/inference/models
- https://docs.wandb.ai/inference/usage-limits
- https://site.wandb.ai/pricing/tokens/

First inspect the plan, then pilot one repository:

```bash
uv run python scripts/run_repo_benchmark.py
uv run python scripts/run_repo_benchmark.py --run --limit 1
```

Once the pilot has a verified adapter, a measured baseline, and a W&B URL,
resume the full set:

```bash
uv run python scripts/run_repo_benchmark.py --run --resume
```

`--profile RUN --generations N --spend-cap USD` adjusts each repository's
search budget. The default is a two-generation DEV pilot with a $3 estimated
LLM spend cap *per repository*, so 25 runs can consume substantially more
time and credits than one run. W&B billing and the harness's estimated cap
are distinct. The runner checks the live model catalog and makes one small
completion before cloning anything. It then executes repositories serially
on the GPU to avoid contention and W&B concurrency spikes.

Each repository gets a shallow checkout, commit SHA, attempt log, archive,
`benchmark.json`, and any W&B URL under `benchmark-runs/`. Failed attempts
remain in separate directories when `--resume` retries them. `summary.json`
records every attempted run, including failures. A reported kernel improvement
uses the accepted candidate's *paired in-session* baseline; a failed ingest
or absent measurement never becomes a performance claim. The reported kernel
comparison uses the initial incumbent and fastest accepted candidate, matching
the search loop's final summary. Use the same GPU,
dataset, objective, profile, and budget when comparing repositories or reruns.
At the end, one W&B summary run publishes the complete table and denominator;
its URL is saved in `summary.json` when upload succeeds.

Some repositories are framework collections or require custom CUDA extensions,
large datasets, or multiple GPUs. Their adapter failures should be recorded
and triaged rather than dropped from the denominator. The list is an intake
set, not a published 30-repository result.
