# kernel evolution

Point it at a PyTorch repo. It figures out how to run one training step of the
repo's model, finds the ops worth attacking, and runs an evolutionary search
where LLM agents write Triton kernels for them. Every candidate must survive a
deterministic four-gate verifier — compiles, matches eager outputs **and
gradients**, beats the `torch.compile` incumbent in isolation, and speeds up
the real training step — before it's accepted. No model ever grades its own
output; correctness is a constraint, speed is the objective.

```
repo ──▶ adapter agent ──▶ profile ──▶ calibrate ──▶ ┌ planner ─▶ subagents ─▶ 4-gate verifier ┐
        (writes+debugs      (hot ops    (noise floor, │            ▲                    │        │
         the harness         become      planted-cheat│            └──── lessons ◀─ curator ◀────┘
         contract)           lineages)   self-test)   └───────── generations ─────────────────────┘
```

Models are routed onto a **fixed op vocabulary** (norms, gated/plain MLP
epilogues, cross-entropy, …) by behavioral pattern matching: modules are
probed and classified by what they compute, never by class or attribute names.
No per-repo code, no runtime vocabulary growth.

## Quickstart

```bash
uv pip install -r requirements.txt
cp .env.example .env        # fill in keys (W&B key covers logging + inference)

python web.py               # http://127.0.0.1:8420 — submit jobs, watch logs,
                            # view generated kernels
# or headless:
python search.py --repo https://github.com/karpathy/nanochat --profile DEV
python search.py --adapter adapters/jepa.py --llm stub    # $0 harness smoke test
```

Needs a CUDA GPU. Runs locally or on a [molab](https://molab.marimo.run)
notebook's GPU — paste the notebook's "Pair with agent" prompt into the web
form and jobs are dispatched, streamed, and synced back automatically.

Observability: runs/metrics/artifacts in W&B, every LLM call and candidate
lifecycle traced in Weave, plus a local sqlite archive
(`marimo run notebooks/viewer.py`).

## Layout

- `search.py` / `web.py` — CLI entry / job frontend
- `kernelevo/` — the harness: op registry + behavioral routing, gates and
  subprocess workers, planner/subagent/curator, adapter-writing agent, molab
  dispatch, sqlite archive
- `adapters/` — bundled demo models (JEPA, small LM)
- `config.py` — `DEV` (smoke) and `RUN` (real search) profiles

First verified result: an agent-written Triton RMSNorm accepted through all
four gates on karpathy/nanochat — training step 8.95ms → 8.58ms on an RTX
PRO 6000, with the gen-2 follow-up correctly *rejected* when its microbenchmark
win fell inside the calibrated noise margin in-model.
