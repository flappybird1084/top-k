# Kernel Evolution

Setup decisions and the scope of the first implemented GPU pilot. See README.md
for the remaining v3 work; this pilot is not a claim of full-spec completion.

- Repository: `top-k`, branch `andre-branch`.
- GPU execution: paired molab notebook, RTX PRO 6000 Blackwell Server Edition.
- First live run: at most five generations, four candidates per generation,
  $100 API-equivalent ceiling, ideally much less. User selected Sol for the pilot.
- Codex OAuth provider targets `gpt-5.6-sol` for all three roles, verified on
  molab. Subscription consumption and dollar estimates must be
  recorded separately; estimates are not invoices.
- Reserve estimated call costs across concurrent jobs before dispatch, including
  repairs, search, and curator calls. Stop new calls when the remaining budget
  cannot cover a reservation. Finish already-authorized GPU gates and archive a
  budget stop. Surface exhaustion to the user.
- W&B replacement account verified: user `andre-de-la-cruz`, entity
  `stephenslee0127-acme`, project `kernel-evolution`. RUN enables mirroring; DEV
  stays local. Credentials are stored outside the repository by W&B login.
  Molab authentication is verified. Do not use the earlier
  `ratatouille/kernel-evolution` project for this run.
- SQLite remains authoritative. W&B/Weave are optional outbound mirrors.
- Proposed demos, flagged before implementation: small JEPA-style ViT using
  CIFAR-10 resized to 64px; approximately 30M-parameter decoder using a pinned
  TinyStories subset, sequence length 512. Measure actual parameter counts and
  eligible operator time shares; do not promise a qualifying lineage or speedup.
- The first profile selected EMA (~9.7% of step time). Normalization backward
  (~1.1%) did not qualify; no threshold was lowered to force its inclusion.
- Five-minute calibration raised margins to 10.31% isolation and 6.06% step time.
  Both planted cheats were rejected at gate 2. The complete four-fixture stub
  run had one valid gate-4 acceptance and the expected three rejections.
- The five-generation Sol pilot finished at $2.7142312 token-equivalent. A
  direct-Inductor audit invalidated its one original acceptance. No confirmed
  speedup remains. See PILOT_RESULTS.md; the old calibration cannot be reused
  after the baseline correction.

Verification decisions:

- Gate 2 performs target-level output, gradient, and mutation correctness checks
  on fresh seeded inputs and valid observed/unseen shapes. It does not train a
  model or evaluate a validation dataset.
- Gate 3 measures the target operation only. A backward-kernel target includes
  that backward computation. Never include unrelated whole-model backpropagation,
  optimizer steps, data loading, or validation metrics in isolation timing.
- Gate 4 measures complete forward/loss/backward/optimizer steps and any adapter
  post-step update, with matched restored state. Full-step correctness checks run
  outside timed regions. No validation-epoch metrics in timed regions.
- The spec's 10-second estimates are expected gate costs, not fixed sampling
  windows or guarantees. Use warmed, synchronized, repeated measurements with
  baseline/candidate ordering balanced across measurement blocks. Record timing
  spread; use a predeclared bounded second measurement for borderline cases.
  If results remain uncertain, archive as inconclusive rather than accepting.
- Keep measurements serial on the GPU. Compile/LLM work can be concurrent, but
  do not allow overlapping candidate GPU benchmarks.
- Calibrate after ingest/profile so real shapes and seeds exist. Gate margins
  account for measurement noise; do not lower them to create an acceptance.
- Add an optional post-optimizer adapter hook for EMA. Use a graph integration
  layer with argument, gradient, and mutation contracts for replacement/fusion.
- Disposable CUDA workers permit recovery when the GPU remains healthy; device
  failure may require stopping rather than claiming a guaranteed GPU reset.

References:

- https://marimo.io/features/vs-colab-alternative
- https://learn.chatgpt.com/docs/auth
- https://learn.chatgpt.com/docs/codex-sdk
- https://cave.cs.toronto.edu/kriz/cifar.html
- https://huggingface.co/datasets/roneneldan/TinyStories
- https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
- https://docs.pytorch.org/tutorials/recipes/recipes/benchmark
