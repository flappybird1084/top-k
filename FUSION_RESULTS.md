# EMA fusion contract check

The user clarified that fusion may combine multiple call sites of the same
accepted kernel. The implementation now discovers 77 independent EMA updates in
the JEPA adapter and exposes them as a fusion target from generation 2. There is
no configured cap on the number of sites in the group.

A **handwritten Triton fixture** passed all four gates against a fresh
per-region Inductor baseline. A subsequent whole-step compiler audit found a
stronger baseline, so this result does not establish a win over whole-step
`torch.compile`:

[W&B fixture check](https://wandb.ai/stephenslee0127-acme/kernel-evolution/runs/f0otw8pb)
contains the raw paired measurements and source artifact.

| Measurement | Unfused direct Inductor | Fused fixture |
| --- | ---: | ---: |
| Complete EMA group, median | 1.6765 ms | 0.3064 ms |
| Complete training step, median | 14.3732 ms | 12.3341 ms |
| Step reduction | | **14.19%** |
| Samples per second | | 1,297.2 |

The four paired step gains were 15.12%, 12.71%, 12.42%, and 16.41%; every pair
exceeded the calibrated 6.91% step margin. The isolated margin was 29.08%, raised
by a noisy calibration sample; every isolation pair still exceeded it by a wide
margin. No margin was lowered to accept the fixture.

Gate 2 checked the one observed group shape and an unseen group with changed
tensor dimensions, using three fresh input draws each. Maximum absolute error
was 4.77e-7. Gate 4 also compared loss, model state, and AdamW state outside the
timed region. The candidate issued one Triton compute launch for all 77 pairs;
copying its outputs into the target parameters remains part of the full step.

This is evidence that the fusion implementation works. It is **not an evolved Sol acceptance** and is
recorded as a handwritten-fixture event, not added to the search population.
No additional search-model calls were made. The original Sol pilot remains at
five evaluated generations, 20 candidates, and zero confirmed acceptances.

Raw results, fresh calibration, the fixture source, and the test log live under
`runs/fusion-dev/`. The expanded 24-test suite passed on molab. The CPU compilation
test ran with `CUDA_VISIBLE_DEVICES` empty and checked cache reuse, a recoverable
autotune configuration failure, and rejection of an invalid Triton API.

The new benchmark protocol excludes profiler labels from normal step timing.
Prepared archives using earlier protocols cannot be reused. Compare the paired
measurements within this experiment; do not infer improvement by subtracting
times from the earlier Sol pilot.

Sol remains the default planner, curator, and kernel agent. The pilot supports
using it for straightforward EMA work: all 20 candidates compiled and passed
correctness without repairs. It does not establish Sol's quality on harder
normalization or GEMM fusion, nor prove that another model would find a speedup.
An Astra comparison should use matched jobs and the corrected verifier.

The whole-step compiler diagnostic measured 13.4242 ms for the per-region
baseline versus 6.9520 ms for whole-step compilation in the same session.
All four paired improvements exceeded 48%. Loss and numerical training state
matched eager; only equivalent AdamW execution metadata was normalized
(the location of exactly equal step counters and the `capturable` flag).
Raw diagnostic results are in `runs/compiled-baseline/`. The default gate-4
backend now compiles the full step, including candidate integration, and checks
parameter gradients as well as loss, parameters, and optimizer state. It
requires a new calibration; earlier fusion timing is retained as historical
integration evidence.

The fresh whole-step recheck completed in `runs/strong-baseline-check/`.
Calibration set a 5.05% isolation margin and an 8.54% step margin. Both planted
cheats were rejected for each target, and all 33 tests passed on Molab.
The same handwritten fixture passed correctness and isolation (1.6075 ms to
0.3080 ms), but was **rejected at gate 4**: the compiled baseline took 7.0108 ms
versus 7.3480 ms for the compiled fixture integration. Every paired block was
slower. There is no confirmed fusion speedup against this stronger baseline.
The result was saved to SQLite before being mirrored to the original fixture
W&B run. No additional search-model calls were made.
