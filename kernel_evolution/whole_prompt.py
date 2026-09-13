"""Prompt assembly for the entire model; no optimization implementations."""
import json
from pathlib import Path


CONTRACT = '''Return one self-contained Python file with install(model, optimizer) -> None.
The harness owns the existing adapter, batches, scalar loss, zero_grad, backward,
optimizer.step, optional post_optimizer_step, and torch.compile of the complete step.
Install instance-local replacements for any model forward/backward or optimizer
computation you choose. Existing PyTorch behavior may remain for unchanged work;
new computational implementations must use Triton. Include your own autograd
bridges when replacing differentiable computation. Every candidate must include
backward-pass optimization: at least one new Triton kernel from your source must
execute during autograd backward. Forward-only or optimizer-only changes do not
satisfy this task. Choose the backward computation yourself from the full profile.
Preserve named parameter
objects, names, shapes, dtypes, requires_grad, buffer schema, optimizer groups,
hyperparameters, parameter tying, and exact mathematical training semantics.
Do not modify shared classes, global torch functions, precision flags, loss/data,
or the harness. Do not call torch.compile yourself. No file/network/process I/O,
benchmark detection, output caches, training shortcuts, or self-grading. You may
use Python helpers/types.MethodType and torch allocation/autograd scaffolding.
Support fresh batches, unseen batch size, and consecutive optimizer updates.
Every candidate is a complete installation, not a diff; preserve useful parent
changes when extending a parent. Actual new Triton launches must occur in eager
execution of a training step before the harness compiles it. The verifier checks
loss, all gradients, parameters, optimizer state, and parameter update deltas
against original eager across multiple real batches and an unseen batch size.
Timing is against the complete compiled incumbent using balanced paired blocks.
Return JSON with source only. Do not execute tools.'''


def source_prompt(job, archive, target):
    parents = []
    for cid in job['parents']:
        row = archive.rows('SELECT * FROM candidates WHERE id=?', (cid,))[0]
        parents.append({'id': cid, 'source': Path(row['code_path']).read_text(),
                        'step_time_ms': row['step_time_ms']})
    return json.dumps(dict(task=CONTRACT, strategy=job['strategy'], parents=parents,
                          target=target, lessons=archive.lessons(),
                          references=json.loads((archive.path.parent/'references/manifest.json').read_text())
                          if (archive.path.parent/'references/manifest.json').exists() else []))
