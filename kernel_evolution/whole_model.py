"""Platform contracts for agent-authored whole-model replacements.

This module contains no optimization implementations. The harness continues to
own data, loss, optimizer hyperparameters, step order, and measurement.
"""
import copy
import importlib.util
import sys
from pathlib import Path


def load_install(path):
    from kernel_evolution.archive import source_hash
    name = 'whole_candidate_' + source_hash(Path(path).read_text())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'install', None)):
        raise TypeError('Whole-model candidate must expose install(model, optimizer)')
    return module.install


def _classes(model):
    result = {}
    for module in model.modules():
        for cls in type(module).__mro__:
            for name, value in vars(cls).items():
                if callable(value):
                    result[(cls, name)] = value
    return result


def _precision():
    import torch
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision(), torch.get_default_dtype(),
            torch.is_grad_enabled())


def _global_callables():
    import torch
    import torch.nn.functional as functional
    return {(module, name): value for module in (torch, functional)
            for name, value in vars(module).items() if callable(value)}


def _assert_invariants(harness):
    guard = getattr(harness, '_whole_model_guard', None)
    if guard is None:
        return
    classes, functions, precision = guard
    if _precision() != precision:
        raise AssertionError('Candidate changed global precision or gradient settings during execution')
    if any(vars(cls).get(name) is not value for (cls, name), value in classes.items()):
        raise AssertionError('Candidate patched shared classes during execution')
    if any(vars(module).get(name) is not value for (module, name), value in functions.items()):
        raise AssertionError('Candidate patched global torch functions')


def _groups(optimizer):
    return [{k: copy.deepcopy(v) for k, v in g.items() if k != 'params'}
            for g in optimizer.param_groups]


def install_candidate(harness, path):
    """Install only instance-local behavior and reject immediate state changes."""
    from kernel_evolution.runtime import clone
    from kernel_evolution.verifier import compare
    model, optimizer = harness.model, harness.optimizer
    named = {name: id(p) for name, p in model.named_parameters()}
    trainable = {name: p.requires_grad for name, p in model.named_parameters()}
    buffers = {name: (tuple(b.shape), b.dtype) for name, b in model.named_buffers()}
    groups = _groups(optimizer)
    group_params = [[id(p) for p in g['params']] for g in optimizer.param_groups]
    state = clone((model.state_dict(), optimizer.state_dict()))
    classes, precision = _classes(model), _precision()
    functions = _global_callables()
    for cls in type(optimizer).__mro__:
        classes.update({(cls, name): value for name, value in vars(cls).items() if callable(value)})
    # Snapshot before importing as candidate top-level code also executes.
    result = load_install(path)(model, optimizer)
    if result is not None:
        raise TypeError('install(model, optimizer) must return None')
    if named != {name: id(p) for name, p in model.named_parameters()}:
        raise AssertionError('Candidate changed named parameter identities')
    if trainable != {name: p.requires_grad for name, p in model.named_parameters()}:
        raise AssertionError('Candidate changed gradient requirements')
    if buffers != {name: (tuple(b.shape), b.dtype) for name, b in model.named_buffers()}:
        raise AssertionError('Candidate changed buffer schema')
    if groups != _groups(optimizer) or group_params != [[id(p) for p in g['params']] for g in optimizer.param_groups]:
        raise AssertionError('Candidate changed optimizer parameter groups or hyperparameters')
    if precision != _precision():
        raise AssertionError('Candidate changed global precision or gradient settings')
    if any(vars(cls).get(name) is not value for (cls, name), value in classes.items()):
        raise AssertionError('Candidate patched shared model classes')
    compare((model.state_dict(), optimizer.state_dict()), state, 0., 0.)
    harness._whole_model_guard = (classes, functions, precision)
    _assert_invariants(harness)
    harness.compiled_step = None
    return {'parameter_count': len(named), 'buffer_count': len(buffers), 'contract': 'instance_install_v1'}


def make_harness(adapter, config, run_dir, path=None):
    from kernel_evolution.runtime import Harness
    harness = Harness(adapter, config, run_dir)
    if path:
        install_candidate(harness, path)
    return harness


def _capture(harness, fn):
    from kernel_evolution.runtime import clone
    import torch
    _assert_invariants(harness)
    loss = getattr(harness, 'capture_step', lambda call: call())(fn)
    torch.cuda.synchronize()
    _assert_invariants(harness)
    gradients = {name: p.grad for name, p in harness.model.named_parameters() if p.requires_grad}
    return clone((loss, harness.model.state_dict(), harness.optimizer.state_dict(), gradients))


def correctness_cases(harness, draws=3):
    """Real batches plus one valid unseen batch size; never fabricate token IDs."""
    import torch
    from torch.utils._pytree import tree_flatten, tree_map
    from kernel_evolution.runtime import move, clone
    cases = []
    iterator = iter(harness.adapter.get_dataloader('train'))
    for _ in range(draws):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        cases.append(move(batch, 'cuda'))
    if not cases:
        raise ValueError('No training batches available for correctness')
    first = cases[0]
    leaves = [x for x in tree_flatten(first)[0] if isinstance(x, torch.Tensor) and x.ndim]
    if not leaves or leaves[0].shape[0] < 2:
        raise ValueError('Whole-model unseen batch check requires batch size >= 2')
    size = leaves[0].shape[0]
    unseen = tree_map(lambda x: x[:size-1].clone() if isinstance(x, torch.Tensor) and x.ndim and x.shape[0] == size else clone(x), first)
    return cases + [unseen]


def capture_reference(harness, config, *, eager=True, draws=3, steps=2):
    """Capture oracle before candidate code executes, including sequential updates."""
    from kernel_evolution.runtime import clone, seed_all
    original_batch = harness.batch
    cases = correctness_cases(harness, draws)
    records = []
    try:
        for index, batch in enumerate(cases):
            harness.restore()
            harness.batch = clone(batch)
            seed_all(config['seed'] + 1009 * index)
            fn = harness.step_callable({}, eager=eager)
            initial = clone(harness.model.state_dict())
            states = [_capture(harness, fn) for _ in range(steps)]
            records.append({'batch': clone(batch), 'initial': initial, 'states': states, 'seed': config['seed'] + 1009 * index})
    finally:
        harness.batch = original_batch
        harness.restore()
    return records


def check_reference(harness, records, config, *, eager=False):
    from kernel_evolution.runtime import clone, seed_all
    from kernel_evolution.verifier import compare, compare_step_state
    original_batch = harness.batch
    errors = []
    try:
        for record in records:
            harness.restore()
            harness.batch = clone(record['batch'])
            seed_all(record['seed'])
            fn = harness.step_callable({}, eager=eager)
            actual_previous = clone(harness.model.state_dict())
            expected_previous = record['initial']
            for expected in record['states']:
                batch_before = clone(harness.batch)
                actual = _capture(harness, fn)
                compare(harness.batch, batch_before, 0., 0.)
                errors.append(compare_step_state(actual, expected, config['rtol'], config['atol']))
                # Absolute parameter tolerance can exceed an AdamW update. Compare
                # updates separately so doing no optimizer work cannot pass.
                actual_delta = {name: actual[1][name] - actual_previous[name] for name, p in harness.model.named_parameters() if p.requires_grad}
                expected_delta = {name: expected[1][name] - expected_previous[name] for name in actual_delta}
                compare(actual_delta, expected_delta, config.get('update_rtol', config['rtol']), config.get('update_atol', 1e-7))
                actual_previous, expected_previous = actual[1], expected[1]
    finally:
        harness.batch = original_batch
        harness.restore()
    return {'checks': len(errors), 'batches': len(records), 'sequential_steps': len(records[0]['states']),
            'unseen_batches': 1, 'max_abs': max(x['max_abs'] for x in errors), 'max_rel': max(x['max_rel'] for x in errors)}


def compare_training_paths(baseline, candidate, config, *, baseline_eager=True,
                           candidate_eager=False, draws=3, steps=2):
    records = capture_reference(baseline, config, eager=baseline_eager, draws=draws, steps=steps)
    return check_reference(candidate, records, config, eager=candidate_eager)


def selftest(harness, records, config):
    """Reject stateful, shape-hardcoded and forward-only planted cheats.

    These deliberately broken step wrappers exercise the external verifier;
    they are never search candidates or optimization implementations.
    """
    import torch
    from torch.utils._pytree import tree_flatten
    if len(records) < 2 or min(len(r['states']) for r in records) < 2:
        raise ValueError('Verifier self-test requires multiple shapes and sequential updates')
    def shape(batch):
        return tuple(tuple(x.shape) for x in tree_flatten(batch)[0] if isinstance(x, torch.Tensor))
    modal = shape(records[0]['batch'])
    if not any(shape(r['batch']) != modal for r in records):
        raise ValueError('Verifier self-test requires an unseen batch shape')
    original_step = harness.step
    # Establish that the legitimate oracle path passes before testing cheats.
    check_reference(harness, records, config, eager=True)
    cached = []
    def caching():
        if not cached:
            cached.append(original_step())
        return cached[0]
    def hardcoded():
        if shape(harness.batch) == modal:
            return original_step()
        return torch.zeros_like(records[0]['states'][0][0])
    def forward_only():
        harness.optimizer.zero_grad(set_to_none=True)
        return harness.adapter.loss_fn(harness.model, harness.batch).detach()
    rejected = []
    try:
        for name, cheat in [('cached_output', caching), ('hardcoded_shape', hardcoded), ('missing_backward', forward_only)]:
            harness.step = cheat
            harness.compiled_step = None
            try:
                check_reference(harness, records, config, eager=True)
            except (AssertionError, RuntimeError) as exc:
                rejected.append({'name': name, 'gate': 2, 'error': str(exc)[:2000]})
            else:
                raise RuntimeError(f'Verifier self-test FAILED: {name} survived correctness')
    finally:
        harness.step = original_step
        harness.compiled_step = None
        harness.restore()
    return rejected
