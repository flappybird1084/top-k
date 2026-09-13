import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class WholeModelTests(unittest.TestCase):
    def harness(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return SimpleNamespace(model=model, optimizer=optimizer, compiled_step='old')

    def install(self, harness, body):
        from kernel_evolution.whole_model import install_candidate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidate.py'
            path.write_text('def install(model, optimizer):\n'+body)
            return install_candidate(harness, path)

    def test_install_keeps_parameters_and_clears_compiled_wrapper(self):
        harness = self.harness()
        self.install(harness, '    model.forward = model.forward\n')
        self.assertIsNone(harness.compiled_step)

    def test_hyperparameter_change_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'hyperparameters'):
            self.install(self.harness(), '    optimizer.param_groups[0]["lr"] = 0.0\n')

    def test_gradient_disabling_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'gradient requirements'):
            self.install(self.harness(), '    model.weight.requires_grad_(False)\n')

    def test_parameter_replacement_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'identities'):
            self.install(self.harness(), '    model.weight = type(model.weight)(model.weight.detach().clone())\n')

    def test_missing_optimizer_update_is_rejected_even_with_loose_absolute_tolerance(self):
        from kernel_evolution.whole_model import capture_reference, check_reference
        from kernel_evolution.runtime import clone
        baseline = self.harness()
        candidate = self.harness()
        for h in (baseline, candidate):
            h.batch = torch.ones(4, 3)
            initial = clone(h.model.state_dict())
            initial_optimizer = copy.deepcopy(h.optimizer.state_dict())
            def restore(h=h, initial=initial, initial_optimizer=initial_optimizer):
                h.model.load_state_dict(initial)
                h.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
                h.optimizer.zero_grad(set_to_none=True)
            h.restore = restore
            def step(h=h):
                h.optimizer.zero_grad(set_to_none=True)
                loss = h.model(h.batch).square().mean()
                loss.backward()
                h.optimizer.step()
                return loss.detach()
            h.step_callable = lambda replacements, eager=False, step=step: step
        config = {'seed': 1729, 'rtol': 1e-4, 'atol': 1e-2}
        with patch('kernel_evolution.whole_model.correctness_cases', return_value=[torch.ones(4, 3)]), patch('torch.cuda.synchronize'):
            records = capture_reference(baseline, config)
            baseline_floor = check_reference(candidate, records, config, update_floor=True)
            self.assertEqual(baseline_floor['update_max_abs'], 0.)
            self.assertTrue(baseline_floor['update_floor_measurement'])
            check_reference(candidate, records, config)
            original_step = candidate.optimizer.step
            # Preserve optimizer moment evolution while suppressing parameter updates.
            def skip_update(*args, **kwargs):
                saved = clone(candidate.model.state_dict())
                original_step(*args, **kwargs)
                candidate.model.load_state_dict(saved)
            candidate.optimizer.step = skip_update
            floor = check_reference(candidate, records, config, update_floor=True)
            self.assertGreater(floor['update_max_abs'], 1e-5)
            # Measuring the numerical floor is explicit; normal verification
            # continues to reject skipped updates.
            with self.assertRaises(AssertionError):
                check_reference(candidate, records, config)

    def test_planted_step_cheats_rejected_and_harness_restored(self):
        from kernel_evolution.whole_model import capture_reference, check_reference, selftest
        from kernel_evolution.runtime import clone
        h = self.harness()
        h.batch = torch.ones(4, 3)
        h.adapter = SimpleNamespace(loss_fn=lambda model, batch: model(batch).square().mean())
        initial = clone(h.model.state_dict())
        initial_optimizer = copy.deepcopy(h.optimizer.state_dict())
        def restore():
            h.model.load_state_dict(initial)
            h.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            h.optimizer.zero_grad(set_to_none=True)
        h.restore = restore
        def step():
            h.optimizer.zero_grad(set_to_none=True)
            loss = h.adapter.loss_fn(h.model, h.batch)
            loss.backward()
            h.optimizer.step()
            return loss.detach()
        h.step = step
        h.step_callable = lambda replacements, eager=False: h.step
        cfg = {'seed': 1729, 'rtol': 1e-4, 'atol': 1e-5}
        with patch('kernel_evolution.whole_model.correctness_cases', return_value=[torch.ones(4, 3), torch.ones(3, 3)]), patch('torch.cuda.synchronize'):
            records = capture_reference(h, cfg)
            result = selftest(h, records, cfg)
            self.assertEqual({r['name'] for r in result}, {'cached_output', 'hardcoded_shape', 'missing_backward'})
            self.assertTrue(all(r['gate'] == 2 for r in result))
            self.assertIs(h.step, step)
            self.assertEqual(check_reference(h, records, cfg)['checks'], 4)
            with self.assertRaisesRegex(ValueError, 'multiple shapes'):
                selftest(h, records[:1], cfg)

    def test_compiler_constructor_initializes_before_install_guard_without_running_step(self):
        from kernel_evolution.whole_model import _assert_invariants
        h = self.harness()
        h.config = {'step_backend': 'inductor'}
        calls = []
        def step():
            calls.append('executed')
            return h.model(torch.ones(2, 3)).sum()
        def step_callable(replacements):
            h.compiled_step = torch.compile(step)
            return h.compiled_step
        h.step_callable = step_callable
        self.install(h, '    return None\n')
        self.assertIsNone(h.compiled_step)
        h.step_callable({})
        _assert_invariants(h)
        self.assertEqual(calls, [])
        self.assertEqual(h.optimizer.state, {})

    def test_shared_class_patch_still_rejected_with_member_name(self):
        h = self.harness()
        cls = type(h.model)
        original = cls.forward
        try:
            with self.assertRaisesRegex(AssertionError, 'Linear.forward'):
                self.install(h, '    type(model).forward = lambda self, x: x\n')
        finally:
            cls.forward = original
