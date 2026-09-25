"""A FLOP-counter limitation must not abort a measured training step."""

from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from kernelevo.profiling import _count_flops_per_sample


class BrokenSDPACounter:
    def __init__(self, display=False):
        self.display = display

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_total_flops(self):
        raise AssertionError(
            "sdpa_flop_count: query/key/value shapes are incompatible")


def test_sdpa_counter_failure_leaves_mfu_unknown(monkeypatch, capsys):
    monkeypatch.setattr(torch.utils.flop_counter, "FlopCounterMode", BrokenSDPACounter)
    model = torch.tensor(2.0, requires_grad=True)
    adapter = SimpleNamespace(loss_fn=lambda model, batch: model.square())

    assert _count_flops_per_sample(adapter, model, None, 1) is None
    assert "MFU will be omitted" in capsys.readouterr().out


def test_unrelated_model_error_still_fails(monkeypatch):
    monkeypatch.setattr(torch.utils.flop_counter, "FlopCounterMode", BrokenSDPACounter)
    adapter = SimpleNamespace(loss_fn=lambda model, batch: (_ for _ in ()).throw(
        AssertionError("model shape mismatch")))

    with pytest.raises(AssertionError, match="model shape mismatch"):
        _count_flops_per_sample(adapter, object(), None, 1)
