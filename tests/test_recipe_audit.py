"""Audit fixes from run ef48abdf: ground-truth model report in prompts,
harness-owned precision, mechanical label-vs-diff records."""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[1]))

from kernelevo import recipes


class TinyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(nn.init.normal_(nn.Parameter(nn.Linear(4, 4).weight.detach())))

    def forward(self, x):
        return x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.norm = TinyNorm()


def test_module_inventory_and_diff():
    m = TinyModel()
    inv = recipes.module_inventory(m)
    assert inv["Linear"] == 3 and inv["TinyNorm"] == 1
    cand = dict(inv, Linear=5)          # candidate added 2 Linears
    cand.pop("TinyNorm")                # and removed the norm
    d = recipes.inventory_diff(inv, cand)
    assert "+2 Linear" in d and "-1 TinyNorm" in d
    assert recipes.inventory_diff(inv, dict(inv)) == ""


def test_architecture_fingerprint_detects_module_structure_change():
    base = nn.Sequential(nn.Linear(4, 4), nn.Identity())
    changed = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    assert recipes.arch_fingerprint(base) != recipes.arch_fingerprint(changed)


def test_model_report_contains_structure_and_source():
    rep = recipes.model_report(TinyModel())
    assert "total parameters" in rep
    assert "Linear: 3" in rep
    assert "class TinyNorm" in rep          # non-torch class source included
    assert "class Linear" not in rep        # torch classes skipped


def test_model_report_respects_cap():
    rep = recipes.model_report(TinyModel(), max_chars=200)
    assert len(rep) < 1200  # inventory lines + omission note only


def test_planner_prompt_carries_model_report():
    phase = dict(kind="architecture", train_seconds=60)
    msgs = recipes.recipe_planner_prompt(phase, {}, [], [], 4, [],
                                         model_report="MODEL-GROUND-TRUTH")
    assert "MODEL-GROUND-TRUTH" in msgs[1]["content"]
    assert "ground truth" in msgs[1]["content"]
    msgs2 = recipes.recipe_planner_prompt(phase, {}, [], [], 4, [])
    assert "MODEL-GROUND-TRUTH" not in msgs2[1]["content"]


def test_subagent_prompt_carries_model_report_and_no_smuggling_rule():
    phase = dict(kind="architecture", train_seconds=60)
    msgs = recipes.recipe_subagent_prompt(phase, "strat", "base", None, "loss",
                                          10_000, [], model_report="MODEL-GT")
    body = msgs[1]["content"]
    assert "MODEL-GT" in body
    assert "do not invent a substitute" in body  # honest-no-op rule (ARCH_RULES)
    assert "Do not bundle extras" in body


def test_precision_autocast_policy():
    import torch
    from kernelevo.recipe_worker import _apply_precision, _precision_context
    m = nn.Linear(4, 4)
    # cpu: never cast (keeps stub/CPU harness tests exact)
    out = _apply_precision(m, {"precision": "bf16"}, "cpu")
    assert out.weight.dtype == torch.float32
    # explicit off: no cast even on cuda-labelled device strings
    out = _apply_precision(nn.Linear(4, 4), {"precision": "off"}, "cuda")
    assert out.weight.dtype == torch.float32
    # bf16 + cuda device string: fp32 parameters and bf16 operations
    out = _apply_precision(nn.Linear(4, 4), {"precision": "bf16"}, "cuda")
    assert out.weight.dtype == torch.float32
    assert _precision_context({"precision": "bf16"}, "cuda").fast_dtype == torch.bfloat16
    # default when key absent is bf16
    out = _apply_precision(nn.Linear(4, 4), {}, "cuda")
    assert out.weight.dtype == torch.float32


def test_author_recipe_accumulates_tokens(tmp_path):
    from kernelevo.llm import Response
    from kernelevo.recipes import author_recipe

    class FakeLLM:
        model = "fake"
        calls = 0
        def complete(self, msgs, meta=None):
            FakeLLM.calls += 1
            return Response("```python\nx=1\n```", input_tokens=100, output_tokens=40)

    checks = iter([(False, "nope"), (True, "")])
    out = author_recipe(FakeLLM(), dict(kind="architecture", train_seconds=1),
                        dict(strategy="s"), "base", None, "loss", 10, [],
                        max_repairs=1,
                        save_fn=lambda src, a: str(tmp_path / f"c{a}.py"),
                        check_fn=lambda p: next(checks))
    assert out["tokens_in"] == 200 and out["tokens_out"] == 80  # 2 attempts summed
    assert out["load_ok"] and out["repairs_used"] == 1


def test_candidate_token_columns_migrate(tmp_path):
    import sqlite3
    from kernelevo.archive import Archive
    old = tmp_path / "old.sqlite"
    db = sqlite3.connect(old)
    db.executescript("CREATE TABLE candidates (id INTEGER PRIMARY KEY, lineage_id INTEGER, created_at REAL);"
                     "CREATE TABLE generations (id INTEGER PRIMARY KEY, model_id INTEGER, started_at REAL);")
    db.close()
    a = Archive(str(old))
    cid = a.add_candidate(lineage_id=1, tokens_in=123, tokens_out=45)
    row = a.candidate(cid)
    assert row["tokens_in"] == 123 and row["tokens_out"] == 45
