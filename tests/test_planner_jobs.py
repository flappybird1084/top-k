import json
from types import SimpleNamespace

from kernelevo.planner import plan


def test_planner_retries_when_jobs_name_no_active_lineage():
    class LLM:
        replies = [
            {"jobs": [{"lineage": "rms_norm", "strategy": "try it"}]},
            {"jobs": [{"lineage": "gelu_mlp", "strategy": "tile forward matmul"}]},
        ]

        def complete(self, messages, **kwargs):
            assert kwargs["json_mode"]
            if len(self.replies) == 1:
                assert "gelu_mlp" in messages[-1]["content"]
            return SimpleNamespace(text=json.dumps(self.replies.pop(0)))

    targets = {"lineages": [{"op": "gelu_mlp", "pct_step_time": 9.0,
                             "shapes": [], "signature": "gelu(x)", "retired": False}]}
    jobs = plan(LLM(), targets, {}, [], 2, 1)

    assert len(jobs) == 1
    assert jobs[0]["lineage"] == "gelu_mlp"
    assert jobs[0]["parent"] is None
