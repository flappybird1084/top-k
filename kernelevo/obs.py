"""One-way observability mirror (spec §10). The loop writes SQLite first, then
calls into here; every method is fire-and-forget — a mirror failure must never
break the loop. Nothing in the loop reads from W&B."""

from __future__ import annotations

import os

try:
    import weave as _weave
except Exception:  # noqa: BLE001 — optional dependency at runtime
    _weave = None


def weave_op(fn):
    """@weave.op if weave is importable, identity otherwise. Weave ops run fine
    (untraced) when weave.init was never called."""
    if _weave is not None:
        try:
            return _weave.op(fn)
        except Exception:  # noqa: BLE001
            return fn
    return fn


def current_trace_url() -> str | None:
    # TODO(user): verify against current Weave docs — accessor names have moved
    # between versions; this tries the ones I know of.
    if _weave is None:
        return None
    try:
        for name in ("require_current_call", "get_current_call"):
            getter = getattr(_weave, name, None)
            if getter:
                call = getter()
                if call is not None:
                    return getattr(call, "ui_url", None)
    except Exception:  # noqa: BLE001
        return None
    return None


def weave_attributes(attrs: dict):
    if _weave is not None:
        try:
            return _weave.attributes(attrs)
        except Exception:  # noqa: BLE001
            pass
    from contextlib import nullcontext
    return nullcontext()


_weave_initialized = False


def init_weave():
    """Idempotent; safe to call from search.py before the adapter-writing stage
    so those traces are captured too (the Mirror also calls it)."""
    global _weave_initialized
    if _weave_initialized or _weave is None or not os.environ.get("WANDB_API_KEY"):
        return
    entity = os.environ.get("WANDB_ENTITY") or None
    project = os.environ.get("WANDB_PROJECT") or "kernel-evolution"
    weave_project = (os.environ.get("WEAVE_PROJECT")
                     or (f"{entity}/{project}" if entity else project))
    try:
        _weave.init(weave_project)
        _weave_initialized = True
        print(f"[obs] weave traces -> https://wandb.ai/{weave_project}/weave")
    except Exception as e:  # noqa: BLE001
        print(f"[obs] weave.init failed ({e}); continuing without Weave tracing")


class Mirror:
    def __init__(self, cfg: dict, model_name: str):
        self.run = None
        self.model_name = model_name
        if not os.environ.get("WANDB_API_KEY"):
            print("[obs] WANDB_API_KEY empty — running without W&B/Weave mirror")
            return
        entity = os.environ.get("WANDB_ENTITY") or None
        project = os.environ.get("WANDB_PROJECT") or "kernel-evolution"
        try:
            import wandb
            self.run = wandb.init(
                project=project, entity=entity,
                name=f"{model_name}-{cfg['profile'].lower()}",
                config={k: v for k, v in cfg.items() if not k.startswith("_")})
        except Exception as e:  # noqa: BLE001
            print(f"[obs] wandb.init failed ({e}); continuing without W&B")
        init_weave()

    def _log(self, data: dict):
        if self.run is None:
            return
        try:
            self.run.log(data)
        except Exception:  # noqa: BLE001
            pass

    def log_startup(self, targets: dict, calib: dict):
        if self.run is None:
            return
        try:
            import wandb
            prof = wandb.Table(
                columns=["op", "pct_step_time", "per_step_us", "seed_kind"],
                data=[[l["op"], l["pct_step_time"], l["per_step_us"], l["seed_kind"]]
                      for l in targets["lineages"]])
            cal = wandb.Table(columns=["key", "value"],
                              data=[[k, str(v)] for k, v in calib.items()])
            self.run.log({"profiler": prof, "calibration": cal})
        except Exception:  # noqa: BLE001
            pass

    def log_candidate(self, op: str, row: dict):
        data = {f"{op}/gate_reached": row.get("gate_reached"),
                f"{op}/accepted": int(bool(row.get("accepted")))}
        for k in ("latency_us", "step_time_ms", "samples_per_s", "mfu"):
            if row.get(k) is not None:
                data[f"{op}/{k}"] = row[k]
        self._log(data)

    def log_generation(self, gen: int, stats: dict, best: dict):
        data = {"generation": gen, **{f"gen/{k}": v for k, v in stats.items()}}
        for op, ms in best.items():
            if ms is not None:
                data[f"best/{op}/step_time_ms"] = ms
        self._log(data)

    def log_lessons(self, lessons: list[str]):
        if self.run is None:
            return
        try:
            import wandb
            self.run.log({"lessons": wandb.Table(columns=["lesson"],
                                                 data=[[t] for t in lessons])})
        except Exception:  # noqa: BLE001
            pass

    def log_kernel_artifact(self, op: str, code_path: str, generation: int):
        if self.run is None:
            return
        try:
            import wandb
            art = wandb.Artifact(f"{self.model_name}-{op}", type="kernel",
                                 metadata={"generation": generation})
            art.add_file(code_path)
            self.run.log_artifact(art)
        except Exception:  # noqa: BLE001
            pass

    def finish(self, stop_reason: str):
        if self.run is None:
            return
        try:
            self.run.summary["stop_reason"] = stop_reason
            self.run.finish()
        except Exception:  # noqa: BLE001
            pass
