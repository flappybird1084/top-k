"""Recipe evaluation worker: load a candidate recipe (or the BASELINE
sentinel), train it for a wall-clock budget with the HARNESS-owned data/loss/
seeds, then report mean held-out validation loss. Runs in its own process so a
hung candidate is killed by the orchestrator's timeout.

Usage: python -m kernelevo.recipe_worker <job.json>
job = {base_adapter, candidate_path|"BASELINE", train_seconds, eval_batches,
       seed, device, param_cap, expected_arch_fp?, check_only?}
Prints 'KEVO_RESULT {json}'. gate on failure: load | param_cap | arch_lock |
sanity; on success gate="trained" with val_loss.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time

import torch

from kernelevo.ingest import load_adapter
from kernelevo.recipes import arch_fingerprint


def _load_recipe(path):
    spec = importlib.util.spec_from_file_location("recipe_candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("build_model", "make_optimizer"):
        if not hasattr(mod, fn):
            raise AttributeError(f"recipe file must define {fn}()")
    return mod


def _cycled(adapter):
    it = iter(adapter.get_dataloader("train"))
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(adapter.get_dataloader("train"))


def _holdout_loss(adapter, model, n_batches):
    # grad stays ENABLED (never backward, detach immediately): the adapter
    # contract requires loss to require grad, and generated adapters
    # legitimately assert that inside loss_fn — no_grad here tripped them.
    was_training = model.training
    model.eval()
    losses = []
    for i, batch in enumerate(adapter.get_dataloader("val")):
        if i >= n_batches:
            break
        losses.append(float(adapter.loss_fn(model, batch).detach()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _wandb_run(job):
    """Per-candidate W&B run (own charts), named <run-id>-g<gen>-s<idx>.
    Entirely optional/fire-and-forget."""
    import os
    wcfg = job.get("wandb")
    if not wcfg or not os.environ.get("WANDB_API_KEY") or job.get("check_only"):
        return None
    try:
        import wandb
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT") or "kernel-evolution",
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=wcfg.get("name"), group=wcfg.get("group"),
            tags=wcfg.get("tags"), config=wcfg.get("config"),
            settings=wandb.Settings(silent=True))
    except Exception:  # noqa: BLE001
        return None


def main():
    job = json.load(open(sys.argv[1]))
    device, seed = job["device"], job["seed"]
    res = dict(ok=False, gate="load", val_loss=None, n_params=None,
               steps=None, arch_fp=None, note=None)
    wb = _wandb_run(job)

    def out():
        if wb is not None:
            try:
                if res.get("val_loss") is not None:
                    wb.log({"val/loss": res["val_loss"]})
                    wb.summary["val_loss"] = res["val_loss"]
                wb.summary["gate"] = res["gate"]
                wb.finish()
            except Exception:  # noqa: BLE001
                pass
        print("KEVO_RESULT " + json.dumps(res))

    adapter, _ = load_adapter(job["base_adapter"])
    torch.manual_seed(seed)
    try:
        if job["candidate_path"] == "BASELINE":
            model = adapter.build_model().to(device)
            opt = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
            lr_schedule, hints = None, {}
        else:
            recipe = _load_recipe(job["candidate_path"])
            model = recipe.build_model().to(device)
            opt = recipe.make_optimizer(model)
            lr_schedule = getattr(recipe, "lr_schedule", None)
            hints = getattr(recipe, "TRAIN_HINTS", {}) or {}
    except Exception as e:  # noqa: BLE001 — candidate code
        import traceback
        res["note"] = f"{type(e).__name__}: {e}\n" + \
            "\n".join(traceback.format_exc().splitlines()[-6:])
        return out()

    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["arch_fp"] = arch_fingerprint(model)
    if job.get("param_cap") and res["n_params"] > job["param_cap"]:
        res.update(gate="param_cap",
                   note=f"{res['n_params']:,} params exceeds cap "
                        f"{job['param_cap']:,}")
        return out()
    if job.get("expected_arch_fp") and res["arch_fp"] != job["expected_arch_fp"]:
        res.update(gate="arch_lock",
                   note="architecture changed in a hyperparameter phase "
                        f"(fingerprint {res['arch_fp']} != parent "
                        f"{job['expected_arch_fp']})")
        return out()

    clip = hints.get("grad_clip")
    base_lrs = [g["lr"] for g in opt.param_groups]
    batches = _cycled(adapter)
    deadline = time.time() + job["train_seconds"]
    step = 0
    try:
        first_loss = None
        while time.time() < deadline or (job.get("check_only") and step < 2):
            batch = next(batches)
            opt.zero_grad(set_to_none=True)
            loss = adapter.loss_fn(model, batch)
            if first_loss is None:
                first_loss = float(loss)
                if not torch.isfinite(loss):
                    res.update(gate="sanity", note=f"initial loss not finite: {first_loss}")
                    return out()
                if job.get("check_only"):
                    deadline = 0  # two steps then stop
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if lr_schedule is not None:
                scale = float(lr_schedule(step))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * scale
            opt.step()
            if hasattr(model, "post_optimizer_step"):
                model.post_optimizer_step()  # e.g. EMA target updates (JEPA)
            if wb is not None and step % 5 == 0:
                try:
                    wb.log({"train/loss": float(loss),
                            "train/lr": opt.param_groups[0]["lr"]}, step=step)
                except Exception:  # noqa: BLE001
                    pass
            step += 1
            if job.get("check_only") and step >= 2:
                break
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        import traceback
        res.update(gate="sanity", note=f"training raised {type(e).__name__}: {e}\n"
                   + "\n".join(traceback.format_exc().splitlines()[-6:]))
        return out()

    res["steps"] = step
    if job.get("check_only"):
        res.update(ok=True, gate="loaded")
        return out()
    val = _holdout_loss(adapter, model, job["eval_batches"])
    if not (val == val and abs(val) < 1e6):  # NaN/inf guard
        res.update(gate="sanity", note=f"validation loss not finite: {val}")
        return out()
    res.update(ok=True, gate="trained", val_loss=val)
    return out()


if __name__ == "__main__":
    main()
