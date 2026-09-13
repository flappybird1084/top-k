"""Reproducible search profiles; credentials never belong in this module."""

import os
from copy import deepcopy


DEV = dict(
    max_generations=2,
    candidates_per_gen=2,
    max_repairs=1,
    gen_wallclock_s=180,
    spend_cap_usd=3.0,
    retire_after=2,
    systemic_halt_after=2,
    llm="stub",
    wandb_mode="disabled",
)

RUN = dict(
    max_generations=5,
    candidates_per_gen=4,
    max_repairs=3,
    gen_wallclock_s=900,
    spend_cap_usd=100.0,
    spend_accounting="api_equivalent_estimate",
    retire_after=5,
    systemic_halt_after=3,
    llm="codex_oauth",
    planner_llm="gpt-5.6-sol",
    subagent_llm="gpt-5.6-sol",
    curator_llm="gpt-5.6-sol",
    wandb_mode="online",
    wandb_entity="stephenslee0127-acme",
    wandb_project="kernel-evolution",
)


def active_config():
    profile = os.environ.get("KERNEL_EVOLUTION_PROFILE", "DEV").upper()
    if profile not in {"DEV", "RUN"}:
        raise ValueError("KERNEL_EVOLUTION_PROFILE must be DEV or RUN")
    defaults = dict(seed=1729, dtype="float32", batch_size=16,
        benchmark_protocol="direct_inductor_v2",
        subagent_models=[], planner_effort="medium",subagent_effort="medium",curator_effort="medium",
        allowed_ops=["layer_norm_backward", "ema_update", "masked_gather_add", "gelu_mlp"],
        min_pct_step_time=5.0, profile_warmup=20, profile_steps=10,
        gate3_margin=0.03, gate4_margin=0.01,
        warmup_steps=10, timed_steps=30, micro_reps=50,
        calibration_repeats=5, calibration_seconds=300,
        rtol=1e-4, atol=1e-5, run_wallclock_s=7200,
        peak_flops=None, max_call_seconds=240, max_call_reservation_usd=3.0,
        max_lessons=20, compile_workers=2, llm_concurrency=2)
    return {"profile": profile, **defaults, **deepcopy({"DEV": DEV, "RUN": RUN}[profile])}
