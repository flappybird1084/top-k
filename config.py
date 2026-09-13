"""Config profiles (spec §8). Selected by KERNELEVO_PROFILE env var or --profile.

Every run logs its active config as the first archive row and as the W&B run config.
calibrate() may raise gate margins and tolerances above these defaults, never lower.
"""

import os

# bf16/fp16 tensor-core dense peak FLOPs, keyed on substring of torch.cuda.get_device_name().
# Approximate; override with peak_flops_override in config for exact numbers.
PEAK_FLOPS = {
    "RTX PRO 6000": 250e12,  # Blackwell workstation, bf16 dense (approx)
    "H100": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A10G": 70e12,
    "L40S": 362e12,
    "L4": 121e12,
    "T4": 65e12,
    "V100": 125e12,
    "4090": 165e12,
    "3090": 71e12,
}

# USD per million tokens: {model substring: (input, output)}. Fallback is _DEFAULT_PRICE.
PRICE_PER_MTOK = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "gpt-5": (2.5, 10.0),
    "gpt-4o": (2.5, 10.0),
}
_DEFAULT_PRICE = (5.0, 25.0)

_COMMON = dict(
    seed=1234,
    device="cuda",
    # lineage set = profiler output ∩ allowed_ops (spec §3.3)
    allowed_ops=["ema_update", "masked_gather_add", "layer_norm", "gelu_mlp",
                 "rms_norm", "relu2_mlp", "swiglu_mlp", "geglu_mlp",
                 "cross_entropy"],
    min_pct_step_time=5.0,
    profile_warmup=20,
    profile_steps=10,
    # gate 2: top-3 observed shapes + 1 unseen, 2 trials each (fresh inputs)
    gate2_shapes=3,
    gate2_trials=2,
    # defaults; calibrate() raises to 2x observed eager-vs-inductor floor if larger
    tol=dict(fp32=(1e-4, 1e-5), bf16=(1e-2, 1e-2), fp16=(1e-2, 1e-2)),
    # gate 3: median-of-50 isolation benchmark, incumbent re-measured in-session
    gate3_warmup=10,
    gate3_iters=50,
    gate3_margin=0.03,  # candidate must be >3% faster; raised by calibrate() if noise larger
    # gate 4: in-model, 10 warmup + 30 timed steps
    gate4_warmup=10,
    gate4_steps=30,
    gate4_margin=0.01,
    verify_timeout_s=600,       # hard cap per verify subprocess (hang kill)
    compile_timeout_s=180,
    peak_flops_override=None,   # float, wins over PEAK_FLOPS lookup
    calib_reps=5,               # noise-floor repetitions
    run_deadline_s=12 * 3600,
    # LLM roles. Provider string: stub|anthropic|openai|wandb, optionally "provider:model".
    # subagent_llm may be a list -> jobs alternate between entries (model diversity, spec §7).
    planner_llm=None,           # None -> falls back to `llm`
    subagent_llm=None,
    curator_llm=None,
    adapter_llm=None,           # adapter-writing agent; falls back to subagent_llm
    max_debug_turns=5,          # adapter-writer ingest-repair attempts (repo pipeline)
    anthropic_model="claude-sonnet-5",
    anthropic_subagent_model="claude-opus-5",
    openai_model="gpt-5",
    # W&B Inference model ids (from the live /v1/models catalog). Kimi-K2.7-Code
    # is the coding-tuned pick for subagent-style work; override per role with
    # e.g. planner_llm="wandb:deepseek-ai/DeepSeek-V4-Flash".
    wandb_inference_model="moonshotai/Kimi-K2.7-Code",
    max_llm_tokens=8192,
    lessons_tail=20,
    planner_web_search=True,    # planner may use the provider's native web-search tool
)

DEV = {
    **_COMMON,
    **dict(
        max_generations=2,
        candidates_per_gen=2,
        max_repairs=1,
        # Per-generation phase budgets (replaces the spec's single gen_wallclock):
        # authoring (planner + subagent LLM calls + compile + gate 2) and
        # evaluation (gates 3-4) each get their own slice, so slow authoring can
        # never starve the benchmarks. Sized from measured molab reality: LLM
        # calls 60-120s each, first gate-3 inductor compile 30-90s.
        llm_budget_s=300,
        eval_budget_s=300,
        spend_cap_usd=3,
        retire_after=2,
        systemic_halt_after=2,
        llm="stub",
        calib_span_s=10,
        calib_reps=3,
        gate3_iters=20,
        gate4_steps=10,
        min_pct_step_time=0.5,
        run_deadline_s=3600,
    ),
}

RUN = {
    **_COMMON,
    **dict(
        max_generations=50,
        candidates_per_gen=8,
        max_repairs=3,
        llm_budget_s=900,
        eval_budget_s=900,
        spend_cap_usd=100,
        retire_after=5,
        systemic_halt_after=3,
        llm="anthropic",
        calib_span_s=300,
    ),
}

PROFILES = {"DEV": DEV, "RUN": RUN}


def load(profile: str | None = None) -> dict:
    name = (profile or os.environ.get("KERNELEVO_PROFILE") or "DEV").strip().upper()
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; expected one of {list(PROFILES)}")
    cfg = dict(PROFILES[name])
    cfg["profile"] = name
    return cfg


def peak_flops_for(gpu_name: str, cfg: dict) -> float | None:
    if cfg.get("peak_flops_override"):
        return float(cfg["peak_flops_override"])
    for key, val in PEAK_FLOPS.items():
        if key.lower() in gpu_name.lower():
            return val
    return None


def price_for(model_name: str) -> tuple[float, float]:
    for key, val in PRICE_PER_MTOK.items():
        if key in (model_name or ""):
            return val
    return _DEFAULT_PRICE
