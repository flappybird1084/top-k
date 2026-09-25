"""Entry point (spec §2): owns the GPU, runs the loop unattended.

  python search.py --adapter adapters/jepa.py                # profile from env or DEV
  KERNELEVO_PROFILE=RUN python search.py --adapter adapters/lm.py
  python search.py --adapter adapters/jepa.py --llm stub     # harness smoke test
  python search.py --adapter adapters/jepa.py --lineage ema_update
  python search.py --repo https://github.com/x/y --comments "optimize the ViT" \\
      --max-debug-turns 5                                    # repo pipeline
"""

import argparse
import datetime
import os
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters/jepa.py",
                    help="adapter module (adapters/jepa.py or adapters.jepa)")
    ap.add_argument("--repo", default=None,
                    help="git URL or local path — the adapter-writing agent builds "
                         "an adapter for the repo's stated model (overrides --adapter)")
    ap.add_argument("--comments", default="",
                    help="freeform guidance for the adapter-writing agent")
    ap.add_argument("--max-debug-turns", type=int, default=None,
                    help="adapter-writer repair attempts (default from config)")
    ap.add_argument("--profile", default=None, choices=["DEV", "RUN"],
                    help="config profile; default from KERNELEVO_PROFILE, else DEV")
    ap.add_argument("--llm", default=None,
                    help="override LLM provider for all roles (stub|anthropic|openai|wandb"
                         "|codex_oauth|claude_oauth[:model])")
    ap.add_argument("--lineage", default=None, help="restrict the run to one lineage")
    ap.add_argument("--out", default=None, help="output dir (default runs/<name>-<ts>)")
    ap.add_argument("--max-generations", type=int, default=None)
    ap.add_argument("--spend-cap", type=float, default=None,
                    help="override spend_cap_usd for this run")
    ap.add_argument("--mode", default="kernel", choices=["kernel", "recipe"],
                    help="kernel evolution (default) or recipe-golf")
    ap.add_argument("--recipe-json", default=None,
                    help="JSON overrides merged into cfg['recipe']")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required — run this on the GPU box (molab), "
                         "not the laptop. Even --llm stub exercises real compiles/gates.")

    import config
    cfg = config.load(args.profile)
    if args.llm:
        cfg["llm"] = args.llm
        cfg["planner_llm"] = cfg["subagent_llm"] = cfg["curator_llm"] = None
    if args.max_generations:
        cfg["max_generations"] = args.max_generations
    if args.spend_cap:
        cfg["spend_cap_usd"] = args.spend_cap
    if args.recipe_json:
        import json as _json
        cfg["recipe"].update(_json.loads(args.recipe_json))

    if args.repo:
        name = os.path.basename(args.repo.rstrip("/")).replace(".git", "") or "repo"
    else:
        name = os.path.basename(args.adapter).replace(".py", "")
    out_dir = args.out or os.path.join(
        "runs", f"{name}-{cfg['profile'].lower()}-"
                f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    latest = os.path.join("runs", "latest")
    os.makedirs(out_dir, exist_ok=True)
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(out_dir), latest)
    except OSError:
        pass

    # W&B identity: name says what/how, group ties relaunches of one job
    # together (the work dir, e.g. kevo_<jobid8>), tags make filtering easy.
    cfg["wandb_run_name"] = (f"{name}-{args.mode}-{cfg['profile'].lower()}-"
                             f"{datetime.datetime.now().strftime('%H%M%S')}")
    cfg["wandb_group"] = os.path.basename(
        os.path.dirname(os.path.abspath(out_dir))) or name
    cfg["wandb_tags"] = [args.mode, cfg["profile"].lower(), name]

    print(f"[search] profile={cfg['profile']} llm={cfg['llm']} out={out_dir}")
    from kernelevo import loop
    try:
        adapter_spec, pool = args.adapter, None
        if args.repo:
            from kernelevo import adapter_writer
            from kernelevo.llm import LLMPool
            from kernelevo.obs import init_weave
            init_weave()  # capture adapter-writer traces too
            pool = LLMPool(cfg)
            adapter_spec, _ = adapter_writer.prepare(
                args.repo, args.comments,
                args.max_debug_turns or cfg["max_debug_turns"],
                out_dir, pool.adapter, cfg["device"], cfg["seed"], mode=args.mode)
        if args.mode == "recipe":
            from kernelevo import recipe_loop
            recipe_loop.run(cfg, adapter_spec, out_dir, pool=pool)
        else:
            loop.run(cfg, adapter_spec, out_dir, only_lineage=args.lineage,
                     pool=pool)
    except KeyboardInterrupt:
        print("\n[search] interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
