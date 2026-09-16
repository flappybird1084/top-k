"""Recipe-golf orchestrator: architecture generations → hyperparameter
generations (arch-locked to parents) → finals. Signal = held-out val loss
after wall-clock-budgeted training; baseline (base adapter + harness AdamW)
re-measured at every distinct budget. Reuses the kernel harness's archive,
mirror, molab transport, and LLM pool."""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from kernelevo import ingest, recipes
from kernelevo.archive import Archive
from kernelevo.gates import REPO_ROOT
from kernelevo.llm import LLMPool
from kernelevo.obs import Mirror, weave_op, current_trace_url


RECIPE_IDLE_TIMEOUT_S = 90   # kill an eval silent this long; worker heartbeats every ~8s


def _log_failed_wandb(meta, gate, note):
    """Candidates that fail AUTHORING never reach recipe_worker, so they'd have
    no W&B run at all — give them one carrying the error trace, marked Failed,
    so every slot in the group is inspectable. Fire-and-forget."""
    if not meta or not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb
        r = wandb.init(project=os.environ.get("WANDB_PROJECT") or "kernel-evolution",
                       entity=os.environ.get("WANDB_ENTITY") or None,
                       name=meta.get("name"), group=meta.get("group"),
                       tags=(meta.get("tags") or []) + ["failed"],
                       config=meta.get("config"),
                       reinit="create_new",   # never disturbs the Mirror run
                       settings=wandb.Settings(silent=True))
        r.summary["gate"] = gate
        r.summary["error"] = (note or "no note")[:4000]
        r.finish(exit_code=1)
    except Exception:  # noqa: BLE001 — observability must never break the loop
        pass


def _execute_worker(job: dict, path_hint: str, timeout: int):
    from kernelevo.procstream import run_result_worker
    job_path = path_hint + ".rjob.json"
    with open(job_path, "w") as f:
        json.dump(job, f)
    r = run_result_worker(
        [sys.executable, "-m", "kernelevo.recipe_worker", job_path],
        cwd=REPO_ROOT, total_timeout=timeout, idle_timeout=RECIPE_IDLE_TIMEOUT_S)
    if r.result is not None:
        return r.result
    if r.timed_out:
        why = (f"went silent for >{RECIPE_IDLE_TIMEOUT_S}s"
               if r.status == "idle_timeout" else f"exceeded {timeout}s")
        return dict(ok=False, gate="hang", note=f"[infra] evaluation {why}; killed")
    return dict(ok=False, gate="crash",
                note="[infra] worker crashed:\n" + (r.stderr or "")[-1500:])


@weave_op
def _run_worker(job: dict, path_hint: str, timeout: int):
    # authoring load-checks are plumbing, not evaluations — no event, or the
    # diagram fills with dozens of 'Checking recipe' bubbles per generation
    if job.get('check_only'):
        result = _execute_worker(job, path_hint, timeout)
        result['weave_trace_url'] = current_trace_url()
        return result
    import uuid
    wcfg = (job.get('wandb') or {}).get('config') or {}
    eid = uuid.uuid4().hex
    event = dict(id=eid, kind=wcfg.get('phase') or 'architecture',
                 kernel=os.path.basename(job['candidate_path']),
                 stage='Training and evaluating held-out data',
                 strategy=wcfg.get('strategy')
                 or str(job['train_seconds']) + 's training budget',
                 started_at=time.time())
    print('[evaluation] '+json.dumps(event),flush=True)
    try:
        result=_execute_worker(job,path_hint,timeout)
        result['weave_trace_url']=current_trace_url()
        return result
    finally:
        print('[evaluation] '+json.dumps(dict(id=eid,finished=True)),flush=True)


def _base_source(adapter_spec: str, adapter_mod) -> str:
    try:
        path = adapter_spec if adapter_spec.endswith(".py") else \
            inspect.getsourcefile(adapter_mod)
        return open(path).read()
    except (OSError, TypeError):
        return "(source unavailable)"


def _loss_source(adapter_mod) -> str:
    try:
        return inspect.getsource(adapter_mod.loss_fn)
    except (OSError, TypeError):
        return "(source unavailable)"


def _parse_jobs(text: str):
    s, e = text.find("{"), text.rfind("}")
    obj = json.loads(text[s:e + 1])
    return obj.get("jobs")


def run(cfg: dict, adapter_spec: str, out_dir: str, pool: LLMPool | None = None) -> str:
    rc = cfg["recipe"]
    os.makedirs(out_dir, exist_ok=True)
    cand_dir = os.path.join(out_dir, "recipes")
    os.makedirs(cand_dir, exist_ok=True)
    run_deadline = time.time() + cfg["run_deadline_s"]

    adapter, adapter_name = ingest.load_adapter(adapter_spec)
    model_name = (os.path.basename(os.path.dirname(adapter_name)) or "repo") \
        if adapter_name.endswith(".py") else adapter_name.split(".")[-1]
    archive = Archive(os.path.join(out_dir, "archive.sqlite"))
    mirror = Mirror(cfg, model_name + "-recipe")
    trail_path = os.path.join(out_dir, "adapter_attempts.json")
    if os.path.exists(trail_path):
        mirror.log_adapter_trail(json.load(open(trail_path)))
    print(f"[ingest] {adapter_name}")
    info = ingest.ingest(adapter, cfg)
    print(f"[ingest] {info['n_params']/1e6:.1f}M params, "
          f"{info['samples_per_batch']} samples/batch, loss0={info['loss0']:.4f}")
    param_cap = int(info["n_params"] * rc["param_budget_ratio"])
    base_source = _base_source(adapter_spec, adapter)
    loss_source = _loss_source(adapter)
    precision = rc.get("precision", "bf16")

    # ground-truth model report for the planner/subagents + baseline module
    # inventory for the label-vs-diff line (audit: run ef48abdf spent two
    # generations "introducing" features the repo already shipped, sight unseen)
    try:
        _m = adapter.build_model()
        model_report = recipes.model_report(_m)
        base_modules = recipes.module_inventory(_m)
        del _m
        import gc
        gc.collect()
    except Exception as e:  # noqa: BLE001 — the report is an aid, never a blocker
        print(f"[recipe] model report unavailable ({e}); prompts go without it")
        model_report, base_modules = None, {}

    model_id = archive.add_model(
        name=model_name, adapter_path=adapter_spec, n_params=info["n_params"],
        flops_per_sample=0, gpu_name="", peak_flops=None,
        config_json=json.dumps({k: v for k, v in cfg.items()}, default=str))
    lineage_ids = {}
    for kind in ("architecture", "mixed", "hyperparam", "finals"):
        lineage_ids[kind] = archive.add_lineage(model_id, kind, [], 0.0)

    pool = pool or LLMPool(cfg)
    run_prefix = (cfg.get("wandb_group") or os.path.basename(out_dir) or "run")
    run_prefix = run_prefix.removeprefix("kevo_")

    def wmeta(suffix, phase_kind, strategy=None, train_seconds=None):
        return dict(name=f"{run_prefix}-{suffix}", group=cfg.get("wandb_group"),
                    tags=["recipe", phase_kind],
                    config=dict(strategy=(strategy or "")[:250],
                                phase=phase_kind, train_seconds=train_seconds))

    active_phases = [p for p in rc["phases"] if p["generations"] > 0]
    budgets = sorted({p["train_seconds"] for p in active_phases}
                     | {rc["finals_train_seconds"]})

    # -------- baseline at every budget (the incumbent, measured in-session)
    baseline = {}
    for secs in budgets:
        job = dict(base_adapter=adapter_name, candidate_path="BASELINE",
                   train_seconds=secs, eval_batches=rc["eval_batches"],
                   seed=cfg["seed"], device=cfg["device"], param_cap=None,
                   precision=precision,
                   wandb=wmeta(f"baseline-{secs}s", "baseline",
                               "baseline", secs))
        r = _run_worker(job, os.path.join(cand_dir, f"baseline_{secs}s"),
                        secs + rc["eval_timeout_grace_s"])
        if not r.get("ok"):
            raise SystemExit(f"[recipe] baseline evaluation failed at {secs}s: "
                             f"{r.get('note')}")
        baseline[secs] = r["val_loss"]
        mirror._log({"recipe/phase":"baseline","recipe/train_secs":secs,"recipe/baseline_val_loss":r["val_loss"],"recipe/model_params":r["n_params"]})
        print(f"[baseline] {secs}s train -> val loss {r['val_loss']:.4f} "
              f"({r['steps']} steps)")
        mirror._log({f"baseline/val_loss_{secs}s": r["val_loss"],
                     f"baseline/steps_{secs}s": r["steps"]})
        archive.add_candidate(
            lineage_id=lineage_ids["architecture"], generation=0,
            strategy=f"baseline (base adapter + harness AdamW) @{secs}s",
            source_kind="baseline", gate_reached=4, compile_ok=1, correct_ok=1,
            repairs_used=0, accepted=1, val_loss=r["val_loss"], phase="baseline",
            train_secs=secs, model_params=r["n_params"], arch_fp=r["arch_fp"],weave_trace_url=r.get("weave_trace_url"))

    results_all = []   # dicts with id, val_loss, phase, code_path, strategy
    gen_index = 0

    last_gen_id = None

    def author_and_eval(phase, jobs, parents_by_id):
        nonlocal gen_index, last_gen_id
        gen_index += 1
        gen_id = archive.start_generation(model_id)
        last_gen_id = gen_id
        tok_in0, tok_out0 = pool.total_tokens()
        outs = []
        n_acc = 0
        # authoring runs 8-wide; each candidate's (serial, GPU-exclusive)
        # evaluation starts the moment ITS authoring completes — later
        # candidates keep generating while earlier ones train
        ex = ThreadPoolExecutor(max_workers=rc["subagent_parallelism"])
        fut_to_job = {}
        for i, job in enumerate(jobs):
            parent = parents_by_id.get(job.get("parent"))
            parent_src = None
            if parent and parent.get("code_path") and \
                    os.path.exists(parent["code_path"]):
                parent_src = open(parent["code_path"]).read()

            def save_fn(src, attempt, _i=i):
                src = "".join(m + "\n" for m in re.findall(
                    r"^from __future__ import .*$", src, re.MULTILINE)) + \
                    re.sub(r"^from __future__ import .*$\n?", "", src,
                           flags=re.MULTILINE)
                p = os.path.join(cand_dir,
                                 f"g{gen_index}_{phase['kind']}_{_i}_a{attempt}.py")
                with open(p, "w") as f:
                    f.write(src)
                return p

            def check_fn(path):
                r = _run_worker(dict(
                    base_adapter=adapter_name, candidate_path=path,
                    train_seconds=0, eval_batches=1, seed=cfg["seed"],
                    device=cfg["device"], param_cap=param_cap,
                    precision=precision, check_only=True), path, 240)
                return bool(r.get("ok")), (r.get("note") or r.get("gate") or "")

            llm = pool.subagent_for(i)
            job["_idx"] = i
            fut = ex.submit(
                recipes.author_recipe, llm, phase, job, base_source,
                parent_src, loss_source, param_cap, lessons,
                rc["recipe_max_repairs"], save_fn, check_fn,
                recipes.baseline_recipe_source(adapter_name),
                model_report)
            fut_to_job[fut] = job

        authored = []
        for fut in as_completed(fut_to_job):
            job = fut_to_job[fut]
            try:
                authored.append((job, fut.result()))
            except Exception as e:  # noqa: BLE001
                authored.append((job, dict(
                    strategy=job["strategy"], parent=job.get("parent"),
                    code_path=None, repairs_used=0, load_ok=False,
                    model_name=None,
                    failure_note=f"[infra] author crashed: {e}")))
            # evaluate the just-authored candidate now, while others generate
            job, a = authored[-1]
            row = dict(lineage_id=lineage_ids[phase["kind"]],
                       generation=gen_index, strategy=a["strategy"],
                       parent_id=a.get("parent") if isinstance(a.get("parent"), int) else None,
                       source_kind="recipe", code_path=a.get("code_path"),
                       model_name=a.get("model_name"),
                       repairs_used=a.get("repairs_used", 0),
                       phase=phase["kind"], train_secs=phase["train_seconds"],
                       compile_ok=int(a.get("load_ok", False)), correct_ok=0,
                       gate_reached=0, accepted=0,
                       tokens_in=a.get("tokens_in", 0),
                       tokens_out=a.get("tokens_out", 0),
                       failure_note=a.get("failure_note"))
            if a.get("load_ok"):
                parent = parents_by_id.get(job.get("parent"))
                expected_fp = (parent or {}).get("arch_fp") \
                    if phase["kind"] == "hyperparam" else None
                r = _run_worker(dict(
                    base_adapter=adapter_name, candidate_path=a["code_path"],
                    train_seconds=phase["train_seconds"],
                    eval_batches=rc["eval_batches"], seed=cfg["seed"],
                    device=cfg["device"], param_cap=param_cap,
                    precision=precision, expected_arch_fp=expected_fp,
                    wandb=wmeta(f"g{gen_index}-s{job.get('_idx', 0)}",
                                phase["kind"], a["strategy"],
                                phase["train_seconds"])), a["code_path"],
                    phase["train_seconds"] + rc["eval_timeout_grace_s"])
                row["weave_trace_url"]=r.get("weave_trace_url")
                if r.get("ok"):
                    accepted = r["val_loss"] < baseline[phase["train_seconds"]] * \
                        (1 - rc["loss_margin_rel"])
                    row.update(gate_reached=4 if accepted else 3, correct_ok=1,
                               accepted=int(accepted), val_loss=r["val_loss"],
                               model_params=r["n_params"], arch_fp=r["arch_fp"],
                               failure_note=None)
                    n_acc += int(accepted)
                    delta = 100 * (baseline[phase["train_seconds"]] - r["val_loss"]) \
                        / baseline[phase["train_seconds"]]
                    print(f"[recipe] {phase['kind']} g{gen_index}: val "
                          f"{r['val_loss']:.4f} vs baseline "
                          f"{baseline[phase['train_seconds']]:.4f} ({delta:+.2f}%)"
                          f"{' ACCEPTED' if accepted else ''} — "
                          f"{a['strategy'][:70]}")
                    # mechanical label-vs-diff line: what ACTUALLY changed,
                    # independent of the strategy prose
                    dparams = r["n_params"] - info["n_params"]
                    bits = [f"params {dparams / 1e6:+.1f}M" if dparams else
                            "params unchanged"]
                    mod_d = recipes.inventory_diff(base_modules,
                                                   r.get("modules") or {})
                    bits.append(f"modules: {mod_d}" if mod_d
                                else "modules unchanged")
                    if r.get("has_lr_schedule"):
                        bits.append("+lr_schedule")
                    if r.get("hints"):
                        bits.append(f"hints={r['hints']}")
                    if r.get("steps") is not None:
                        bits.append(f"{r['steps']} steps")
                    row["diff"] = " | ".join(bits)
                    print(f"[recipe-diff] #{job.get('_idx', 0)}: {row['diff']}")
                else:
                    note = r.get("note") or r.get("gate")
                    row.update(gate_reached=1, correct_ok=0,
                               failure_note=str(note)[:1500])
                    print(f"[recipe] {phase['kind']} g{gen_index}: "
                          f"{r.get('gate')} — {str(note)[:100]}")
            else:
                _log_failed_wandb(
                    wmeta(f"g{gen_index}-s{job.get('_idx', 0)}", phase["kind"],
                          a["strategy"], phase["train_seconds"]),
                    "author_failed", a.get("failure_note"))
            cid = archive.add_candidate(**{k: v for k, v in row.items()
                                           if k != "diff"})
            row["id"] = cid
            outs.append(row)
            base_v = baseline[phase["train_seconds"]]
            v = row.get("val_loss")
            mirror._log({
                "recipe/val_loss": row.get("val_loss"),
                "recipe/baseline_val_loss": baseline[phase["train_seconds"]],
                "recipe/phase": phase["kind"],
                "recipe/train_secs": phase["train_seconds"],
                "recipe/candidate_id": cid,
                "recipe/accepted": row["accepted"],
                "recipe/model_params": row.get("model_params"),
                f"recipe/{phase['kind']}/val_loss": row.get("val_loss"),
                "cand/id": cid, "cand/generation": gen_index,
                f"cand/{phase['kind']}/val_loss": v,
                "cand/delta_pct": (100 * (base_v - v) / base_v) if v else None,
                "cand/accepted": row["accepted"],
                "cand/params": row.get("model_params"),
                "best/val_loss": min([r["val_loss"] for r in results_all + outs
                                      if r.get("val_loss")] + ([v] if v else []),
                                     default=None),
            })
        ex.shutdown(wait=True)
        tok_in, tok_out = pool.total_tokens()
        gen_tok_in, gen_tok_out = tok_in - tok_in0, tok_out - tok_out0
        archive.finish_generation(gen_id, len(outs), n_acc, pool.total_usd(),
                                  tokens_in=gen_tok_in, tokens_out=gen_tok_out)
        print(f"[generation] g{gen_index} done: {len(outs)} candidate(s), "
              f"{n_acc} accepted — {gen_tok_in:,} tokens in / "
              f"{gen_tok_out:,} out")
        mirror._log({"gen/tokens_in": gen_tok_in, "gen/tokens_out": gen_tok_out,
                     "generation": gen_index})

        # curator lessons
        if pool.total_usd() < cfg["spend_cap_usd"]:
            from kernelevo import curator
            digest = dict(phase=phase["kind"], generation=gen_index,
                          n_failed=sum(1 for o in outs if not o.get("val_loss")),
                          baseline_val_loss=baseline[phase["train_seconds"]],
                          candidates=[{k: o.get(k) for k in
                                       ("strategy", "val_loss", "model_params",
                                        "failure_note")} for o in outs])
            for line in curator.curate(pool.curator, digest):
                archive.add_lesson(gen_id, model_id, line)
        return outs

    # ---------------------------------------------------------------- phases
    lessons: list[str] = []
    for phase in active_phases:
        for _ in range(phase["generations"]):
            if time.time() > run_deadline or pool.total_usd() >= cfg["spend_cap_usd"]:
                print("[recipe] stopping early (deadline or spend cap)")
                if last_gen_id is not None:
                    archive.set_stop_reason(last_gen_id, "deadline_or_spend_cap")
                break
            lessons = archive.lessons_tail(model_id, cfg["lessons_tail"])
            evaluated = [r for r in results_all if r.get("val_loss")]
            parents = sorted(evaluated, key=lambda r: r["val_loss"])[:rc.get("parent_pool", 4)]
            parents_by_id = {p["id"]: p for p in parents}
            base_summary = dict(
                model=model_name, params=info["n_params"], param_cap=param_cap,
                baseline_val_loss_by_budget=baseline,
                batch=info["samples_per_batch"])
            # 'diff' is the mechanical record of what each candidate actually
            # changed — the planner reasons from it, not from strategy prose
            outcomes = [{k: r.get(k) for k in ("phase", "strategy", "val_loss",
                                               "failure_note", "diff")}
                        for r in results_all[-16:]]
            print(f"\n=== recipe {phase['kind']} generation {gen_index + 1} — "
                  f"spend ${pool.total_usd():.2f} ===")
            from kernelevo import researcher as researchmod
            from kernelevo import websearch
            can_research = websearch.available()
            msgs = recipes.recipe_planner_prompt(phase, base_summary, outcomes,
                                                 lessons, phase["candidates"],
                                                 parents,
                                                 research_enabled=can_research,
                                                 model_report=model_report)
            jobs, research_used, empty_retry = [], 0, False
            for _ in range(5):
                resp = pool.planner.complete(
                    msgs, json_mode=True,
                    meta={"active_lineages": [phase["kind"]],
                          "n_jobs": phase["candidates"]})
                try:
                    s, e = resp.text.find("{"), resp.text.rfind("}")
                    obj = json.loads(resp.text[s:e + 1])
                except (ValueError, json.JSONDecodeError):
                    break
                question = obj.get("research")
                if question and can_research and research_used < 2:
                    research_used += 1
                    brief = researchmod.research(pool.researcher, str(question))
                    print(f"[planner] research: {str(question)[:70]} -> "
                          f"{len(brief)} char brief")
                    msgs = msgs + [
                        {"role": "assistant", "content": resp.text},
                        {"role": "user", "content":
                         "Research brief:\n" + brief +
                         f'\n\nContinue: one more {{"research": "..."}} '
                         f'({2 - research_used} left) or the final '
                         f'{{"jobs": [...]}}.'}]
                    continue
                jobs = (obj.get("jobs") or [])[:phase["candidates"]]
                if not jobs and not empty_retry:
                    empty_retry = True
                    print("[recipe] planner returned zero jobs; re-prompting once")
                    msgs = msgs + [
                        {"role": "assistant", "content": resp.text},
                        {"role": "user", "content":
                         f"Zero jobs wastes the generation. Propose between 1 "
                         f'and {phase["candidates"]} jobs now as '
                         f'{{"jobs": [...]}}.'}]
                    continue
                break
            if not jobs:
                print("[recipe] planner produced no jobs; skipping generation")
                continue
            for j in jobs:
                j.setdefault("parent", parents[0]["id"] if parents and
                             phase["kind"] == "hyperparam" else None)
            print(f"[planner] {len(jobs)} job(s): "
                  + "; ".join(j["strategy"][:60] for j in jobs[:3]) + " …")
            try:
                results_all += author_and_eval(phase, jobs, parents_by_id)
            except Exception as e:
                # a crashed generation must still close its archive row —
                # NULL n_candidates rows were audit finding 17's second half
                try:
                    row = archive.db.execute(
                        "SELECT finished_at FROM generations WHERE id=?",
                        (last_gen_id,)).fetchone()
                    if row and row["finished_at"] is None:
                        archive.finish_generation(
                            last_gen_id, 0, 0, pool.total_usd(),
                            stop_reason=f"crashed: {e}"[:200])
                except Exception:  # noqa: BLE001 — recording must not mask the crash
                    pass
                raise

    # ---------------------------------------------------------------- finals
    finalists = sorted([r for r in results_all if r.get("val_loss")],
                       key=lambda r: r["val_loss"])[:rc["finals_top_k"]]
    fsecs = rc["finals_train_seconds"]
    print(f"\n=== finals: {len(finalists)} candidate(s) at {fsecs}s each ===")
    winner = None
    for fi, fr in enumerate(finalists):
        r = _run_worker(dict(
            base_adapter=adapter_name, candidate_path=fr["code_path"],
            train_seconds=fsecs, eval_batches=rc["eval_batches"],
            seed=cfg["seed"], device=cfg["device"], param_cap=param_cap,
            precision=precision,
            wandb=wmeta(f"final-s{fi}", "finals", fr["strategy"], fsecs)),
            fr["code_path"], fsecs + rc["eval_timeout_grace_s"])
        if not r.get("ok"):
            print(f"[finals] candidate {fr['id']} failed: {r.get('note')}")
            continue
        accepted = r["val_loss"] < baseline[fsecs] * (1 - rc["loss_margin_rel"])
        cid = archive.add_candidate(
            lineage_id=lineage_ids["finals"], generation=gen_index + 1,
            strategy=f"FINAL @{fsecs}s of: {fr['strategy']}"[:400],
            source_kind="recipe", code_path=fr["code_path"],
            parent_id=fr["id"], phase="finals", train_secs=fsecs,
            compile_ok=1, correct_ok=1, gate_reached=4 if accepted else 3,
            accepted=int(accepted), val_loss=r["val_loss"],
            model_params=r["n_params"], arch_fp=r["arch_fp"], repairs_used=0,weave_trace_url=r.get("weave_trace_url"))
        mirror._log({"recipe/phase":"finals","recipe/train_secs":fsecs,"recipe/val_loss":r["val_loss"],"recipe/baseline_val_loss":baseline[fsecs],"recipe/accepted":int(accepted),"recipe/candidate_id":cid,"recipe/model_params":r["n_params"]})
        delta = 100 * (baseline[fsecs] - r["val_loss"]) / baseline[fsecs]
        print(f"[finals] val {r['val_loss']:.4f} vs baseline "
              f"{baseline[fsecs]:.4f} ({delta:+.2f}%)"
              f"{' ACCEPTED' if accepted else ''} — {fr['strategy'][:70]}")
        mirror._log({"finals/val_loss": r["val_loss"], "finals/delta_pct": delta})
        if recipes.is_better_final(r["val_loss"], accepted, winner):
            winner = dict(fr, final_val_loss=r["val_loss"], final_id=cid)

    print("\n[result] ================= final performance =================")
    print(f"[result] baseline: val loss {baseline[fsecs]:.4f} after {fsecs}s "
          f"(base adapter + AdamW)")
    if winner:
        d = 100 * (baseline[fsecs] - winner["final_val_loss"]) / baseline[fsecs]
        print(f"[result] winner: val loss {winner['final_val_loss']:.4f} "
              f"({d:+.2f}% vs baseline) — {winner['strategy'][:120]}")
        print(f"[result] winning recipe: {winner['code_path']}")
    else:
        print("[result] no recipe beat the baseline at finals scale")
    if mirror.run is not None:
        try:
            import wandb
            mirror.run.summary["baseline_300s"] = baseline[fsecs]
            if winner:
                mirror.run.summary["winner_300s"] = winner["final_val_loss"]
                mirror.run.summary["improvement_pct"] = \
                    100 * (baseline[fsecs] - winner["final_val_loss"]) / baseline[fsecs]
                mirror.run.summary["winner_strategy"] = winner["strategy"][:250]
            rows = [dict(r) for r in archive.db.execute(
                "SELECT c.id, c.generation, c.parent_id, c.phase, c.strategy, "
                "c.val_loss, c.accepted, c.train_secs FROM candidates c "
                "JOIN lineages l ON c.lineage_id=l.id WHERE l.model_id=? "
                "ORDER BY c.id", (model_id,))]
            mirror.run.log({"lineage": wandb.Table(
                columns=list(rows[0].keys()),
                data=[list(r.values()) for r in rows])})
        except Exception:  # noqa: BLE001 — mirror is fire-and-forget
            pass
    # "Every stop writes stop_reason" held for kernel runs only — recipe
    # archives all had it empty (audit finding 17). SQLite is authoritative.
    if last_gen_id is not None:
        archive.set_stop_reason(last_gen_id, "recipe_complete")
    mirror.finish("recipe_complete")
    print(f"[loop] stopped: recipe_complete; total LLM spend "
          f"${pool.total_usd():.2f}; archive at "
          f"{os.path.join(out_dir, 'archive.sqlite')}")
    return "recipe_complete"