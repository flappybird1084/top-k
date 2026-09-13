"""The generation loop (spec §4.5) and its failure handling / stop conditions
(spec §6). SQLite is written first; the mirror is fire-and-forget."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from kernelevo import calibrate as calibmod
from kernelevo import curator, ingest, planner, profiling, subagent
from kernelevo.archive import Archive
from kernelevo.gates import GateRunner, source_hash
from kernelevo.llm import LLMPool
from kernelevo.obs import Mirror

MIN_VERIFY_WINDOW_S = 30


class GenContext:
    """Everything a subagent lifecycle needs; also the budget checks that abort
    work at the next safe boundary (spec §6.1/§6.3)."""

    def __init__(self, cfg, runner, pool, targets, lessons, candidates_dir,
                 archive, incumbents, gen_deadline, run_deadline):
        self.cfg = cfg
        self.runner = runner
        self.pool = pool
        self.targets = targets
        self.lessons = lessons
        self.candidates_dir = candidates_dir
        self.archive = archive
        self._incumbents = incumbents
        self.gen_deadline = gen_deadline
        self.run_deadline = run_deadline

    def lineage_info(self, op):
        return next(l for l in self.targets["lineages"] if l["op"] == op)

    def incumbents_snapshot(self):
        return {op: dict(entry) for op, entry in self._incumbents.items()}

    def parent_info(self, job):
        ids = job.get("parents") or ([job["parent"]] if job.get("parent") else [])
        sources, lat = [], None
        for cid in ids:
            row = self.archive.candidate(cid) if isinstance(cid, int) else None
            if row and row.get("code_path") and os.path.exists(row["code_path"]):
                sources.append(open(row["code_path"]).read())
                lat = lat or row.get("latency_us")
        if not sources:  # fall back to the lineage seed (inductor extraction)
            seed = self.lineage_info(job["lineage"])["seed_path"]
            if os.path.exists(seed):
                sources.append(open(seed).read()[:12000])
        return ("\n\n# ---- next parent ----\n\n".join(sources) or None), lat

    def budget_exceeded(self):
        if time.time() > self.gen_deadline:
            return (f"gen_timeout: LLM/authoring budget "
                    f"{self.cfg['llm_budget_s']}s exceeded")
        if time.time() > self.run_deadline:
            return "run_deadline reached"
        if self.pool.total_usd() >= self.cfg["spend_cap_usd"]:
            return f"spend_cap_usd ({self.cfg['spend_cap_usd']}) exceeded"
        return None


def _cfg_json(cfg):
    return json.dumps({k: v for k, v in cfg.items()}, default=str)


def _summary(archive, model_id, targets, active_ops):
    out = {}
    for lin in archive.lineages(model_id):
        op = lin["op_name"]
        if op not in active_ops:
            continue
        inc = archive.candidate(lin["incumbent_id"]) if lin["incumbent_id"] else None
        out[op] = dict(
            pct_step_time=lin["pct_step_time"],
            incumbent=(dict(id=inc["id"], strategy=inc["strategy"],
                            source_kind=inc["source_kind"],
                            latency_us=inc["latency_us"],
                            step_time_ms=inc["step_time_ms"]) if inc else None),
            recent=[dict(strategy=r["strategy"], gate_reached=r["gate_reached"],
                         accepted=bool(r["accepted"]),
                         note=(r["failure_note"] or "")[:160])
                    for r in archive.recent_outcomes(lin["id"], 3)],
            parents=[dict(id=p["id"], strategy=p["strategy"],
                          latency_us=p["latency_us"], accepted=bool(p["accepted"]))
                     for p in archive.parents_for(lin["id"])],
        )
    return out


def run(cfg: dict, adapter_spec: str, out_dir: str, only_lineage: str | None = None,
        pool: LLMPool | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    candidates_dir = os.path.join(out_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    run_start = time.time()
    run_deadline = run_start + cfg["run_deadline_s"]

    from kernelevo import patch
    patch.install()
    adapter, adapter_name = ingest.load_adapter(adapter_spec)
    if adapter_name.endswith(".py"):  # generated adapter: name it after its job dir
        model_name = os.path.basename(os.path.dirname(adapter_name)) or "repo"
    else:
        model_name = adapter_name.split(".")[-1]
    archive = Archive(os.path.join(out_dir, "archive.sqlite"))
    mirror = Mirror(cfg, model_name)

    print(f"[ingest] {adapter_name}")
    info = ingest.ingest(adapter, cfg)
    print(f"[ingest] {info['n_params']/1e6:.1f}M params, "
          f"{info['samples_per_batch']} samples/batch, loss0={info['loss0']:.4f}")

    print("[profile] profiling one training step…")
    targets = profiling.run_profile(adapter, cfg, info, out_dir)
    if only_lineage:
        targets["lineages"] = [l for l in targets["lineages"] if l["op"] == only_lineage]
    if not targets["lineages"]:
        raise SystemExit("no target lineages: profiler output ∩ allowed_ops ∩ "
                         f"≥{cfg['min_pct_step_time']}% step time is empty")
    print(f"[profile] step {targets['step_time_ms']:.2f}ms; lineages: "
          + ", ".join(f"{l['op']} ({l['pct_step_time']}%)" for l in targets["lineages"]))

    runner = GateRunner(cfg, out_dir, adapter_name, targets)
    print("[calibrate] noise/numerical floors + verifier self-test…")
    calib = calibmod.calibrate(cfg, runner, targets, out_dir)

    model_id = archive.add_model(
        name=model_name, adapter_path=adapter_spec, n_params=info["n_params"],
        flops_per_sample=targets["flops_per_sample"], gpu_name=calib["gpu_name"],
        peak_flops=calib["peak_flops"], config_json=_cfg_json(cfg))
    archive.add_calibration(
        model_id=model_id, noise_spread=calib["noise_spread"],
        gate3_margin=calib["gate3_margin"], gate4_margin=calib["gate4_margin"],
        tol_json=json.dumps(calib["tol"]), cheats_rejected=1,
        detail_json=json.dumps(calib, default=str))
    mirror.log_startup(targets, calib)

    # generation 0: the inductor seed is the incumbent (spec §4.4)
    lineage_ids, incumbents = {}, {}
    for lin in targets["lineages"]:
        lid = archive.add_lineage(model_id, lin["op"], lin["shapes"], lin["pct_step_time"])
        lineage_ids[lin["op"]] = lid
        cid = archive.add_candidate(
            lineage_id=lid, generation=0, strategy="inductor baseline",
            source_kind="inductor", code_path=lin["seed_path"],
            source_hash=source_hash(open(lin["seed_path"]).read()),
            gate_reached=4, compile_ok=1, correct_ok=1, repairs_used=0, accepted=1)
        archive.set_incumbent(lid, cid)
        incumbents[lin["op"]] = {"kind": "inductor"}

    pool = pool or LLMPool(cfg)
    stop_reason = None
    systemic_fail_gens = 0
    gen_id = None

    for gen in range(1, cfg["max_generations"] + 1):
        if time.time() > run_deadline:
            stop_reason = "run_deadline"
            break
        if pool.total_usd() >= cfg["spend_cap_usd"]:
            stop_reason = "spend_cap"
            break
        active = [l["op"] for l in targets["lineages"]
                  if not next(x for x in archive.lineages(model_id)
                              if x["op_name"] == l["op"])["retired"]]
        if not active:
            stop_reason = "all_lineages_retired"
            break

        gen_id = archive.start_generation(model_id)
        gen_start = time.time()
        lessons = archive.lessons_tail(model_id, cfg["lessons_tail"])
        active_targets = dict(targets)
        active_targets["lineages"] = [l for l in targets["lineages"] if l["op"] in active]
        summary = _summary(archive, model_id, targets, active)

        print(f"\n=== generation {gen} — active: {', '.join(active)}, "
              f"spend ${pool.total_usd():.2f} ===")
        jobs = planner.plan(pool.planner, active_targets, summary, lessons,
                            cfg["candidates_per_gen"], gen, cfg)
        print(f"[planner] {len(jobs)} job(s): "
              + "; ".join(f"{j['lineage']}: {j['strategy'][:60]}" for j in jobs))

        ctx = GenContext(cfg, runner, pool, active_targets, lessons, candidates_dir,
                         archive, incumbents, gen_start + cfg["llm_budget_s"],
                         run_deadline)
        results = []
        for job in jobs:  # resolve archive-backed context on THIS thread only
            job["_parent_source"], job["_parent_latency"] = ctx.parent_info(job)
        if jobs:
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
                futs = [ex.submit(subagent.run_candidate, job, i, gen, ctx)
                        for i, job in enumerate(jobs)]
                for job, fut in zip(jobs, futs):
                    try:
                        results.append(fut.result())
                    except Exception as e:  # noqa: BLE001 — archive, don't kill the run
                        results.append(dict(
                            lineage=job["lineage"], strategy=job["strategy"],
                            parent=job.get("parent"), parents=job.get("parents"),
                            model_name=None, source_kind="mutation", gate_reached=0,
                            compile_ok=False, correct_ok=False, repairs_used=0,
                            code_path=None, source_hash=None, flags=None,
                            weave_trace_url=None,
                            failure_note=f"[infra] subagent crashed: "
                                         f"{type(e).__name__}: {e}"))

        n_accepted = 0
        accepted_ops = set()
        # evaluation phase gets its OWN budget, starting now — authoring can
        # never starve the benchmarks
        eval_deadline = time.time() + cfg["eval_budget_s"]
        for res in results:
            op = res["lineage"]
            if res["gate_reached"] >= 2:
                remaining = eval_deadline - time.time()
                if remaining < MIN_VERIFY_WINDOW_S:
                    res["failure_note"] = ("[infra] eval budget exhausted before "
                                           "gates 3-4")
                else:
                    v = runner.verify(res["code_path"], op, upto=4,
                                      incumbents=ctx.incumbents_snapshot(),
                                      timeout=int(min(cfg["verify_timeout_s"], remaining)))
                    res.update({k: v[k] for k in
                                ("gate_reached", "latency_us", "incumbent_latency_us",
                                 "step_time_ms", "incumbent_step_time_ms",
                                 "samples_per_s", "mfu", "accepted", "failure_note")
                                if k in v})
            cid = archive.add_candidate(
                lineage_id=lineage_ids[op], generation=gen,
                parent_id=res.get("parent") if isinstance(res.get("parent"), int) else None,
                parents_json=json.dumps(res.get("parents")) if res.get("parents") else None,
                strategy=res["strategy"], source_kind=res["source_kind"],
                code_path=res.get("code_path"), source_hash=res.get("source_hash"),
                model_name=res.get("model_name"), weave_trace_url=res.get("weave_trace_url"),
                gate_reached=res["gate_reached"], compile_ok=int(res["compile_ok"]),
                correct_ok=int(res["correct_ok"]), repairs_used=res["repairs_used"],
                latency_us=res.get("latency_us"),
                incumbent_latency_us=res.get("incumbent_latency_us"),
                step_time_ms=res.get("step_time_ms"),
                incumbent_step_time_ms=res.get("incumbent_step_time_ms"),
                samples_per_s=res.get("samples_per_s"), mfu=res.get("mfu"),
                accepted=int(bool(res.get("accepted"))),
                failure_note=res.get("failure_note"), flags=res.get("flags"))
            mirror.log_candidate(op, res)
            if res.get("accepted"):
                n_accepted += 1
                accepted_ops.add(op)
                incumbents[op] = {"kind": "candidate", "path": res["code_path"]}
                archive.set_incumbent(lineage_ids[op], cid)
                mirror.log_kernel_artifact(op, res["code_path"], gen)
                print(f"[gate4] ACCEPTED {op}: step {res['step_time_ms']:.2f}ms "
                      f"(was {res['incumbent_step_time_ms']:.2f}ms)"
                      + (f", MFU {res['mfu']:.3f}" if res.get("mfu") else ""))
            else:
                print(f"[gates] {op}: gate_reached={res['gate_reached']} "
                      f"{(res.get('failure_note') or '')[:100]}")

        # §6.2 — barren lineages and retirement
        for op in active:
            if op not in accepted_ops:
                if archive.bump_barren(lineage_ids[op], cfg["retire_after"]):
                    print(f"[loop] lineage {op} retired after "
                          f"{cfg['retire_after']} barren generations")

        # §6.2 — systemic failure: nothing passed gate 1
        if not any(r["gate_reached"] >= 1 for r in results):
            systemic_fail_gens += 1
        else:
            systemic_fail_gens = 0

        # curator (skipped if the spend cap is already blown — no new LLM calls)
        if pool.total_usd() < cfg["spend_cap_usd"]:
            digest = dict(
                generation=gen,
                n_failed=sum(1 for r in results if not r.get("accepted")),
                candidates=[dict(lineage=r["lineage"], strategy=r["strategy"],
                                 gate_reached=r["gate_reached"],
                                 accepted=bool(r.get("accepted")),
                                 repairs_used=r["repairs_used"],
                                 latency_us=r.get("latency_us"),
                                 incumbent_latency_us=r.get("incumbent_latency_us"),
                                 failure_note=(r.get("failure_note") or "")[:300])
                            for r in results])
            for line in curator.curate(pool.curator, digest):
                archive.add_lesson(gen_id, model_id, line)
            mirror.log_lessons(archive.lessons_tail(model_id, cfg["lessons_tail"]))

        archive.finish_generation(gen_id, len(results), n_accepted, pool.total_usd())
        best = {}
        for lin in archive.lineages(model_id):
            inc = archive.candidate(lin["incumbent_id"]) if lin["incumbent_id"] else None
            best[lin["op_name"]] = inc.get("step_time_ms") if inc else None
        mirror.log_generation(gen, dict(n_candidates=len(results), n_accepted=n_accepted,
                                        llm_usd=round(pool.total_usd(), 2)), best)

        if systemic_fail_gens >= cfg["systemic_halt_after"]:
            stop_reason = "systemic_halt"
            print("[loop] SYSTEMIC HALT: no candidate passed gate 1 for "
                  f"{systemic_fail_gens} consecutive generations — check the "
                  "environment (Triton version, reference snippet), not the search")
            break

    stop_reason = stop_reason or "max_generations"
    if gen_id is not None:
        archive.set_stop_reason(gen_id, stop_reason)
    mirror.finish(stop_reason)
    print(f"\n[loop] stopped: {stop_reason}; total LLM spend ${pool.total_usd():.2f}; "
          f"archive at {os.path.join(out_dir, 'archive.sqlite')}")
    return stop_reason
