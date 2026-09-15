"""Local loop state (spec §9). The loop reads ONLY from here; W&B/Weave is a
one-way mirror. Single writer: the orchestrator thread."""

from __future__ import annotations

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY, name TEXT, adapter_path TEXT, n_params INTEGER,
  flops_per_sample REAL, gpu_name TEXT, peak_flops REAL, config_json TEXT,
  created_at REAL);
CREATE TABLE IF NOT EXISTS calibration (
  model_id INTEGER, noise_spread REAL, gate3_margin REAL, gate4_margin REAL,
  tol_json TEXT, cheats_rejected INTEGER, detail_json TEXT);
CREATE TABLE IF NOT EXISTS lineages (
  id INTEGER PRIMARY KEY, model_id INTEGER, op_name TEXT, shapes_json TEXT,
  pct_step_time REAL, incumbent_id INTEGER, barren_generations INTEGER DEFAULT 0,
  retired INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY, lineage_id INTEGER, generation INTEGER,
  parent_id INTEGER, parents_json TEXT, strategy TEXT,
  source_kind TEXT,   -- inductor | retrieved:<url> | mutation | fusion | planted_cheat
  code_path TEXT, source_hash TEXT, model_name TEXT, weave_trace_url TEXT,
  gate_reached INTEGER,  -- highest gate PASSED (0 = failed compile)
  compile_ok INTEGER, correct_ok INTEGER, repairs_used INTEGER,
  latency_us REAL, incumbent_latency_us REAL,
  step_time_ms REAL, incumbent_step_time_ms REAL, samples_per_s REAL, mfu REAL,
  accepted INTEGER, failure_note TEXT, flags TEXT, created_at REAL,
  -- recipe-golf columns (null for kernel candidates)
  val_loss REAL, phase TEXT, train_secs REAL, model_params INTEGER,
  arch_fp TEXT,
  -- authoring LLM usage for this candidate (all attempts summed)
  tokens_in INTEGER, tokens_out INTEGER);
CREATE TABLE IF NOT EXISTS generations (
  id INTEGER PRIMARY KEY, model_id INTEGER, started_at REAL, finished_at REAL,
  n_candidates INTEGER, n_accepted INTEGER, llm_usd REAL, stop_reason TEXT,
  tokens_in INTEGER, tokens_out INTEGER);
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY, generation_id INTEGER, model_id INTEGER, text TEXT);
"""


class Archive:
    def __init__(self, path: str):
        # Orchestrator-thread only. Worker threads must never touch this
        # connection (python sqlite3 does not serialize concurrent use of one
        # connection); the loop pre-resolves anything subagents need before
        # fanning out, and the thread check makes violations fail loudly.
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        # archives persist across relaunches of a job; add columns introduced
        # after the archive was created (no-op error when they already exist)
        for table in ("generations", "candidates"):
            for col in ("tokens_in INTEGER", "tokens_out INTEGER"):
                try:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
        self.db.commit()

    def _insert(self, table: str, row: dict) -> int:
        keys = list(row)
        cur = self.db.execute(
            f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
            [row[k] for k in keys])
        self.db.commit()
        return cur.lastrowid

    # -- models / calibration
    def add_model(self, **kw) -> int:
        kw.setdefault("created_at", time.time())
        return self._insert("models", kw)

    def add_calibration(self, **kw) -> int:
        return self._insert("calibration", kw)

    # -- lineages
    def add_lineage(self, model_id, op_name, shapes, pct) -> int:
        return self._insert("lineages", dict(
            model_id=model_id, op_name=op_name,
            shapes_json=json.dumps(shapes), pct_step_time=pct))

    def lineages(self, model_id, active_only=False):
        q = "SELECT * FROM lineages WHERE model_id=?"
        if active_only:
            q += " AND retired=0"
        return [dict(r) for r in self.db.execute(q, (model_id,))]

    def set_incumbent(self, lineage_id, cand_id):
        self.db.execute("UPDATE lineages SET incumbent_id=?, barren_generations=0 WHERE id=?",
                        (cand_id, lineage_id))
        self.db.commit()

    def bump_barren(self, lineage_id, retire_after) -> bool:
        """Increment barren counter; retire and return True if threshold hit."""
        self.db.execute(
            "UPDATE lineages SET barren_generations=barren_generations+1 WHERE id=?",
            (lineage_id,))
        row = self.db.execute(
            "SELECT barren_generations FROM lineages WHERE id=?", (lineage_id,)).fetchone()
        retired = row["barren_generations"] >= retire_after
        if retired:
            self.db.execute("UPDATE lineages SET retired=1 WHERE id=?", (lineage_id,))
        self.db.commit()
        return retired

    # -- candidates
    def add_candidate(self, **kw) -> int:
        kw.setdefault("created_at", time.time())
        return self._insert("candidates", kw)

    def candidate(self, cand_id):
        r = self.db.execute("SELECT * FROM candidates WHERE id=?", (cand_id,)).fetchone()
        return dict(r) if r else None

    def parents_for(self, lineage_id, frac=0.5):
        """Selection (spec §9): top-frac accepted by step time; always the incumbent;
        plus one candidate per distinct strategy that passed gate 2 (diversity for fusion)."""
        accepted = [dict(r) for r in self.db.execute(
            "SELECT * FROM candidates WHERE lineage_id=? AND accepted=1 "
            "ORDER BY step_time_ms ASC", (lineage_id,))]
        keep = {c["id"]: c for c in accepted[: max(1, int(len(accepted) * frac))]}
        inc = self.db.execute(
            "SELECT incumbent_id FROM lineages WHERE id=?", (lineage_id,)).fetchone()
        if inc and inc["incumbent_id"]:
            c = self.candidate(inc["incumbent_id"])
            if c:
                keep[c["id"]] = c
        seen_strategies = {c["strategy"] for c in keep.values()}
        for r in self.db.execute(
                "SELECT * FROM candidates WHERE lineage_id=? AND gate_reached>=2 "
                "ORDER BY created_at ASC", (lineage_id,)):
            c = dict(r)
            if c["strategy"] not in seen_strategies:
                keep[c["id"]] = c
                seen_strategies.add(c["strategy"])
        return list(keep.values())

    def recent_outcomes(self, lineage_id, n_generations=3):
        gens = [r["generation"] for r in self.db.execute(
            "SELECT DISTINCT generation FROM candidates WHERE lineage_id=? "
            "ORDER BY generation DESC LIMIT ?", (lineage_id, n_generations))]
        if not gens:
            return []
        rows = self.db.execute(
            f"SELECT generation, strategy, gate_reached, accepted, failure_note, "
            f"latency_us, incumbent_latency_us, step_time_ms FROM candidates "
            f"WHERE lineage_id=? AND generation IN ({','.join('?' * len(gens))})",
            [lineage_id, *gens])
        return [dict(r) for r in rows]

    # -- generations / lessons
    def start_generation(self, model_id) -> int:
        return self._insert("generations", dict(model_id=model_id, started_at=time.time()))

    def finish_generation(self, gen_id, n_candidates, n_accepted, llm_usd,
                          stop_reason=None, tokens_in=None, tokens_out=None):
        self.db.execute(
            "UPDATE generations SET finished_at=?, n_candidates=?, n_accepted=?, "
            "llm_usd=?, stop_reason=?, tokens_in=?, tokens_out=? WHERE id=?",
            (time.time(), n_candidates, n_accepted, llm_usd, stop_reason,
             tokens_in, tokens_out, gen_id))
        self.db.commit()

    def set_stop_reason(self, gen_id, reason):
        self.db.execute("UPDATE generations SET stop_reason=? WHERE id=?", (reason, gen_id))
        self.db.commit()

    def add_lesson(self, generation_id, model_id, text):
        return self._insert("lessons", dict(
            generation_id=generation_id, model_id=model_id, text=text))

    def lessons_tail(self, model_id, n=20):
        # Scoped to one model/run: a persisted archive may hold lessons from
        # earlier runs (possibly about since-fixed harness bugs) — never inject
        # those into a new run's prompts.
        rows = self.db.execute(
            "SELECT text FROM lessons WHERE model_id=? ORDER BY id DESC LIMIT ?",
            (model_id, n)).fetchall()
        return [r["text"] for r in reversed(rows)]
