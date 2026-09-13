"""SQLite is authoritative. No network calls are made from this module."""
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS run_config(id INTEGER PRIMARY KEY, created_at REAL, config_json TEXT);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, name TEXT, adapter_path TEXT, n_params INTEGER,
 flops_per_sample REAL, gpu_name TEXT, peak_flops REAL);
CREATE TABLE IF NOT EXISTS calibration(model_id TEXT PRIMARY KEY, noise_spread REAL, gate3_margin REAL,
 gate4_margin REAL, tol_json TEXT, cheats_rejected INTEGER, details_json TEXT);
CREATE TABLE IF NOT EXISTS lineages(id TEXT PRIMARY KEY, model_id TEXT, op_name TEXT, shapes_json TEXT,
 pct_step_time REAL, incumbent_id TEXT, barren_generations INTEGER DEFAULT 0, retired INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, lineage_id TEXT, generation INTEGER,
 parent_id TEXT, parents_json TEXT, strategy TEXT, source_kind TEXT, code_path TEXT, source_hash TEXT,
 model_name TEXT, weave_trace_url TEXT, wandb_run_url TEXT, gate_reached INTEGER DEFAULT 0,
 compile_ok INTEGER DEFAULT 0, correct_ok INTEGER DEFAULT 0, repairs_used INTEGER DEFAULT 0,
 latency_us REAL, incumbent_latency_us REAL, step_time_ms REAL, incumbent_step_time_ms REAL,
 samples_per_s REAL, mfu REAL, accepted INTEGER DEFAULT 0, failure_note TEXT, created_at REAL,
 details_json TEXT);
CREATE TABLE IF NOT EXISTS generations(id INTEGER PRIMARY KEY, model_id TEXT, started_at REAL,
 finished_at REAL, n_candidates INTEGER, n_accepted INTEGER, llm_usd REAL, stop_reason TEXT);
CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, generation_id INTEGER, model_id TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS llm_calls(id TEXT PRIMARY KEY, role TEXT, model TEXT, generation INTEGER,
 input_tokens INTEGER, cached_input_tokens INTEGER, output_tokens INTEGER, usd REAL,
 reserved_usd REAL, status TEXT, created_at REAL, details_json TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created_at REAL, kind TEXT, payload_json TEXT);
CREATE TABLE IF NOT EXISTS search_cache(query TEXT PRIMARY KEY, results_json TEXT);
CREATE TABLE IF NOT EXISTS generation_attempts(id INTEGER PRIMARY KEY, generation INTEGER,
 created_at REAL, reason TEXT, row_json TEXT);
CREATE TABLE IF NOT EXISTS candidate_audits(id INTEGER PRIMARY KEY, candidate_id TEXT,
 created_at REAL, decision TEXT, reason TEXT, original_row_json TEXT, evidence_json TEXT);
'''


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript(SCHEMA)

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args)]

    def put(self, table, **values):
        with self.lock, self.db:
            columns = {r['name'] for r in self.db.execute(f'PRAGMA table_info({table})')}
            if not columns or not set(values) <= columns:
                raise ValueError(f'Unknown archive fields: {table}: {set(values) - columns}')
            names = ','.join(values)
            self.db.execute(f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({",".join("?" for _ in values)})',
                            list(values.values()))

    def execute(self, sql, args=()):
        with self.lock, self.db:
            return self.db.execute(sql, args)

    def event(self, kind, payload):
        self.put('events', created_at=time.time(), kind=kind, payload_json=json.dumps(payload, default=str))

    def invalidate_acceptance(self, candidate_id, reason, evidence):
        """Preserve original evidence when a later verifier audit invalidates a win."""
        with self.lock, self.db:
            row=self.db.execute('SELECT * FROM candidates WHERE id=?',(candidate_id,)).fetchone()
            if row is None:raise ValueError('Unknown candidate '+candidate_id)
            if not row['accepted']:return
            self.db.execute('INSERT INTO candidate_audits '
                '(candidate_id,created_at,decision,reason,original_row_json,evidence_json) VALUES (?,?,?,?,?,?)',
                (candidate_id,time.time(),'acceptance_invalidated',reason,json.dumps(dict(row)),json.dumps(evidence)))
            self.db.execute('UPDATE candidates SET accepted=0,failure_note=? WHERE id=?',
                            ('audit_unconfirmed: '+reason,candidate_id))
            self.db.execute('UPDATE generations SET n_accepted=(SELECT COUNT(*) FROM candidates '
                            'WHERE generation=generations.id AND accepted=1) WHERE id=?',(row['generation'],))
            # Revert only if this candidate remains the incumbent; never overwrite another lineage.
            fallback=self.db.execute('SELECT id FROM candidates WHERE lineage_id=? AND accepted=1 '
                                     'ORDER BY generation DESC LIMIT 1',(row['lineage_id'],)).fetchone()
            self.db.execute('UPDATE lineages SET incumbent_id=? WHERE incumbent_id=?',
                            (fallback['id'] if fallback else None,candidate_id))

    def summary(self, generation):
        return dict(lineages=self.rows('SELECT * FROM lineages'),
            recent=self.rows('SELECT id,lineage_id,generation,strategy,accepted,gate_reached,failure_note,step_time_ms '
                             'FROM candidates WHERE generation>=?', (max(0, generation-3),)),
            parents=self.parents())

    def parents(self):
        selected = {}
        for line in self.rows('SELECT * FROM lineages WHERE retired=0'):
            accepted = self.rows('SELECT * FROM candidates WHERE lineage_id=? AND accepted=1 '
                                 'ORDER BY step_time_ms IS NULL,step_time_ms', (line['id'],))
            top = accepted[:max(1, (len(accepted)+1)//2)]
            diverse = self.rows('SELECT * FROM candidates WHERE lineage_id=? AND correct_ok=1 '
                                'ORDER BY generation DESC', (line['id'],))
            seen = {x['strategy'] for x in top}
            for item in diverse:
                if item['strategy'] not in seen or item['id'] == line['incumbent_id']:
                    top.append(item)
                    seen.add(item['strategy'])
            selected[line['id']] = list({x['id']: x for x in top}.values())
        return selected

    def lessons(self, n=20):
        return [x['text'] for x in reversed(self.rows('SELECT text FROM lessons ORDER BY id DESC LIMIT ?', (n,)))]


def source_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()
