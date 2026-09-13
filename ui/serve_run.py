"""Local read-only archive bridge. Production must serve this API behind run authorization."""
import argparse
import json
import sqlite3
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def profile_summary(archive):
    path=archive.parent/'profile.json'
    try:
        profile=json.loads(path.read_text())
    except (OSError,ValueError):return None
    import math
    def positive(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0
    if not positive(profile.get('step_time_ms')):return None
    targets=profile.get('targets',[])
    result={'compiled_step_ms':profile['step_time_ms'], 'operations_inspected':len(targets),
            'eligible_operations':sum(bool(t.get('eligible')) for t in targets)}
    if positive(profile.get('eager_step_time_ms')):result['eager_step_ms']=profile['eager_step_time_ms']
    return result


def snapshot(archive, repo=None, data=None, report=None, weave_embed=None, marimo=None):
    result = dict(repo=repo, data=data, status='waiting', candidates=[], traces=[], integrations=({'marimo_url': marimo} if marimo else {}))
    if not archive or not archive.is_file():
        return result
    try:
        comparison=json.loads((archive.parent/'result.json').read_text()).get('bpd_comparison',{})
        import math
        result['bpd_comparison']={k:v for k,v in comparison.items() if k in ('baseline','kernel') and isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0}
    except (OSError,ValueError,AttributeError):pass
    profile=profile_summary(archive)
    if profile:result['profiling']=profile
    with sqlite3.connect(archive.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def rows(sql, table):
            return [dict(row) for row in db.execute(sql)] if table in tables else []
        result['candidates'] = rows('SELECT id,lineage_id,generation,source_kind,strategy,gate_reached,accepted,step_time_ms,incumbent_step_time_ms,wandb_run_url,weave_trace_url,created_at FROM candidates ORDER BY created_at,id', 'candidates')
        baselines=[r['step_time_ms'] for r in result['candidates'] if r.get('generation')==0 and isinstance(r.get('step_time_ms'),(int,float)) and r['step_time_ms']>0]
        if baselines: result['baseline_ms']=baselines[0]
        result['traces'] = rows('SELECT id,parent_id,name,started_at,ended_at,error,url FROM trace_outbox ORDER BY started_at DESC LIMIT 100', 'trace_outbox')[::-1]
        result['active_evaluations']=[]
        for span in rows("SELECT id,inputs_json,started_at FROM trace_outbox WHERE name='candidate_lifecycle' AND ended_at IS NULL ORDER BY started_at",'trace_outbox'):
            try:job=json.loads(span['inputs_json']).get('job',{})
            except (ValueError,TypeError):job={}
            result['active_evaluations'].append({'id':span['id'],'kernel':job.get('lineage','Kernel'),'strategy':job.get('strategy',''),'started_at':span['started_at'],'stage':'Generating / evaluating'})
        generations = rows('SELECT id,finished_at,stop_reason FROM generations ORDER BY id DESC LIMIT 1', 'generations')
        events = rows("SELECT kind,payload_json FROM events WHERE kind IN ('run_finished','wandb_connected','weave_ready','no_targets','prepare_failed') ORDER BY id", 'events')
        if generations:
            result['generation'] = generations[0]['id']
        result['status'] = 'running' if result['candidates'] or result['traces'] or generations else 'waiting'
        for event in events:
            try:
                payload = json.loads(event['payload_json'])
            except (ValueError, TypeError):
                continue
            if event['kind'] == 'run_finished':
                result['status'] = 'complete' if payload.get('stop_reason') not in ('interrupted','systemic_halt') else 'failed'
            elif event['kind'] == 'no_targets':
                result['status'] = 'complete'
                result['stop_reason'] = 'no_targets'
            elif event['kind'] == 'prepare_failed':
                result['status'] = 'failed'
                result['stop_reason'] = 'prepare_failed'
            elif event['kind'] == 'wandb_connected':
                result['integrations']['wandb_url'] = payload.get('url')
            elif event['kind'] == 'weave_ready':
                result['integrations']['weave_url'] = payload.get('url')
        for row in result['candidates']:
            for field, key in [('wandb_run_url','wandb_url'),('weave_trace_url','weave_url')]:
                if row.get(field):
                    result['integrations'][key] = row[field]
        if report:
            result['integrations']['wandb_embed_url'] = report
            result['integrations'].setdefault('wandb_url', report)
        if weave_embed:
            result['integrations']['weave_embed_url'] = weave_embed
        # No fabricated baseline: the integration can supply a calibrated baseline later.
        return result


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith('/api/'):
            if path != '/api/runs/current':
                self.send_error(404)
                return
            try:
                state = snapshot(self.server.archive, self.server.repo, self.server.data,
                                 self.server.report, self.server.weave_embed, self.server.marimo)
                body = json.dumps(state, allow_nan=False).encode()
            except (sqlite3.Error, ValueError):
                self.send_error(503, 'Archive temporarily unavailable')
                return
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--repo')
    parser.add_argument('--data')
    parser.add_argument('--wandb-report-url')
    parser.add_argument('--weave-embed-url')
    parser.add_argument('--marimo-url')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(Path(__file__).parent)))
    server.archive, server.repo, server.data = args.archive, args.repo, args.data
    server.report, server.weave_embed = args.wandb_report_url, args.weave_embed_url
    server.marimo = args.marimo_url
    print(f'http://127.0.0.1:{args.port}/workspace.html?run=current', flush=True)
    server.serve_forever()
