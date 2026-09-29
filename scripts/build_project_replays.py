"""Export only the already-public benchmark evidence used by the project planets."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.build_landing_benchmarks import ROOT, build

DESTINATION = ROOT / 'ui/assets/project-replays.json'
FIELDS = ('accepted', 'compile_ok', 'correct_ok', 'gate_reached', 'generation',
          'model_params', 'phase', 'train_secs', 'val_loss', 'step_time_ms',
          'incumbent_step_time_ms')
RESULT_FIELDS = ('measured', 'accepted', 'architecture_changed', 'baseline_val_loss',
                 'candidate_val_loss', 'final_budget_s', 'baseline_ms', 'candidate_ms',
                 'improvement_pct')


def payload():
    planets = json.loads(build().split('=', 1)[1].rstrip(';\n'))['results']
    projects = []
    for planet in planets:
        folder = planet['evidence_url'].rsplit('/', 1)[-1]
        source = ROOT / 'benchmarks/results' / folder
        if not (source / 'evidence.json').exists():
            continue
        evidence = json.loads((source / 'evidence.json').read_text(encoding='utf-8'))
        summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
        record = next(r for r in evidence['repositories'] if r['repo'] == planet['name'])
        slug = hashlib.sha256(planet['name'].encode()).hexdigest()[:12]
        project = {'repo': planet['name'], 'name': planet['short'], 'slug': slug,
                   'evidence_url': planet['evidence_url'],
                   'wandb_url': summary.get('wandb_summary_url'),
                   'notebook': f'assets/benchmarks/{slug}.html',
                   'caveat': planet.get('architecture', {}).get('caveat'), 'modes': {}}
        for side in ('architecture', 'kernel'):
            state = record[side]
            candidates = [{k: c[k] for k in FIELDS if k in c}
                          for c in state.get('candidate_gates', [])]
            # Ordinals identify exported rows, not invented archive IDs or ancestry.
            for i, candidate in enumerate(candidates):
                candidate['ordinal'] = i + 1
            project['modes'][side] = {
                'status': state['status'], 'model': state.get('llm'),
                'commit': state.get('commit'), 'data_source': state.get('data_source'),
                'timing_scope': state.get('timing_scope'),
                'result': {k: v for k, v in state.get('result', {}).items() if k in RESULT_FIELDS},
                'candidates': candidates,
            }
        projects.append(project)
    return {'schema': 'top-k-project-replays-v1', 'projects': projects}


def write(notebooks=False):
    result = payload()
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if notebooks:
        for project in result['projects']:
            dest = ROOT / 'ui' / project['notebook']
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'marimo', 'export', 'html',
                            str(ROOT / 'ui/project_notebook.py'), '-o', str(dest), '--force',
                            '--no-include-code', '--', '--project', project['repo']], check=True)
            html = dest.read_text(encoding='utf-8')
            html = html.replace('"theme": "light"', '"theme": "dark"')
            dest.write_text(html, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', action='store_true')
    args = parser.parse_args()
    write(notebooks=args.notebooks)


if __name__ == '__main__':
    main()
