import json
from scripts.build_project_replays import ROOT, DESTINATION, payload, FIELDS, RESULT_FIELDS


def test_replay_exports_match_published_measurements_without_private_fields():
    generated = payload()
    assert json.loads(DESTINATION.read_text(encoding='utf-8')) == generated
    source = json.loads((ROOT / 'benchmarks/results/top10-2026-09-25/evidence.json').read_text())
    by_repo = {r['repo']: r for r in source['repositories']}
    assert len(generated['projects']) == 5
    for project in generated['projects']:
        for side, mode in project['modes'].items():
            original = by_repo[project['repo']][side]
            assert mode['model'] == original['llm']
            assert len(mode['candidates']) == len(original.get('candidate_gates', []))
            for row, raw in zip(mode['candidates'], original.get('candidate_gates', [])):
                assert set(row) <= set(FIELDS) | {'ordinal'}
                assert {k: v for k, v in row.items() if k != 'ordinal'} == {k: v for k, v in raw.items() if k in FIELDS}
            assert mode['result'] == {k: v for k, v in original['result'].items() if k in RESULT_FIELDS}
        assert (ROOT / 'ui' / project['notebook']).is_file()
