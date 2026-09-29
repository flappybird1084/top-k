"""A marimo view of public benchmark evidence. Exported for project replay."""
import marimo

app = marimo.App(width='full')


@app.cell
def _():
    import marimo as mo
    import json
    import html
    from pathlib import Path
    return mo, json, html, Path


@app.cell
def _(mo, json, html, Path):
    _projects = json.loads((Path(__file__).parent / 'assets/project-replays.json').read_text())['projects']
    _project = next(p for p in _projects if p['repo'] == mo.cli_args().get('project'))
    _panels = [mo.md('# ' + _project['repo'])]
    for _side, _mode in _project['modes'].items():
        _result = _mode['result']
        _baseline = _result.get('baseline_val_loss' if _side == 'architecture' else 'baseline_ms')
        _candidate = _result.get('candidate_val_loss' if _side == 'architecture' else 'candidate_ms')
        _panels.append(mo.md('## ' + _side.title()))
        if _baseline is not None and _candidate is not None:
            _unit = 'validation loss' if _side == 'architecture' else 'ms / training step'
            _budget = f" · {_result['final_budget_s']:g}s training" if _side == 'architecture' else ''
            _maximum = max(_baseline, _candidate, 1e-30)
            _svg = f'<svg viewBox="0 0 700 160" role="img" aria-label="{html.escape(_unit)}"><text x="0" y="20" fill="currentColor">Baseline · {_baseline:.8g}</text><rect y="32" width="{650*_baseline/_maximum}" height="30" rx="5" fill="#959aa6"/><text x="0" y="95" fill="currentColor">Candidate · {_candidate:.8g}</text><rect y="107" width="{650*_candidate/_maximum}" height="30" rx="5" fill="#65ac91"/></svg>'
            _panels.extend([mo.md(f'**{_unit}{_budget}**'), mo.Html(_svg)])
        _rows = [{k: c.get(k) for k in ('ordinal', 'generation', 'phase', 'train_secs', 'val_loss', 'step_time_ms', 'gate_reached', 'accepted')} for c in _mode['candidates']]
        if _rows:
            _panels.append(mo.ui.table(_rows, selection=None, page_size=len(_rows)))
        else:
            _panels.append(mo.md('No candidate measurements recorded.'))
        _panels.append(mo.md(f"**Inference model:** `{_mode['model']}`\n\n**Data:** {_mode['data_source']}\n\n**Source:** `{_mode['commit']}`"))
    mo.vstack(_panels)
    return


if __name__ == '__main__':
    app.run()
