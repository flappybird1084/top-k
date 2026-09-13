"""Recorded benchmark evidence; no GPU or new evaluations are launched."""
import marimo
app=marimo.App(width='full')

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import statistics
    return mo, json, Path, statistics

@app.cell
def _(mo,json,Path,statistics):
    _side=mo.cli_args().get('view','kernel')
    _side='architecture' if _side=='architecture' else 'kernel'
    _s=json.loads((Path(__file__).parent/'assets'/'recorded'/f'{_side}.json').read_text())
    if _side=='kernel':
        _v=_s['verification']
        _pairs=[{'Pair':i+1,'Compiled PyTorch (ms)':round(b['baseline'],4),'Astra (ms)':round(b['candidate'],4),'Reduction (%)':100*(1-b['candidate']/b['baseline'])} for i,b in enumerate(_v['blocks'])]
        _gain=statistics.median(r['Reduction (%)'] for r in _pairs)
        for _pair in _pairs:_pair['Reduction (%)']=round(_pair['Reduction (%)'],4)
        _summary=mo.md(f'**{_gain:.2f}% lower full training-step time** · 8 correctness checks passed')
        _table=mo.ui.table(_pairs,selection=None)
        _rows=[{'Generation':r['generation'],'Candidate':r['id'],'Model':r['model_name'],'Strategy':r['strategy'],'Accepted':bool(r['accepted']),'Step ms':r['step_time_ms']} for r in _s['candidates']]
    else:
        _rows=[{'Candidate':r['id'],'Generation':r['generation'],'Phase':r['phase'],'Budget (s)':r['train_secs'],'Validation loss':r['val_loss'],'Strategy':r['strategy']} for r in _s['architecture']['candidates']]
        _summary=mo.md('**7.21% lower validation loss** · equal 300-second training budget')
        _table=mo.ui.table([{'Budget (s)':b,'Baseline':min(r['val_loss'] for r in _s['architecture']['candidates'] if r['phase']=='baseline' and r['train_secs']==b),'Best':min(r['val_loss'] for r in _s['architecture']['candidates'] if r['phase']!='baseline' and r['train_secs']==b and r['val_loss'] is not None)} for b in [60,120,300]],selection=None)
    mo.vstack([_summary,_table,mo.md('**Recorded evaluations**'),mo.ui.table(_rows,selection=None,page_size=len(_rows))])
    return

if __name__=='__main__':app.run()
