import marimo
app=marimo.App(width='full')

@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from ui_server import snapshot
    refresh=mo.ui.refresh(options=[5,10,30],default_interval=5)
    return mo,snapshot,refresh

@app.cell
def _(mo,snapshot,refresh):
    refresh
    _request=mo.app_meta().request
    _run=_request.query_params.get('run','') if _request else ''
    if isinstance(_run,(list,tuple)):_run=_run[0] if _run else ''
    try:_state=snapshot(_run)
    except (ValueError,FileNotFoundError):_state={}
    _rows=[{'kernel':r['lineage_id'],'candidate':r['id'],'generation':r['generation'],
            'status':'Baseline' if r['generation']==0 else 'Verified' if r['accepted'] else 'Rejected',
            'step_ms':r['step_time_ms']} for r in _state.get('candidates',[])]
    mo.vstack([refresh,mo.md('Status: **'+_state.get('status','waiting')+'**'),
               mo.ui.table(_rows,selection=None) if _rows else mo.md('Waiting for recorded kernel measurements.')])
    return

if __name__=='__main__':app.run()
