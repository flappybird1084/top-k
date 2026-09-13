"""Copy authoritative run state and candidate sources back to the local workspace."""
import base64
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
name=sys.argv[1] if len(sys.argv)>1 else 'sol-pilot'
if not name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run name')
code='''import base64,json,sqlite3
from pathlib import Path
root=Path('/marimo/top-k/runs')/NAME
files={}
if (root/'archive.sqlite').exists():
    source=sqlite3.connect('file:'+str(root/'archive.sqlite')+'?mode=ro',uri=True)
    dest=sqlite3.connect(':memory:')
    source.backup(dest)
    files['archive.sqlite']=base64.b64encode(dest.serialize()).decode()
for pattern in ['prepared.json','targets.json','profile.json','candidates/*.py','run.log','prepare.log',
                'baseline_audit.json','audit.log','audit_summary.json','postrun-tests.log',
                'language-smoke.log','audit-mirror.log']:
    for p in root.glob(pattern):
        files[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
print('KE_TRANSFER:'+json.dumps(files))
'''.replace('NAME',repr(name))
proc=subprocess.run([sys.executable,str(root/'scripts/molab.py')],input=code,text=True,capture_output=True,check=True)
line=next(x for x in proc.stdout.splitlines() if x.startswith('KE_TRANSFER:'))
files=json.loads(line.removeprefix('KE_TRANSFER:'))
dest=root/'runs'/name
for name,data in files.items():
    path=dest/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(data))
print('Copied',len(files),'run files to',dest)
