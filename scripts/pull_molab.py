"""Copy authoritative run state and candidate sources back to the local workspace."""
import base64
import json
import subprocess
import sys
import zlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
name=sys.argv[1] if len(sys.argv)>1 else 'sol-pilot'
if not name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run name')
code='''import base64,json,sqlite3,zlib
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
                'language-smoke.log','audit-mirror.log','fusion_compile.json','fusion_validation.json',
                'fixture_fused_ema.py','tests.log','validation.log',
                'diagnostic*.json','diagnose.py','compiled_region_*.py','compiled_steps/*.py',
                'input/adapter*.py','input/manifest.json','launch.json','result.json']:
    for p in root.glob(pattern):
        files[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
print('KE_TRANSFER_Z:'+base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())
'''.replace('NAME',repr(name))
proc=subprocess.run([sys.executable,str(root/'scripts/molab.py')],input=code,text=True,capture_output=True,check=True)
line=next(x for x in proc.stdout.splitlines() if x.startswith('KE_TRANSFER_Z:'))
files=json.loads(zlib.decompress(base64.b64decode(line.removeprefix('KE_TRANSFER_Z:'))))
dest=root/'runs'/name
for name,data in files.items():
    path=dest/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(data))
print('Copied',len(files),'run files to',dest)
