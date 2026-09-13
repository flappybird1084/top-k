"""Install platform helpers on the existing authenticated GPU runtime; no jobs launched."""
import base64
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
files={name:base64.b64encode((root/name).read_bytes()).decode() for name in ('worker.py','control.py','launch.py')}
files['archive_bridge.py']=base64.b64encode((root.parent/'serve_run.py').read_bytes()).decode()
code="""import base64,json
from pathlib import Path
root=Path('/marimo/top-k-platform');root.mkdir(exist_ok=True)
for name,content in FILES.items():
    path=root/name
    path.write_bytes(base64.b64decode(content))
print('Platform worker installed. No runs launched.')
""".replace('FILES',repr(files))
subprocess.run([sys.executable,str((root.parent.parent if (root.parent.parent/'search.py').exists() else root.parent.parent/'top-k')/'scripts/molab.py')],input=code,text=True,check=True)
