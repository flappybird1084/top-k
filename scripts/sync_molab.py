"""Upload source files only; never notebook.py, credentials, data, or run state."""
import base64
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = {}
for folder in ['kernel_evolution', 'adapters', 'tests']:
    for p in (root/folder).rglob('*.py'):
        files[str(p.relative_to(root))] = base64.b64encode(p.read_bytes()).decode()
for name in ['config.py', 'search.py', 'pyproject.toml', 'README.md', 'IMPLEMENTATION_NOTES.md','PILOT_RESULTS.md','FUSION_RESULTS.md']:
    p = root/name
    if p.exists():
        files[name] = base64.b64encode(p.read_bytes()).decode()
code = '''import base64, json
from pathlib import Path
root = Path('/marimo/top-k')
for name, data in json.loads(PAYLOAD).items():
    p = root/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(data))
print('Uploaded source files:', COUNT)
'''.replace('PAYLOAD', repr(json.dumps(files))).replace('COUNT', str(len(files)))
subprocess.run([sys.executable, str(root/'scripts/molab.py')], input=code, text=True, check=True)
