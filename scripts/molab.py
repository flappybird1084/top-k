"""Execute scratchpad code via the installed marimo-pair skill. Secrets stay local."""
import json
import subprocess
import sys
from pathlib import Path

connection = json.loads((Path.home()/'.local/state/kernel-evolution/molab.json').read_text())
script = Path.home()/'.agents/skills/marimo-pair/scripts/execute-code.sh'
result = subprocess.run(['bash', str(script), '--url', connection['url'],
                         '--token', connection['token'], '-'],
                        input=sys.stdin.read(), text=True)
raise SystemExit(result.returncode)
