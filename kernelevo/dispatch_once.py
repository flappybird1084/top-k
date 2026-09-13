"""Cross-process launch protection for immutable UI run IDs."""
import fcntl
import json
from pathlib import Path

def run_once(job_path, launch):
    root=Path(job_path).parent
    with (root/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=root/'dispatch-result.json'
        if result.exists():return int(json.loads(result.read_text())['exit_code'])
        started=root/'dispatch-started'
        if started.exists():
            raise RuntimeError('Previous dispatcher stopped without a result; inspect remote run before retrying.')
        started.touch()
        code=launch()
        tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps({'exit_code':int(code)}));tmp.replace(result)
        return code
