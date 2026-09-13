import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def run_worker(module,request,directory,timeout,*,cuda=True):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    tag=uuid.uuid4().hex
    inp,out,log=[directory/(tag+suffix) for suffix in ['.input.json','.output.json','.log']]
    inp.write_text(json.dumps(request,default=str))
    env=os.environ.copy()
    env['PYTHONUNBUFFERED']='1'
    if not cuda: env['CUDA_VISIBLE_DEVICES']=''
    with log.open('w') as f:
        proc=subprocess.Popen([sys.executable,'-m',module,str(inp),str(out)],env=env,
                              stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try: proc.wait(timeout=max(.01,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            return dict(status='timeout',failure_note='gen_timeout',log_path=str(log))
    if not out.exists():
        return dict(status='crash',failure_note=log.read_text()[-16000:],log_path=str(log))
    result=json.loads(out.read_text())
    result['log_path']=str(log)
    return result
