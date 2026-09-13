"""Run a worker subprocess that reports its answer as a ``KEVO_RESULT {json}``
line, with two liveness guarantees ``subprocess.run(timeout=...)`` cannot give:

  * **Result-line reap.** The parent acts on the ``KEVO_RESULT`` line the
    instant it appears and kills the child — so a worker that has finished its
    work but wedges at *exit* (e.g. joining a ``datasets`` streaming pool that
    never returns) no longer blocks until the full timeout. This is the exact
    failure that froze an ingest for 25+ minutes after ~40s of real work.
  * **Idle timeout.** A worker that goes silent (no stdout/stderr) for longer
    than ``idle_timeout`` is killed — a genuine hang is caught in the idle
    window instead of the (necessarily generous, download-sized) total budget.
    Workers keep it alive by emitting ``KEVO_HEARTBEAT ...`` lines during long
    quiet stretches; any output resets the idle clock.

``subprocess.run`` only ever observes one event — process exit — which is why
neither guarantee is possible with it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time

RESULT_PREFIX = "KEVO_RESULT "
HEARTBEAT_PREFIX = "KEVO_HEARTBEAT"


class WorkerOutcome:
    """status in {result, crash, timeout, idle_timeout}. ``result`` is the
    parsed dict when a KEVO_RESULT line was seen (even if the child then
    crashed or was reaped)."""

    __slots__ = ("status", "result", "returncode", "stdout", "stderr")

    def __init__(self, status, result=None, returncode=None, stdout="", stderr=""):
        self.status = status
        self.result = result
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def timed_out(self) -> bool:
        return self.status in ("timeout", "idle_timeout")


def _kill_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 — already dead / no pgid on this platform
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def run_result_worker(cmd, *, cwd=None, env=None, total_timeout,
                      idle_timeout=None, on_line=None, poll=0.5):
    """Launch ``cmd``, stream its output, and return a WorkerOutcome.

    on_line(str) is called for every output line (result/heartbeat/other) as it
    arrives — use it to surface progress to a human log. The child is started
    in its own session so the whole group can be reaped even if it spawned
    helpers or GPU processes.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, start_new_session=True)

    out_lines, err_lines = [], []
    holder = {}                      # {"line": <KEVO_RESULT line>}
    last = [time.time()]
    lock = threading.Lock()

    def reader(stream, sink, is_stdout):
        try:
            for line in iter(stream.readline, ""):
                with lock:
                    last[0] = time.time()
                    sink.append(line)
                    if is_stdout and line.startswith(RESULT_PREFIX) and "line" not in holder:
                        holder["line"] = line
                if on_line is not None:
                    try:
                        on_line(line.rstrip("\n"))
                    except Exception:  # noqa: BLE001 — logging must not kill the read
                        pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    threads = [threading.Thread(target=reader, args=(proc.stdout, out_lines, True), daemon=True),
               threading.Thread(target=reader, args=(proc.stderr, err_lines, False), daemon=True)]
    for t in threads:
        t.start()

    start = time.time()
    status = None
    while True:
        with lock:
            have_result = "line" in holder
            idle = time.time() - last[0]
        if have_result:
            status = "result"
            break
        if proc.poll() is not None:
            for t in threads:                     # let readers drain the pipes
                t.join(timeout=2)
            with lock:
                status = "result" if "line" in holder else "crash"
            break
        if time.time() - start > total_timeout:
            status = "timeout"
            break
        if idle_timeout is not None and idle > idle_timeout:
            status = "idle_timeout"
            break
        time.sleep(poll)

    if proc.poll() is None:
        _kill_group(proc)

    with lock:
        stdout, stderr, line = "".join(out_lines), "".join(err_lines), holder.get("line")
    result = None
    if line:
        try:
            result = json.loads(line[len(RESULT_PREFIX):])
        except ValueError:
            if status == "result":
                status = "crash"
    return WorkerOutcome(status, result, proc.returncode, stdout, stderr)
