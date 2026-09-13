"""Adapter-writing agent (spec §11 extension, now in scope): given a git repo,
produce an adapter.py satisfying the §3.1 contract, verified by ingest() in a
subprocess with up to max_debug_turns repairs — same philosophy as the kernel
repair loop: external deterministic verifier, raw feedback, no diagnosis.

Once ingest passes ("data is flowing into the model"), the normal search loop
takes over. NOTE: this clones and executes arbitrary repo + LLM-written code;
run it only on a box you'd run the repo itself on.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

from kernelevo import prompts
from kernelevo.gates import REPO_ROOT
from kernelevo.llm import extract_code
from kernelevo.obs import weave_op

SURVEY_MAX_FILES = 25
SURVEY_MAX_FILE_CHARS = 6000
SURVEY_MAX_TOTAL_CHARS = 60000
INGEST_TIMEOUT_S = 900

_SCORE_WORDS = ("train", "model", "main", "data", "dataset", "loss", "config", "net")


def fetch_repo(repo: str, dest_dir: str, log=print) -> str:
    """Accepts a git URL or an existing local path."""
    if os.path.isdir(repo):
        return os.path.abspath(repo)
    target = os.path.join(dest_dir, "repo")
    if os.path.isdir(os.path.join(target, ".git")):
        return target
    log(f"[adapter] cloning {repo}")
    from urllib.parse import urlsplit, unquote
    u = urlsplit(repo)
    base, marker, ref = repo.partition('/tree/') if u.hostname == 'github.com' else (repo, '', '')
    branch_args = ['--branch', unquote(ref), '--single-branch'] if marker else []
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--depth", "1", *branch_args, '--', base, target],
                   check=True, capture_output=True, text=True, timeout=600)
    return target


def survey_repo(repo_dir: str) -> str:
    paths = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("node_modules", "__pycache__", "venv", ".venv")]
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, repo_dir)
            if f.lower().startswith("readme") or f.endswith((".py", ".yaml", ".yml", ".toml")):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                score = sum(w in rel.lower() for w in _SCORE_WORDS)
                if f.lower().startswith("readme"):
                    score += 10
                score -= rel.count(os.sep) * 0.5   # prefer shallow files
                if size > 200_000:
                    score -= 5
                paths.append((score, rel, p))
    paths.sort(key=lambda t: -t[0])

    tree = "\n".join(sorted(rel for _, rel, _ in paths))
    chunks = [f"## File tree (surveyed files)\n{tree[:4000]}\n"]
    total = len(chunks[0])
    for _, rel, p in paths[:SURVEY_MAX_FILES]:
        try:
            text = open(p, errors="replace").read()[:SURVEY_MAX_FILE_CHARS]
        except OSError:
            continue
        chunk = f"\n## {rel}\n```\n{text}\n```\n"
        if total + len(chunk) > SURVEY_MAX_TOTAL_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "".join(chunks)


def _run_ingest(adapter_path: str, device: str, seed: int) -> tuple[dict | None, str]:
    job_path = adapter_path + ".ingest.json"
    with open(job_path, "w") as f:
        json.dump(dict(adapter=adapter_path, device=device, seed=seed), f)
    try:
        proc = subprocess.run([sys.executable, "-m", "kernelevo.ingest_worker", job_path],
                              cwd=REPO_ROOT, timeout=INGEST_TIMEOUT_S,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, f"ingest exceeded {INGEST_TIMEOUT_S}s (hung dataloader or download?); killed"
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("KEVO_RESULT "):
            return json.loads(line[len("KEVO_RESULT "):]), ""
    return None, ((proc.stderr or proc.stdout or "no output").strip())[-4000:]


@weave_op
def prepare(repo: str, comments: str, max_debug_turns: int, out_dir: str,
            llm, device: str, seed: int = 1234, log=print) -> tuple[str, dict]:
    """Returns (adapter_path, ingest_info) or raises SystemExit with the last error."""
    os.makedirs(out_dir, exist_ok=True)
    repo_dir = fetch_repo(repo, out_dir, log)
    log(f"[adapter] surveying {repo_dir}")
    survey = survey_repo(repo_dir)
    adapter_path = os.path.join(out_dir, "adapter.py")
    verified_path = adapter_path + ".ok"  # last version that passed ingest
    for candidate in (verified_path, adapter_path):
        if os.path.exists(candidate):
            if candidate != adapter_path:
                shutil.copyfile(candidate, adapter_path)
            info, _ = _run_ingest(adapter_path, device, seed)
            if info is not None:
                log(f"[adapter] reusing previously verified adapter "
                    f"({info['n_params']/1e6:.1f}M params) — skipping the writing agent")
                return adapter_path, info
    if os.path.exists(adapter_path):
        log("[adapter] existing adapter no longer passes ingest; rewriting")
    header = (f"import sys\nsys.path.insert(0, {repo_dir!r})\n"
              f"sys.path.insert(0, {os.path.join(repo_dir, 'src')!r})\n\n")
    messages = prompts.adapter_writer_prompt(survey, comments, device)
    last_err = "no attempts made"
    for attempt in range(max_debug_turns):
        log(f"[adapter] attempt {attempt + 1}/{max_debug_turns}")
        resp = llm.complete(messages, meta={"role": "adapter", "attempt": attempt,
                                            "repo_dir": repo_dir})
        src = extract_code(resp.text)
        src = re.sub(r"^import sys\nsys\.path.*\n", "", src)  # header owns sys.path
        # __future__ imports must stay on top — hoist them above the header
        futures = re.findall(r"^from __future__ import .*$", src, re.MULTILINE)
        src = re.sub(r"^from __future__ import .*$\n?", "", src, flags=re.MULTILINE)
        with open(adapter_path, "w") as f:
            f.write("".join(f + "\n" for f in futures) + header + src)
        info, err = _run_ingest(adapter_path, device, seed)
        if info is not None:
            log(f"[adapter] ingest OK: {info['n_params']/1e6:.1f}M params, "
                f"{info['samples_per_batch']} samples/batch, loss0={info['loss0']:.4f} "
                f"— data is flowing")
            shutil.copyfile(adapter_path, verified_path)
            return adapter_path, info
        last_err = err
        log(f"[adapter] ingest failed:\n{err[-600:]}")
        messages = messages + [
            {"role": "assistant", "content": resp.text},
            {"role": "user", "content":
             f"Running the adapter through the harness's ingest check failed:\n\n"
             f"{err}\n\nFix the adapter. Respond with exactly one Python code block "
             f"containing the full corrected file."}]
    raise SystemExit(f"[adapter] could not produce a working adapter for {repo} in "
                     f"{max_debug_turns} attempt(s). Last error:\n{last_err}")
