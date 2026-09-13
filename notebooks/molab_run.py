# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "torch", "triton", "numpy", "pandas", "python-dotenv",
#                 "wandb", "weave", "anthropic", "openai"]
# ///
"""Launcher notebook for molab (marimo cloud), where the GPU is (spec §2/§10).

Upload this notebook plus the repo to molab, set secrets (ANTHROPIC_API_KEY,
WANDB_API_KEY, ...) in the molab environment, and run. It shells out to
search.py so the loop runs exactly as it does headless; the notebook is just
where the GPU lives.

TODO(user): verify molab specifics — how the repo is mounted alongside the
notebook and how secrets are exposed — against current molab docs; the cells
below assume the repo dir is the working directory and secrets are env vars.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import sys

    import marimo as mo

    repo = os.getcwd()  # adjust if the repo is mounted elsewhere on molab
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "no GPU visible"
    mo.md(f"**Kernel evolution launcher** — repo `{repo}`, GPU: `{gpu}`")
    return mo, os, repo, subprocess, sys


@app.cell
def _(mo):
    adapter = mo.ui.dropdown(["adapters/jepa.py", "adapters/lm.py"],
                             value="adapters/jepa.py", label="adapter")
    profile = mo.ui.dropdown(["DEV", "RUN"], value="DEV", label="profile")
    llm = mo.ui.dropdown(["(from profile)", "stub", "anthropic", "openai", "wandb"],
                         value="(from profile)", label="llm override")
    go = mo.ui.run_button(label="launch search")
    mo.hstack([adapter, profile, llm, go])
    return adapter, go, llm, profile


@app.cell
def _(adapter, go, llm, mo, profile, repo, subprocess, sys):
    mo.stop(not go.value, mo.md("*press **launch search** to start*"))
    cmd = [sys.executable, "search.py", "--adapter", adapter.value,
           "--profile", profile.value]
    if llm.value != "(from profile)":
        cmd += ["--llm", llm.value]
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        print(line, end="")
    mo.md(f"```\n" + "\n".join(lines[-40:]) + "\n```")
    return


if __name__ == "__main__":
    app.run()
