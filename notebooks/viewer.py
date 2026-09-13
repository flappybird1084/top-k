# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "pandas", "matplotlib"]
# ///
"""Optional viewer (spec §10): three cells, reads archive.sqlite, no state of
its own. W&B panels cover the same ground; this is the local alternative.

  marimo run notebooks/viewer.py -- --archive runs/latest/archive.sqlite
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sqlite3
    import sys

    import marimo as mo
    import pandas as pd

    path = "runs/latest/archive.sqlite"
    if "--archive" in sys.argv:
        path = sys.argv[sys.argv.index("--archive") + 1]
    db = sqlite3.connect(path)
    cands = pd.read_sql("SELECT c.*, l.op_name FROM candidates c "
                        "JOIN lineages l ON c.lineage_id = l.id", db)
    lessons = pd.read_sql("SELECT * FROM lessons", db)
    gens = pd.read_sql("SELECT * FROM generations", db)
    mo.vstack([mo.md(f"# Kernel evolution — `{path}`"),
               mo.md(f"{len(cands)} candidates, {len(gens)} generations, "
                     f"stop: `{gens.stop_reason.dropna().iloc[-1] if gens.stop_reason.notna().any() else '(running)'}`")])
    return cands, gens, lessons, mo, pd


@app.cell
def _(cands, mo):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for op, grp in cands[cands.accepted == 1].groupby("op_name"):
        g = grp.sort_values("generation")
        best = g.step_time_ms.cummin()
        ax1.plot(g.generation, best, marker="o", label=op)
    ax1.set(title="best-so-far step time per lineage", xlabel="generation",
            ylabel="step_time_ms")
    ax1.legend(fontsize=8)
    funnel = cands.gate_reached.value_counts().sort_index()
    ax2.bar([f"gate {int(g)}" for g in funnel.index], funnel.values)
    ax2.set(title="gate funnel (highest gate passed)")
    fig.tight_layout()
    mo.mpl.interactive(fig) if hasattr(mo, "mpl") else fig
    return


@app.cell
def _(cands, lessons, mo):
    mo.vstack([
        mo.md("## Lessons"),
        mo.ui.table(lessons[["generation_id", "text"]], page_size=10),
        mo.md("## Candidates"),
        mo.ui.table(cands[["generation", "op_name", "strategy", "gate_reached",
                           "accepted", "repairs_used", "latency_us", "step_time_ms",
                           "mfu", "failure_note"]], page_size=15),
    ])
    return


if __name__ == "__main__":
    app.run()
