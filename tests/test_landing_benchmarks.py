"""Keep public landing claims aligned with the measured benchmark archive."""

import json
import copy

from scripts.build_landing_benchmarks import DESTINATION, ROOT, build, entries


def test_new_reports_add_projects_and_keep_protocols_separate(tmp_path, monkeypatch):
    from scripts import build_landing_benchmarks as catalog
    original = json.loads((ROOT / "benchmarks/results/top10-2026-09-25/summary.json").read_text())
    row = next(r for r in original["results"] if r["repo"] == "huggingface/pytorch-image-models")
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    paths = []
    for folder, rows in [("top10-2026-09-25", [row]), ("2026-09-28-new", [copy.deepcopy(row)])]:
        path = tmp_path / folder / "summary.json"
        path.parent.mkdir()
        if "new" in folder:
            rows[0]["kernel"]["status"] = "failed"
            added = copy.deepcopy(rows[0])
            added["repo"] = "example/new-project"
            rows.append(added)
        path.write_text(json.dumps({"schema": "top-k-dual-molab-benchmark-v1", "results": rows}))
        paths.append(path)
    payload = json.loads(catalog.build(paths).split("=", 1)[1].rstrip(";\n"))
    assert len(payload["results"]) == 2
    assert "kernel" not in payload["results"][0]  # no old kernel mixed into new protocol
    assert "2026-09-28-new" in payload["results"][0]["evidence_url"]
    assert payload["results"][1]["name"] == "example/new-project"


def test_nonfinite_or_nonimproving_metrics_never_become_planets():
    from scripts.build_landing_benchmarks import lower, accepted
    assert not lower(1, float("nan"))
    assert not lower(float("inf"), 1)
    assert not lower(1, 2)
    assert not lower(-1, -2)
    assert not accepted({"measured": "false", "accepted": True})
    assert accepted({"measured": True, "accepted": 3})
    assert not accepted({"measured": True, "accepted": "3"})


def test_single_mode_summary_does_not_break_catalog(tmp_path):
    report = tmp_path / "summary.json"
    report.write_text(json.dumps({"schema": "single-mode", "results": [{"repo": "x/y"}]}))
    assert entries(report) == []


def test_landing_benchmark_data_matches_published_summary():
    assert DESTINATION.read_text(encoding="utf-8") == build()
    payload = json.loads(build().split("=", 1)[1].rstrip(";\n"))
    rows = payload["results"]
    assert len({row["name"] for row in rows}) == len(rows)
    assert {row["name"] for row in rows} >= {
        "huggingface/pytorch-image-models",
        "karpathy/nanochat",
        "huggingface/transformers",
        "huggingface/diffusers",
        "Lightning-AI/litgpt",
    }
    original = entries(ROOT / "benchmarks/results/top10-2026-09-25/summary.json")
    assert original[0]["kernel"]["improvement_pct"] == 9.499
    assert original[0]["architecture"]["improvement_pct"] == 4.535
    assert original[-1]["architecture"]["caveat"] == "loss-floor"


def test_timeline_candidate_ids_and_gains_match_recorded_run():
    recorded = (ROOT / "ui/landing-data.js").read_text(encoding="utf-8")
    timeline = (ROOT / "ui/landing-timeline.js").read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    recorded_candidates = decoder.raw_decode(recorded.split("=", 1)[1])[0]["arch"]["candidates"]
    generations = decoder.raw_decode(timeline.split("=", 1)[1])[0]["gens"]
    by_id = {candidate["id"]: candidate for candidate in recorded_candidates}
    for generation in generations:
        ids = [agent["id"] for agent in generation["agents"]]
        assert len(ids) == len(set(ids))
        for agent in generation["agents"]:
            assert agent["gain"] == by_id[agent["id"]]["gain"]
