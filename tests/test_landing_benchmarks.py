"""Keep public landing claims aligned with the measured benchmark archive."""

import json

from scripts.build_landing_benchmarks import DESTINATION, ROOT, build


def test_landing_benchmark_data_matches_published_summary():
    assert DESTINATION.read_text(encoding="utf-8") == build()
    payload = json.loads(build().split("=", 1)[1].rstrip(";\n"))
    rows = payload["results"]
    assert len(rows) == 5
    assert len({row["name"] for row in rows}) == len(rows)
    assert {row["name"] for row in rows} == {
        "huggingface/pytorch-image-models",
        "karpathy/nanochat",
        "huggingface/transformers",
        "huggingface/diffusers",
        "Lightning-AI/litgpt",
    }
    assert rows[0]["kernel"]["improvement_pct"] == 9.499
    assert rows[0]["architecture"]["improvement_pct"] == 4.535
    assert rows[-1]["architecture"]["caveat"] == "loss-floor"


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
