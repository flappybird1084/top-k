"""Research subagent (spec §4.5's search tool, agentified per user design):
the planner asks a question; this agent runs its own search loop against the
SearXNG instance and returns a distilled brief. The planner never touches the
search API itself.
"""

from __future__ import annotations

import json

from kernelevo import websearch
from kernelevo.obs import weave_op

MAX_SEARCHES = 4

_SYSTEM = (
    "You are the research subagent of a GPU-kernel evolution system. You are "
    "given one question from the planner. Use web search to find prior art, "
    "papers, repositories, or documentation that answer it. To search, respond "
    f"with ONLY {{\"search\": \"<query>\"}} (up to {MAX_SEARCHES} searches). "
    "When you have enough, respond with ONLY {\"brief\": \"<your findings>\"} — "
    "at most ~400 words, citing URLs, quoting exact code fragments or formulas "
    "you found verbatim where relevant. Report only what the sources say; do "
    "not add your own implementation advice."
)


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return {}


@weave_op
def research(llm, question: str) -> str:
    if not websearch.available():
        return "(research unavailable: SEARXNG_URL not configured)"
    msgs = [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question}]
    resp = None
    for round_ in range(MAX_SEARCHES + 1):
        resp = llm.complete(msgs, json_mode=True,
                            meta={"question": question, "round": round_})
        obj = _parse(resp.text)
        query = obj.get("search")
        if query and round_ < MAX_SEARCHES:
            try:
                results = websearch.search(str(query))
            except Exception as e:  # noqa: BLE001 — instance down ≠ crashed plan
                results = [{"error": f"search failed: {e}"}]
            print(f"[research] {str(query)[:80]} -> {len(results)} result(s)")
            msgs = msgs + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content":
                 "Results:\n" + json.dumps(results, indent=1) +
                 f'\n\nContinue: another {{"search": ...}} '
                 f'({MAX_SEARCHES - 1 - round_} left) or the final '
                 f'{{"brief": ...}}.'}]
            continue
        return str(obj.get("brief") or resp.text)
    return str(resp.text if resp else "(no research produced)")
