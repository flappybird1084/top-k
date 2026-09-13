"""Provider-independent web search via a SearXNG instance (SEARXNG_URL env).

The planner uses this through a prompt-level protocol (it replies
{"search": "..."} and gets results back), so it works with any chat model —
no provider tool-calling support required. Results are cached by query for the
process lifetime (spec §4.5). The instance must allow the JSON API
(formats: [html, json] in settings.yml).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

_cache: dict[str, list] = {}


def available() -> bool:
    return bool(os.environ.get("SEARXNG_URL"))


def search(query: str, n: int = 5, timeout: int = 20) -> list[dict]:
    if query in _cache:
        return _cache[query]
    base = os.environ["SEARXNG_URL"].rstrip("/")
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "kernelevo/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:500]}
               for r in data.get("results", [])[:n]]
    _cache[query] = results
    return results
