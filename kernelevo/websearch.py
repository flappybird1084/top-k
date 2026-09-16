"""Web search for the research subagent.

Two paths, tried in order:
1. Direct: query the SearXNG instance at SEARXNG_URL (works on machines that
   can reach it — e.g. the box running the web server, on the tailnet).
2. Relay: on machines that cannot reach it (molab notebooks), drop a request
   file into KEVO_RELAY_DIR and wait; the local molab dispatcher sees it on its
   next poll, runs the query against SearXNG locally, and writes the response
   file back. SearXNG never has to be publicly exposed.

Results are cached by query for the process lifetime. The SearXNG instance
must allow the JSON API (formats: [html, json] in settings.yml).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid

_cache: dict[str, list] = {}

RELAY_TIMEOUT_S = 90


def available() -> bool:
    return bool(os.environ.get("SEARXNG_URL") or os.environ.get("KEVO_RELAY_DIR"))


def direct_search(query: str, n: int = 5, timeout: int = 10) -> list[dict]:
    base = os.environ["SEARXNG_URL"].rstrip("/")
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "kernelevo/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:500]}
            for r in data.get("results", [])[:n]]


def _relay_search(query: str, n: int) -> list[dict]:
    relay = os.environ["KEVO_RELAY_DIR"]
    os.makedirs(relay, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    req_path = os.path.join(relay, f"{rid}.req.json")
    res_path = os.path.join(relay, f"{rid}.res.json")
    # The launch environment's relay secret is what makes a request
    # attributable to this run; the dispatcher's policy refuses any search
    # without it (mirrors kernelevo.codex_oauth.relay_complete).
    with open(req_path + ".tmp", "w") as f:
        json.dump({"query": query, "n": n,
                   "relay_token": os.environ.get("KEVO_RELAY_TOKEN", "")}, f)
    os.replace(req_path + ".tmp", req_path)
    deadline = time.time() + RELAY_TIMEOUT_S
    while time.time() < deadline:
        if os.path.exists(res_path):
            try:
                return json.load(open(res_path))
            except ValueError:
                time.sleep(0.5)  # torn read of a mid-write file
                continue
        time.sleep(1.0)
    raise TimeoutError(f"search relay: no response within {RELAY_TIMEOUT_S}s "
                       "(is the dispatcher still attached?)")


def search(query: str, n: int = 5) -> list[dict]:
    if query in _cache:
        return _cache[query]
    # a relay dir means we're on a box that can't reach SearXNG (molab) — go
    # straight to the relay rather than burning a doomed direct attempt
    if os.environ.get("KEVO_RELAY_DIR"):
        return _store(query, _relay_search(query, n))
    if os.environ.get("SEARXNG_URL"):
        return _store(query, direct_search(query, n))
    raise RuntimeError("no search path available (SEARXNG_URL unset)")


def _store(query: str, results: list) -> list:
    _cache[query] = results
    return results
