"""Web search and page fetch.

Search defaults to DISABLED. Published solutions to competition problems are on
the open web, so with search on a "proof" may be retrieval rather than reasoning,
and the transcript will not distinguish the two. For genuinely open problems
there is nothing to leak, so enabling it is free upside.
"""

import re

import requests

from . import config

_used = 0
_enabled = config.SEARCH_ENABLED


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def enabled() -> bool:
    return _enabled


def used() -> int:
    return _used


def web_search(query: str, num: int = 5) -> dict:
    global _used
    if not _enabled:
        return {"error": "web search disabled for this run (leak-free proof evaluation)"}
    if not config.SERPER_KEY:
        return {"error": "SERPER_API_KEY not set (put it in .env)"}
    if _used >= config.SEARCH_BUDGET:
        return {"error": f"search budget exhausted ({config.SEARCH_BUDGET})"}
    _used += 1
    try:
        r = requests.post(config.SERPER_URL,
                          headers={"X-API-KEY": config.SERPER_KEY,
                                   "Content-Type": "application/json"},
                          json={"q": query, "num": num}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return {"error": f"search failed: {exc}"}
    return {"results": [{"title": it.get("title"), "link": it.get("link"),
                         "snippet": it.get("snippet")}
                        for it in data.get("organic", [])[:num]],
            "budget_left": config.SEARCH_BUDGET - _used}


def fetch_url(url: str) -> dict:
    # Gated exactly like web_search. Without this a model that INVENTS a
    # fetch_url call -- observed on IMO 2026 P1, reaching for its own journal --
    # gets an arbitrary GET executed in a run labelled search=False. Withholding
    # the spec only controls what is offered, never what is asked for.
    if not _enabled:
        return {"error": "web fetch disabled for this run (leak-free proof evaluation)"}
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (autoresearch)"})
        r.raise_for_status()
    except Exception as exc:
        return {"error": f"fetch failed: {exc}"}

    text = r.text
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "text": text[:config.FETCH_MAX_CHARS],
            "truncated": len(text) > config.FETCH_MAX_CHARS}
