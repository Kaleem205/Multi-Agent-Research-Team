"""
A minimal web search tool for the Researcher agent, backed by DuckDuckGo's
free search endpoint (no API key required).

The Researcher only depends on the `search()` function's signature below —
if you later want to swap in a paid provider (Tavily, SerpAPI, Bing), just
implement a function with the same signature and import that instead.
"""



from __future__ import annotations

from typing import List, TypedDict

from duckduckgo_search import DDGS


class SearchResult(TypedDict):
    title: str
    snippet: str
    url: str


def search(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Run a web search and return a list of lightweight results.

    Any failure (network issue, rate limit, etc.) is caught and results in
    an empty list rather than crashing the pipeline — the Researcher agent
    is written to handle sparse or empty results gracefully.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    results: List[SearchResult] = []
    for item in raw_results:
        results.append(
            SearchResult(
                title=item.get("title", "").strip(),
                snippet=item.get("body", "").strip(),
                url=item.get("href", "").strip(),
            )
        )
    return results