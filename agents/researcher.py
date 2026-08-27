"""
The Researcher agent.

Given a list of research angles (from the Orchestrator), this agent runs a
web search for each angle and asks Claude to summarize the findings into a
concise, source-linked research note. These notes become the raw material
the Writer agent turns into a report.
"""

from __future__ import annotations

from typing import List

from agents.config import call_claude
from agents.state import ResearchNote
from tools.web_search import search

SYSTEM_PROMPT = """You are a meticulous research analyst. You are given a \
research angle and a set of raw web search results (titles, snippets, and \
URLs). Your job is to synthesize those results into a clear, factual \
summary of what is currently known about that angle.

Rules:
- Base your summary ONLY on the information in the provided search results.
- If the search results are sparse or contradictory, say so explicitly \
rather than filling gaps from general knowledge.
- Write 3-6 sentences: dense, factual, no filler.
- Do not include a sources list in your summary text — sources are tracked \
separately.
"""


def research_angle(angle: str) -> ResearchNote:
    """
    Research a single angle: search the web, then have Claude synthesize
    the raw results into a clean summary.
    """
    raw_results = search(angle, max_results=5)

    if not raw_results:
        return ResearchNote(
            angle=angle,
            summary=(
                "No web search results were available for this angle. "
                "This section could not be independently researched."
            ),
            sources=[],
        )

    results_block = "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}"
        for r in raw_results
    )
    user_message = (
        f"Research angle: {angle}\n\n"
        f"Search results:\n\n{results_block}\n\n"
        "Summarize what is known about this angle based on these results."
    )

    reply = call_claude(SYSTEM_PROMPT, user_message)

    return ResearchNote(
        angle=angle,
        summary=reply.text,
        sources=[r["url"] for r in raw_results if r["url"]],
    )


def run_researcher(research_angles: List[str]) -> List[ResearchNote]:
    """
    Research every angle produced by the Orchestrator and return the full
    list of research notes. This is the function the LangGraph node calls.
    """
    return [research_angle(angle) for angle in research_angles]