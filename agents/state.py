"""
The shared state object that flows through the LangGraph graph.

Every agent reads from and writes to this single TypedDict. This is what
makes the system a genuine *multi-agent* pipeline rather than five
independent scripts: the Researcher's output becomes the Writer's input, the
Critic's feedback becomes the Writer's input on the next pass, and so on.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class ResearchNote(TypedDict):
    """One piece of research gathered by the Researcher agent."""

    angle: str
    summary: str
    sources: List[str]


class CriticVerdict(TypedDict):
    """The Critic agent's judgment on a single draft."""

    approved: bool
    feedback: str


class ReportState(TypedDict, total=False):
    """
    The full pipeline state.

    total=False means individual keys can be absent until the agent
    responsible for them has run — LangGraph fills this in incrementally as
    the graph executes.
    """

    # Input
    topic: str
    max_revisions: int

    # Orchestrator output
    research_angles: List[str]

    # Researcher output
    research_notes: List[ResearchNote]

    # Writer output (overwritten on each revision pass)
    draft: str
    revision_count: int

    # Critic output (overwritten on each pass)
    critic_verdict: Optional[CriticVerdict]
    critic_history: List[CriticVerdict]

    # Editor output
    final_report: str