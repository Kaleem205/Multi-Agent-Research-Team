"""
Shared configuration for all agents: the Anthropic client, default model,
and a single helper every agent uses to talk to Claude.

Centralizing this here means each agent file stays focused on *what* it asks
Claude to do, not *how* the API call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 4096


def get_client() -> Anthropic:
    """
    Build an Anthropic client from the ANTHROPIC_API_KEY environment
    variable. Raises a clear error early if the key is missing, rather than
    letting every agent fail with a confusing 401 later.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
            "add your Anthropic API key before running the pipeline."
        )
    return Anthropic(api_key=api_key)


@dataclass
class AgentReply:
    """Normalized result of a single call to Claude for one agent turn."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


def call_claude(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.4,
) -> AgentReply:
    """
    Send one message to Claude with a given system prompt and return the
    plain-text reply along with basic token usage, so the CLI can report
    cost/usage if desired.

    Every agent in this project calls through this single function, which
    keeps retry/error-handling logic in one place.
    """
    client = get_client()
    resolved_model = model or DEFAULT_MODEL

    response = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    combined_text = "\n".join(text_parts).strip()

    return AgentReply(
        text=combined_text,
        model=resolved_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )