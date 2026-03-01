"""
Chat-first requirements discovery flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ...config import PlanningConfig


DISCOVERY_SYSTEM_PROMPT = """You help users discover planning requirements.

Process:
1) Ask 3-5 targeted questions.
2) Accept partial answers and ask concise follow-ups.
3) After 2-4 exchanges, return JSON marker:
   {"discovery":"complete","summary":"..."}
4) Keep interactions short and practical.
"""


@dataclass
class DiscoveryTurnResult:
    reply: str
    complete: bool
    summary: str | None = None


class DiscoveryManager:
    def __init__(self, config: PlanningConfig):
        self.config = config
        self.turn_count = 0

    def _llm_call(self, messages: list[dict[str, Any]]) -> str:
        import litellm

        response = litellm.completion(
            model=self.config.author_model,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
        )
        return str(response.choices[0].message.content or "").strip()

    def opening_prompt(self) -> str:
        return (
            "I will help define your plan scope. "
            "What problem are you solving, who are users, and what constraints matter most?"
        )

    def _try_extract_completion(self, text: str) -> DiscoveryTurnResult:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return DiscoveryTurnResult(reply=text, complete=False, summary=None)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return DiscoveryTurnResult(reply=text, complete=False, summary=None)

        if parsed.get("discovery") == "complete":
            summary = str(parsed.get("summary", "")).strip()
            prose = (text[: match.start()] + text[match.end() :]).strip()
            return DiscoveryTurnResult(
                reply=prose or "Discovery complete.",
                complete=True,
                summary=summary or None,
            )
        return DiscoveryTurnResult(reply=text, complete=False, summary=None)

    async def handle_turn(self, conversation: list[dict[str, Any]]) -> DiscoveryTurnResult:
        self.turn_count += 1

        if self.turn_count >= self.config.discovery_max_turns:
            summary = await self.force_summary(conversation)
            return DiscoveryTurnResult(
                reply=(
                    f"Discovery limit reached ({self.config.discovery_max_turns} turns). "
                    "Proceeding with draft."
                ),
                complete=True,
                summary=summary,
            )

        try:
            import litellm  # noqa: F401
        except ImportError:
            return DiscoveryTurnResult(
                reply="LLM unavailable; proceeding with provided requirements context.",
                complete=True,
                summary="\n".join(m.get("content", "") for m in conversation if m.get("role") == "user"),
            )

        messages = [{"role": "system", "content": DISCOVERY_SYSTEM_PROMPT}, *conversation]
        response = self._llm_call(messages)
        return self._try_extract_completion(response)

    async def force_summary(self, conversation: list[dict[str, Any]]) -> str:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return "\n".join(
                m.get("content", "")
                for m in conversation
                if m.get("role") == "user"
            )[:2000]

        messages = [
            {
                "role": "system",
                "content": "Summarize the discovered planning requirements in concise markdown.",
            },
            *conversation,
        ]
        return self._llm_call(messages)
