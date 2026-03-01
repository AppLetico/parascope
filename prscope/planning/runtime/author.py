"""
Author runtime: plan drafting/refinement with tool-use enforcement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...config import PlanningConfig
from .tools import CODEBASE_TOOLS, ToolExecutor, extract_file_references


@dataclass
class AuthorResult:
    plan: str
    unverified_references: set[str]
    accessed_paths: set[str]


AUTHOR_SYSTEM_PROMPT = """You are an expert software architect creating implementation plans.

Rules:
1. Verify assumptions against the repository using tools before finalizing.
2. Output markdown with sections: Goals, Non-Goals, Architecture, Implementation Steps, TODOs, Open Questions.
3. Reference concrete file paths in backticks where relevant.
4. Keep reasoning concise and implementation-focused.
"""


class AuthorAgent:
    def __init__(self, config: PlanningConfig, tool_executor: ToolExecutor):
        self.config = config
        self.tool_executor = tool_executor

    def _fallback_plan(self, user_context: str) -> str:
        return (
            "# Plan Draft\n\n"
            "## Goals\n- Define implementation steps aligned with requirements.\n\n"
            "## Non-Goals\n- No code changes in planning phase.\n\n"
            "## Architecture\n- Reuse existing project structure.\n\n"
            "## Implementation Steps\n- [ ] Inspect target modules and interfaces\n"
            "- [ ] Draft changes by file path\n"
            "- [ ] Define tests and rollout\n\n"
            "## TODOs\n- [ ] Replace fallback with model-generated plan once LLM is available\n\n"
            "## Open Questions\n- Clarify acceptance criteria.\n\n"
            f"Context summary:\n{user_context[:1000]}"
        )

    def _llm_call(self, messages: list[dict[str, Any]]):
        import litellm

        return litellm.completion(
            model=self.config.author_model,
            messages=messages,
            tools=CODEBASE_TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=2400,
        )

    async def author_loop(
        self,
        messages: list[dict[str, Any]],
        require_tool_calls: bool = True,
        max_attempts: int = 6,
    ) -> AuthorResult:
        self.tool_executor.accessed_paths.clear()
        conversation = [
            {"role": "system", "content": AUTHOR_SYSTEM_PROMPT},
            *messages,
        ]

        try:
            import litellm  # noqa: F401
        except ImportError:
            fallback = self._fallback_plan("\n".join(m.get("content", "") for m in messages))
            refs = extract_file_references(fallback)
            return AuthorResult(
                plan=fallback,
                unverified_references=refs - self.tool_executor.accessed_paths,
                accessed_paths=self.tool_executor.accessed_paths.copy(),
            )

        for attempt in range(max_attempts):
            response = self._llm_call(conversation)
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "")
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": getattr(tc, "id", None),
                                "type": "function",
                                "function": {
                                    "name": getattr(tc.function, "name", ""),
                                    "arguments": getattr(tc.function, "arguments", "{}"),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    result = self.tool_executor.execute(tc)
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "name": result["name"],
                            "content": json.dumps(result["result"]),
                        }
                    )
                continue

            if require_tool_calls and attempt == 0 and not self.tool_executor.accessed_paths:
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "You must verify your assumptions against real files. "
                            "Call search_codebase/read_file/list_dir before finalizing."
                        ),
                    }
                )
                continue

            plan_content = content.strip() or self._fallback_plan("")
            referenced = extract_file_references(plan_content)
            unverified = referenced - self.tool_executor.accessed_paths
            return AuthorResult(
                plan=plan_content,
                unverified_references=unverified,
                accessed_paths=self.tool_executor.accessed_paths.copy(),
            )

        fallback = self._fallback_plan("")
        refs = extract_file_references(fallback)
        return AuthorResult(
            plan=fallback,
            unverified_references=refs - self.tool_executor.accessed_paths,
            accessed_paths=self.tool_executor.accessed_paths.copy(),
        )
