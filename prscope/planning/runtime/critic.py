"""
Critic runtime with strict JSON contract validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config import PlanningConfig, RepoProfile
from ...memory import ParsedConstraint

EXPECTED_CRITIC_FIELDS = {
    "major_issues_remaining": int,
    "minor_issues_remaining": int,
    "hard_constraint_violations": list,
    "critique_complete": bool,
}


class CriticParseError(RuntimeError):
    """Raised on malformed critic contract responses."""


@dataclass
class CriticResult:
    major_issues_remaining: int
    minor_issues_remaining: int
    hard_constraint_violations: list[str]
    critique_complete: bool
    prose: str
    parse_error: str | None = None


CRITIC_SYSTEM_PROMPT = """You are an adversarial senior reviewer for implementation plans.

Return a JSON block FIRST, then prose critique.
Required JSON fields:
- major_issues_remaining (int)
- minor_issues_remaining (int)
- hard_constraint_violations (list[str])
- critique_complete (bool)

Rules:
- major_issues_remaining must be > 0 if hard constraints are violated.
- critique_complete must be true only when major_issues_remaining is 0.
- Use only valid hard-constraint IDs.
"""


class CriticAgent:
    def __init__(self, config: PlanningConfig, repo: RepoProfile):
        self.config = config
        self.repo = repo

    def _llm_call(self, messages: list[dict[str, Any]], temperature: float) -> str:
        import litellm

        response = litellm.completion(
            model=self.config.critic_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1800,
        )
        return str(response.choices[0].message.content or "").strip()

    def _parse_critic_response(self, raw: str, valid_constraint_ids: set[str]) -> CriticResult:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise CriticParseError("No JSON block found in critic response")

        json_text = match.group(0)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise CriticParseError(f"Malformed critic JSON: {exc}") from exc

        for field, expected_type in EXPECTED_CRITIC_FIELDS.items():
            if field not in data:
                raise CriticParseError(f"Missing required field: {field}")
            if not isinstance(data[field], expected_type):
                raise CriticParseError(
                    f"Wrong type for {field}: expected {expected_type.__name__}"
                )

        violations = [str(v) for v in data["hard_constraint_violations"]]
        unknown = set(violations) - valid_constraint_ids
        if unknown:
            raise CriticParseError(f"Unknown constraint IDs in violations: {sorted(unknown)}")

        return CriticResult(
            major_issues_remaining=int(data["major_issues_remaining"]),
            minor_issues_remaining=int(data["minor_issues_remaining"]),
            hard_constraint_violations=violations,
            critique_complete=bool(data["critique_complete"]),
            prose=raw[match.end() :].strip(),
        )

    async def run_critic(
        self,
        *,
        requirements: str,
        plan_content: str,
        manifesto: str,
        architecture: str,
        constraints: list[ParsedConstraint],
        max_retries: int = 2,
        temperature: float | None = None,
    ) -> CriticResult:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return CriticResult(
                major_issues_remaining=0,
                minor_issues_remaining=1,
                hard_constraint_violations=[],
                critique_complete=True,
                prose="LiteLLM unavailable; critic fallback used.",
            )

        valid_ids = {
            c.id
            for c in constraints
            if c.severity == "hard" and not c.optional
        }
        constraints_blob = "\n".join(
            f"- {c.id}: {c.text} (severity={c.severity}, optional={c.optional})"
            for c in constraints
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Manifesto\n{manifesto}\n\n"
                    f"## Constraints\n{constraints_blob}\n\n"
                    f"## Architecture\n{architecture}\n\n"
                    f"## Requirements\n{requirements}\n\n"
                    f"## Current Plan\n{plan_content}"
                ),
            },
        ]

        temp = self.config.validate_temperature if temperature is None else temperature
        for attempt in range(max_retries):
            raw = self._llm_call(messages, temperature=temp)
            try:
                return self._parse_critic_response(raw, valid_ids)
            except CriticParseError as exc:
                if attempt < max_retries - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Formatting error: {exc}. Respond again with JSON first, "
                                "then prose critique."
                            ),
                        }
                    )
                    continue
                return CriticResult(
                    major_issues_remaining=1,
                    minor_issues_remaining=0,
                    hard_constraint_violations=[],
                    critique_complete=False,
                    prose=raw,
                    parse_error=str(exc),
                )

        return CriticResult(
            major_issues_remaining=1,
            minor_issues_remaining=0,
            hard_constraint_violations=[],
            critique_complete=False,
            prose="Critic failed to return a valid contract.",
            parse_error="unreachable retry guard",
        )

    async def validate_headless(
        self,
        *,
        session_id: str,
        round_number: int,
        plan_sha: str,
        requirements: str,
        plan_content: str,
        manifesto: str,
        architecture: str,
        constraints: list[ParsedConstraint],
    ) -> CriticResult:
        result = await self.run_critic(
            requirements=requirements,
            plan_content=plan_content,
            manifesto=manifesto,
            architecture=architecture,
            constraints=constraints,
            max_retries=2,
            temperature=self.config.validate_temperature,
        )

        if self.config.validate_audit_log:
            audit_dir = Path.home() / ".prscope" / "repos" / self.repo.name / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            audit_path = audit_dir / f"{session_id}-validate-{stamp}.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "round": round_number,
                        "plan_sha": plan_sha,
                        "critic_result": asdict(result),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result
