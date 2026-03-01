"""
Planning runtime orchestration for start modes and adversarial rounds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...config import PlanningConfig, PrscopeConfig, RepoProfile
from ...memory import MemoryStore, ParsedConstraint
from ...profile import build_profile
from ...store import PlanVersion, PlanningSession, PullRequest, Store
from ..core import ApprovalBlockedError, ConvergenceResult, PlanningCore
from ..render import export_plan_documents
from .author import AuthorAgent, AuthorResult
from .critic import CriticAgent, CriticResult
from .discovery import DiscoveryManager, DiscoveryTurnResult
from .tools import ToolExecutor


class PlanningRuntime:
    def __init__(self, store: Store, config: PrscopeConfig, repo: RepoProfile):
        self.store = store
        self.config = config
        self.repo = repo
        self.planning_config: PlanningConfig = config.planning
        self.memory = MemoryStore(repo, self.planning_config)
        self.tools = ToolExecutor(repo.resolved_path)
        self.author = AuthorAgent(self.planning_config, self.tools)
        self.critic = CriticAgent(self.planning_config, repo)
        self.discovery = DiscoveryManager(self.planning_config)

    def _core(self, session_id: str) -> PlanningCore:
        return PlanningCore(self.store, session_id, self.planning_config)

    async def _prepare_memory(self, rebuild_memory: bool = False) -> dict[str, str]:
        profile = build_profile(self.repo.resolved_path)
        await self.memory.ensure_memory(profile, rebuild=rebuild_memory)
        blocks = self.memory.load_all_blocks()
        blocks["manifesto"] = self.memory.load_manifesto()
        return blocks

    def _constraints(self) -> list[ParsedConstraint]:
        return self.memory.load_constraints(self.repo.resolved_manifesto)

    async def start_from_requirements(
        self,
        requirements: str,
        title: str | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        await self._prepare_memory(rebuild_memory=rebuild_memory)
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title=title or (requirements.splitlines()[0][:80] if requirements.strip() else "New Plan"),
            requirements=requirements,
            seed_type="requirements",
            status="drafting",
        )
        core = self._core(session.id)
        author_result = await self._run_initial_draft(core, requirements=requirements)
        core.add_turn("author", author_result.plan, round_number=0)
        core.save_plan_version(author_result.plan, round_number=0)
        core.transition("refining")
        return self.store.get_planning_session(session.id) or session

    def _build_pr_seed_context(self, pr: PullRequest, evaluation: Any, files: list[Any]) -> str:
        sections: list[str] = []
        sections.append(f"## PR #{pr.number}: {pr.title}\n{pr.body or ''}")
        file_list = files[:30]
        file_lines = "\n".join(f"- {f.path}" for f in file_list)
        if len(files) > 30:
            file_lines += f"\n- ... and {len(files) - 30} more files (omitted)"
        sections.append(f"## Changed Files\n{file_lines}")

        if evaluation is not None:
            llm_summary = ""
            if evaluation.llm_json:
                try:
                    llm = json.loads(evaluation.llm_json)
                    llm_summary = json.dumps(llm, indent=2)[:2000]
                except json.JSONDecodeError:
                    llm_summary = str(evaluation.llm_json)[:2000]
            sections.append(
                f"## Prior Analysis\n"
                f"Decision: {evaluation.decision}\n"
                f"Rule score: {evaluation.rule_score}\n"
                f"Final score: {evaluation.final_score}\n"
                f"LLM summary:\n{llm_summary}"
            )

        combined = "\n\n".join(sections)
        cap = max(self.planning_config.seed_token_budget, 500) * 4
        if len(combined) > cap:
            combined = combined[:cap] + "\n\n[Seed context truncated to fit token budget]"
        return combined

    async def start_from_pr(
        self,
        upstream_repo: str,
        pr_number: int,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        await self._prepare_memory(rebuild_memory=rebuild_memory)
        upstream = self.store.get_upstream_repo(upstream_repo)
        if upstream is None:
            raise ValueError(f"Unknown upstream repo in store: {upstream_repo}")
        pr = self.store.get_pull_request(upstream.id, pr_number)
        if pr is None:
            raise ValueError(f"PR not found in store: {upstream_repo}#{pr_number}")
        evaluation = None
        if pr.head_sha:
            evaluations = self.store.list_evaluations(limit=200)
            for candidate in evaluations:
                if candidate.pr_id == pr.id:
                    evaluation = candidate
                    break
        files = self.store.get_pr_files(pr.id)
        requirements = self._build_pr_seed_context(pr, evaluation, files)
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title=f"PR #{pr_number}: {pr.title}",
            requirements=requirements,
            seed_type="upstream_pr",
            seed_ref=f"{upstream_repo}#{pr_number}",
            status="drafting",
        )
        core = self._core(session.id)
        author_result = await self._run_initial_draft(core, requirements=requirements)
        core.add_turn("author", author_result.plan, round_number=0)
        core.save_plan_version(author_result.plan, round_number=0)
        core.transition("refining")
        return self.store.get_planning_session(session.id) or session

    async def start_from_chat(self, rebuild_memory: bool = False) -> tuple[PlanningSession, str]:
        await self._prepare_memory(rebuild_memory=rebuild_memory)
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title="New Plan (discovery)",
            requirements="",
            seed_type="chat",
            status="discovery",
        )
        opening = self.discovery.opening_prompt()
        self._core(session.id).add_turn("author", opening, round_number=0)
        return session, opening

    async def handle_discovery_turn(self, session_id: str, user_message: str) -> DiscoveryTurnResult:
        core = self._core(session_id)
        session = core.get_session()
        if session.status != "discovery":
            raise ValueError("Session is not in discovery mode")

        current_round = session.current_round
        core.add_turn("user", user_message, round_number=current_round)
        conversation = [
            {"role": turn.role, "content": turn.content}
            for turn in core.get_conversation()
        ]
        result = await self.discovery.handle_turn(conversation)
        core.add_turn("author", result.reply, round_number=current_round)

        if result.complete:
            summary = result.summary or user_message
            self.store.update_planning_session(session_id, requirements=summary)
            core.transition("drafting")
            author_result = await self._run_initial_draft(core, requirements=summary)
            core.add_turn("author", author_result.plan, round_number=0)
            core.save_plan_version(author_result.plan, round_number=0)
            core.transition("refining")

        return result

    async def _run_initial_draft(self, core: PlanningCore, requirements: str) -> AuthorResult:
        blocks = self.memory.load_all_blocks()
        messages = [
            {
                "role": "user",
                "content": (
                    f"Manifesto:\n{self.memory.load_manifesto()}\n\n"
                    f"Architecture memory:\n{blocks.get('architecture', '')}\n\n"
                    f"Module memory:\n{blocks.get('modules', '')}\n\n"
                    f"Requirements:\n{requirements}"
                ),
            }
        ]
        result = await self.author.author_loop(messages, require_tool_calls=True)
        return result

    async def run_adversarial_round(
        self,
        session_id: str,
        user_input: str | None = None,
    ) -> tuple[CriticResult, AuthorResult, ConvergenceResult]:
        core = self._core(session_id)
        session = core.get_session()
        if session.status not in {"drafting", "refining", "approved"}:
            raise ValueError(f"Session is not in refining state: {session.status}")

        current = core.get_current_plan()
        if current is None:
            raise ValueError("Cannot run adversarial round without initial plan")

        round_number = core.advance_round()
        blocks = self.memory.load_all_blocks()
        manifesto = self.memory.load_manifesto()
        constraints = self._constraints()
        requirements = (session.requirements or "") + (f"\n\nUser input:\n{user_input}" if user_input else "")
        if user_input:
            core.add_turn("user", user_input, round_number=round_number)

        critic_result = await self.critic.run_critic(
            requirements=requirements,
            plan_content=current.plan_content,
            manifesto=manifesto,
            architecture=blocks.get("architecture", ""),
            constraints=constraints,
        )
        critic_content = (
            json.dumps(
                {
                    "major_issues_remaining": critic_result.major_issues_remaining,
                    "minor_issues_remaining": critic_result.minor_issues_remaining,
                    "hard_constraint_violations": critic_result.hard_constraint_violations,
                    "critique_complete": critic_result.critique_complete,
                },
                indent=2,
            )
            + "\n\n"
            + critic_result.prose
        )
        core.add_turn(
            "critic",
            critic_content,
            round_number=round_number,
            major_issues_remaining=critic_result.major_issues_remaining,
            minor_issues_remaining=critic_result.minor_issues_remaining,
            hard_constraint_violations=critic_result.hard_constraint_violations,
            parse_error=critic_result.parse_error,
        )

        author_messages = [
            {
                "role": "user",
                "content": (
                    f"Manifesto:\n{manifesto}\n\n"
                    f"Architecture:\n{blocks.get('architecture', '')}\n\n"
                    f"Modules:\n{blocks.get('modules', '')}\n\n"
                    f"Requirements:\n{requirements}\n\n"
                    f"Current plan:\n{current.plan_content}\n\n"
                    f"Critique:\n{critic_content}"
                ),
            }
        ]
        author_result = await self.author.author_loop(author_messages, require_tool_calls=False)
        core.add_turn("author", author_result.plan, round_number=round_number)
        core.save_plan_version(author_result.plan, round_number=round_number)
        convergence = core.check_convergence()
        return critic_result, author_result, convergence

    def approve(self, session_id: str, unverified_references: set[str] | None = None) -> None:
        core = self._core(session_id)
        core.approve(unverified_references=unverified_references)

    async def validate_session(self, session_id: str) -> CriticResult:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan version found for validation")
        constraints = self._constraints()
        blocks = self.memory.load_all_blocks()
        manifesto = self.memory.load_manifesto()

        return await self.critic.validate_headless(
            session_id=session.id,
            round_number=plan.round,
            plan_sha=plan.plan_sha,
            requirements=session.requirements,
            plan_content=plan.plan_content,
            manifesto=manifesto,
            architecture=blocks.get("architecture", ""),
            constraints=constraints,
        )

    def export(self, session_id: str, output_dir: Path | None = None) -> dict[str, Path]:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan to export")
        paths = export_plan_documents(
            repo=self.repo,
            session=session,
            plan=plan,
            output_dir=output_dir,
        )
        core.mark_exported()
        return paths

    def status(self, session_id: str, merged_pr_files: set[str]) -> dict[str, Any]:
        core = self._core(session_id)
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan to compare")
        planned_files = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan.plan_content))
        implemented = planned_files & merged_pr_files
        missing = planned_files - merged_pr_files
        unplanned = merged_pr_files - planned_files
        return {
            "planned_total": len(planned_files),
            "implemented_count": len(implemented),
            "missing_count": len(missing),
            "unplanned_count": len(unplanned),
            "implemented": sorted(implemented),
            "missing": sorted(missing),
            "unplanned": sorted(unplanned),
        }

    def plan_diff(self, session_id: str, round_number: int | None = None) -> str:
        import difflib

        if round_number is not None:
            current = self.store.get_plan_version(session_id, round_number)
            previous = self.store.get_plan_version(session_id, max(round_number - 1, 0))
            if current is None or previous is None:
                return ""
        else:
            versions = self.store.get_plan_versions(session_id, limit=2)
            if len(versions) < 2:
                return ""
            current, previous = versions[0], versions[1]
        return "\n".join(
            difflib.unified_diff(
                previous.plan_content.splitlines(),
                current.plan_content.splitlines(),
                fromfile=f"round-{previous.round}",
                tofile=f"round-{current.round}",
                lineterm="",
            )
        )
