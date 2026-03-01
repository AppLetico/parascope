"""
Textual TUI for planning sessions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .planning.runtime.orchestration import PlanningRuntime

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, Markdown, RichLog, Static
except ImportError:  # pragma: no cover
    App = object  # type: ignore
    ComposeResult = object  # type: ignore
    Binding = object  # type: ignore
    Horizontal = object  # type: ignore
    Vertical = object  # type: ignore
    Footer = object  # type: ignore
    Header = object  # type: ignore
    Input = object  # type: ignore
    Markdown = object  # type: ignore
    RichLog = object  # type: ignore
    Static = object  # type: ignore


@dataclass
class SessionViewState:
    session_id: str
    round_number: int
    status: str
    convergence_pct: float = 0.0
    seeded_ref: str | None = None


class PlanningTUI(App):  # type: ignore[misc]
    TITLE = "Prscope Planning Mode"
    BINDINGS = [
        Binding("ctrl+k", "critique", "Critique"),
        Binding("ctrl+d", "toggle_diff", "Diff"),
        Binding("ctrl+a", "approve", "Approve"),
        Binding("ctrl+e", "export", "Export"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
      layout: vertical;
    }
    #main {
      height: 1fr;
    }
    #plan_panel {
      width: 60%;
      border: solid $surface;
      padding: 1;
    }
    #right_col {
      width: 40%;
    }
    #log_panel {
      height: 1fr;
      border: solid $surface;
    }
    #input_panel {
      height: 3;
      border: solid $surface;
    }
    #status_bar {
      height: 1;
      content-align: left middle;
    }
    """

    def __init__(self, runtime: PlanningRuntime, session_id: str):
        super().__init__()
        self.runtime = runtime
        self.session_id = session_id
        self.showing_diff = False
        self.last_unverified: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield Markdown("", id="plan_panel")
            with Vertical(id="right_col"):
                yield RichLog(id="log_panel", wrap=True, highlight=True, markup=True)
                yield Input(placeholder="Add context or ask a question...", id="input_panel")
        yield Static("", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_view()

    def _status_line(self) -> str:
        session = self.runtime.store.get_planning_session(self.session_id)
        if session is None:
            return "[repo: unknown] session missing"
        seed = f" [Seeded: {session.seed_ref}]" if session.seed_ref else ""
        return (
            f"[repo: {session.repo_name}]  Round: {session.current_round}/"
            f"{self.runtime.planning_config.max_adversarial_rounds}  "
            f"{session.status.upper()}  Δ{self._latest_change_pct():.1f}%{seed}"
        )

    def _latest_change_pct(self) -> float:
        core = self.runtime._core(self.session_id)  # noqa: SLF001
        conv = core.check_convergence()
        return conv.change_pct * 100

    def _refresh_view(self) -> None:
        plan_widget = self.query_one("#plan_panel", Markdown)
        log_widget = self.query_one("#log_panel", RichLog)
        status = self.query_one("#status_bar", Static)

        core = self.runtime._core(self.session_id)  # noqa: SLF001
        plan = core.get_current_plan()
        if self.showing_diff:
            diff = self.runtime.plan_diff(self.session_id)
            plan_widget.update(f"```diff\n{diff or 'No previous version to diff against.'}\n```")
        else:
            plan_widget.update(plan.plan_content if plan else "_No plan yet._")

        log_widget.clear()
        for turn in core.get_conversation():
            role = turn.role.upper()
            log_widget.write(f"[{role}] {turn.content}")
        status.update(self._status_line())

    async def _run_critique(self) -> None:
        critic, author, conv = await self.runtime.run_adversarial_round(self.session_id)
        self.last_unverified = author.unverified_references
        self._refresh_view()
        if conv.reason == "regression":
            self.notify(f"Plan regressed: {conv.regression}", severity="warning")
        elif conv.converged:
            self.notify(
                f"Plan stabilized (Δ {conv.change_pct*100:.1f}%, "
                f"{conv.major_issues or 0} major issues).",
            )
        if self.last_unverified:
            refs = ", ".join(sorted(self.last_unverified)[:3])
            self.notify(f"Unverified file references: {refs}", severity="warning")

    def action_critique(self) -> None:
        asyncio.create_task(self._run_critique())

    def action_toggle_diff(self) -> None:
        self.showing_diff = not self.showing_diff
        self._refresh_view()

    def action_approve(self) -> None:
        try:
            self.runtime.approve(self.session_id, unverified_references=self.last_unverified)
            self.notify("Plan approved.")
            self._refresh_view()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    def action_export(self) -> None:
        try:
            paths = self.runtime.export(self.session_id)
            self.notify(f"Exported PRD and RFC to {paths['prd'].parent}")
            self._refresh_view()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore[name-defined]
        text = event.value.strip()
        if not text:
            return
        session = self.runtime.store.get_planning_session(self.session_id)
        if session is None:
            return
        if session.status == "discovery":
            result = await self.runtime.handle_discovery_turn(self.session_id, text)
            self.notify(result.reply)
        else:
            core = self.runtime._core(self.session_id)  # noqa: SLF001
            core.add_turn("user", text, round_number=session.current_round)
        event.input.value = ""
        self._refresh_view()
