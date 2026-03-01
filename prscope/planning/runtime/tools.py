"""
Planning runtime tool definitions and sandboxed execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLOCKED_PATTERNS = {".env", "id_rsa", "credentials", "token", ".secret"}
IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".prscope"}


class ToolSafetyError(RuntimeError):
    """Raised when a tool invocation violates sandbox policy."""


CODEBASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories under a path in the repository",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_entries": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search the repository for a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutor:
    """Sandboxed file-system tools for planning."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.accessed_paths: set[str] = set()

    def _safe_path(self, raw_path: str | None) -> Path:
        if not raw_path:
            return self.repo_root
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ToolSafetyError(f"Path escapes repo root: {raw_path}") from exc

        lower_name = candidate.name.lower()
        if any(token in lower_name for token in BLOCKED_PATTERNS):
            raise ToolSafetyError(f"Blocked sensitive file: {raw_path}")
        return candidate

    def list_dir(self, path: str | None = None, max_entries: int = 200) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.exists() or not safe.is_dir():
            raise ToolSafetyError(f"Directory not found: {path or '.'}")
        entries = []
        for child in sorted(safe.iterdir(), key=lambda p: p.name)[:max_entries]:
            rel = str(child.relative_to(self.repo_root))
            entries.append({"path": rel, "type": "dir" if child.is_dir() else "file"})
            self.accessed_paths.add(rel)
        return {"path": str(safe.relative_to(self.repo_root)), "entries": entries}

    def read_file(self, path: str, max_lines: int = 200) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.exists() or not safe.is_file():
            raise ToolSafetyError(f"File not found: {path}")
        lines = safe.read_text(encoding="utf-8", errors="ignore").splitlines()
        snippet = lines[:max_lines]
        rel = str(safe.relative_to(self.repo_root))
        self.accessed_paths.add(rel)
        return {
            "path": rel,
            "truncated": len(lines) > max_lines,
            "line_count": len(lines),
            "content": "\n".join(snippet),
        }

    def search_codebase(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 40,
    ) -> dict[str, Any]:
        root = self._safe_path(path)
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    rel_parts = file_path.relative_to(self.repo_root).parts
                except ValueError:
                    continue
                if any(part in IGNORED_DIRS for part in rel_parts):
                    continue
                candidates.append(file_path)

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolSafetyError(f"Invalid regex pattern: {pattern}") from exc

        matches: list[dict[str, Any]] = []
        for file_path in candidates:
            if len(matches) >= max_results:
                break
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line_num, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    rel = str(file_path.relative_to(self.repo_root))
                    self.accessed_paths.add(rel)
                    matches.append({"path": rel, "line": line_num, "text": line.strip()})
                    if len(matches) >= max_results:
                        break

        return {"pattern": pattern, "results": matches, "count": len(matches)}

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> ToolCall:
        call_id = getattr(raw_call, "id", None) or raw_call.get("id", "tool-call")
        func = getattr(raw_call, "function", None) or raw_call.get("function", {})
        name = getattr(func, "name", None) or func.get("name")
        raw_args = getattr(func, "arguments", None) or func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return ToolCall(id=call_id, name=name or "", arguments=args)

    def execute(self, raw_tool_call: Any) -> dict[str, Any]:
        parsed = self._parse_tool_call(raw_tool_call)
        if parsed.name == "list_dir":
            result = self.list_dir(
                path=parsed.arguments.get("path"),
                max_entries=int(parsed.arguments.get("max_entries", 200)),
            )
        elif parsed.name == "read_file":
            result = self.read_file(
                path=str(parsed.arguments.get("path", "")),
                max_lines=int(parsed.arguments.get("max_lines", 200)),
            )
        elif parsed.name == "search_codebase":
            result = self.search_codebase(
                pattern=str(parsed.arguments.get("pattern", "")),
                path=parsed.arguments.get("path"),
                max_results=int(parsed.arguments.get("max_results", 40)),
            )
        else:
            raise ToolSafetyError(f"Unknown tool: {parsed.name}")

        return {"tool_call_id": parsed.id, "name": parsed.name, "result": result}


def extract_file_references(text: str) -> set[str]:
    refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", text))
    return {ref for ref in refs if "/" in ref}
