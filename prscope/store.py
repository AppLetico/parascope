"""
SQLite database storage for Prscope.

Schema:
- repo_profiles: Local repo profile snapshots
- upstream_repos: Tracked upstream repositories
- pull_requests: PR metadata
- pr_files: Files changed per PR
- evaluations: Evaluation results (with deduplication)
- artifacts: Generated PRD files
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from .config import get_prscope_dir


DB_FILENAME = "prscope.db"

SCHEMA = """
-- Repo profile snapshots keyed by git HEAD SHA
CREATE TABLE IF NOT EXISTS repo_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_root TEXT NOT NULL,
    profile_sha TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(repo_root, profile_sha)
);

-- Tracked upstream repositories
CREATE TABLE IF NOT EXISTS upstream_repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT UNIQUE NOT NULL,
    last_synced_at TEXT,
    last_seen_updated_at TEXT
);

-- PR metadata
CREATE TABLE IF NOT EXISTS pull_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    number INTEGER NOT NULL,
    state TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    author TEXT,
    labels_json TEXT,
    updated_at TEXT,
    merged_at TEXT,
    head_sha TEXT,
    html_url TEXT,
    FOREIGN KEY (repo_id) REFERENCES upstream_repos(id),
    UNIQUE(repo_id, number)
);

-- Files changed per PR
CREATE TABLE IF NOT EXISTS pr_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    FOREIGN KEY (pr_id) REFERENCES pull_requests(id)
);

-- Evaluation results with deduplication key
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_id INTEGER NOT NULL,
    local_profile_sha TEXT NOT NULL,
    pr_head_sha TEXT NOT NULL,
    rule_score REAL,
    final_score REAL,
    matched_features_json TEXT,
    signals_json TEXT,
    llm_json TEXT,
    decision TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (pr_id) REFERENCES pull_requests(id),
    UNIQUE(pr_id, local_profile_sha, pr_head_sha)
);

-- Generated artifacts (PRDs, etc.)
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);

-- Planning sessions (interactive plan lifecycle)
CREATE TABLE IF NOT EXISTS planning_sessions (
    id TEXT PRIMARY KEY,
    repo_name TEXT NOT NULL,
    title TEXT NOT NULL,
    requirements TEXT NOT NULL,
    status TEXT NOT NULL,
    seed_type TEXT NOT NULL,
    seed_ref TEXT,
    current_round INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Planning conversation turns
CREATE TABLE IF NOT EXISTS planning_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    round INTEGER NOT NULL,
    major_issues_remaining INTEGER,
    minor_issues_remaining INTEGER,
    hard_constraint_violations TEXT,
    parse_error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
);

-- Plan snapshots by round
CREATE TABLE IF NOT EXISTS plan_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    plan_content TEXT NOT NULL,
    plan_sha TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_pr_repo ON pull_requests(repo_id);
CREATE INDEX IF NOT EXISTS idx_pr_state ON pull_requests(state);
CREATE INDEX IF NOT EXISTS idx_pr_updated ON pull_requests(updated_at);
CREATE INDEX IF NOT EXISTS idx_eval_pr ON evaluations(pr_id);
CREATE INDEX IF NOT EXISTS idx_eval_decision ON evaluations(decision);
CREATE INDEX IF NOT EXISTS idx_turns_session_round ON planning_turns(session_id, round);
CREATE INDEX IF NOT EXISTS idx_versions_session_round ON plan_versions(session_id, round);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON planning_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_repo ON planning_sessions(repo_name);
"""


@dataclass
class RepoProfile:
    """Stored repo profile."""
    id: int | None
    repo_root: str
    profile_sha: str
    profile_json: str
    created_at: str
    
    @property
    def profile_data(self) -> dict[str, Any]:
        return json.loads(self.profile_json)


@dataclass
class UpstreamRepo:
    """Stored upstream repo."""
    id: int | None
    full_name: str
    last_synced_at: str | None
    last_seen_updated_at: str | None


@dataclass
class PullRequest:
    """Stored pull request."""
    id: int | None
    repo_id: int
    number: int
    state: str
    title: str
    body: str | None
    author: str | None
    labels_json: str | None
    updated_at: str | None
    merged_at: str | None
    head_sha: str | None
    html_url: str | None
    
    @property
    def labels(self) -> list[str]:
        if self.labels_json:
            return json.loads(self.labels_json)
        return []


@dataclass
class PRFile:
    """Stored PR file change."""
    id: int | None
    pr_id: int
    path: str
    additions: int
    deletions: int


@dataclass
class Evaluation:
    """Stored evaluation result."""
    id: int | None
    pr_id: int
    local_profile_sha: str
    pr_head_sha: str
    rule_score: float | None
    final_score: float | None
    matched_features_json: str | None
    signals_json: str | None
    llm_json: str | None
    decision: str | None
    created_at: str
    
    @property
    def matched_features(self) -> list[str]:
        if self.matched_features_json:
            return json.loads(self.matched_features_json)
        return []
    
    @property
    def signals(self) -> dict[str, Any]:
        if self.signals_json:
            return json.loads(self.signals_json)
        return {}


@dataclass
class Artifact:
    """Stored artifact (PRD file, etc.)."""
    id: int | None
    evaluation_id: int
    type: str
    path: str
    created_at: str


@dataclass
class PlanningSession:
    """Stored planning session."""

    id: str
    repo_name: str
    title: str
    requirements: str
    status: str
    seed_type: str
    seed_ref: str | None
    current_round: int
    created_at: str
    updated_at: str


@dataclass
class PlanningTurn:
    """Stored planning conversation turn."""

    id: int | None
    session_id: str
    role: str
    content: str
    round: int
    major_issues_remaining: int | None = None
    minor_issues_remaining: int | None = None
    hard_constraint_violations: list[str] | None = None
    parse_error: str | None = None
    created_at: str = ""


@dataclass
class PlanVersion:
    """Stored plan snapshot."""

    id: int | None
    session_id: str
    round: int
    plan_content: str
    plan_sha: str
    created_at: str


class Store:
    """SQLite storage manager for Prscope."""
    
    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = get_prscope_dir() / DB_FILENAME
        self.db_path = db_path
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Ensure database schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
    
    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    # =========================================================================
    # Repo Profiles
    # =========================================================================
    
    def save_profile(self, repo_root: str, profile_sha: str, profile_json: str) -> RepoProfile:
        """Save or update a repo profile."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repo_profiles (repo_root, profile_sha, profile_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo_root, profile_sha) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    created_at = excluded.created_at
                """,
                (repo_root, profile_sha, profile_json, self._now())
            )
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? AND profile_sha = ?",
                (repo_root, profile_sha)
            ).fetchone()
            return RepoProfile(**dict(row))
    
    def get_profile(self, repo_root: str, profile_sha: str) -> RepoProfile | None:
        """Get a specific profile by SHA."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? AND profile_sha = ?",
                (repo_root, profile_sha)
            ).fetchone()
            return RepoProfile(**dict(row)) if row else None
    
    def get_latest_profile(self, repo_root: str) -> RepoProfile | None:
        """Get the most recent profile for a repo."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? ORDER BY created_at DESC LIMIT 1",
                (repo_root,)
            ).fetchone()
            return RepoProfile(**dict(row)) if row else None
    
    # =========================================================================
    # Upstream Repos
    # =========================================================================
    
    def upsert_upstream_repo(self, full_name: str) -> UpstreamRepo:
        """Insert or get an upstream repo."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO upstream_repos (full_name) VALUES (?)",
                (full_name,)
            )
            row = conn.execute(
                "SELECT * FROM upstream_repos WHERE full_name = ?",
                (full_name,)
            ).fetchone()
            return UpstreamRepo(**dict(row))
    
    def update_repo_sync_time(self, repo_id: int, last_seen_updated_at: str | None = None) -> None:
        """Update last sync time for a repo."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE upstream_repos 
                SET last_synced_at = ?, last_seen_updated_at = COALESCE(?, last_seen_updated_at)
                WHERE id = ?
                """,
                (self._now(), last_seen_updated_at, repo_id)
            )
    
    def get_upstream_repo(self, full_name: str) -> UpstreamRepo | None:
        """Get upstream repo by full name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM upstream_repos WHERE full_name = ?",
                (full_name,)
            ).fetchone()
            return UpstreamRepo(**dict(row)) if row else None
    
    def list_upstream_repos(self) -> list[UpstreamRepo]:
        """List all upstream repos."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM upstream_repos ORDER BY full_name").fetchall()
            return [UpstreamRepo(**dict(row)) for row in rows]
    
    # =========================================================================
    # Pull Requests
    # =========================================================================
    
    def upsert_pull_request(
        self,
        repo_id: int,
        number: int,
        state: str,
        title: str,
        body: str | None = None,
        author: str | None = None,
        labels: list[str] | None = None,
        updated_at: str | None = None,
        merged_at: str | None = None,
        head_sha: str | None = None,
        html_url: str | None = None,
    ) -> PullRequest:
        """Insert or update a pull request."""
        labels_json = json.dumps(labels) if labels else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pull_requests 
                    (repo_id, number, state, title, body, author, labels_json, 
                     updated_at, merged_at, head_sha, html_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, number) DO UPDATE SET
                    state = excluded.state,
                    title = excluded.title,
                    body = excluded.body,
                    author = excluded.author,
                    labels_json = excluded.labels_json,
                    updated_at = excluded.updated_at,
                    merged_at = excluded.merged_at,
                    head_sha = excluded.head_sha,
                    html_url = excluded.html_url
                """,
                (repo_id, number, state, title, body, author, labels_json,
                 updated_at, merged_at, head_sha, html_url)
            )
            row = conn.execute(
                "SELECT * FROM pull_requests WHERE repo_id = ? AND number = ?",
                (repo_id, number)
            ).fetchone()
            return PullRequest(**dict(row))
    
    def get_pull_request(self, repo_id: int, number: int) -> PullRequest | None:
        """Get a specific PR."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pull_requests WHERE repo_id = ? AND number = ?",
                (repo_id, number)
            ).fetchone()
            return PullRequest(**dict(row)) if row else None
    
    def get_pull_request_by_id(self, pr_id: int) -> PullRequest | None:
        """Get a PR by its database ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pull_requests WHERE id = ?",
                (pr_id,)
            ).fetchone()
            return PullRequest(**dict(row)) if row else None
    
    def list_pull_requests(
        self,
        repo_id: int | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[PullRequest]:
        """List pull requests with optional filters."""
        with self._connect() as conn:
            query = "SELECT * FROM pull_requests WHERE 1=1"
            params: list[Any] = []
            
            if repo_id is not None:
                query += " AND repo_id = ?"
                params.append(repo_id)
            if state is not None:
                query += " AND state = ?"
                params.append(state)
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [PullRequest(**dict(row)) for row in rows]
    
    # =========================================================================
    # PR Files
    # =========================================================================
    
    def save_pr_files(self, pr_id: int, files: list[dict[str, Any]]) -> None:
        """Save files for a PR (replaces existing)."""
        with self._connect() as conn:
            # Delete existing files
            conn.execute("DELETE FROM pr_files WHERE pr_id = ?", (pr_id,))
            # Insert new files
            for f in files:
                conn.execute(
                    "INSERT INTO pr_files (pr_id, path, additions, deletions) VALUES (?, ?, ?, ?)",
                    (pr_id, f.get("path", ""), f.get("additions", 0), f.get("deletions", 0))
                )
    
    def get_pr_files(self, pr_id: int) -> list[PRFile]:
        """Get files for a PR."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pr_files WHERE pr_id = ?",
                (pr_id,)
            ).fetchall()
            return [PRFile(**dict(row)) for row in rows]
    
    # =========================================================================
    # Evaluations
    # =========================================================================
    
    def evaluation_exists(self, pr_id: int, local_profile_sha: str, pr_head_sha: str) -> bool:
        """Check if an evaluation already exists (deduplication)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM evaluations 
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha)
            ).fetchone()
            return row is not None
    
    def save_evaluation(
        self,
        pr_id: int,
        local_profile_sha: str,
        pr_head_sha: str,
        rule_score: float,
        final_score: float,
        matched_features: list[str],
        signals: dict[str, Any],
        llm_result: dict[str, Any] | None,
        decision: str,
    ) -> Evaluation:
        """Save an evaluation result."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations 
                    (pr_id, local_profile_sha, pr_head_sha, rule_score, final_score,
                     matched_features_json, signals_json, llm_json, decision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pr_id, local_profile_sha, pr_head_sha) DO UPDATE SET
                    rule_score = excluded.rule_score,
                    final_score = excluded.final_score,
                    matched_features_json = excluded.matched_features_json,
                    signals_json = excluded.signals_json,
                    llm_json = excluded.llm_json,
                    decision = excluded.decision,
                    created_at = excluded.created_at
                """,
                (
                    pr_id, local_profile_sha, pr_head_sha, rule_score, final_score,
                    json.dumps(matched_features), json.dumps(signals),
                    json.dumps(llm_result) if llm_result else None,
                    decision, self._now()
                )
            )
            row = conn.execute(
                """
                SELECT * FROM evaluations 
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha)
            ).fetchone()
            return Evaluation(**dict(row))
    
    def get_evaluation(self, pr_id: int, local_profile_sha: str, pr_head_sha: str) -> Evaluation | None:
        """Get a specific evaluation."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM evaluations 
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha)
            ).fetchone()
            return Evaluation(**dict(row)) if row else None
    
    def get_evaluation_by_id(self, evaluation_id: int) -> Evaluation | None:
        """Get evaluation by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE id = ?",
                (evaluation_id,)
            ).fetchone()
            return Evaluation(**dict(row)) if row else None
    
    def list_evaluations(
        self,
        decision: str | None = None,
        limit: int = 100,
    ) -> list[Evaluation]:
        """List evaluations with optional filters."""
        with self._connect() as conn:
            query = "SELECT * FROM evaluations WHERE 1=1"
            params: list[Any] = []
            
            if decision is not None:
                query += " AND decision = ?"
                params.append(decision)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [Evaluation(**dict(row)) for row in rows]
    
    # =========================================================================
    # Artifacts
    # =========================================================================
    
    def save_artifact(self, evaluation_id: int, artifact_type: str, path: str) -> Artifact:
        """Save an artifact record."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (evaluation_id, type, path, created_at) VALUES (?, ?, ?, ?)",
                (evaluation_id, artifact_type, path, self._now())
            )
            row = conn.execute(
                "SELECT * FROM artifacts WHERE evaluation_id = ? AND type = ? AND path = ?",
                (evaluation_id, artifact_type, path)
            ).fetchone()
            return Artifact(**dict(row))
    
    def get_artifacts(self, evaluation_id: int) -> list[Artifact]:
        """Get artifacts for an evaluation."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE evaluation_id = ?",
                (evaluation_id,)
            ).fetchall()
            return [Artifact(**dict(row)) for row in rows]

    # =========================================================================
    # Planning sessions and versions
    # =========================================================================

    def _row_to_planning_turn(self, row: sqlite3.Row) -> PlanningTurn:
        data = dict(row)
        violations_raw = data.get("hard_constraint_violations")
        violations: list[str] | None = None
        if violations_raw:
            try:
                loaded = json.loads(violations_raw)
                if isinstance(loaded, list):
                    violations = [str(v) for v in loaded]
            except json.JSONDecodeError:
                violations = None

        return PlanningTurn(
            id=data.get("id"),
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            round=data["round"],
            major_issues_remaining=data.get("major_issues_remaining"),
            minor_issues_remaining=data.get("minor_issues_remaining"),
            hard_constraint_violations=violations,
            parse_error=data.get("parse_error"),
            created_at=data.get("created_at", ""),
        )

    def create_planning_session(
        self,
        repo_name: str,
        title: str,
        requirements: str,
        seed_type: str,
        seed_ref: str | None = None,
        status: str = "drafting",
        session_id: str | None = None,
    ) -> PlanningSession:
        """Create and persist a planning session."""
        sid = session_id or str(uuid4())
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_sessions
                    (id, repo_name, title, requirements, status, seed_type, seed_ref,
                     current_round, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, repo_name, title, requirements, status, seed_type, seed_ref, 0, now, now),
            )
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            return PlanningSession(**dict(row))

    def get_planning_session(self, session_id: str) -> PlanningSession | None:
        """Get a planning session by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return PlanningSession(**dict(row)) if row else None

    def list_planning_sessions(
        self,
        repo_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[PlanningSession]:
        """List planning sessions with optional filters."""
        query = "SELECT * FROM planning_sessions WHERE 1=1"
        params: list[Any] = []
        if repo_name is not None:
            query += " AND repo_name = ?"
            params.append(repo_name)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [PlanningSession(**dict(row)) for row in rows]

    def update_planning_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        requirements: str | None = None,
        current_round: int | None = None,
        seed_ref: str | None = None,
    ) -> PlanningSession:
        """Update mutable fields on a planning session."""
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if requirements is not None:
            updates.append("requirements = ?")
            params.append(requirements)
        if current_round is not None:
            updates.append("current_round = ?")
            params.append(current_round)
        if seed_ref is not None:
            updates.append("seed_ref = ?")
            params.append(seed_ref)

        updates.append("updated_at = ?")
        params.append(self._now())
        params.append(session_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE planning_sessions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Planning session not found: {session_id}")
            return PlanningSession(**dict(row))

    def add_planning_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        round_number: int,
        major_issues_remaining: int | None = None,
        minor_issues_remaining: int | None = None,
        hard_constraint_violations: list[str] | None = None,
        parse_error: str | None = None,
    ) -> PlanningTurn:
        """Insert a planning conversation turn."""
        violations_json = json.dumps(hard_constraint_violations or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_turns
                    (session_id, role, content, round, major_issues_remaining,
                     minor_issues_remaining, hard_constraint_violations, parse_error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    round_number,
                    major_issues_remaining,
                    minor_issues_remaining,
                    violations_json,
                    parse_error,
                    self._now(),
                ),
            )
            row = conn.execute("SELECT * FROM planning_turns WHERE id = last_insert_rowid()").fetchone()
            return self._row_to_planning_turn(row)

    def get_planning_turns(self, session_id: str, round_number: int | None = None) -> list[PlanningTurn]:
        """Fetch planning turns for a session."""
        with self._connect() as conn:
            if round_number is None:
                rows = conn.execute(
                    """
                    SELECT * FROM planning_turns
                    WHERE session_id = ?
                    ORDER BY round ASC, id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM planning_turns
                    WHERE session_id = ? AND round = ?
                    ORDER BY id ASC
                    """,
                    (session_id, round_number),
                ).fetchall()
            return [self._row_to_planning_turn(row) for row in rows]

    def get_latest_critic_turn(self, session_id: str) -> PlanningTurn | None:
        """Get the latest critic turn for semantic convergence checks."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_turns
                WHERE session_id = ? AND role = 'critic'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return self._row_to_planning_turn(row) if row else None

    def save_plan_version(
        self,
        session_id: str,
        round_number: int,
        plan_content: str,
        plan_sha: str,
    ) -> PlanVersion:
        """Persist a plan snapshot."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plan_versions (session_id, round, plan_content, plan_sha, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, round_number, plan_content, plan_sha, self._now()),
            )
            row = conn.execute("SELECT * FROM plan_versions WHERE id = last_insert_rowid()").fetchone()
            return PlanVersion(**dict(row))

    def get_plan_versions(self, session_id: str, limit: int = 20) -> list[PlanVersion]:
        """Get latest plan versions for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM plan_versions
                WHERE session_id = ?
                ORDER BY round DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [PlanVersion(**dict(row)) for row in rows]

    def get_plan_version(self, session_id: str, round_number: int) -> PlanVersion | None:
        """Get a specific plan version by round."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM plan_versions
                WHERE session_id = ? AND round = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id, round_number),
            ).fetchone()
            return PlanVersion(**dict(row)) if row else None
