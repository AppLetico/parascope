"""
Configuration management for Parascope.

Loads and validates:
- parascope.yml: Main configuration (upstream repos, thresholds)
- parascope.features.yml: Feature definitions for matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UpstreamRepo:
    """Configuration for an upstream repository to monitor."""
    repo: str  # full_name like "owner/repo"
    filters: dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        return self.repo


@dataclass
class ScoringConfig:
    """Scoring thresholds and weights."""
    min_rule_score: float = 0.3
    min_final_score: float = 0.5
    keyword_weight: float = 0.4
    path_weight: float = 0.6


@dataclass
class SyncConfig:
    """Default sync settings."""
    state: str = "merged"  # merged, open, closed, all
    max_prs: int = 100
    fetch_files: bool = True
    # Date windowing for initial sync (ISO date or relative like "90d", "6m")
    since: str = "90d"  # Default: last 90 days
    # Incremental mode: only fetch PRs newer than last sync
    incremental: bool = True
    # Batch size for evaluation (limits LLM calls per run)
    eval_batch_size: int = 25


@dataclass
class LLMConfig:
    """LLM configuration using LiteLLM."""
    enabled: bool = False
    model: str = "gpt-4o"  # LiteLLM model string (e.g., gpt-4o, claude-3-opus, gemini-pro)
    temperature: float = 0.3
    max_tokens: int = 2000
    # API keys are read from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)


@dataclass
class Feature:
    """A feature definition for matching PRs."""
    name: str
    keywords: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)  # glob patterns
    description: str = ""


@dataclass
class ParascopeConfig:
    """Complete Parascope configuration."""
    local_repo: str | None = None  # Path to local repo (defaults to current git root)
    upstream: list[UpstreamRepo] = field(default_factory=list)
    sync: SyncConfig = field(default_factory=SyncConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    features: list[Feature] = field(default_factory=list)
    
    def get_local_repo_path(self) -> Path:
        """Get the local repo path, resolving relative paths."""
        if self.local_repo:
            path = Path(self.local_repo).expanduser()
            if not path.is_absolute():
                # Relative to config file location (repo root)
                path = get_repo_root() / path
            return path.resolve()
        return get_repo_root()
    
    @classmethod
    def load(cls, repo_root: Path) -> "ParascopeConfig":
        """Load configuration from repo root directory."""
        config = cls()
        
        # Load main config
        main_config_path = repo_root / "parascope.yml"
        if main_config_path.exists():
            with open(main_config_path) as f:
                data = yaml.safe_load(f) or {}
            config = cls._parse_main_config(data)
        
        # Load features config
        features_path = repo_root / "parascope.features.yml"
        if features_path.exists():
            with open(features_path) as f:
                data = yaml.safe_load(f) or {}
            config.features = cls._parse_features(data)
        
        return config
    
    @classmethod
    def _parse_main_config(cls, data: dict[str, Any]) -> "ParascopeConfig":
        """Parse main configuration dictionary."""
        config = cls()
        
        # Parse local repo path
        config.local_repo = data.get("local_repo")
        
        # Parse upstream repos
        for repo_data in data.get("upstream", []):
            if isinstance(repo_data, str):
                config.upstream.append(UpstreamRepo(repo=repo_data))
            elif isinstance(repo_data, dict):
                config.upstream.append(UpstreamRepo(
                    repo=repo_data.get("repo", ""),
                    filters=repo_data.get("filters", {})
                ))
        
        # Parse scoring config
        scoring_data = data.get("scoring", {})
        config.scoring = ScoringConfig(
            min_rule_score=scoring_data.get("min_rule_score", 0.3),
            min_final_score=scoring_data.get("min_final_score", 0.5),
            keyword_weight=scoring_data.get("keyword_weight", 0.4),
            path_weight=scoring_data.get("path_weight", 0.6),
        )
        
        # Parse sync config
        sync_data = data.get("sync", {})
        config.sync = SyncConfig(
            state=sync_data.get("state", "merged"),
            max_prs=sync_data.get("max_prs", 100),
            fetch_files=sync_data.get("fetch_files", True),
            since=sync_data.get("since", "90d"),
            incremental=sync_data.get("incremental", True),
            eval_batch_size=sync_data.get("eval_batch_size", 25),
        )
        
        # Parse LLM config
        llm_data = data.get("llm", {})
        config.llm = LLMConfig(
            enabled=llm_data.get("enabled", False),
            model=llm_data.get("model", "gpt-4o"),
            temperature=llm_data.get("temperature", 0.3),
            max_tokens=llm_data.get("max_tokens", 2000),
        )
        
        return config
    
    @classmethod
    def _parse_features(cls, data: dict[str, Any]) -> list[Feature]:
        """Parse features configuration dictionary."""
        features = []
        features_data = data.get("features", {})
        
        for name, feature_data in features_data.items():
            if isinstance(feature_data, dict):
                features.append(Feature(
                    name=name,
                    keywords=feature_data.get("keywords", []),
                    paths=feature_data.get("paths", []),
                    description=feature_data.get("description", ""),
                ))
        
        return features


def get_repo_root() -> Path:
    """Find the repository root (directory containing .git)."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # No .git found, use current directory
    return Path.cwd()


def get_parascope_dir(repo_root: Path | None = None) -> Path:
    """Get the .parascope directory path."""
    if repo_root is None:
        repo_root = get_repo_root()
    return repo_root / ".parascope"


def ensure_parascope_dir(repo_root: Path | None = None) -> Path:
    """Ensure .parascope directory exists and return its path."""
    parascope_dir = get_parascope_dir(repo_root)
    parascope_dir.mkdir(parents=True, exist_ok=True)
    return parascope_dir
