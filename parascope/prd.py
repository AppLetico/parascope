"""
PRD (Product Requirements Document) generator for Parascope.

Renders markdown PRD files from evaluated PRs using Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, PackageLoader

import json as json_module

from .config import get_parascope_dir, ParascopeConfig
from .store import Store, PullRequest, Evaluation, PRFile


def _parse_llm_json(llm_json: str | dict | None) -> dict | None:
    """Parse LLM JSON from string or dict."""
    if not llm_json:
        return None
    if isinstance(llm_json, dict):
        return llm_json
    try:
        return json_module.loads(llm_json)
    except Exception:
        return None


def get_template_env() -> Environment:
    """Get Jinja2 environment with template loaders."""
    # Try package loader first, fall back to file loader
    try:
        return Environment(
            loader=PackageLoader("parascope", "templates"),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    except Exception:
        # Fall back to file-based loader
        template_dir = Path(__file__).parent / "templates"
        return Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )


def get_prd_dir(repo_name: str | None = None) -> Path:
    """Get the PRD output directory."""
    prd_dir = get_parascope_dir() / "prd"
    if repo_name:
        # Sanitize repo name for filesystem
        safe_name = repo_name.replace("/", "_")
        prd_dir = prd_dir / safe_name
    prd_dir.mkdir(parents=True, exist_ok=True)
    return prd_dir


def build_prd_context(
    pr: PullRequest,
    files: list[PRFile],
    evaluation: Evaluation,
    config: ParascopeConfig,
    repo_name: str,
) -> dict[str, Any]:
    """
    Build the template context for PRD rendering.
    
    Args:
        pr: Pull request data
        files: Files changed in the PR
        evaluation: Evaluation result
        config: Parascope configuration
        repo_name: Repository name
    
    Returns:
        Dictionary context for template rendering
    """
    # Build feature matches table
    feature_matches = []
    matched_set = set(evaluation.matched_features)
    
    for feature in config.features:
        if feature.name in matched_set:
            # Get match details from signals if available
            feature_matches.append({
                "name": feature.name,
                "description": feature.description or f"Matches feature: {feature.name}",
                "matched": True,
                "keywords": feature.keywords,
                "paths": feature.paths,
            })
    
    # Summarize file changes
    file_summary = []
    for f in files[:20]:  # Limit to first 20 files
        file_summary.append({
            "path": f.path,
            "additions": f.additions,
            "deletions": f.deletions,
            "change_type": "+" if f.additions > f.deletions else "-" if f.deletions > f.additions else "~",
        })
    
    # Build body summary (first 500 chars)
    body_summary = ""
    if pr.body:
        body_summary = pr.body[:500]
        if len(pr.body) > 500:
            body_summary += "..."
    
    return {
        # PR info
        "pr_number": pr.number,
        "pr_title": pr.title,
        "pr_body": pr.body or "",
        "pr_body_summary": body_summary,
        "pr_author": pr.author or "unknown",
        "pr_url": pr.html_url or "",
        "pr_state": pr.state,
        "pr_labels": pr.labels,
        "pr_updated_at": pr.updated_at or "",
        "pr_merged_at": pr.merged_at or "",
        "pr_head_sha": pr.head_sha or "",
        
        # Repository info
        "repo_name": repo_name,
        
        # Files
        "files": file_summary,
        "file_count": len(files),
        "total_additions": sum(f.additions for f in files),
        "total_deletions": sum(f.deletions for f in files),
        
        # Evaluation
        "rule_score": evaluation.rule_score,
        "final_score": evaluation.final_score,
        "decision": evaluation.decision,
        "matched_features": evaluation.matched_features,
        "feature_matches": feature_matches,
        "signals": evaluation.signals,
        
        # LLM analysis (if available)
        "llm_analysis": _parse_llm_json(evaluation.llm_json),
        
        # Metadata
        "evaluation_id": evaluation.id,
        "local_profile_sha": evaluation.local_profile_sha,
        "evaluated_at": evaluation.created_at,
    }


def render_prd(
    pr: PullRequest,
    files: list[PRFile],
    evaluation: Evaluation,
    config: ParascopeConfig,
    repo_name: str,
    template_name: str = "prd.md.j2",
) -> str:
    """
    Render a PRD from a template.
    
    Args:
        pr: Pull request data
        files: Files changed in the PR
        evaluation: Evaluation result
        config: Parascope configuration
        repo_name: Repository name
        template_name: Template file name
    
    Returns:
        Rendered markdown string
    """
    env = get_template_env()
    template = env.get_template(template_name)
    context = build_prd_context(pr, files, evaluation, config, repo_name)
    return template.render(**context)


def should_generate_prd(
    evaluation: Evaluation,
    min_confidence: float = 0.7,
    include_all: bool = False,
) -> bool:
    """
    Determine if a PRD should be generated for this evaluation.
    
    Args:
        evaluation: The evaluation to check
        min_confidence: Minimum LLM confidence required
        include_all: If True, include all evaluated PRs regardless of confidence
    
    Returns:
        True if PRD should be generated
    """
    if include_all:
        return True
    
    # Check LLM analysis if available
    if evaluation.llm_json:
        import json
        try:
            llm_data = json.loads(evaluation.llm_json) if isinstance(evaluation.llm_json, str) else evaluation.llm_json
            relevance = llm_data.get("relevance", {})
            decision = relevance.get("decision", "skip")
            confidence = relevance.get("confidence", 0.0)
            
            # Only generate for high-confidence implement decisions
            if decision == "implement" and confidence >= min_confidence:
                return True
            if decision == "partial" and confidence >= min_confidence + 0.1:
                return True
            
            return False
        except Exception:
            pass
    
    # Fall back to rule-based decision
    return evaluation.decision == "relevant" and (evaluation.final_score or 0) >= min_confidence


def generate_prd(
    store: Store,
    config: ParascopeConfig,
    evaluation_id: int | None = None,
    pr_id: int | None = None,
    repo_name: str | None = None,
    min_confidence: float = 0.7,
    include_all: bool = False,
) -> list[Path]:
    """
    Generate PRD files for high-confidence evaluated PRs.
    
    Args:
        store: Parascope store
        config: Parascope configuration
        evaluation_id: Specific evaluation to generate PRD for
        pr_id: Specific PR to generate PRD for
        repo_name: Filter by repository name
        min_confidence: Minimum LLM confidence to generate PRD
        include_all: If True, include all evaluated PRs
    
    Returns:
        List of generated PRD file paths
    """
    generated = []
    
    # Get evaluations to process
    if evaluation_id:
        evaluation = store.get_evaluation_by_id(evaluation_id)
        evaluations = [evaluation] if evaluation else []
    else:
        # Get relevant evaluations
        evaluations = store.list_evaluations(decision="relevant")
        # Also check 'maybe' if include_all
        if include_all:
            evaluations.extend(store.list_evaluations(decision="maybe"))
    
    for evaluation in evaluations:
        if not evaluation:
            continue
        
        # Check if PRD should be generated
        if not should_generate_prd(evaluation, min_confidence, include_all):
            continue
        
        # Filter by pr_id if specified
        if pr_id and evaluation.pr_id != pr_id:
            continue
        
        # Get PR data
        pr = store.get_pull_request_by_id(evaluation.pr_id)
        if not pr:
            continue
        
        # Get repo name
        repos = store.list_upstream_repos()
        pr_repo_name = None
        for repo in repos:
            if repo.id == pr.repo_id:
                pr_repo_name = repo.full_name
                break
        
        if not pr_repo_name:
            continue
        
        # Filter by repo_name if specified
        if repo_name and pr_repo_name != repo_name:
            continue
        
        # Get files
        files = store.get_pr_files(pr.id)
        
        # Render PRD
        prd_content = render_prd(pr, files, evaluation, config, pr_repo_name)
        
        # Save to file
        prd_dir = get_prd_dir(pr_repo_name)
        prd_path = prd_dir / f"PR-{pr.number}.md"
        prd_path.write_text(prd_content)
        
        # Record artifact
        store.save_artifact(evaluation.id, "prd", str(prd_path))
        
        generated.append(prd_path)
    
    return generated
