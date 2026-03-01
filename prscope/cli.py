"""
Prscope CLI - Planning-first workflow with upstream PR intelligence.

Commands:
    init      - Initialize Prscope in current repository
    profile   - Scan and profile local codebase
    upstream  - Upstream sync/evaluate/history utilities
    plan      - Interactive planning (PRD + RFC generation)
    repos     - Repo profile management
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file from current directory or repo root
load_dotenv()  # Loads from current directory
load_dotenv(Path.cwd() / ".env")  # Explicit current dir
# Also try repo root
from .config import get_repo_root
try:
    load_dotenv(get_repo_root() / ".env")
except Exception:
    pass

from . import __version__
from .config import (
    PrscopeConfig,
    get_repo_root,
    ensure_prscope_dir,
)
from .store import Store
from .profile import build_profile, hash_profile
from .github import GitHubClient, sync_repo_prs, GitHubAPIError
from .scoring import evaluate_pr
from .planning.runtime import PlanningRuntime


# Sample configuration files
SAMPLE_CONFIG = """\
# Prscope Configuration
# See: https://github.com/prscope/prscope

# Local repository to profile and compare against upstream PRs
# Can be absolute path or relative to this config file
local_repo: .  # Current directory (default)
# local_repo: ~/workspace/my-project
# local_repo: /absolute/path/to/repo

# Upstream repositories to monitor
upstream:
  - repo: openclaw/openclaw
    filters:
      # Override sync defaults per-repo (optional)
      # state: merged  # merged (default), open, closed, all
      # labels: [security, agents]  # Optional label filter

# Sync settings (how to fetch PRs from upstream)
sync:
  state: merged       # merged (default), open, closed, all
  max_prs: 100        # Maximum PRs to fetch per repo
  fetch_files: true   # Fetch file changes (needed for path matching)
  since: 90d          # Initial date window (90d = 90 days, 6m = 6 months, or ISO date)
  incremental: true   # Only fetch new PRs after initial sync
  eval_batch_size: 25 # Max PRs to evaluate per run (controls LLM costs)

# Scoring thresholds
scoring:
  min_rule_score: 0.3    # Minimum score to consider (0-1)
  min_final_score: 0.5   # Threshold for "relevant" decision (0-1)
  keyword_weight: 0.4    # Weight for keyword matching
  path_weight: 0.6       # Weight for path matching

# LLM configuration (RECOMMENDED for high-quality, noise-free results)
# Without LLM, you only get rule-based matching (more noise)
# API keys are read from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
# See: https://docs.litellm.ai/docs/providers
llm:
  enabled: true          # Enable for production use
  model: gpt-4o          # LiteLLM model string
  # model: claude-3-opus # Anthropic
  # model: gemini-pro    # Google
  # model: ollama/llama2 # Local Ollama
  temperature: 0.2       # Lower = more consistent decisions
  max_tokens: 3000

# Planning mode configuration (PRD + RFC generation)
planning:
  author_model: gpt-4o
  critic_model: claude-3-5-sonnet-20241022
  max_adversarial_rounds: 10
  convergence_threshold: 0.05
  output_dir: ./plans
  require_verified_file_references: false

# Optional multi-repo profiles
# repos:
#   my-repo:
#     path: ~/workspace/my-repo
#     upstream:
#       - repo: owner/upstream-repo
"""

SAMPLE_FEATURES = """\
# Prscope Feature Definitions
# Define features to match against upstream PRs

features:
  # Security-related changes
  security:
    keywords:
      - security
      - auth
      - authentication
      - authorization
      - jwt
      - tls
      - injection
      - vulnerability
      - cve
    paths:
      - "**/security/**"
      - "**/auth/**"
      - "**/*auth*.py"
      - "**/*auth*.ts"

  # Streaming/real-time features
  streaming:
    keywords:
      - stream
      - streaming
      - sse
      - websocket
      - real-time
      - realtime
      - chunk
      - flush
    paths:
      - "**/streaming*"
      - "**/stream*"
      - "**/*sse*"

  # Tool/function calling
  tools:
    keywords:
      - tool
      - function
      - function_call
      - tool_call
      - proxy
    paths:
      - "**/tools/**"
      - "**/*tool*"
      - "**/*proxy*"

  # API changes
  api:
    keywords:
      - api
      - endpoint
      - route
      - handler
      - rest
      - graphql
    paths:
      - "**/api/**"
      - "**/routes/**"
      - "**/handlers/**"
      - "**/routers/**"
"""


@click.group()
@click.version_option(version=__version__)
def main():
    """Prscope - Planning-first tool with upstream PR intelligence."""
    pass


def _warn_legacy_command(old: str, new: str) -> None:
    click.echo(
        f"⚠️  Deprecated command `{old}`. Please use `{new}` instead.",
        err=True,
    )


@main.group(name="upstream")
def upstream_group() -> None:
    """Upstream PR ingestion and evaluation commands."""


@main.command()
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
def init(force: bool):
    """Initialize Prscope in the current repository."""
    repo_root = get_repo_root()
    click.echo(f"Initializing Prscope in: {repo_root}")
    
    # Create .prscope directory
    prscope_dir = ensure_prscope_dir(repo_root)
    click.echo(f"  Created: {prscope_dir}")
    
    # Initialize database
    store = Store()
    click.echo(f"  Database: {store.db_path}")
    
    # Create sample config files
    config_path = repo_root / "prscope.yml"
    if not config_path.exists() or force:
        config_path.write_text(SAMPLE_CONFIG)
        click.echo(f"  Created: {config_path}")
    else:
        click.echo(f"  Skipped: {config_path} (already exists)")
    
    features_path = repo_root / "prscope.features.yml"
    if not features_path.exists() or force:
        features_path.write_text(SAMPLE_FEATURES)
        click.echo(f"  Created: {features_path}")
    else:
        click.echo(f"  Skipped: {features_path} (already exists)")
    
    # Add to .gitignore
    gitignore_path = repo_root / ".gitignore"
    gitignore_entry = "\n# Prscope\n.prscope/\n.env\n"
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if ".prscope" not in content:
            with open(gitignore_path, "a") as f:
                f.write(gitignore_entry)
            click.echo(f"  Updated: {gitignore_path}")
    else:
        gitignore_path.write_text(gitignore_entry)
        click.echo(f"  Created: {gitignore_path}")
    
    # Create env templates if not exists
    env_example_path = repo_root / "env.example"
    env_sample_path = repo_root / "env.sample"
    if not env_example_path.exists() or force:
        # Copy from package
        import importlib.resources
        try:
            # Try to read from installed package
            pkg_env_example = Path(__file__).parent.parent / "env.example"
            if pkg_env_example.exists():
                env_example_path.write_text(pkg_env_example.read_text())
                click.echo(f"  Created: {env_example_path}")
        except Exception:
            pass
    if not env_sample_path.exists() or force:
        pkg_env_sample = Path(__file__).parent.parent / "env.sample"
        if pkg_env_sample.exists():
            env_sample_path.write_text(pkg_env_sample.read_text())
            click.echo(f"  Created: {env_sample_path}")
    
    click.echo("\nPrscope initialized! Next steps:")
    click.echo("  1. Edit prscope.yml to add upstream repositories")
    click.echo("  2. Edit prscope.features.yml to define features")
    click.echo("  3. Set GITHUB_TOKEN environment variable")
    click.echo("  4. Run: prscope profile")
    click.echo("  5. Run: prscope sync")
    click.echo("  6. Run: prscope evaluate")
    click.echo("  7. Start planning: prscope plan chat")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def profile(as_json: bool):
    """Scan and profile the local codebase."""
    config = PrscopeConfig.load(get_repo_root())
    repo_root = config.get_local_repo_path()
    
    if not as_json:
        click.echo(f"Profiling repository: {repo_root}")
    
    # Build profile
    profile_data = build_profile(repo_root)
    profile_sha = hash_profile(profile_data)
    
    # Save to store
    store = Store()
    store.save_profile(
        repo_root=str(repo_root),
        profile_sha=profile_sha,
        profile_json=json.dumps(profile_data),
    )
    
    if as_json:
        click.echo(json.dumps({
            "profile_sha": profile_sha,
            "git_sha": profile_data.get("git_sha"),
            "total_files": profile_data["file_tree"]["total_files"],
            "extensions": profile_data["file_tree"]["extensions"],
            "import_stats": profile_data["import_stats"],
        }, indent=2))
    else:
        click.echo(f"\nProfile saved:")
        click.echo(f"  SHA: {profile_sha}")
        click.echo(f"  Git HEAD: {profile_data.get('git_sha', 'unknown')}")
        click.echo(f"  Total files: {profile_data['file_tree']['total_files']}")
        
        ext_counts = profile_data["file_tree"]["extensions"]
        top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:5]
        if top_exts:
            click.echo(f"  Top extensions: {', '.join(f'{e}({c})' for e, c in top_exts)}")
        
        stats = profile_data["import_stats"]
        click.echo(f"  Files analyzed: {stats['files_analyzed']}")
        click.echo(f"  Python imports: {stats['python_imports']}")
        click.echo(f"  JS imports: {stats['js_imports']}")


@main.command(hidden=True)
@click.option("--repo", help="Sync specific repository (owner/repo)")
@click.option("--state", default=None, help="PR state filter (merged, open, closed, all)")
@click.option("--max-prs", default=None, type=int, help="Maximum PRs to fetch")
@click.option("--since", default=None, help="Only fetch PRs after date (ISO date or 90d/6m/1y)")
@click.option("--full", is_flag=True, help="Full sync (ignore incremental watermark)")
@click.option("--no-files", is_flag=True, help="Skip fetching file lists")
def sync(
    repo: str | None,
    state: str | None,
    max_prs: int | None,
    since: str | None,
    full: bool,
    no_files: bool,
    warn_legacy: bool = True,
):
    """Fetch PRs from upstream repositories.
    
    By default, uses incremental sync (only PRs newer than last sync).
    First sync uses --since window (default: 90 days).
    
    Examples:
    
        prscope sync                    # Incremental (new PRs only)
        prscope sync --since 30d        # Last 30 days
        prscope sync --since 2024-01-01 # Since specific date
        prscope sync --full             # Ignore watermark, use --since window
    """
    if warn_legacy:
        _warn_legacy_command("prscope sync", "prscope upstream sync")

    repo_root = get_repo_root()
    config = PrscopeConfig.load(repo_root)
    
    if not config.upstream:
        click.echo("No upstream repositories configured.")
        click.echo("Add repositories to prscope.yml")
        sys.exit(1)
    
    # Filter repos if specified
    repos_to_sync = config.upstream
    if repo:
        repos_to_sync = [r for r in repos_to_sync if r.full_name == repo]
        if not repos_to_sync:
            click.echo(f"Repository not found in config: {repo}")
            sys.exit(1)
    
    # Initialize GitHub client
    client = GitHubClient()
    store = Store()
    
    # Use config defaults, allow CLI overrides
    default_state = state or config.sync.state
    default_max_prs = max_prs or config.sync.max_prs
    default_since = since or config.sync.since
    default_incremental = config.sync.incremental and not full
    default_fetch_files = config.sync.fetch_files and not no_files
    
    click.echo(f"Sync settings: state={default_state}, max_prs={default_max_prs}, since={default_since}")
    if default_incremental:
        click.echo("  Mode: incremental (only new PRs since last sync)")
    else:
        click.echo("  Mode: full (using date window)")
    
    total_new = 0
    total_updated = 0
    total_skipped = 0
    
    for upstream in repos_to_sync:
        click.echo(f"\n{'─' * 60}")
        click.echo(f"📦 Syncing: {upstream.full_name}")
        click.echo(f"{'─' * 60}")
        
        # Per-repo filters can override defaults
        pr_state = upstream.filters.get("state", default_state)
        
        # Progress callback for real-time updates
        last_stage = [None]
        def progress(stage: str, current: int, total: int, message: str):
            if stage == "fetch" and current == 0:
                click.echo(f"  ⏳ {message}")
            elif stage == "fetch" and current > 0:
                click.echo(f"  ✓ {message}")
            elif stage == "process":
                # Show progress every 5 PRs or for small batches
                if total <= 10 or current % 5 == 0 or current == total:
                    pct = int(current / total * 100) if total > 0 else 0
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    click.echo(f"\r  [{bar}] {current}/{total} PRs processed", nl=False)
                    if current == total:
                        click.echo()  # Newline at end
            elif stage == "done":
                pass  # Handled below
        
        try:
            new_count, updated_count, skipped_count = sync_repo_prs(
                client=client,
                store=store,
                repo_name=upstream.full_name,
                state=pr_state,
                max_prs=default_max_prs,
                fetch_files=default_fetch_files,
                since=default_since,
                incremental=default_incremental,
                progress_callback=progress,
            )
            
            click.echo(f"\n  ✅ Done: {new_count} new, {updated_count} updated, {skipped_count} unchanged")
            
            total_new += new_count
            total_updated += updated_count
            total_skipped += skipped_count
            
        except GitHubAPIError as e:
            click.echo(f"\n  ❌ Error: {e}", err=True)
            continue
    
    click.echo(f"\n{'═' * 60}")
    click.echo(f"📊 Sync Summary: {total_new} new, {total_updated} updated, {total_skipped} unchanged")
    click.echo(f"{'═' * 60}")


@main.command(hidden=True)
@click.option("--repo", help="Evaluate PRs from specific repository")
@click.option("--pr", "pr_number", type=int, help="Evaluate specific PR number")
@click.option("--batch", default=None, type=int, help="Max PRs to evaluate (limits LLM calls)")
@click.option("--force", is_flag=True, help="Re-evaluate even if already done")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def evaluate(
    repo: str | None,
    pr_number: int | None,
    batch: int | None,
    force: bool,
    as_json: bool,
    warn_legacy: bool = True,
):
    """Evaluate PRs for relevance using multi-stage analysis.
    
    Uses 3-stage pipeline:
    1. Rule-based filtering (keywords + paths)
    2. Semantic similarity (detect already-implemented)
    3. LLM analysis (final decision with reasoning)
    
    Examples:
    
        prscope evaluate               # Evaluate all unevaluated PRs
        prscope evaluate --batch 10    # Limit to 10 PRs (control LLM costs)
        prscope evaluate --pr 123      # Evaluate specific PR
        prscope evaluate --force       # Re-evaluate all
    """
    if warn_legacy:
        _warn_legacy_command("prscope evaluate", "prscope upstream evaluate")

    config = PrscopeConfig.load(get_repo_root())
    local_repo_path = config.get_local_repo_path()
    store = Store()
    
    if not config.features:
        click.echo("No features defined in prscope.features.yml")
        sys.exit(1)
    
    # Get current profile
    profile = store.get_latest_profile(str(local_repo_path))
    if not profile:
        click.echo("No profile found. Run: prscope profile")
        sys.exit(1)
    
    local_profile_sha = profile.profile_sha
    
    # Determine batch size
    batch_size = batch or config.sync.eval_batch_size
    
    if not as_json:
        click.echo(f"\n{'═' * 60}")
        click.echo(f"🔍 Evaluating PRs")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Profile: {local_profile_sha[:8]}")
        if config.llm.enabled:
            click.echo(f"  LLM: {config.llm.model}")
            click.echo(f"  Mode: semantic + AI analysis (3-stage pipeline)")
            click.echo(f"  Batch limit: {batch_size} PRs")
        else:
            click.echo("  Mode: rule-based only (enable LLM for better accuracy)")
    
    # Get PRs to evaluate
    prs = store.list_pull_requests()
    if repo:
        upstream = store.get_upstream_repo(repo)
        if upstream:
            prs = [pr for pr in prs if pr.repo_id == upstream.id]
    
    if pr_number:
        prs = [pr for pr in prs if pr.number == pr_number]
    
    # Count already-evaluated PRs upfront
    pending_prs = []
    already_evaluated = 0
    for pr in prs:
        if not force and pr.head_sha:
            if store.evaluation_exists(pr.id, local_profile_sha, pr.head_sha):
                already_evaluated += 1
                continue
        pending_prs.append(pr)
    
    total_pending = len(pending_prs)
    to_process = min(total_pending, batch_size)
    
    if not as_json:
        click.echo(f"{'─' * 60}")
        click.echo(f"  Total PRs in database: {len(prs)}")
        click.echo(f"  Already evaluated: {already_evaluated}")
        click.echo(f"  Pending evaluation: {total_pending}")
        click.echo(f"  Will process: {to_process} (batch limit: {batch_size})")
        click.echo(f"{'─' * 60}")
    
    evaluated = []
    batch_limited = 0
    
    # Load full profile data for LLM
    profile_data = profile.profile_data if config.llm.enabled else None
    
    for i, pr in enumerate(pending_prs[:batch_size], 1):
        if not as_json:
            pct = int(i / to_process * 100) if to_process > 0 else 0
            click.echo(f"\n  [{i}/{to_process}] PR #{pr.number}: {pr.title[:45]}...")
            click.echo(f"      ⏳ Analyzing...", nl=False)
        
        files = store.get_pr_files(pr.id)
        result = evaluate_pr(
            pr=pr,
            files=files,
            config=config,
            local_profile_sha=local_profile_sha,
            store=store,
            local_profile=profile_data,
            local_repo_path=local_repo_path,
        )
        
        if result:
            evaluated.append({
                "pr_id": pr.id,
                "number": pr.number,
                "title": pr.title,
                "rule_score": result.rule_score,
                "final_score": result.final_score,
                "final_decision": result.final_decision,
                "llm_decision": result.llm_decision,
                "llm_confidence": result.llm_confidence,
                "llm_reasoning": result.llm_reasoning,
                "matched_features": result.matched_features,
                "has_existing_impl": result.has_existing_implementation,
                "should_seed_plan": result.should_seed_plan(),
            })
            
            if not as_json:
                icon = {"relevant": "✅", "maybe": "⚠️", "skip": "❌"}.get(result.final_decision, "  ")
                conf_pct = int(result.llm_confidence * 100)
                click.echo(f"\r      {icon} {result.final_decision.upper()} ({conf_pct}% confidence)")
                if result.llm_reasoning:
                    click.echo(f"      💬 {result.llm_reasoning[:70]}")
    
    # Calculate remaining
    batch_limited = total_pending - len(evaluated)
    
    if as_json:
        click.echo(json.dumps({
            "evaluated": evaluated,
            "skipped": already_evaluated,
            "batch_limited": batch_limited,
            "profile_sha": local_profile_sha,
        }, indent=2))
    else:
        # Summary by decision
        implement = [r for r in evaluated if r["llm_decision"] == "implement"]
        partial = [r for r in evaluated if r["llm_decision"] == "partial"]
        skip_prs = [r for r in evaluated if r["llm_decision"] == "skip"]
        plan_seed_ready = [r for r in evaluated if r["should_seed_plan"]]
        
        click.echo(f"\n{'═' * 60}")
        click.echo(f"📊 Evaluation Summary")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Evaluated this run: {len(evaluated)}")
        click.echo(f"  Previously evaluated: {already_evaluated}")
        if batch_limited > 0:
            click.echo(f"  Remaining (batch limit): {batch_limited}")
        
        click.echo(f"\n  📈 Results:")
        click.echo(f"      ✅ Implement: {len(implement)}")
        click.echo(f"      ⚠️  Partial:   {len(partial)}")
        click.echo(f"      ❌ Skip:      {len(skip_prs)}")
        click.echo(f"      🗺️  Plan-seed ready: {len(plan_seed_ready)}")
        
        if implement:
            click.echo(f"\n{'─' * 60}")
            click.echo(f"✅ RECOMMENDED TO IMPLEMENT ({len(implement)})")
            click.echo(f"{'─' * 60}")
            for r in sorted(implement, key=lambda x: -x["llm_confidence"]):
                conf_pct = int(r['llm_confidence'] * 100)
                click.echo(f"  PR #{r['number']}: {r['title'][:50]}")
                click.echo(f"    Confidence: {conf_pct}%")
                if r['llm_reasoning']:
                    click.echo(f"    Reason: {r['llm_reasoning'][:65]}")
                click.echo()
        
        if partial:
            click.echo(f"\n{'─' * 60}")
            click.echo(f"⚠️  PARTIAL RELEVANCE ({len(partial)})")
            click.echo(f"{'─' * 60}")
            for r in partial[:5]:  # Limit to 5
                conf_pct = int(r['llm_confidence'] * 100)
                click.echo(f"  PR #{r['number']}: {r['title'][:50]} ({conf_pct}%)")
        
        if batch_limited > 0:
            click.echo(f"\n💡 Tip: Run `prscope evaluate` again to process {batch_limited} more PRs")
        
        if plan_seed_ready:
            click.echo(
                "\n💡 Tip: seed planning directly from upstream context with:\n"
                "   `prscope plan start --from-pr <owner/repo> <pr-number>`"
            )


@main.command(hidden=True)
@click.option("--limit", default=10, help="Number of PRs to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def digest(limit: int, as_json: bool, warn_legacy: bool = True):
    """Show summary of relevant PRs."""
    if warn_legacy:
        _warn_legacy_command("prscope digest", "prscope upstream digest")
    store = Store()
    
    # Get recent relevant evaluations
    evaluations = store.list_evaluations(decision="relevant", limit=limit)
    
    digest_data = []
    for eval in evaluations:
        pr = store.get_pull_request_by_id(eval.pr_id)
        if not pr:
            continue
        
        # Get repo name
        repos = store.list_upstream_repos()
        repo_name = None
        for r in repos:
            if r.id == pr.repo_id:
                repo_name = r.full_name
                break
        
        digest_data.append({
            "repo": repo_name,
            "number": pr.number,
            "title": pr.title,
            "author": pr.author,
            "score": eval.final_score,
            "matched_features": eval.matched_features,
            "url": pr.html_url,
            "evaluated_at": eval.created_at,
        })
    
    if as_json:
        click.echo(json.dumps(digest_data, indent=2))
    else:
        if not digest_data:
            click.echo("No relevant PRs found.")
            click.echo("Run: prscope sync && prscope evaluate")
            return
        
        click.echo(f"Top {len(digest_data)} Relevant PRs:\n")
        for item in digest_data:
            click.echo(f"[{item['repo']}] PR #{item['number']}: {item['title']}")
            click.echo(f"  Score: {item['score']:.2f} | Features: {', '.join(item['matched_features'])}")
            click.echo(f"  Author: {item['author']} | {item['url']}")
            click.echo()


@main.command(hidden=True)
@click.option("--limit", default=20, help="Number of evaluations to show")
@click.option("--decision", type=click.Choice(["relevant", "maybe", "skip"]), help="Filter by decision")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(limit: int, decision: str | None, as_json: bool, warn_legacy: bool = True):
    """View evaluation history."""
    if warn_legacy:
        _warn_legacy_command("prscope history", "prscope upstream history")
    store = Store()
    
    evaluations = store.list_evaluations(decision=decision, limit=limit)
    
    history_data = []
    for eval in evaluations:
        pr = store.get_pull_request_by_id(eval.pr_id)
        if not pr:
            continue
        
        history_data.append({
            "id": eval.id,
            "pr_number": pr.number,
            "pr_title": pr.title,
            "rule_score": eval.rule_score,
            "final_score": eval.final_score,
            "decision": eval.decision,
            "matched_features": eval.matched_features,
            "profile_sha": eval.local_profile_sha[:8],
            "pr_sha": eval.pr_head_sha[:8] if eval.pr_head_sha else "unknown",
            "created_at": eval.created_at,
        })
    
    if as_json:
        click.echo(json.dumps(history_data, indent=2))
    else:
        if not history_data:
            click.echo("No evaluation history found.")
            return
        
        click.echo(f"Evaluation History ({len(history_data)} entries):\n")
        for item in history_data:
            decision_icon = {"relevant": "✓", "maybe": "?", "skip": "✗"}.get(item["decision"], " ")
            click.echo(f"[{decision_icon}] #{item['id']} - PR #{item['pr_number']}: {item['pr_title'][:40]}")
            click.echo(f"    Score: {item['final_score']:.2f} | Profile: {item['profile_sha']} | PR: {item['pr_sha']}")
            click.echo(f"    Features: {', '.join(item['matched_features']) or 'none'}")
            click.echo(f"    Date: {item['created_at']}")
            click.echo()


@upstream_group.command("sync")
@click.option("--repo", help="Sync specific repository (owner/repo)")
@click.option("--state", default=None, help="PR state filter (merged, open, closed, all)")
@click.option("--max-prs", default=None, type=int, help="Maximum PRs to fetch")
@click.option("--since", default=None, help="Only fetch PRs after date (ISO date or 90d/6m/1y)")
@click.option("--full", is_flag=True, help="Full sync (ignore incremental watermark)")
@click.option("--no-files", is_flag=True, help="Skip fetching file lists")
def upstream_sync(
    repo: str | None,
    state: str | None,
    max_prs: int | None,
    since: str | None,
    full: bool,
    no_files: bool,
) -> None:
    """Fetch upstream PR metadata for planning seeds."""
    sync(
        repo=repo,
        state=state,
        max_prs=max_prs,
        since=since,
        full=full,
        no_files=no_files,
        warn_legacy=False,
    )


@upstream_group.command("evaluate")
@click.option("--repo", help="Evaluate PRs from specific repository")
@click.option("--pr", "pr_number", type=int, help="Evaluate specific PR number")
@click.option("--batch", default=None, type=int, help="Max PRs to evaluate (limits LLM calls)")
@click.option("--force", is_flag=True, help="Re-evaluate even if already done")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_evaluate(
    repo: str | None,
    pr_number: int | None,
    batch: int | None,
    force: bool,
    as_json: bool,
) -> None:
    """Score upstream PRs for planning relevance."""
    evaluate(
        repo=repo,
        pr_number=pr_number,
        batch=batch,
        force=force,
        as_json=as_json,
        warn_legacy=False,
    )


@upstream_group.command("digest")
@click.option("--limit", default=10, help="Number of PRs to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_digest(limit: int, as_json: bool) -> None:
    """Summarize high-signal upstream PR candidates."""
    digest(limit=limit, as_json=as_json, warn_legacy=False)


@upstream_group.command("history")
@click.option("--limit", default=20, help="Number of evaluations to show")
@click.option("--decision", type=click.Choice(["relevant", "maybe", "skip"]), help="Filter by decision")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_history(limit: int, decision: str | None, as_json: bool) -> None:
    """Show stored upstream evaluation history."""
    history(limit=limit, decision=decision, as_json=as_json, warn_legacy=False)


def _load_planning_runtime(repo_name: str | None) -> tuple[PrscopeConfig, Store, PlanningRuntime]:
    repo_root = get_repo_root()
    config = PrscopeConfig.load(repo_root)
    repo_profile = config.resolve_repo(repo_name, cwd=Path.cwd())
    store = Store()
    runtime = PlanningRuntime(store=store, config=config, repo=repo_profile)
    return config, store, runtime


def _format_age(path: Path) -> str:
    if not path.exists():
        return "never"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    delta = datetime.now(tz=timezone.utc) - mtime
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _run_tui(runtime: PlanningRuntime, session_id: str) -> None:
    try:
        from .tui import PlanningTUI
    except Exception as exc:  # pragma: no cover
        raise click.ClickException(
            "Textual UI is unavailable. Install planning dependencies and retry."
        ) from exc
    PlanningTUI(runtime, session_id).run()


@main.group(name="repos")
def repos_group() -> None:
    """Repository profile utilities."""


@repos_group.command("list")
def repos_list() -> None:
    """List configured repo profiles."""
    config = PrscopeConfig.load(get_repo_root())
    repos = config.list_repos()
    if not repos:
        click.echo("No repositories configured.")
        return
    click.echo("Name\tPath\tUpstreams\tMemory")
    for repo in repos:
        meta_path = repo.memory_dir / "_meta.json"
        click.echo(
            f"{repo.name}\t{repo.resolved_path}\t{len(repo.upstream)}\t{_format_age(meta_path)}"
        )


@main.group(name="plan")
def plan_group() -> None:
    """Interactive planning commands."""


@plan_group.command("start")
@click.argument("requirements", required=False)
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--from-pr", "from_pr", nargs=2, type=str, help="Seed from upstream repo + PR number")
@click.option("--no-tui", is_flag=True, help="Do not launch the interactive TUI")
@click.option("--rebuild-memory", is_flag=True, help="Force memory rebuild")
def plan_start(
    requirements: str | None,
    repo_name: str | None,
    from_pr: tuple[str, str] | None,
    no_tui: bool,
    rebuild_memory: bool,
) -> None:
    """Start a planning session from requirements, PR seed, or discovery chat."""
    _, _, runtime = _load_planning_runtime(repo_name)

    if from_pr:
        upstream_repo, pr_num_raw = from_pr
        session = asyncio.run(
            runtime.start_from_pr(
                upstream_repo=upstream_repo,
                pr_number=int(pr_num_raw),
                rebuild_memory=rebuild_memory,
            )
        )
    elif requirements:
        session = asyncio.run(
            runtime.start_from_requirements(
                requirements=requirements,
                rebuild_memory=rebuild_memory,
            )
        )
    else:
        session, opening = asyncio.run(runtime.start_from_chat(rebuild_memory=rebuild_memory))
        click.echo(opening)

    click.echo(f"Created planning session: {session.id}")
    if not no_tui:
        _run_tui(runtime, session.id)


@plan_group.command("chat")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--no-tui", is_flag=True, help="Do not launch the interactive TUI")
@click.option("--rebuild-memory", is_flag=True, help="Force memory rebuild")
def plan_chat(repo_name: str | None, no_tui: bool, rebuild_memory: bool) -> None:
    """Start chat-first discovery mode."""
    _, _, runtime = _load_planning_runtime(repo_name)
    session, opening = asyncio.run(runtime.start_from_chat(rebuild_memory=rebuild_memory))
    click.echo(opening)
    click.echo(f"Created planning session: {session.id}")
    if not no_tui:
        _run_tui(runtime, session.id)


@plan_group.command("resume")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Optional override for repo profile")
def plan_resume(session_id: str, repo_name: str | None) -> None:
    """Resume an existing planning session in the TUI."""
    config = PrscopeConfig.load(get_repo_root())
    store = Store()
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    effective_repo = repo_name or session.repo_name
    repo_profile = config.resolve_repo(effective_repo, cwd=Path.cwd())
    runtime = PlanningRuntime(store=store, config=config, repo=repo_profile)
    _run_tui(runtime, session_id)


@plan_group.command("list")
@click.option("--repo", "repo_name", help="Filter sessions by repo profile")
def plan_list(repo_name: str | None) -> None:
    """List planning sessions."""
    store = Store()
    sessions = store.list_planning_sessions(repo_name=repo_name, limit=200)
    if not sessions:
        click.echo("No planning sessions found.")
        return
    click.echo("Session ID\tRepo\tStatus\tRound\tTitle")
    for session in sessions:
        click.echo(
            f"{session.id}\t{session.repo_name}\t{session.status}\t"
            f"{session.current_round}\t{session.title}"
        )


@plan_group.command("export")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
def plan_export(session_id: str, repo_name: str | None) -> None:
    """Export PRD and RFC markdown files."""
    _, store, runtime = _load_planning_runtime(repo_name)
    if store.get_planning_session(session_id) is None:
        raise click.ClickException(f"Session not found: {session_id}")
    paths = runtime.export(session_id)
    click.echo(f"Exported:\n- {paths['prd']}\n- {paths['rfc']}\n- {paths['conversation']}")


@plan_group.command("diff")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--round", "round_number", type=int, default=None, help="Diff specific round against previous")
def plan_diff(session_id: str, repo_name: str | None, round_number: int | None) -> None:
    """Show plan diff as unified text."""
    _, store, runtime = _load_planning_runtime(repo_name)
    if store.get_planning_session(session_id) is None:
        raise click.ClickException(f"Session not found: {session_id}")
    diff_text = runtime.plan_diff(session_id, round_number=round_number)
    click.echo(diff_text or "No diff available.")


@plan_group.command("memory")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--rebuild", is_flag=True, help="Force rebuild memory blocks")
def plan_memory(repo_name: str | None, rebuild: bool) -> None:
    """Build/show memory blocks for active repo."""
    _, _, runtime = _load_planning_runtime(repo_name)
    profile = build_profile(runtime.repo.resolved_path)
    asyncio.run(runtime.memory.ensure_memory(profile, rebuild=rebuild))
    click.echo(f"Memory dir: {runtime.repo.memory_dir}")
    for block in ("architecture", "modules", "patterns", "entrypoints"):
        path = runtime.repo.memory_dir / f"{block}.md"
        click.echo(f"- {block}: {path} ({_format_age(path)})")


@plan_group.command("manifesto")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--edit", "open_editor", is_flag=True, help="Open manifesto in editor")
def plan_manifesto(repo_name: str | None, open_editor: bool) -> None:
    """Create or open repo manifesto file."""
    _, _, runtime = _load_planning_runtime(repo_name)
    path = runtime.memory.ensure_manifesto()
    if open_editor:
        click.edit(filename=str(path))
    else:
        click.echo(path)


@plan_group.command("validate")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
def plan_validate(session_id: str, repo_name: str | None) -> None:
    """Headless validation for CI."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    result = asyncio.run(runtime.validate_session(session_id))
    hard_count = len(result.hard_constraint_violations)
    major = result.major_issues_remaining
    if hard_count > 0 and major == 0:
        click.echo(
            "FAILED: hard constraint violations only "
            f"({', '.join(result.hard_constraint_violations)})"
        )
        sys.exit(2)
    if major > 0 or hard_count > 0:
        click.echo(
            f"FAILED: {major} major issues, hard violations: "
            f"{', '.join(result.hard_constraint_violations) or 'none'}"
        )
        sys.exit(1)
    click.echo("Plan validated - 0 major issues, 0 hard constraint violations")


@plan_group.command("status")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--pr-number", type=int, default=None, help="Merged PR number for drift detection")
def plan_status(session_id: str, repo_name: str | None, pr_number: int | None) -> None:
    """Compare planned file refs with merged PR files."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")

    chosen_pr = pr_number
    upstream_repo = None
    if chosen_pr is None and session.seed_ref and "#" in session.seed_ref:
        upstream_repo, pr_raw = session.seed_ref.rsplit("#", 1)
        if pr_raw.isdigit():
            chosen_pr = int(pr_raw)

    if chosen_pr is None:
        raise click.ClickException("Provide --pr-number or use a session seeded from an upstream PR.")
    if upstream_repo is None:
        raise click.ClickException("Session does not include upstream repo name in seed_ref.")

    upstream = store.get_upstream_repo(upstream_repo)
    if upstream is None:
        raise click.ClickException(f"Upstream repo not found in DB: {upstream_repo}")
    pr = store.get_pull_request(upstream.id, chosen_pr)
    if pr is None:
        raise click.ClickException(f"PR not found in DB: {upstream_repo}#{chosen_pr}")
    merged_files = {f.path for f in store.get_pr_files(pr.id)}
    drift = runtime.status(session_id, merged_pr_files=merged_files)

    click.echo(f"Plan Status: {upstream_repo}#{chosen_pr}")
    click.echo(f"Implemented as planned: {drift['implemented_count']} files")
    click.echo(f"Planned but not touched: {drift['missing_count']} files")
    for item in drift["missing"][:20]:
        click.echo(f"  - {item}")
    click.echo(f"Unplanned changes: {drift['unplanned_count']} files")
    for item in drift["unplanned"][:20]:
        click.echo(f"  - {item}")
    if drift["missing_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
