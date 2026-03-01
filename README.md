# Prscope

<p align="center">
  <img src="prscope-banner.png" alt="Prscope" width="100%" />
</p>

<h2 align="center">Planning-First PR Intelligence</h2>

<p align="center">
  <b>PLAN. REFINE. SHIP.</b>
  <br />
  <i>Turn upstream PRs into structured plans. Interactive TUI, adversarial Author/Critic rounds, and high-quality PRD/RFC outputs.</i>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/built_with-Python-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/status-Alpha-yellow.svg" alt="Status">
</p>

---

Planning-first PR intelligence for local repositories.

Prscope now treats upstream PR analysis as **input** to an interactive planning system that produces high-quality `PRD.md` and `RFC.md` outputs.

## What Changed

- Planning is now the primary workflow (`prscope plan ...`)
- Upstream sync/evaluate remain, but mainly to seed planning sessions
- Standalone legacy PRD generation command was removed (`prscope prd`)
- Multi-repo profiles, manifesto constraints, adversarial Author/Critic rounds, and headless validation are built in

## Core Workflow

1. `prscope profile` – profile local repo structure
2. `prscope upstream sync` – pull upstream PR metadata/files
3. `prscope upstream evaluate` – score upstream PRs for planning relevance
4. `prscope plan start --from-pr <owner/repo> <pr-number>` or `prscope plan chat`
5. Refine interactively in TUI, then `Ctrl+A` approve, `Ctrl+E` export
6. Optional CI checks: `prscope plan validate <session-id>`

## Planning Features

- Interactive Textual UI for Q&A and iterative refinement
- Adversarial planning loop (Author LLM ↔ Critic LLM)
- Structured memory blocks per repo:
  - `architecture.md`
  - `modules.md`
  - `patterns.md`
  - `entrypoints.md`
- Manifesto-driven constraints (`<repo>/.prscope/manifesto.md`)
- Convergence based on:
  - plan hash / diff
  - structural regression checks
  - critic `major_issues_remaining`
- Strict verified-reference mode:
  - `planning.require_verified_file_references: true`
- Drift detection:
  - `prscope plan status <session-id> --pr-number <N>`

## CLI Reference

### Upstream Input Commands

- `prscope init`
- `prscope profile`
- `prscope upstream sync`
- `prscope upstream evaluate`
- `prscope upstream digest`
- `prscope upstream history`

### Planning Commands

- `prscope repos list`
- `prscope plan start "requirements text" [--repo <name>]`
- `prscope plan start --from-pr <owner/repo> <number> [--repo <name>]`
- `prscope plan chat [--repo <name>]`
- `prscope plan resume <session-id>`
- `prscope plan list [--repo <name>]`
- `prscope plan memory [--repo <name>] [--rebuild]`
- `prscope plan manifesto [--repo <name>] [--edit]`
- `prscope plan diff <session-id> [--round N]`
- `prscope plan export <session-id>`
- `prscope plan validate <session-id>`
- `prscope plan status <session-id> --pr-number <N>`

## Configuration (`prscope.yml`)

```yaml
# Backward-compatible single repo
local_repo: ~/workspace/my-repo
upstream:
  - repo: openclaw/openclaw

sync:
  state: merged
  max_prs: 100
  fetch_files: true
  since: 90d
  incremental: true
  eval_batch_size: 25

llm:
  enabled: true
  model: gpt-4o
  temperature: 0.2
  max_tokens: 3000

planning:
  author_model: gpt-4o
  critic_model: claude-3-5-sonnet-20241022
  max_adversarial_rounds: 10
  convergence_threshold: 0.05
  output_dir: ./plans
  memory_concurrency: 2
  discovery_max_turns: 5
  seed_token_budget: 4000
  require_verified_file_references: false
  validate_temperature: 0.0
  validate_audit_log: true

# Optional multi-repo profiles
repos:
  my-repo:
    path: ~/workspace/my-repo
    upstream:
      - repo: openclaw/openclaw
```

## Manifesto Constraints

Create/edit with:

```bash
prscope plan manifesto --repo my-repo --edit
```

Machine-readable block example:

```yaml
extends: org-default # V1 stub, parsed but not resolved
constraints:
  - id: C-001
    text: "No synchronous I/O on the main thread"
    severity: hard
  - id: C-002
    text: "Prefer stdlib over new dependencies"
    severity: soft
    optional: true
```

## Installation

```bash
pip install -e ".[dev]"
```

## Environment Setup

```bash
cp env.sample .env
# or: cp env.example .env
```

Set `GITHUB_TOKEN` and any LLM provider keys needed by your configured planning models.

## Development

```bash
make test
make lint
make check
```
