"""
Prscope - Monitor upstream GitHub PRs and generate implementation PRDs.

A CLI tool that:
1. Tracks upstream GitHub repositories for relevant PRs
2. Profiles your local codebase to understand its structure
3. Evaluates PR relevance using feature matching
4. Generates implementation PRDs as markdown

Usage:
    prscope init          # Initialize in current repo
    prscope profile       # Scan and profile local codebase
    prscope sync          # Fetch PRs from upstream repos
    prscope evaluate      # Score PRs for relevance
    prscope prd           # Generate PRD documents
    prscope digest        # Summary of relevant PRs
    prscope history       # View evaluation history
"""

__version__ = "0.1.0"
__author__ = "Prscope"
