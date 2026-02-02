"""
Parascope - Monitor upstream GitHub PRs and generate implementation PRDs.

A CLI tool that:
1. Tracks upstream GitHub repositories for relevant PRs
2. Profiles your local codebase to understand its structure
3. Evaluates PR relevance using feature matching
4. Generates implementation PRDs as markdown

Usage:
    parascope init          # Initialize in current repo
    parascope profile       # Scan and profile local codebase
    parascope sync          # Fetch PRs from upstream repos
    parascope evaluate      # Score PRs for relevance
    parascope prd           # Generate PRD documents
    parascope digest        # Summary of relevant PRs
    parascope history       # View evaluation history
"""

__version__ = "0.1.0"
__author__ = "Parascope"
