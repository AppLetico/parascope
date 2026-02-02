# Parascope Makefile
# Common development tasks

# Use python3 on macOS, python elsewhere
PYTHON := $(shell command -v python3 2>/dev/null || echo python)

.PHONY: install dev test lint format check clean help

# Default target
help:
	@echo "Parascope Development Commands"
	@echo ""
	@echo "  make install      Install package (editable)"
	@echo "  make dev          Install with dev dependencies"
	@echo "  make test         Run unit tests"
	@echo "  make lint         Run linter (ruff check)"
	@echo "  make format       Format code (ruff format)"
	@echo "  make check        Run lint + format check + tests"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Parascope Workflow Commands"
	@echo ""
	@echo "  make run          Incremental: profile -> sync -> evaluate (1 batch) -> prd"
	@echo "  make run-full     Full: profile -> sync-full -> evaluate-all -> prd"
	@echo "  make sync         Sync PRs (incremental, only new since last sync)"
	@echo "  make sync-full    Sync PRs (full, ignores watermark)"
	@echo "  make evaluate     Evaluate PRs (one batch)"
	@echo "  make evaluate-all Evaluate all pending PRs (loops until done)"
	@echo "  make prd          Generate PRD documents"
	@echo "  make digest       Show PR digest"
	@echo ""

# Install package in editable mode
install:
	$(PYTHON) -m pip install -e .

# Install with dev dependencies
dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Run tests
test:
	pytest -v

# Run tests with coverage
test-cov:
	pytest --cov=parascope --cov-report=term-missing

# Lint code
lint:
	ruff check .

# Format code
format:
	ruff format .

# Check all (lint, format check, tests)
check: lint
	ruff format --check .
	pytest -q

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Initialize parascope in current directory
init:
	$(PYTHON) -m parascope.cli init

# Profile local codebase
profile:
	$(PYTHON) -m parascope.cli profile

# Sync upstream PRs (incremental - only new since last sync)
sync:
	$(PYTHON) -m parascope.cli sync

# Sync all PRs (full - ignores watermark, uses since window)
sync-full:
	$(PYTHON) -m parascope.cli sync --full

# Evaluate PRs (one batch)
evaluate:
	$(PYTHON) -m parascope.cli evaluate

# Evaluate all pending PRs (loops until done)
evaluate-all:
	@while $(PYTHON) -m parascope.cli evaluate 2>&1 | grep -q "Remaining"; do \
		echo "Processing next batch..."; \
	done
	@echo "All PRs evaluated"

# Generate PRDs
prd:
	$(PYTHON) -m parascope.cli prd

# Show digest
digest:
	$(PYTHON) -m parascope.cli digest

# Incremental workflow: profile -> sync (incremental) -> evaluate (one batch) -> prd
run: profile sync evaluate prd
	@echo "Parascope workflow complete"

# Full workflow: profile -> sync (full) -> evaluate (all) -> prd
run-full: profile sync-full evaluate-all prd
	@echo "Parascope full workflow complete"
