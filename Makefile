# Parascope Makefile
# Common development tasks

# Use python3 on macOS, python elsewhere
PYTHON := $(shell command -v python3 2>/dev/null || echo python)

.PHONY: install dev test lint format check clean help

# Default target
help:
	@echo "Parascope Development Commands"
	@echo ""
	@echo "  make install    Install package (editable)"
	@echo "  make dev        Install with dev dependencies"
	@echo "  make test       Run unit tests"
	@echo "  make lint       Run linter (ruff check)"
	@echo "  make format     Format code (ruff format)"
	@echo "  make check      Run lint + format check + tests"
	@echo "  make clean      Remove build artifacts"
	@echo "  make init       Initialize parascope in current dir"
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

# Sync upstream PRs
sync:
	$(PYTHON) -m parascope.cli sync

# Evaluate PRs
evaluate:
	$(PYTHON) -m parascope.cli evaluate

# Generate PRDs
prd:
	$(PYTHON) -m parascope.cli prd

# Show digest
digest:
	$(PYTHON) -m parascope.cli digest

# Full workflow: profile -> sync -> evaluate -> prd
run: profile sync evaluate prd
	@echo "Parascope workflow complete"
