# Task runner for the local quality workflow — and the exact commands the
# GitHub Actions pipeline invokes, so CI and local
# runs cannot drift. On Windows machines without `make`, run the underlying
# commands directly (they are documented in README.md).

PYTHON ?= python

.PHONY: lint test coverage coverage-gate audit secrets quality smoke dev all

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests/

coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-report=html

## coverage-gate: same as CI's src/-only enforcement — reuses
## the .coverage data `coverage` just wrote, no second test run.
coverage-gate: coverage
	$(PYTHON) -m coverage report --include="src/*" --fail-under=80

## audit: dependency vulnerability scan — same command CI
## runs, report-only (see ci.yml's own comment for why it isn't a hard gate
## yet). Requires the [dev] extra (pip-audit).
audit:
	$(PYTHON) -m pip_audit

## secrets: local secret scan — requires the gitleaks binary
## on PATH (not a pip package; https://github.com/gitleaks/gitleaks/releases).
## CI runs the equivalent via gitleaks/gitleaks-action.
secrets:
	gitleaks detect --source . -v

## quality: lint + full test suite (the pre-commit bar)
quality: lint test

## smoke: end-to-end serving workflow on a synthetic stack (no data/ needed)
smoke:
	$(PYTHON) scripts/smoke.py

## dev: single-command local dev — starts the API if needed, then the dashboard
dev:
	$(PYTHON) scripts/dev.py

all: quality smoke
