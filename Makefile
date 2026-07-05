# Task runner for the local quality workflow — and the exact commands the
# future GitHub Actions pipeline (backlog C2) will invoke, so CI and local
# runs cannot drift. On Windows machines without `make`, run the underlying
# commands directly (they are documented in README.md and .ai/AI_AGENT.md).

PYTHON ?= python

.PHONY: lint test coverage quality smoke all

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests/

coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-report=html

## quality: lint + full test suite (the pre-commit bar)
quality: lint test

## smoke: end-to-end serving workflow on a synthetic stack (no data/ needed)
smoke:
	$(PYTHON) scripts/smoke.py

all: quality smoke
