.PHONY: help check test smoke lint type fmt clean

help:
	@echo "check  - ruff + mypy --strict over src/ and tests/"
	@echo "test   - pytest, CPU only, no GPU / network / HF token"
	@echo "smoke  - dry-run every stage config on both profiles"
	@echo "fmt    - ruff format + import sort"

check: lint type

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

type:
	uv run mypy

test:
	uv run pytest

# Deselects anything needing a GPU or the downloaded release files, which is what
# CI runs. Kept separate from `test` so local runs still exercise the real data.
test-ci:
	uv run pytest -m "not gpu and not dataset"

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Resolution is a pure function, so a config typo surfaces here rather than
# thirty minutes into a run.
smoke:
	@for stage in profile-data prepare-sft train-sft evaluate serve bench sweep; do \
		for profile in l4 a100; do \
			uv run smolqwen $$stage --profile $$profile --dry-run > /dev/null \
				|| exit 1; \
		done; \
	done
	uv run smolqwen probe --no-write > /dev/null
	@echo "smoke ok"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
