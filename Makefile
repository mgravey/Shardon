PYTHON ?= python3

.PHONY: setup test admin router web demo validate dev

setup:
	./scripts/bootstrap.sh

test:
	uv run --all-packages --group dev pytest

admin:
	uv run --package shardon-admin-api shardon-admin-api

router:
	uv run --package shardon-router-api shardon-router-api

web:
	npm --workspace apps/admin_web run dev

demo:
	./scripts/run_demo.sh

dev:
	./scripts/run-local.sh

validate:
	uv run --package shardon-admin-api python scripts/validate_config.py
