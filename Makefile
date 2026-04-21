PYTHON ?= python3

.PHONY: test admin router web demo validate

test:
	uv run --package shardon-admin-api pytest

admin:
	uv run --package shardon-admin-api shardon-admin-api

router:
	uv run --package shardon-router-api shardon-router-api

web:
	npm --workspace apps/admin_web run dev

demo:
	./scripts/run_demo.sh

validate:
	uv run --package shardon-admin-api python scripts/validate_config.py

