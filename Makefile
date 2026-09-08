BUILD_DIR ?= build
BUILD_TYPE ?= Release
PYTHON ?= python3

.PHONY: build test lint fmt dev clean docs gate gate-update abi-gate abi-gate-update

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

test: build
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check chronovec/ tests/
	$(PYTHON) -m ruff format --check chronovec/ tests/

fmt:
	$(PYTHON) -m ruff check --fix chronovec/ tests/
	$(PYTHON) -m ruff format chronovec/ tests/

typecheck:
	$(PYTHON) -m mypy chronovec/ --ignore-missing-imports

dev: build
	$(PYTHON) -m pip install -e ".[dev]"

gate: build
	$(PYTHON) benchmarks/regression_gate.py

gate-update: build
	$(PYTHON) benchmarks/regression_gate.py --update

abi-gate:
	$(PYTHON) tools/abi_gate.py

abi-gate-update:
	$(PYTHON) tools/abi_gate.py --update

docs:
	@echo "Docs are plain markdown in docs/. Open any .md file to read them."
	@echo "To serve locally: python -m http.server --directory docs 8080"

clean:
	rm -rf $(BUILD_DIR) dist/ *.egg-info/ **/__pycache__/
