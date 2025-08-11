PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test lint format ruff

install:
	$(PIP) install --upgrade pip || true
	$(PIP) install -e . --break-system-packages
	$(PIP) install -r requirements.txt --break-system-packages || true
	$(PIP) install pytest black isort ruff --break-system-packages

test:
	~/.local/bin/pytest -q || pytest -q

lint:
	~/.local/bin/ruff . || ruff .
	~/.local/bin/black --check . || black --check .
	~/.local/bin/isort --check-only . || isort --check-only .

format: ruff
	~/.local/bin/isort . --profile black || isort . --profile black
	~/.local/bin/black . || black .

ruff:
	~/.local/bin/ruff . --fix || ruff . --fix