.PHONY: install test test-quick lint format typecheck clean all docker-build docker-up docker-down

all: lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v \
		--cov=aumos_drift_detector \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=80

test-quick:
	pytest tests/ -x -q --no-header

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/aumos_drift_detector/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info coverage.xml .coverage

docker-build:
	docker build -t aumos-drift-detector:dev .

docker-up:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker-compose.dev.yml down

dev:
	uvicorn aumos_drift_detector.main:app --reload --host 0.0.0.0 --port 8000
