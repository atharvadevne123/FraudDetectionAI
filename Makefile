.PHONY: install install-dev test test-cov lint lint-fix type-check \
        run train evaluate docker-build docker-up clean check-data \
        format format-check security generate-data

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest pytest-cov ruff pre-commit

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short \
	  --cov=api --cov=pipeline --cov=models --cov=utils --cov=config \
	  --cov-report=term-missing --cov-report=html

lint:
	ruff check . --select E,F,W,I --ignore E501,E402

lint-fix:
	ruff check . --select E,F,W,I --ignore E501,E402 --fix

type-check:
	python -m mypy api/ models/ pipeline/ utils/ --ignore-missing-imports --no-error-summary

run:
	python api/app.py

train:
	python scripts/train.py

evaluate:
	@echo "Usage: make evaluate INPUT=data/test.csv"
	python scripts/evaluate.py --test-data $(INPUT)

check-data:
	@echo "Usage: make check-data INPUT=data/raw/transactions.csv"
	python scripts/check_data_quality.py --input $(INPUT)

docker-build:
	docker build -f docker/Dockerfile.api -t fraud-detection-api:latest .

docker-up:
	docker compose up --build

format:
	ruff format .

format-check:
	ruff format --check .

security:
	pip install bandit -q && bandit -r api/ models/ pipeline/ utils/ -ll --exit-zero

generate-data:
	python scripts/generate_synthetic_data.py --rows 50000 --out data/raw/transactions.parquet

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml dist build *.egg-info
