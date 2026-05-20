.PHONY: install test lint run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=api --cov=pipeline --cov=models --cov-report=term-missing

lint:
	ruff check api/ pipeline/ models/ tests/ --ignore E501,E402

lint-fix:
	ruff check api/ pipeline/ models/ tests/ --ignore E501,E402 --fix

run:
	python api/app.py

docker-build:
	docker build -f docker/Dockerfile.api -t fraud-detection-api:latest .

docker-up:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
