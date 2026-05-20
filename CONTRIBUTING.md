# Contributing to FraudDetectionAI

## Setup

```bash
git clone https://github.com/atharvadevne123/FraudDetectionAI.git
cd FraudDetectionAI
pip install -r requirements.txt
pip install pre-commit && pre-commit install
```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes and write tests
3. Run lint: `make lint`
4. Run tests: `make test`
5. Push and open a pull request

## Code Standards

- All functions must have type annotations
- All public classes/functions must have docstrings
- Minimum 80% test coverage for new code
- Ruff lint must pass with zero errors

## Tests

```bash
pytest tests/ -v --cov=api --cov=pipeline --cov=models
```

## Commit Convention

```
type(scope): short description

Types: feat, fix, refactor, test, docs, chore, ci
```
