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
# Run all tests
make test

# Run with coverage report
make test-cov

# Run a specific test file
pytest tests/test_api.py -v

# Run only tests matching a keyword
pytest tests/ -k "parametrize" -v
```

Coverage must not fall below 70% for core packages. Add at least three
parametrized cases for any new public function.

## Code Style

- Use Google-style docstrings for all public classes and functions.
- Type-annotate every parameter and return value.
- Keep functions under 40 lines; extract helpers when they grow.
- No `print()` calls — use `loguru.logger` throughout.

```bash
# Auto-fix lint and formatting
make lint-fix
make format
```

## Security Scan

```bash
make security
```

## Commit Convention

```
type(scope): short description

Types: feat, fix, refactor, test, docs, chore, ci
```

**Examples:**
- `feat(api): add /metrics/summary endpoint`
- `fix(model): handle NaN in feature matrix before SMOTE`
- `test(drift): add parametrized KS-threshold tests`
