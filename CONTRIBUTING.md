# Contributing to AumOS Drift Detector

Thank you for contributing to `aumos-drift-detector`. This service underpins ML model
monitoring for the AumOS Enterprise platform.

## License Notice

**IMPORTANT**: This project is licensed under Apache 2.0. By contributing, you agree
your contributions will be licensed under Apache 2.0.

**PROHIBITED**: Do NOT introduce any code, dependencies, or sub-components licensed
under AGPL, GPL, or any other copyleft license. AumOS Enterprise must remain free to
distribute and use commercially. This rule applies to statistical libraries too —
always check before adding a new dependency.

Approved licenses for dependencies: Apache 2.0, BSD, MIT, ISC.

## Development Setup

```bash
git clone https://github.com/aumos/aumos-drift-detector.git
cd aumos-drift-detector
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d
```

## Running Tests

```bash
make test          # Full test suite with coverage
make test-quick    # Fast run (stops on first failure)
make lint          # Ruff linter + format check
make typecheck     # mypy strict mode
make all           # lint + typecheck + test
```

## Code Standards

- **Type hints**: Required on ALL function signatures and return types
- **Docstrings**: Google style on all public classes and functions
- **Line length**: 120 characters maximum
- **Linter**: ruff (configured in pyproject.toml)
- **Type checker**: mypy in strict mode
- **Import order**: stdlib, then third-party, then local
- **No print()**: Use structlog for all logging
- **Async by default**: All I/O operations must be async

## Statistical Algorithm Contributions

When adding or modifying drift detection algorithms:
1. Algorithms must be **pure functions** — no side effects, no async
2. Include unit tests with **known distributions** to verify correctness
3. Document the algorithm, parameters, and interpretation of results
4. Cite the academic paper or reference implementation
5. Include a numerical example in the docstring

## Architecture Rules

- `core/` — no framework imports (no FastAPI, no SQLAlchemy, no Kafka)
- `adapters/` — concrete implementations only, depend on `core/` interfaces
- `api/` — thin layer: validate inputs, call services, return response schemas
- Statistical tests in `adapters/statistical_tests/` — pure Python + scipy/numpy
- Concept drift in `adapters/concept_drift/` — stateful detectors with `update()`/`detect()`

## Pull Request Process

1. Branch from `main`: `feature/your-feature` or `fix/your-bug`
2. Write tests alongside implementation (not after)
3. Ensure all CI checks pass: `make all`
4. Update CHANGELOG.md under `[Unreleased]`
5. Request review from at least one maintainer
6. Squash-merge with a descriptive commit message

## Commit Message Format

Follow Conventional Commits:

```
feat: add CUSUM algorithm to concept drift detectors
fix: correct PSI bin edge handling for empty reference buckets
refactor: extract drift threshold evaluation into separate helper
docs: add numerical example to KS test docstring
test: add chi-squared test with imbalanced categories
chore: bump scipy to 1.13.0
```

## Questions?

Open a GitHub issue or reach out in the `#aumos-mlops` Slack channel.
