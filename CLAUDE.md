# CLAUDE.md — AumOS Drift Detector

## Project Context
`aumos-drift-detector` is repo #23 in the AumOS Enterprise platform. It provides
statistical and concept drift detection for deployed ML models, plus automated
retraining triggers that feed into the MLOps lifecycle.

## Position in the AumOS Graph

```
Upstream producers:
  aumos-common      → base models, auth, database, events, observability
  aumos-proto       → Protobuf schemas for drift events
  aumos-mlops-lifecycle → model deployment metadata and retraining workflow

Downstream consumers:
  aumos-mlops-lifecycle → receives drift.detected events to trigger retraining
  aumos-observability   → receives drift metrics for dashboards
```

## Domain Responsibilities
- **Statistical drift**: KS test, PSI, chi-squared over feature distributions
- **Concept drift**: ADWIN, DDM, EDDM algorithms on streaming prediction error
- **Monitor scheduling**: cron-driven periodic drift checks per model
- **Alert management**: severity-based alerting with acknowledgement workflow
- **Retraining triggers**: publish `drift.retraining_required` Kafka events

## Table Prefix
All ORM tables use the `drf_` prefix:
- `drf_monitors`    — DriftMonitor configurations
- `drf_detections`  — DriftDetection results per run
- `drf_alerts`      — DriftAlert notifications

## Tech Stack
- Python 3.11+
- FastAPI 0.110+ with async lifespan
- SQLAlchemy 2.0+ asyncio (asyncpg driver)
- Pydantic 2.6+ for all models and settings
- Apache Kafka via confluent-kafka (drift events)
- scipy + scikit-learn for statistical tests
- evidently for data quality and drift reports
- pytest 8.0+ with pytest-asyncio

## Settings Prefix
All environment variables use the `AUMOS_DRIFT_` prefix (see `.env.example`).

## Coding Standards
- Type hints REQUIRED on ALL function signatures and return types
- Docstrings: Google style on all public classes and functions
- Max line length: 120 characters
- Linter: ruff (configured in pyproject.toml)
- Type checker: mypy strict mode
- Import order: stdlib → third-party → local
- Async by default for all I/O operations
- No raw SQL — always SQLAlchemy ORM
- No print() — always structlog

## Architecture Patterns
- Hexagonal: core/ has no framework imports; adapters/ bridges to infra
- Dependency injection for all services (constructor injection)
- Interfaces (Protocols) in core/interfaces.py consumed by services
- Concrete implementations in adapters/ (repositories, kafka, tests)
- Statistical test algorithms are pure functions (no async, no side effects)
- Concept drift detectors are stateful objects with update/detect interface

## Statistical Tests
- `ks_test.py`: Kolmogorov-Smirnov two-sample test for continuous features
- `psi.py`: Population Stability Index for binned distribution comparison
- `chi_squared.py`: Chi-squared test for categorical features

## Concept Drift Detectors
- `adwin.py`: ADaptive WINdowing — stream-based change detection
- `ddm.py`: Drift Detection Method and its enhanced variant EDDM

## Kafka Topics Published
- `drift.detected` — any drift found (statistical or concept)
- `drift.retraining_required` — threshold crossed, retraining needed
- `drift.alert_raised` — alert created for a drift detection

## Testing Requirements
- Minimum 80% coverage (statistical algorithms have deterministic outputs)
- Unit tests for all statistical algorithms with known distributions
- Mock Kafka/DB in unit tests; use testcontainers for integration tests
- Test file naming: tests/{module}/test_{submodule}.py

## Key Dependencies
- Do NOT introduce AGPL/GPL dependencies — Apache 2.0 only
- evidently is Apache 2.0 licensed — safe to use
- scipy is BSD licensed — safe to use
- scikit-learn is BSD licensed — safe to use
