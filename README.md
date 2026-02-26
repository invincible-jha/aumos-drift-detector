# AumOS Drift Detector

> Statistical and concept drift detection with automated retraining triggers for AumOS Enterprise

[![CI](https://github.com/aumos/aumos-drift-detector/actions/workflows/ci.yml/badge.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()

## Overview

`aumos-drift-detector` is repository #23 in the AumOS Enterprise platform. It monitors
deployed ML models for data drift and concept drift, triggers automated retraining when
drift exceeds configurable thresholds, and surfaces drift analytics in the MLOps dashboard.

The service implements a statistically rigorous, multi-algorithm approach to drift
detection — from classical KS tests and PSI scores to stream-based ADWIN and DDM
detectors for real-time concept drift.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                aumos-drift-detector                │
│                                                   │
│  api/          FastAPI routes + Pydantic schemas  │
│  core/         Domain models + services           │
│  adapters/     Repositories + Kafka + algorithms  │
│    statistical_tests/  KS, PSI, Chi-squared       │
│    concept_drift/      ADWIN, DDM, EDDM           │
└───────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
  aumos-common            Kafka Topics
  (auth, DB, events)      drift.detected
                          drift.retraining_required
                          drift.alert_raised
```

## Features

- **Statistical drift detection** — Kolmogorov-Smirnov, Population Stability Index (PSI),
  and chi-squared tests over feature distributions
- **Concept drift detection** — ADWIN and DDM/EDDM algorithms for streaming prediction error
- **Drift monitors** — Cron-scheduled per-model monitors with configurable thresholds
- **Evidently AI integration** — Rich drift reports via the evidently library
- **Automated retraining triggers** — Kafka events consumed by `aumos-mlops-lifecycle`
- **Alert management** — Severity-based alerts with acknowledgement workflow
- **Multi-tenant** — Full tenant isolation via Row-Level Security (inherited from aumos-common)
- **Dashboard API** — Drift summary endpoint for the MLOps observability dashboard

## Statistical Methods

| Test | Feature Type | Use Case |
|------|-------------|---------|
| Kolmogorov-Smirnov | Continuous | Distribution shift in numeric features |
| Population Stability Index | Continuous (binned) | Population shift, credit-scoring style |
| Chi-squared | Categorical | Category frequency drift |
| ADWIN | Streaming error rate | Concept drift in online models |
| DDM / EDDM | Streaming error rate | Warning and drift level detection |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/monitors` | Create a drift monitor |
| GET | `/api/v1/monitors` | List all monitors |
| GET | `/api/v1/monitors/{id}` | Get monitor details |
| POST | `/api/v1/monitors/{id}/run` | Trigger immediate drift check |
| GET | `/api/v1/detections` | List drift detections |
| GET | `/api/v1/detections/{id}` | Get detection details |
| POST | `/api/v1/alerts/acknowledge/{id}` | Acknowledge a drift alert |
| GET | `/api/v1/dashboard` | Drift dashboard summary |

## Quick Start

```bash
git clone https://github.com/aumos/aumos-drift-detector.git
cd aumos-drift-detector
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d
pip install -e ".[dev]"
uvicorn aumos_drift_detector.main:app --reload
```

## Development

```bash
make install       # pip install -e ".[dev]"
make test          # full test suite with coverage
make test-quick    # fast run, stop on first failure
make lint          # ruff check + format check
make format        # ruff format + autofix
make typecheck     # mypy strict mode
make all           # lint + typecheck + test
```

## Environment Variables

All variables use the `AUMOS_DRIFT_` prefix. See `.env.example` for the full list.

Key settings:

```bash
AUMOS_DRIFT_DATABASE__URL=postgresql+asyncpg://aumos:aumos_dev@localhost:5432/aumos
AUMOS_DRIFT_KAFKA__BOOTSTRAP_SERVERS=localhost:9092
AUMOS_DRIFT_DRIFT_THRESHOLD_KS=0.05
AUMOS_DRIFT_DRIFT_THRESHOLD_PSI=0.2
AUMOS_DRIFT_RETRAINING_TRIGGER_ENABLED=true
```

## Database Schema

Tables are prefixed with `drf_`:

- `drf_monitors` — Drift monitor configurations (cron schedule, thresholds, features)
- `drf_detections` — Drift detection results (score, threshold, is_drifted, details)
- `drf_alerts` — Alert records with severity and acknowledgement tracking

## Integration with AumOS MLOps

When a drift detection score crosses the configured threshold, this service:
1. Creates a `DriftDetection` record with `is_drifted=True`
2. Creates a `DriftAlert` with computed severity
3. Publishes `drift.retraining_required` to Kafka
4. `aumos-mlops-lifecycle` consumes the event and schedules a retraining job

## License

Apache 2.0 — see [LICENSE](./LICENSE)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). All contributions must maintain Apache 2.0
compatibility — no AGPL or GPL dependencies.

## Security

See [SECURITY.md](./SECURITY.md) for the vulnerability reporting policy.
