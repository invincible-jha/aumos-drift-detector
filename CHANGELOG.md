# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for `aumos-drift-detector` service
- `DriftMonitor` ORM model (`drf_monitors`) — per-model cron-scheduled drift monitors
- `DriftDetection` ORM model (`drf_detections`) — drift run results with scores and thresholds
- `DriftAlert` ORM model (`drf_alerts`) — severity-based alert records with acknowledgement
- `DriftDetectionService` — orchestrates statistical and concept drift checks
- `MonitoringService` — CRUD operations for drift monitors
- `AlertingService` — alert creation, severity classification, and acknowledgement
- `DriftMonitorRepository` and `DriftDetectionRepository` — SQLAlchemy async repositories
- `DriftEventPublisher` — Kafka publisher for `drift.detected`, `drift.retraining_required`, `drift.alert_raised`
- `KolmogorovSmirnovTest` — two-sample KS test for continuous feature drift
- `PopulationStabilityIndex` — PSI calculation with configurable binning
- `ChiSquaredTest` — chi-squared test for categorical feature drift
- `AdwinDetector` — ADaptive WINdowing concept drift detector
- `DdmDetector` — Drift Detection Method with warning and drift levels
- `EddmDetector` — Enhanced DDM with error rate window weighting
- `POST /api/v1/monitors` — create drift monitor
- `GET /api/v1/monitors` — list drift monitors (paginated)
- `GET /api/v1/monitors/{id}` — get monitor details
- `POST /api/v1/monitors/{id}/run` — trigger immediate drift check
- `GET /api/v1/detections` — list drift detections
- `GET /api/v1/detections/{id}` — get detection details
- `POST /api/v1/alerts/acknowledge/{id}` — acknowledge drift alert
- `GET /api/v1/dashboard` — drift dashboard summary
- `Settings` class with `AUMOS_DRIFT_` prefix for all configuration
- Multi-stage Dockerfile with non-root `aumos` user
- `docker-compose.dev.yml` with postgres, redis, and kafka services
- GitHub Actions CI pipeline with lint, typecheck, and matrix test jobs
- Standard `Makefile` targets
