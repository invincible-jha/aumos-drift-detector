"""Protocol (interface) definitions for the Drift Detector service.

Defines abstract contracts between the service layer and adapters,
enabling dependency injection and test doubles without coupling to
concrete implementations.
"""

import uuid
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aumos_drift_detector.core.models import DriftAlert, DriftDetection, DriftMonitor


@runtime_checkable
class IDriftMonitorRepository(Protocol):
    """Contract for DriftMonitor persistence operations."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        name: str,
        feature_columns: list[str],
        reference_data_uri: str,
        schedule_cron: str | None,
        thresholds: dict,
    ) -> DriftMonitor:
        """Create a new drift monitor configuration.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Target model UUID from aumos-model-registry.
            name: Human-readable monitor name.
            feature_columns: List of feature column names to watch.
            reference_data_uri: S3/MinIO URI to reference dataset.
            schedule_cron: Cron expression for scheduled runs; None = manual only.
            thresholds: Per-test threshold overrides dict.

        Returns:
            Newly created DriftMonitor ORM instance.
        """
        ...

    async def get_by_id(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftMonitor | None:
        """Fetch a monitor by its UUID within a tenant scope.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Requesting tenant (enforces isolation).

        Returns:
            DriftMonitor or None if not found.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        status: str | None,
    ) -> tuple[list[DriftMonitor], int]:
        """Return paginated monitors for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            status: Optional status filter (active | paused | disabled).

        Returns:
            Tuple of (monitors, total_count).
        """
        ...

    async def list_active_scheduled(self) -> list[DriftMonitor]:
        """Return all active monitors that have a schedule_cron configured.

        Used by the scheduler to find monitors that need to be run.

        Returns:
            List of active, cron-scheduled DriftMonitor instances.
        """
        ...

    async def update_status(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID, status: str
    ) -> DriftMonitor:
        """Update the operational status of a monitor.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.
            status: New status value (active | paused | disabled).

        Returns:
            Updated DriftMonitor.
        """
        ...

    async def delete(self, monitor_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Delete a monitor and cascade to its detection history.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.
        """
        ...


@runtime_checkable
class IDriftDetectionRepository(Protocol):
    """Contract for DriftDetection persistence operations."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        detection_type: str,
        test_name: str,
        score: float,
        threshold: float,
        is_drifted: bool,
        details: dict,
    ) -> DriftDetection:
        """Persist a drift detection result.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Parent monitor UUID.
            detection_type: 'statistical' or 'concept'.
            test_name: Algorithm name (ks, psi, chi2, adwin, ddm, eddm).
            score: Aggregate drift score (p-value or PSI or error rate).
            threshold: Threshold the score was compared against.
            is_drifted: True if drift was detected.
            details: Per-feature breakdown and additional metadata.

        Returns:
            Newly created DriftDetection ORM instance.
        """
        ...

    async def get_by_id(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftDetection | None:
        """Fetch a detection by UUID within a tenant scope.

        Args:
            detection_id: Detection UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftDetection or None.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        monitor_id: uuid.UUID | None,
        is_drifted: bool | None,
    ) -> tuple[list[DriftDetection], int]:
        """Return paginated detections for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            monitor_id: Optional filter by monitor.
            is_drifted: Optional filter (True = drifted only, False = clean only).

        Returns:
            Tuple of (detections, total_count).
        """
        ...

    async def get_drift_summary(
        self, tenant_id: uuid.UUID, days: int
    ) -> dict:
        """Aggregate drift statistics for the dashboard.

        Args:
            tenant_id: Requesting tenant.
            days: Rolling window in days.

        Returns:
            Dict with total_checks, drifted_count, monitors_with_drift, etc.
        """
        ...


@runtime_checkable
class IDriftAlertRepository(Protocol):
    """Contract for DriftAlert persistence operations."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        detection_id: uuid.UUID,
        severity: str,
        channel: str,
        message: str,
    ) -> DriftAlert:
        """Create a new drift alert.

        Args:
            tenant_id: Owning tenant.
            detection_id: Parent detection UUID.
            severity: Alert severity (info | warning | critical).
            channel: Notification channel.
            message: Human-readable alert message.

        Returns:
            Newly created DriftAlert ORM instance.
        """
        ...

    async def get_by_id(
        self, alert_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftAlert | None:
        """Fetch an alert by UUID within a tenant scope.

        Args:
            alert_id: Alert UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftAlert or None.
        """
        ...

    async def acknowledge(
        self,
        alert_id: uuid.UUID,
        tenant_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> DriftAlert:
        """Mark an alert as acknowledged by an operator.

        Args:
            alert_id: Alert UUID.
            tenant_id: Owning tenant.
            acknowledged_by: UUID of the acknowledging user.

        Returns:
            Updated DriftAlert with acknowledged=True.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        acknowledged: bool | None,
        severity: str | None,
    ) -> tuple[list[DriftAlert], int]:
        """Return paginated alerts for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            acknowledged: Optional filter by acknowledgement state.
            severity: Optional filter by severity level.

        Returns:
            Tuple of (alerts, total_count).
        """
        ...


@runtime_checkable
class IDriftEventPublisher(Protocol):
    """Contract for publishing drift events to Kafka."""

    async def publish_drift_detected(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        detection_id: uuid.UUID,
        test_name: str,
        score: float,
        is_drifted: bool,
    ) -> None:
        """Publish a drift.detected event.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Monitor that triggered the detection.
            detection_id: Detection record UUID.
            test_name: Algorithm that detected drift.
            score: Drift score value.
            is_drifted: Whether drift threshold was crossed.
        """
        ...

    async def publish_retraining_required(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID,
        reason: str,
    ) -> None:
        """Publish a drift.retraining_required event.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Monitor that flagged the need for retraining.
            model_id: Model UUID to retrain.
            detection_id: Detection that triggered this retraining request.
            reason: Human-readable reason describing which test failed.
        """
        ...

    async def publish_alert_raised(
        self,
        tenant_id: uuid.UUID,
        alert_id: uuid.UUID,
        severity: str,
        message: str,
    ) -> None:
        """Publish a drift.alert_raised event.

        Args:
            tenant_id: Owning tenant.
            alert_id: Alert record UUID.
            severity: Alert severity level.
            message: Alert message text.
        """
        ...


@runtime_checkable
class IFeatureImportanceAnalyser(Protocol):
    """Contract for SHAP/LIME-based drift feature importance analysis."""

    def rank_features_by_drift(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        drift_scores: dict[str, float],
    ) -> list[Any]:
        """Rank features by their drift contribution.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.
            drift_scores: Pre-computed drift score per feature.

        Returns:
            List of FeatureImportanceResult ordered by importance_rank.
        """
        ...

    def generate_waterfall_data(self, feature_rankings: list[Any]) -> Any:
        """Generate SHAP waterfall chart data.

        Args:
            feature_rankings: Ranked feature importance results.

        Returns:
            WaterfallChartData for visualisation.
        """
        ...

    def record_historical_importance(
        self,
        monitor_id: uuid.UUID,
        feature_rankings: list[Any],
    ) -> None:
        """Append a timestamped importance snapshot to history.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_rankings: Ranked feature importance results.
        """
        ...


@runtime_checkable
class IPerformanceMonitor(Protocol):
    """Contract for model performance monitoring and degradation detection."""

    def add_observation(
        self,
        y_true: float | None,
        y_pred: float,
        y_score: float | None,
        segment_key: str | None,
        timestamp: datetime | None,
    ) -> str:
        """Add a prediction observation to the rolling window.

        Args:
            y_true: Ground truth label (None = delayed label scenario).
            y_pred: Model's predicted label or value.
            y_score: Predicted probability or confidence score.
            segment_key: Optional segment identifier.
            timestamp: UTC timestamp; defaults to now.

        Returns:
            Observation ID string.
        """
        ...

    def compute_window_metrics(self) -> Any:
        """Compute performance metrics over the current rolling window.

        Returns:
            PerformanceMetrics with accuracy, F1, AUC, RMSE, and MAE.
        """
        ...

    def detect_degradation(self, metrics: Any) -> list[Any]:
        """Detect performance degradation against the configured baseline.

        Args:
            metrics: PerformanceMetrics from compute_window_metrics.

        Returns:
            List of DegradationAlert for any metrics crossing thresholds.
        """
        ...


@runtime_checkable
class IAlertSystem(Protocol):
    """Contract for configurable multi-channel drift alerting."""

    async def evaluate_and_dispatch(
        self,
        monitor_id: uuid.UUID,
        metric_values: dict[str, float],
        model_id: uuid.UUID | None,
    ) -> list[Any]:
        """Evaluate rules and dispatch alerts for triggered thresholds.

        Args:
            monitor_id: UUID of the drift monitor.
            metric_values: Dict of metric_name to current value.
            model_id: Optional model UUID for rule scoping.

        Returns:
            List of DispatchedAlert for newly fired alerts.
        """
        ...

    def acknowledge_alert(
        self,
        alert_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> bool:
        """Acknowledge an alert by an operator.

        Args:
            alert_id: UUID of the alert.
            acknowledged_by: UUID of the acknowledging user.

        Returns:
            True if acknowledged successfully.
        """
        ...


@runtime_checkable
class IRetrainTrigger(Protocol):
    """Contract for automated model retraining trigger evaluation."""

    async def evaluate_drift_trigger(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID,
        drift_score: float,
        test_name: str,
    ) -> Any:
        """Evaluate whether drift warrants triggering model retraining.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Drift monitor UUID.
            model_id: Target model UUID.
            detection_id: DriftDetection UUID.
            drift_score: Aggregate drift score.
            test_name: Statistical test name.

        Returns:
            TriggerEvent recording the evaluation outcome.
        """
        ...

    async def evaluate_performance_trigger(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        accuracy: float | None,
        rmse: float | None,
    ) -> Any:
        """Evaluate whether performance degradation warrants retraining.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Monitor UUID.
            model_id: Model UUID.
            accuracy: Current accuracy (None = not applicable).
            rmse: Current RMSE (None = not applicable).

        Returns:
            TriggerEvent recording the evaluation outcome.
        """
        ...


@runtime_checkable
class IBaselineManager(Protocol):
    """Contract for reference distribution baseline management."""

    def capture_baseline(
        self,
        model_id: uuid.UUID,
        model_version: str,
        feature_data: dict[str, np.ndarray],
        data_uri: str,
        tags: dict[str, str] | None,
        window_days: int,
        activate: bool,
    ) -> Any:
        """Capture a new versioned baseline from feature data arrays.

        Args:
            model_id: UUID of the model.
            model_version: Model version string.
            feature_data: Dict of feature_name to numpy array.
            data_uri: S3/MinIO URI of the source dataset.
            tags: Optional metadata tags.
            window_days: Window size in days (0 = full dataset).
            activate: Whether to activate this baseline immediately.

        Returns:
            Newly created BaselineVersion.
        """
        ...

    def get_active_baseline(self, model_id: uuid.UUID) -> Any | None:
        """Return the currently active baseline for a model.

        Args:
            model_id: UUID of the model.

        Returns:
            Active BaselineVersion or None.
        """
        ...

    def compare_baselines(
        self,
        old_baseline_id: uuid.UUID,
        new_baseline_id: uuid.UUID,
        significance_threshold: float,
    ) -> Any:
        """Compare two baseline versions.

        Args:
            old_baseline_id: UUID of the older baseline.
            new_baseline_id: UUID of the newer baseline.
            significance_threshold: Fractional shift threshold for flagging changes.

        Returns:
            BaselineComparison result.
        """
        ...


@runtime_checkable
class IDriftTrendAnalyzer(Protocol):
    """Contract for historical drift trend analysis."""

    def record_drift_score(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        score: float,
        test_name: str,
        is_drifted: bool,
        timestamp: datetime | None,
    ) -> None:
        """Append a drift score observation to the time series.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Name of the feature.
            score: Drift score value.
            test_name: Name of the statistical test.
            is_drifted: Whether the score crossed the threshold.
            timestamp: UTC timestamp; defaults to now.
        """
        ...

    def analyse_feature_trend(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        window_size: int | None,
    ) -> Any:
        """Analyse drift score trend for a feature.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to analyse.
            window_size: Optional number of most-recent observations.

        Returns:
            TrendAnalysis with direction, slope, and change points.
        """
        ...

    def forecast_drift(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        horizon_steps: int,
        drift_threshold: float,
    ) -> Any:
        """Forecast future drift scores via linear extrapolation.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to forecast.
            horizon_steps: Number of future steps to forecast.
            drift_threshold: Drift score above which drift is predicted.

        Returns:
            DriftForecast with predicted scores and confidence intervals.
        """
        ...


@runtime_checkable
class IDriftReportGenerator(Protocol):
    """Contract for automated drift report generation."""

    def generate_report(
        self,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        feature_results: list[dict[str, Any]],
        period_start: datetime,
        baseline_accuracy: float | None,
        current_accuracy: float | None,
        historical_scores: dict[str, list[tuple[datetime, float]]] | None,
        tags: dict[str, str] | None,
    ) -> Any:
        """Generate a complete drift assessment report.

        Args:
            monitor_id: UUID of the drift monitor.
            model_id: UUID of the model.
            tenant_id: UUID of the owning tenant.
            feature_results: List of feature drift result dicts.
            period_start: Start of the assessment period.
            baseline_accuracy: Optional baseline accuracy.
            current_accuracy: Optional current accuracy.
            historical_scores: Optional dict of feature_name to (timestamp, score) pairs.
            tags: Optional metadata tags.

        Returns:
            Generated DriftReport.
        """
        ...

    def export_report_json(self, report_id: uuid.UUID) -> str:
        """Export a report as a JSON string.

        Args:
            report_id: UUID of the report.

        Returns:
            JSON string.
        """
        ...
