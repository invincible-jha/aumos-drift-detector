"""Business logic services for the AumOS Drift Detector.

All services depend on repository and publisher interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code lives here — pure domain logic only.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.observability import get_logger

from aumos_drift_detector.core.interfaces import (
    IAlertSystem,
    IBaselineManager,
    IDriftAlertRepository,
    IDriftDetectionRepository,
    IDriftEventPublisher,
    IDriftMonitorRepository,
    IDriftReportGenerator,
    IDriftTrendAnalyzer,
    IFeatureImportanceAnalyser,
    IPerformanceMonitor,
    IRetrainTrigger,
)
from aumos_drift_detector.core.models import DriftAlert, DriftDetection, DriftMonitor

logger = get_logger(__name__)

# Valid monitor status transitions
_VALID_STATUS_TRANSITIONS: dict[str, list[str]] = {
    "active": ["paused", "disabled"],
    "paused": ["active", "disabled"],
    "disabled": ["active"],
}

# Severity mapping: (psi_range or drift_score) → severity level
def _compute_severity(score: float, threshold: float, test_name: str) -> str:
    """Derive alert severity from the drift score relative to threshold.

    For PSI: score > 2x threshold = critical, > 1.5x = warning, else info.
    For p-value tests (KS, chi2): lower p-value = more severe.
    For concept drift: any detection = critical.

    Args:
        score: The raw drift score.
        threshold: The threshold the score was compared against.
        test_name: Name of the detection algorithm used.

    Returns:
        Severity string: 'info' | 'warning' | 'critical'.
    """
    if test_name in ("adwin", "ddm", "eddm"):
        return "critical"
    if test_name == "psi":
        if score > threshold * 2:
            return "critical"
        if score > threshold * 1.5:
            return "warning"
        return "info"
    # KS / chi2: p-value based — smaller p-value = more severe
    if score < threshold * 0.1:
        return "critical"
    if score < threshold * 0.5:
        return "warning"
    return "info"


class MonitoringService:
    """CRUD and lifecycle management for drift monitors.

    Orchestrates creation, retrieval, status changes, and deletion
    of DriftMonitor configurations.
    """

    def __init__(self, monitor_repo: IDriftMonitorRepository) -> None:
        """Initialise with injected monitor repository.

        Args:
            monitor_repo: DriftMonitor persistence repository.
        """
        self._monitors = monitor_repo

    async def create_monitor(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        name: str,
        feature_columns: list[str],
        reference_data_uri: str,
        schedule_cron: str | None = None,
        thresholds: dict | None = None,
    ) -> DriftMonitor:
        """Create a new drift monitor for a model.

        Args:
            tenant_id: Owning tenant.
            model_id: Target model UUID.
            name: Human-readable monitor name.
            feature_columns: Feature columns to include in drift checks.
            reference_data_uri: URI to the reference/baseline dataset.
            schedule_cron: Optional cron schedule for periodic runs.
            thresholds: Optional per-test threshold overrides.

        Returns:
            Newly created DriftMonitor.

        Raises:
            ConflictError: If a monitor with the same name already exists for this tenant.
        """
        monitor = await self._monitors.create(
            tenant_id=tenant_id,
            model_id=model_id,
            name=name,
            feature_columns=feature_columns,
            reference_data_uri=reference_data_uri,
            schedule_cron=schedule_cron,
            thresholds=thresholds or {},
        )
        logger.info(
            "Drift monitor created",
            monitor_id=str(monitor.id),
            model_id=str(model_id),
            tenant_id=str(tenant_id),
        )
        return monitor

    async def get_monitor(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftMonitor:
        """Retrieve a monitor by ID.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftMonitor ORM instance.

        Raises:
            NotFoundError: If no monitor found.
        """
        monitor = await self._monitors.get_by_id(monitor_id, tenant_id)
        if monitor is None:
            raise NotFoundError(
                message=f"Drift monitor {monitor_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return monitor

    async def list_monitors(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[DriftMonitor], int]:
        """Return paginated drift monitors for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            status: Optional status filter.

        Returns:
            Tuple of (monitors, total_count).
        """
        return await self._monitors.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            status=status,
        )

    async def update_status(
        self,
        monitor_id: uuid.UUID,
        tenant_id: uuid.UUID,
        new_status: str,
    ) -> DriftMonitor:
        """Update a monitor's operational status.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.
            new_status: Target status (active | paused | disabled).

        Returns:
            Updated DriftMonitor.

        Raises:
            NotFoundError: If monitor not found.
            ConflictError: If the status transition is not allowed.
        """
        monitor = await self.get_monitor(monitor_id, tenant_id)
        allowed = _VALID_STATUS_TRANSITIONS.get(monitor.status, [])
        if new_status not in allowed:
            raise ConflictError(
                message=f"Cannot transition monitor from '{monitor.status}' to '{new_status}'.",
                error_code=ErrorCode.INVALID_OPERATION,
            )
        updated = await self._monitors.update_status(monitor_id, tenant_id, new_status)
        logger.info(
            "Monitor status updated",
            monitor_id=str(monitor_id),
            previous_status=monitor.status,
            new_status=new_status,
        )
        return updated

    async def delete_monitor(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        """Delete a monitor and all its detection history.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.

        Raises:
            NotFoundError: If monitor not found.
        """
        await self.get_monitor(monitor_id, tenant_id)
        await self._monitors.delete(monitor_id, tenant_id)
        logger.info("Drift monitor deleted", monitor_id=str(monitor_id))


class DriftDetectionService:
    """Orchestrates drift detection runs and persists results.

    Receives pre-computed drift scores from adapters (statistical tests or
    concept drift detectors) and applies thresholds to determine if drift
    has occurred. Publishes Kafka events for positive detections.
    """

    def __init__(
        self,
        monitor_repo: IDriftMonitorRepository,
        detection_repo: IDriftDetectionRepository,
        alert_repo: IDriftAlertRepository,
        event_publisher: IDriftEventPublisher,
        retraining_trigger_enabled: bool = True,
        retraining_cooldown_seconds: int = 3600,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            monitor_repo: DriftMonitor repository.
            detection_repo: DriftDetection repository.
            alert_repo: DriftAlert repository.
            event_publisher: Kafka event publisher.
            retraining_trigger_enabled: Whether to emit retraining_required events.
            retraining_cooldown_seconds: Minimum seconds between retraining triggers.
        """
        self._monitors = monitor_repo
        self._detections = detection_repo
        self._alerts = alert_repo
        self._publisher = event_publisher
        self._retraining_enabled = retraining_trigger_enabled
        self._cooldown = retraining_cooldown_seconds

    async def record_detection(
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
        """Persist a drift detection result and trigger downstream actions.

        If drift is detected (`is_drifted=True`), an alert is raised and
        a `drift.detected` Kafka event is published. If the monitor is
        configured for retraining triggers, a `drift.retraining_required`
        event is also published.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Parent monitor UUID.
            detection_type: 'statistical' or 'concept'.
            test_name: Algorithm name.
            score: Drift score value.
            threshold: Threshold compared against the score.
            is_drifted: Whether drift was detected.
            details: Per-feature breakdown dict.

        Returns:
            Persisted DriftDetection ORM instance.

        Raises:
            NotFoundError: If the parent monitor is not found.
        """
        monitor = await self._monitors.get_by_id(monitor_id, tenant_id)
        if monitor is None:
            raise NotFoundError(
                message=f"Drift monitor {monitor_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        detection = await self._detections.create(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            detection_type=detection_type,
            test_name=test_name,
            score=score,
            threshold=threshold,
            is_drifted=is_drifted,
            details=details,
        )

        await self._publisher.publish_drift_detected(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            detection_id=detection.id,
            test_name=test_name,
            score=score,
            is_drifted=is_drifted,
        )

        if is_drifted:
            severity = _compute_severity(score, threshold, test_name)
            message = (
                f"Drift detected on monitor '{monitor.name}' using {test_name} "
                f"(score={score:.4f}, threshold={threshold:.4f})."
            )
            alert = await self._alerts.create(
                tenant_id=tenant_id,
                detection_id=detection.id,
                severity=severity,
                channel="internal",
                message=message,
            )

            await self._publisher.publish_alert_raised(
                tenant_id=tenant_id,
                alert_id=alert.id,
                severity=severity,
                message=message,
            )

            if self._retraining_enabled:
                await self._publisher.publish_retraining_required(
                    tenant_id=tenant_id,
                    monitor_id=monitor_id,
                    model_id=monitor.model_id,
                    detection_id=detection.id,
                    reason=f"{test_name} score {score:.4f} exceeded threshold {threshold:.4f}",
                )
                logger.info(
                    "Retraining trigger published",
                    monitor_id=str(monitor_id),
                    model_id=str(monitor.model_id),
                    test_name=test_name,
                )

        logger.info(
            "Drift detection recorded",
            detection_id=str(detection.id),
            monitor_id=str(monitor_id),
            test_name=test_name,
            score=score,
            is_drifted=is_drifted,
        )
        return detection

    async def get_detection(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftDetection:
        """Retrieve a specific detection by ID.

        Args:
            detection_id: Detection UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftDetection ORM instance.

        Raises:
            NotFoundError: If not found.
        """
        detection = await self._detections.get_by_id(detection_id, tenant_id)
        if detection is None:
            raise NotFoundError(
                message=f"Drift detection {detection_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return detection

    async def list_detections(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        monitor_id: uuid.UUID | None = None,
        is_drifted: bool | None = None,
    ) -> tuple[list[DriftDetection], int]:
        """Return paginated drift detections for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            monitor_id: Optional filter by parent monitor.
            is_drifted: Optional filter (True = drifted only, False = clean only).

        Returns:
            Tuple of (detections, total_count).
        """
        return await self._detections.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            monitor_id=monitor_id,
            is_drifted=is_drifted,
        )

    async def get_dashboard_summary(
        self, tenant_id: uuid.UUID, days: int = 7
    ) -> dict:
        """Return aggregated drift statistics for the dashboard.

        Args:
            tenant_id: Requesting tenant.
            days: Rolling window in days (default 7).

        Returns:
            Dict with summary metrics.
        """
        return await self._detections.get_drift_summary(tenant_id=tenant_id, days=days)


class AlertingService:
    """Alert lifecycle management — creation, retrieval, and acknowledgement."""

    def __init__(
        self,
        alert_repo: IDriftAlertRepository,
        event_publisher: IDriftEventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            alert_repo: DriftAlert persistence repository.
            event_publisher: Kafka event publisher.
        """
        self._alerts = alert_repo
        self._publisher = event_publisher

    async def acknowledge_alert(
        self,
        alert_id: uuid.UUID,
        tenant_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> DriftAlert:
        """Acknowledge a drift alert, marking it as reviewed.

        Args:
            alert_id: Alert UUID.
            tenant_id: Owning tenant.
            acknowledged_by: UUID of the acknowledging user.

        Returns:
            Updated DriftAlert with acknowledged=True.

        Raises:
            NotFoundError: If alert not found for tenant.
            ConflictError: If alert has already been acknowledged.
        """
        alert = await self._alerts.get_by_id(alert_id, tenant_id)
        if alert is None:
            raise NotFoundError(
                message=f"Drift alert {alert_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        if alert.acknowledged:
            raise ConflictError(
                message=f"Alert {alert_id} is already acknowledged.",
                error_code=ErrorCode.INVALID_OPERATION,
            )
        updated = await self._alerts.acknowledge(
            alert_id=alert_id,
            tenant_id=tenant_id,
            acknowledged_by=acknowledged_by,
        )
        logger.info(
            "Drift alert acknowledged",
            alert_id=str(alert_id),
            acknowledged_by=str(acknowledged_by),
        )
        return updated

    async def list_alerts(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        acknowledged: bool | None = None,
        severity: str | None = None,
    ) -> tuple[list[DriftAlert], int]:
        """Return paginated drift alerts for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            acknowledged: Optional filter by acknowledgement state.
            severity: Optional filter by severity level.

        Returns:
            Tuple of (alerts, total_count).
        """
        return await self._alerts.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            acknowledged=acknowledged,
            severity=severity,
        )


class FeatureImportanceService:
    """Service that orchestrates SHAP/LIME-based drift feature importance analysis.

    Delegates to an IFeatureImportanceAnalyser implementation and persists
    historical importance snapshots per monitor.
    """

    def __init__(self, analyser: IFeatureImportanceAnalyser) -> None:
        """Initialise with injected feature importance analyser.

        Args:
            analyser: Concrete IFeatureImportanceAnalyser implementation.
        """
        self._analyser = analyser

    def analyse_and_rank(
        self,
        monitor_id: uuid.UUID,
        reference: dict,
        production: dict,
        drift_scores: dict[str, float],
    ) -> list[Any]:
        """Rank features by drift contribution and record to history.

        Args:
            monitor_id: UUID of the drift monitor.
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.
            drift_scores: Pre-computed drift score per feature.

        Returns:
            Ordered list of FeatureImportanceResult.
        """
        rankings = self._analyser.rank_features_by_drift(
            reference=reference,
            production=production,
            drift_scores=drift_scores,
        )
        self._analyser.record_historical_importance(
            monitor_id=monitor_id,
            feature_rankings=rankings,
        )
        logger.info(
            "Feature importance analysis completed",
            monitor_id=str(monitor_id),
            feature_count=len(rankings),
        )
        return rankings

    def get_waterfall_data(self, feature_rankings: list[Any]) -> Any:
        """Return waterfall chart data for the given feature rankings.

        Args:
            feature_rankings: Ranked feature importance results.

        Returns:
            WaterfallChartData for visualisation.
        """
        return self._analyser.generate_waterfall_data(feature_rankings)


class PerformanceMonitoringService:
    """Service that manages model performance monitoring and degradation detection.

    Wraps an IPerformanceMonitor to track prediction accuracy and detect
    degradation. Publishes alerts via the event publisher when thresholds
    are crossed.
    """

    def __init__(
        self,
        performance_monitor: IPerformanceMonitor,
        event_publisher: IDriftEventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            performance_monitor: IPerformanceMonitor implementation.
            event_publisher: Kafka event publisher for alert events.
        """
        self._monitor = performance_monitor
        self._publisher = event_publisher

    def record_prediction(
        self,
        y_true: float | None,
        y_pred: float,
        y_score: float | None = None,
        segment_key: str | None = None,
    ) -> str:
        """Add a prediction observation to the performance monitor.

        Args:
            y_true: Ground truth label (None if not yet available).
            y_pred: Model's predicted value.
            y_score: Predicted probability or confidence.
            segment_key: Optional segment for per-segment tracking.

        Returns:
            Observation ID for later label resolution.
        """
        return self._monitor.add_observation(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            segment_key=segment_key,
            timestamp=None,
        )

    def evaluate_performance(self) -> list[Any]:
        """Compute window metrics and return any degradation alerts.

        Returns:
            List of DegradationAlert if performance has degraded.
        """
        try:
            metrics = self._monitor.compute_window_metrics()
            alerts = self._monitor.detect_degradation(metrics)
            if alerts:
                logger.warning(
                    "Performance degradation alerts generated",
                    alert_count=len(alerts),
                )
            return alerts
        except ValueError as exc:
            logger.debug("Insufficient observations for performance evaluation", reason=str(exc))
            return []


class RetrainTriggerService:
    """Service that evaluates retraining triggers and coordinates with mlops-lifecycle.

    Wraps an IRetrainTrigger to check drift scores and performance metrics,
    enforcing cooldown periods and publishing Kafka events when triggers fire.
    """

    def __init__(self, retrain_trigger: IRetrainTrigger) -> None:
        """Initialise with injected retrain trigger.

        Args:
            retrain_trigger: IRetrainTrigger implementation.
        """
        self._trigger = retrain_trigger

    async def evaluate_drift(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID,
        drift_score: float,
        test_name: str,
    ) -> Any:
        """Evaluate and potentially dispatch a drift-based retraining trigger.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Drift monitor UUID.
            model_id: Target model UUID.
            detection_id: DriftDetection UUID.
            drift_score: Aggregate drift score.
            test_name: Statistical test that produced the score.

        Returns:
            TriggerEvent recording the evaluation outcome.
        """
        event = await self._trigger.evaluate_drift_trigger(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=detection_id,
            drift_score=drift_score,
            test_name=test_name,
        )
        if event.triggered:
            logger.info(
                "Retraining trigger fired",
                model_id=str(model_id),
                reason=event.reason,
                outcome=event.outcome,
            )
        return event

    async def evaluate_performance(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        accuracy: float | None = None,
        rmse: float | None = None,
    ) -> Any:
        """Evaluate and potentially dispatch a performance-based retraining trigger.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Monitor UUID.
            model_id: Target model UUID.
            accuracy: Current accuracy (None = not applicable).
            rmse: Current RMSE (None = not applicable).

        Returns:
            TriggerEvent recording the evaluation outcome.
        """
        return await self._trigger.evaluate_performance_trigger(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            accuracy=accuracy,
            rmse=rmse,
        )


class BaselineManagementService:
    """Service for managing reference distribution baselines.

    Provides business-level validation and orchestration around the
    IBaselineManager adapter, including model version lifecycle coordination.
    """

    def __init__(self, baseline_manager: IBaselineManager) -> None:
        """Initialise with injected baseline manager.

        Args:
            baseline_manager: IBaselineManager implementation.
        """
        self._manager = baseline_manager

    def create_baseline(
        self,
        model_id: uuid.UUID,
        model_version: str,
        feature_data: dict,
        data_uri: str = "",
        tags: dict[str, str] | None = None,
        window_days: int = 0,
    ) -> Any:
        """Capture and activate a new baseline for a model version.

        Args:
            model_id: UUID of the model.
            model_version: Model version string.
            feature_data: Dict of feature_name to numpy array.
            data_uri: Optional source dataset URI.
            tags: Optional metadata tags.
            window_days: Window size in days (0 = full training set).

        Returns:
            Newly captured and activated BaselineVersion.
        """
        baseline = self._manager.capture_baseline(
            model_id=model_id,
            model_version=model_version,
            feature_data=feature_data,
            data_uri=data_uri,
            tags=tags,
            window_days=window_days,
            activate=True,
        )
        logger.info(
            "Baseline created and activated",
            baseline_id=str(baseline.baseline_id),
            model_id=str(model_id),
            model_version=model_version,
        )
        return baseline

    def get_active_baseline(self, model_id: uuid.UUID) -> Any:
        """Return the active baseline for a model.

        Args:
            model_id: UUID of the model.

        Returns:
            Active BaselineVersion.

        Raises:
            NotFoundError: If no active baseline is found.
        """
        baseline = self._manager.get_active_baseline(model_id)
        if baseline is None:
            raise NotFoundError(
                message=f"No active baseline found for model {model_id}.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return baseline

    def compare_baseline_versions(
        self,
        old_baseline_id: uuid.UUID,
        new_baseline_id: uuid.UUID,
    ) -> Any:
        """Compare two baseline versions and return the comparison result.

        Args:
            old_baseline_id: UUID of the older baseline.
            new_baseline_id: UUID of the newer baseline.

        Returns:
            BaselineComparison with per-feature shift analysis.
        """
        return self._manager.compare_baselines(
            old_baseline_id=old_baseline_id,
            new_baseline_id=new_baseline_id,
            significance_threshold=0.1,
        )


class DriftTrendService:
    """Service for historical drift trend analysis and forecasting.

    Wraps IDriftTrendAnalyzer to provide trend direction assessment,
    change point detection, and linear drift forecasting.
    """

    def __init__(self, trend_analyzer: IDriftTrendAnalyzer) -> None:
        """Initialise with injected trend analyser.

        Args:
            trend_analyzer: IDriftTrendAnalyzer implementation.
        """
        self._analyzer = trend_analyzer

    def record_detection_result(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        score: float,
        test_name: str,
        is_drifted: bool,
    ) -> None:
        """Append a drift detection result to the trend time series.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature name.
            score: Drift score.
            test_name: Statistical test name.
            is_drifted: Whether drift was detected.
        """
        self._analyzer.record_drift_score(
            monitor_id=monitor_id,
            feature_name=feature_name,
            score=score,
            test_name=test_name,
            is_drifted=is_drifted,
            timestamp=None,
        )

    def get_feature_trend(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        window_size: int | None = None,
    ) -> Any:
        """Analyse drift trend for a feature.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to analyse.
            window_size: Optional observation window size.

        Returns:
            TrendAnalysis with direction and change point information.

        Raises:
            NotFoundError: If insufficient history is available.
        """
        try:
            return self._analyzer.analyse_feature_trend(
                monitor_id=monitor_id,
                feature_name=feature_name,
                window_size=window_size,
            )
        except ValueError as exc:
            raise NotFoundError(
                message=str(exc),
                error_code=ErrorCode.NOT_FOUND,
            ) from exc

    def forecast_feature_drift(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        horizon_steps: int = 10,
        drift_threshold: float = 0.25,
    ) -> Any:
        """Generate a linear drift forecast for a feature.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to forecast.
            horizon_steps: Number of future observations to forecast.
            drift_threshold: Score threshold for predicted drift.

        Returns:
            DriftForecast with predicted scores and confidence intervals.
        """
        return self._analyzer.forecast_drift(
            monitor_id=monitor_id,
            feature_name=feature_name,
            horizon_steps=horizon_steps,
            drift_threshold=drift_threshold,
        )


class DriftReportService:
    """Service for orchestrating drift report generation and distribution.

    Aggregates drift detection results, feature importance rankings, and
    performance metrics to produce comprehensive reports.
    """

    def __init__(
        self,
        report_generator: IDriftReportGenerator,
        detection_repo: IDriftDetectionRepository,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            report_generator: IDriftReportGenerator implementation.
            detection_repo: IDriftDetectionRepository for fetching detection history.
        """
        self._generator = report_generator
        self._detections = detection_repo

    async def generate_monitor_report(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        period_start: datetime,
        baseline_accuracy: float | None = None,
        current_accuracy: float | None = None,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Generate a drift assessment report for a monitor.

        Fetches recent detections for the monitor and delegates to the
        report generator. The caller provides pre-computed accuracy values
        for performance impact analysis.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Drift monitor UUID.
            model_id: Target model UUID.
            period_start: Start of the reporting period.
            baseline_accuracy: Optional baseline accuracy for impact analysis.
            current_accuracy: Optional current accuracy for impact analysis.
            tags: Optional report metadata tags.

        Returns:
            Generated DriftReport.
        """
        detections, _ = await self._detections.list_by_tenant(
            tenant_id=tenant_id,
            page=1,
            page_size=200,
            monitor_id=monitor_id,
            is_drifted=None,
        )

        feature_results: list[dict[str, Any]] = []
        for detection in detections:
            if detection.detected_at < period_start:
                continue
            details = detection.details or {}
            feature_results.append({
                "feature_name": details.get("feature", "unknown"),
                "drift_score": detection.score,
                "threshold": detection.threshold,
                "is_drifted": detection.is_drifted,
                "test_name": detection.test_name,
                "contribution_pct": details.get("contribution_pct", 0.0),
                "importance_rank": details.get("importance_rank", 999),
                "mean_shift": details.get("mean_shift", 0.0),
            })

        report = self._generator.generate_report(
            monitor_id=monitor_id,
            model_id=model_id,
            tenant_id=tenant_id,
            feature_results=feature_results,
            period_start=period_start,
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            historical_scores=None,
            tags=tags,
        )
        logger.info(
            "Drift report generated via service",
            report_id=str(report.report_id),
            monitor_id=str(monitor_id),
            feature_result_count=len(feature_results),
        )
        return report
