"""Business logic services for the AumOS Drift Detector.

All services depend on repository and publisher interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code lives here — pure domain logic only.
"""

import uuid

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.observability import get_logger

from aumos_drift_detector.core.interfaces import (
    IDriftAlertRepository,
    IDriftDetectionRepository,
    IDriftEventPublisher,
    IDriftMonitorRepository,
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
