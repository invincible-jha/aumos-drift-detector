"""Protocol (interface) definitions for the Drift Detector service.

Defines abstract contracts between the service layer and adapters,
enabling dependency injection and test doubles without coupling to
concrete implementations.
"""

import uuid
from typing import Protocol, runtime_checkable

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
