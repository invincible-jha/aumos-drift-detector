"""SQLAlchemy async repository implementations for the Drift Detector.

Concrete implementations of the repository interfaces defined in core/interfaces.py.
All database operations are tenant-scoped and use parameterised queries.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

from aumos_drift_detector.core.models import DriftAlert, DriftDetection, DriftMonitor

logger = get_logger(__name__)


class DriftMonitorRepository:
    """SQLAlchemy async implementation of IDriftMonitorRepository.

    All operations are tenant-scoped: every query filters on tenant_id to
    enforce Row-Level Security at the application layer.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Active async database session.
        """
        self._session = session

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
        """Create a new DriftMonitor record.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Target model UUID.
            name: Human-readable monitor name.
            feature_columns: Feature columns to monitor.
            reference_data_uri: URI to the reference dataset.
            schedule_cron: Optional cron schedule expression.
            thresholds: Per-test threshold overrides.

        Returns:
            Persisted DriftMonitor instance.
        """
        monitor = DriftMonitor(
            tenant_id=tenant_id,
            model_id=model_id,
            name=name,
            feature_columns=feature_columns,
            reference_data_uri=reference_data_uri,
            schedule_cron=schedule_cron,
            thresholds=thresholds,
            status="active",
        )
        self._session.add(monitor)
        await self._session.flush()
        await self._session.refresh(monitor)
        logger.debug("DriftMonitor created", monitor_id=str(monitor.id))
        return monitor

    async def get_by_id(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftMonitor | None:
        """Fetch a monitor by UUID with tenant isolation.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftMonitor or None if not found.
        """
        result = await self._session.execute(
            select(DriftMonitor).where(
                DriftMonitor.id == monitor_id,
                DriftMonitor.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

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
            status: Optional status filter.

        Returns:
            Tuple of (monitors, total_count).
        """
        query = select(DriftMonitor).where(DriftMonitor.tenant_id == tenant_id)
        if status is not None:
            query = query.where(DriftMonitor.status == status)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        paginated = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(paginated)
        return list(result.scalars().all()), total

    async def list_active_scheduled(self) -> list[DriftMonitor]:
        """Return all active, cron-scheduled monitors across all tenants.

        Returns:
            List of active DriftMonitor instances with a schedule_cron set.
        """
        result = await self._session.execute(
            select(DriftMonitor).where(
                DriftMonitor.status == "active",
                DriftMonitor.schedule_cron.is_not(None),
            )
        )
        return list(result.scalars().all())

    async def update_status(
        self, monitor_id: uuid.UUID, tenant_id: uuid.UUID, status: str
    ) -> DriftMonitor:
        """Update the status of a monitor.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.
            status: New status value.

        Returns:
            Updated DriftMonitor instance.
        """
        await self._session.execute(
            update(DriftMonitor)
            .where(
                DriftMonitor.id == monitor_id,
                DriftMonitor.tenant_id == tenant_id,
            )
            .values(status=status)
        )
        await self._session.flush()
        monitor = await self.get_by_id(monitor_id, tenant_id)
        assert monitor is not None  # noqa: S101 — we just updated it
        return monitor

    async def delete(self, monitor_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Delete a monitor and cascade to all its detections and alerts.

        Args:
            monitor_id: Monitor UUID.
            tenant_id: Owning tenant.
        """
        monitor = await self.get_by_id(monitor_id, tenant_id)
        if monitor is not None:
            await self._session.delete(monitor)
            await self._session.flush()


class DriftDetectionRepository:
    """SQLAlchemy async implementation of IDriftDetectionRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Active async database session.
        """
        self._session = session

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
        """Persist a new drift detection result.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Parent monitor UUID.
            detection_type: 'statistical' or 'concept'.
            test_name: Algorithm name.
            score: Drift score.
            threshold: Comparison threshold.
            is_drifted: Drift verdict.
            details: Per-feature breakdown.

        Returns:
            Persisted DriftDetection instance.
        """
        detection = DriftDetection(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            detection_type=detection_type,
            test_name=test_name,
            score=score,
            threshold=threshold,
            is_drifted=is_drifted,
            details=details,
        )
        self._session.add(detection)
        await self._session.flush()
        await self._session.refresh(detection)
        logger.debug("DriftDetection created", detection_id=str(detection.id))
        return detection

    async def get_by_id(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftDetection | None:
        """Fetch a detection by UUID with tenant isolation.

        Args:
            detection_id: Detection UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftDetection or None.
        """
        result = await self._session.execute(
            select(DriftDetection).where(
                DriftDetection.id == detection_id,
                DriftDetection.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

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
            monitor_id: Optional monitor filter.
            is_drifted: Optional drift verdict filter.

        Returns:
            Tuple of (detections, total_count).
        """
        query = (
            select(DriftDetection)
            .where(DriftDetection.tenant_id == tenant_id)
            .order_by(DriftDetection.detected_at.desc())
        )
        if monitor_id is not None:
            query = query.where(DriftDetection.monitor_id == monitor_id)
        if is_drifted is not None:
            query = query.where(DriftDetection.is_drifted == is_drifted)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        paginated = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(paginated)
        return list(result.scalars().all()), total

    async def get_drift_summary(
        self, tenant_id: uuid.UUID, days: int
    ) -> dict:
        """Aggregate drift statistics over a rolling window.

        Args:
            tenant_id: Requesting tenant.
            days: Rolling window in days.

        Returns:
            Summary dict with aggregate metrics and per-monitor breakdown.
        """
        since = datetime.now(UTC) - timedelta(days=days)

        # Aggregate totals
        total_result = await self._session.execute(
            select(func.count())
            .select_from(DriftDetection)
            .where(
                DriftDetection.tenant_id == tenant_id,
                DriftDetection.detected_at >= since,
            )
        )
        total_checks = total_result.scalar_one()

        drifted_result = await self._session.execute(
            select(func.count())
            .select_from(DriftDetection)
            .where(
                DriftDetection.tenant_id == tenant_id,
                DriftDetection.detected_at >= since,
                DriftDetection.is_drifted.is_(True),
            )
        )
        drifted_checks = drifted_result.scalar_one()

        # Monitor totals
        monitor_result = await self._session.execute(
            select(func.count()).select_from(DriftMonitor).where(
                DriftMonitor.tenant_id == tenant_id
            )
        )
        total_monitors = monitor_result.scalar_one()

        active_result = await self._session.execute(
            select(func.count()).select_from(DriftMonitor).where(
                DriftMonitor.tenant_id == tenant_id,
                DriftMonitor.status == "active",
            )
        )
        active_monitors = active_result.scalar_one()

        # Unacknowledged alerts
        alert_result = await self._session.execute(
            select(func.count()).select_from(DriftAlert).where(
                DriftAlert.tenant_id == tenant_id,
                DriftAlert.acknowledged.is_(False),
            )
        )
        unacknowledged_alerts = alert_result.scalar_one()

        return {
            "total_checks": total_checks,
            "drifted_checks": drifted_checks,
            "total_monitors": total_monitors,
            "active_monitors": active_monitors,
            "unacknowledged_alerts": unacknowledged_alerts,
            "monitors": [],  # Per-monitor breakdown populated by caller
        }


class DriftAlertRepository:
    """SQLAlchemy async implementation of IDriftAlertRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Active async database session.
        """
        self._session = session

    async def create(
        self,
        tenant_id: uuid.UUID,
        detection_id: uuid.UUID,
        severity: str,
        channel: str,
        message: str,
    ) -> DriftAlert:
        """Persist a new drift alert.

        Args:
            tenant_id: Owning tenant.
            detection_id: Parent detection UUID.
            severity: Alert severity level.
            channel: Notification channel.
            message: Human-readable alert message.

        Returns:
            Persisted DriftAlert instance.
        """
        alert = DriftAlert(
            tenant_id=tenant_id,
            detection_id=detection_id,
            severity=severity,
            channel=channel,
            message=message,
            acknowledged=False,
        )
        self._session.add(alert)
        await self._session.flush()
        await self._session.refresh(alert)
        logger.debug("DriftAlert created", alert_id=str(alert.id), severity=severity)
        return alert

    async def get_by_id(
        self, alert_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> DriftAlert | None:
        """Fetch an alert by UUID with tenant isolation.

        Args:
            alert_id: Alert UUID.
            tenant_id: Requesting tenant.

        Returns:
            DriftAlert or None.
        """
        result = await self._session.execute(
            select(DriftAlert).where(
                DriftAlert.id == alert_id,
                DriftAlert.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def acknowledge(
        self,
        alert_id: uuid.UUID,
        tenant_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> DriftAlert:
        """Mark an alert as acknowledged.

        Args:
            alert_id: Alert UUID.
            tenant_id: Owning tenant.
            acknowledged_by: UUID of the acknowledging user.

        Returns:
            Updated DriftAlert instance.
        """
        await self._session.execute(
            update(DriftAlert)
            .where(
                DriftAlert.id == alert_id,
                DriftAlert.tenant_id == tenant_id,
            )
            .values(
                acknowledged=True,
                acknowledged_by=acknowledged_by,
                acknowledged_at=datetime.now(UTC),
            )
        )
        await self._session.flush()
        alert = await self.get_by_id(alert_id, tenant_id)
        assert alert is not None  # noqa: S101
        return alert

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
            acknowledged: Optional acknowledgement filter.
            severity: Optional severity filter.

        Returns:
            Tuple of (alerts, total_count).
        """
        query = (
            select(DriftAlert)
            .where(DriftAlert.tenant_id == tenant_id)
            .order_by(DriftAlert.created_at.desc())
        )
        if acknowledged is not None:
            query = query.where(DriftAlert.acknowledged == acknowledged)
        if severity is not None:
            query = query.where(DriftAlert.severity == severity)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        paginated = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(paginated)
        return list(result.scalars().all()), total
