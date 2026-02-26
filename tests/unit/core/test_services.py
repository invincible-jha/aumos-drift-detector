"""Unit tests for core service classes using mock repositories and publishers.

All external dependencies (database, Kafka) are mocked so tests are fast
and deterministic.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_common.errors import ConflictError, NotFoundError

from aumos_drift_detector.core.models import DriftAlert, DriftDetection, DriftMonitor
from aumos_drift_detector.core.services import (
    AlertingService,
    DriftDetectionService,
    MonitoringService,
    _compute_severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_monitor(
    *,
    monitor_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    name: str = "test-monitor",
    status: str = "active",
) -> DriftMonitor:
    """Build a minimal DriftMonitor ORM mock."""
    m = MagicMock(spec=DriftMonitor)
    m.id = monitor_id or uuid.uuid4()
    m.tenant_id = tenant_id or uuid.uuid4()
    m.model_id = model_id or uuid.uuid4()
    m.name = name
    m.status = status
    m.feature_columns = ["age", "income"]
    m.reference_data_uri = "s3://bucket/reference.parquet"
    m.schedule_cron = None
    m.thresholds = {}
    return m


def make_detection(
    *,
    detection_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID | None = None,
    monitor_id: uuid.UUID | None = None,
    is_drifted: bool = False,
) -> DriftDetection:
    """Build a minimal DriftDetection ORM mock."""
    d = MagicMock(spec=DriftDetection)
    d.id = detection_id or uuid.uuid4()
    d.tenant_id = tenant_id or uuid.uuid4()
    d.monitor_id = monitor_id or uuid.uuid4()
    d.detection_type = "statistical"
    d.test_name = "ks"
    d.score = 0.03 if is_drifted else 0.8
    d.threshold = 0.05
    d.is_drifted = is_drifted
    d.details = {}
    return d


def make_alert(
    *,
    alert_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID | None = None,
    acknowledged: bool = False,
) -> DriftAlert:
    """Build a minimal DriftAlert ORM mock."""
    a = MagicMock(spec=DriftAlert)
    a.id = alert_id or uuid.uuid4()
    a.tenant_id = tenant_id or uuid.uuid4()
    a.detection_id = uuid.uuid4()
    a.severity = "warning"
    a.message = "Drift detected"
    a.acknowledged = acknowledged
    a.acknowledged_by = None
    return a


# ---------------------------------------------------------------------------
# _compute_severity helper tests
# ---------------------------------------------------------------------------


class TestComputeSeverity:
    """Tests for the severity classification helper."""

    def test_adwin_always_critical(self) -> None:
        """ADWIN drift is always classified as critical."""
        assert _compute_severity(0.9, 0.5, "adwin") == "critical"

    def test_ddm_always_critical(self) -> None:
        """DDM drift is always critical."""
        assert _compute_severity(0.9, 0.5, "ddm") == "critical"

    def test_eddm_always_critical(self) -> None:
        """EDDM drift is always critical."""
        assert _compute_severity(0.9, 0.5, "eddm") == "critical"

    def test_psi_critical_when_double_threshold(self) -> None:
        """PSI > 2x threshold is critical."""
        assert _compute_severity(0.5, 0.2, "psi") == "critical"

    def test_psi_warning_between_1_5x_and_2x_threshold(self) -> None:
        """PSI between 1.5x and 2x threshold is warning."""
        assert _compute_severity(0.35, 0.2, "psi") == "warning"

    def test_psi_info_below_1_5x_threshold(self) -> None:
        """PSI just above threshold is info."""
        assert _compute_severity(0.22, 0.2, "psi") == "info"

    def test_ks_critical_when_very_small_p_value(self) -> None:
        """KS p-value < 10% of threshold is critical."""
        # threshold=0.05, score=0.001 → score < 0.05 * 0.1 = 0.005
        assert _compute_severity(0.001, 0.05, "ks") == "critical"

    def test_ks_info_near_threshold(self) -> None:
        """KS p-value just below threshold is info."""
        # threshold=0.05, score=0.04 → between 0.5x and 1x threshold
        assert _compute_severity(0.04, 0.05, "ks") == "info"


# ---------------------------------------------------------------------------
# MonitoringService tests
# ---------------------------------------------------------------------------


class TestMonitoringService:
    """Tests for MonitoringService."""

    @pytest.fixture()
    def monitor_repo(self) -> AsyncMock:
        """Mock monitor repository."""
        return AsyncMock()

    @pytest.fixture()
    def service(self, monitor_repo: AsyncMock) -> MonitoringService:
        """MonitoringService with mocked repository."""
        return MonitoringService(monitor_repo=monitor_repo)

    async def test_create_monitor_returns_monitor(
        self, service: MonitoringService, monitor_repo: AsyncMock
    ) -> None:
        """create_monitor must delegate to the repository and return the result."""
        tenant_id = uuid.uuid4()
        model_id = uuid.uuid4()
        expected = make_monitor(tenant_id=tenant_id, model_id=model_id)
        monitor_repo.create.return_value = expected

        result = await service.create_monitor(
            tenant_id=tenant_id,
            model_id=model_id,
            name="my-monitor",
            feature_columns=["age"],
            reference_data_uri="s3://bucket/ref.parquet",
        )

        assert result is expected
        monitor_repo.create.assert_called_once()

    async def test_get_monitor_not_found_raises(
        self, service: MonitoringService, monitor_repo: AsyncMock
    ) -> None:
        """get_monitor must raise NotFoundError when repository returns None."""
        monitor_repo.get_by_id.return_value = None
        with pytest.raises(NotFoundError):
            await service.get_monitor(uuid.uuid4(), uuid.uuid4())

    async def test_update_status_invalid_transition_raises(
        self, service: MonitoringService, monitor_repo: AsyncMock
    ) -> None:
        """Transitioning from 'active' to 'active' must raise ConflictError."""
        monitor = make_monitor(status="active")
        monitor_repo.get_by_id.return_value = monitor
        with pytest.raises(ConflictError):
            await service.update_status(monitor.id, monitor.tenant_id, "active")

    async def test_update_status_valid_transition_succeeds(
        self, service: MonitoringService, monitor_repo: AsyncMock
    ) -> None:
        """Transitioning from 'active' to 'paused' must succeed."""
        monitor = make_monitor(status="active")
        updated = make_monitor(status="paused")
        monitor_repo.get_by_id.return_value = monitor
        monitor_repo.update_status.return_value = updated

        result = await service.update_status(monitor.id, monitor.tenant_id, "paused")
        assert result.status == "paused"

    async def test_delete_monitor_calls_repo(
        self, service: MonitoringService, monitor_repo: AsyncMock
    ) -> None:
        """delete_monitor must call repository.delete after confirming existence."""
        monitor = make_monitor()
        monitor_repo.get_by_id.return_value = monitor

        await service.delete_monitor(monitor.id, monitor.tenant_id)

        monitor_repo.delete.assert_called_once_with(monitor.id, monitor.tenant_id)


# ---------------------------------------------------------------------------
# DriftDetectionService tests
# ---------------------------------------------------------------------------


class TestDriftDetectionService:
    """Tests for DriftDetectionService."""

    @pytest.fixture()
    def monitor_repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def detection_repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def alert_repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def publisher(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def service(
        self,
        monitor_repo: AsyncMock,
        detection_repo: AsyncMock,
        alert_repo: AsyncMock,
        publisher: AsyncMock,
    ) -> DriftDetectionService:
        return DriftDetectionService(
            monitor_repo=monitor_repo,
            detection_repo=detection_repo,
            alert_repo=alert_repo,
            event_publisher=publisher,
            retraining_trigger_enabled=True,
        )

    async def test_record_detection_publishes_event(
        self,
        service: DriftDetectionService,
        monitor_repo: AsyncMock,
        detection_repo: AsyncMock,
        alert_repo: AsyncMock,
        publisher: AsyncMock,
    ) -> None:
        """record_detection must always publish drift.detected event."""
        monitor = make_monitor()
        detection = make_detection(is_drifted=False)
        monitor_repo.get_by_id.return_value = monitor
        detection_repo.create.return_value = detection

        result = await service.record_detection(
            tenant_id=monitor.tenant_id,
            monitor_id=monitor.id,
            detection_type="statistical",
            test_name="ks",
            score=0.8,
            threshold=0.05,
            is_drifted=False,
            details={},
        )

        publisher.publish_drift_detected.assert_called_once()
        assert result is detection

    async def test_drift_detected_creates_alert(
        self,
        service: DriftDetectionService,
        monitor_repo: AsyncMock,
        detection_repo: AsyncMock,
        alert_repo: AsyncMock,
        publisher: AsyncMock,
    ) -> None:
        """When is_drifted=True, an alert must be created."""
        monitor = make_monitor()
        detection = make_detection(is_drifted=True)
        alert = make_alert()
        monitor_repo.get_by_id.return_value = monitor
        detection_repo.create.return_value = detection
        alert_repo.create.return_value = alert

        await service.record_detection(
            tenant_id=monitor.tenant_id,
            monitor_id=monitor.id,
            detection_type="statistical",
            test_name="psi",
            score=0.5,
            threshold=0.2,
            is_drifted=True,
            details={},
        )

        alert_repo.create.assert_called_once()
        publisher.publish_alert_raised.assert_called_once()
        publisher.publish_retraining_required.assert_called_once()

    async def test_no_retraining_when_disabled(
        self,
        monitor_repo: AsyncMock,
        detection_repo: AsyncMock,
        alert_repo: AsyncMock,
        publisher: AsyncMock,
    ) -> None:
        """When retraining_trigger_enabled=False, no retraining event is published."""
        service = DriftDetectionService(
            monitor_repo=monitor_repo,
            detection_repo=detection_repo,
            alert_repo=alert_repo,
            event_publisher=publisher,
            retraining_trigger_enabled=False,
        )
        monitor = make_monitor()
        detection = make_detection(is_drifted=True)
        alert = make_alert()
        monitor_repo.get_by_id.return_value = monitor
        detection_repo.create.return_value = detection
        alert_repo.create.return_value = alert

        await service.record_detection(
            tenant_id=monitor.tenant_id,
            monitor_id=monitor.id,
            detection_type="statistical",
            test_name="ks",
            score=0.01,
            threshold=0.05,
            is_drifted=True,
            details={},
        )

        publisher.publish_retraining_required.assert_not_called()

    async def test_record_detection_monitor_not_found_raises(
        self,
        service: DriftDetectionService,
        monitor_repo: AsyncMock,
        detection_repo: AsyncMock,
        alert_repo: AsyncMock,
        publisher: AsyncMock,
    ) -> None:
        """NotFoundError must be raised when the monitor does not exist."""
        monitor_repo.get_by_id.return_value = None
        with pytest.raises(NotFoundError):
            await service.record_detection(
                tenant_id=uuid.uuid4(),
                monitor_id=uuid.uuid4(),
                detection_type="statistical",
                test_name="ks",
                score=0.5,
                threshold=0.05,
                is_drifted=False,
                details={},
            )


# ---------------------------------------------------------------------------
# AlertingService tests
# ---------------------------------------------------------------------------


class TestAlertingService:
    """Tests for AlertingService."""

    @pytest.fixture()
    def alert_repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def publisher(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture()
    def service(self, alert_repo: AsyncMock, publisher: AsyncMock) -> AlertingService:
        return AlertingService(alert_repo=alert_repo, event_publisher=publisher)

    async def test_acknowledge_alert_success(
        self, service: AlertingService, alert_repo: AsyncMock
    ) -> None:
        """acknowledge_alert must call repo.acknowledge and return updated alert."""
        alert_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        alert = make_alert(alert_id=alert_id, tenant_id=tenant_id, acknowledged=False)
        updated = make_alert(alert_id=alert_id, tenant_id=tenant_id, acknowledged=True)
        alert_repo.get_by_id.return_value = alert
        alert_repo.acknowledge.return_value = updated

        result = await service.acknowledge_alert(alert_id, tenant_id, user_id)

        alert_repo.acknowledge.assert_called_once()
        assert result.acknowledged is True

    async def test_acknowledge_not_found_raises(
        self, service: AlertingService, alert_repo: AsyncMock
    ) -> None:
        """NotFoundError must be raised when alert does not exist."""
        alert_repo.get_by_id.return_value = None
        with pytest.raises(NotFoundError):
            await service.acknowledge_alert(uuid.uuid4(), uuid.uuid4(), uuid.uuid4())

    async def test_acknowledge_already_acknowledged_raises(
        self, service: AlertingService, alert_repo: AsyncMock
    ) -> None:
        """ConflictError must be raised when alert is already acknowledged."""
        alert = make_alert(acknowledged=True)
        alert_repo.get_by_id.return_value = alert
        with pytest.raises(ConflictError):
            await service.acknowledge_alert(alert.id, alert.tenant_id, uuid.uuid4())
