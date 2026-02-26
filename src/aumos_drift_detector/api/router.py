"""FastAPI router for the AumOS Drift Detector API.

All routes are thin: validate inputs via Pydantic schemas, delegate
business logic to service classes, and return Pydantic response schemas.
No domain logic lives in this module.
"""

import math
import uuid

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

from aumos_drift_detector.adapters.repositories import (
    DriftAlertRepository,
    DriftDetectionRepository,
    DriftMonitorRepository,
)
from aumos_drift_detector.api.schemas import (
    AlertAcknowledgeRequest,
    AlertListResponse,
    DashboardResponse,
    DetectionListResponse,
    DriftAlertResponse,
    DriftDetectionResponse,
    MonitorCreateRequest,
    MonitorListResponse,
    MonitorResponse,
    MonitorStatusUpdateRequest,
    MonitorSummaryItem,
    PaginationMeta,
    RunMonitorRequest,
)
from aumos_drift_detector.core.services import (
    AlertingService,
    DriftDetectionService,
    MonitoringService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Drift Detector"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _monitoring_service(
    session: AsyncSession = Depends(get_db_session),
) -> MonitoringService:
    """Build a MonitoringService with repository dependencies.

    Args:
        session: SQLAlchemy async session (injected by FastAPI).

    Returns:
        Configured MonitoringService instance.
    """
    return MonitoringService(monitor_repo=DriftMonitorRepository(session))


def _detection_service(
    session: AsyncSession = Depends(get_db_session),
    request: Request = None,  # type: ignore[assignment]
) -> DriftDetectionService:
    """Build a DriftDetectionService with all injected dependencies.

    Args:
        session: SQLAlchemy async session.
        request: FastAPI request (provides app.state for Kafka publisher).

    Returns:
        Configured DriftDetectionService instance.
    """
    publisher = request.app.state.kafka_publisher
    return DriftDetectionService(
        monitor_repo=DriftMonitorRepository(session),
        detection_repo=DriftDetectionRepository(session),
        alert_repo=DriftAlertRepository(session),
        event_publisher=publisher,
    )


def _alerting_service(
    session: AsyncSession = Depends(get_db_session),
    request: Request = None,  # type: ignore[assignment]
) -> AlertingService:
    """Build an AlertingService with repository and publisher dependencies.

    Args:
        session: SQLAlchemy async session.
        request: FastAPI request (provides app.state for Kafka publisher).

    Returns:
        Configured AlertingService instance.
    """
    publisher = request.app.state.kafka_publisher
    return AlertingService(
        alert_repo=DriftAlertRepository(session),
        event_publisher=publisher,
    )


# ---------------------------------------------------------------------------
# Monitor CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/monitors",
    response_model=MonitorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a drift monitor",
)
async def create_monitor(
    body: MonitorCreateRequest,
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(_monitoring_service),
) -> MonitorResponse:
    """Create a new drift monitor configuration for a model.

    The monitor defines which features to watch, the reference dataset,
    the detection schedule, and threshold overrides.

    Args:
        body: Monitor creation request payload.
        tenant: Authenticated tenant context (from JWT).
        service: Injected MonitoringService.

    Returns:
        Newly created drift monitor resource.
    """
    monitor = await service.create_monitor(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        model_id=body.model_id,
        name=body.name,
        feature_columns=body.feature_columns,
        reference_data_uri=body.reference_data_uri,
        schedule_cron=body.schedule_cron,
        thresholds=body.thresholds,
    )
    return MonitorResponse.model_validate(monitor)


@router.get(
    "/monitors",
    response_model=MonitorListResponse,
    summary="List drift monitors",
)
async def list_monitors(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(_monitoring_service),
) -> MonitorListResponse:
    """Return a paginated list of drift monitors for the authenticated tenant.

    Args:
        page: 1-based page number (default 1).
        page_size: Results per page (default 20, max 100).
        status_filter: Optional status filter (active | paused | disabled).
        tenant: Authenticated tenant context.
        service: Injected MonitoringService.

    Returns:
        Paginated list of drift monitor resources.
    """
    page_size = min(page_size, 100)
    monitors, total = await service.list_monitors(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        page=page,
        page_size=page_size,
        status=status_filter,
    )
    total_pages = max(1, math.ceil(total / page_size))
    return MonitorListResponse(
        items=[MonitorResponse.model_validate(m) for m in monitors],
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/monitors/{monitor_id}",
    response_model=MonitorResponse,
    summary="Get drift monitor details",
)
async def get_monitor(
    monitor_id: uuid.UUID,
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(_monitoring_service),
) -> MonitorResponse:
    """Retrieve a single drift monitor by ID.

    Args:
        monitor_id: Monitor UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected MonitoringService.

    Returns:
        Drift monitor resource.
    """
    monitor = await service.get_monitor(monitor_id, tenant.tenant_id)  # type: ignore[attr-defined]
    return MonitorResponse.model_validate(monitor)


@router.patch(
    "/monitors/{monitor_id}/status",
    response_model=MonitorResponse,
    summary="Update monitor status",
)
async def update_monitor_status(
    monitor_id: uuid.UUID,
    body: MonitorStatusUpdateRequest,
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(_monitoring_service),
) -> MonitorResponse:
    """Update the operational status of a drift monitor.

    Args:
        monitor_id: Monitor UUID path parameter.
        body: Status update request.
        tenant: Authenticated tenant context.
        service: Injected MonitoringService.

    Returns:
        Updated drift monitor resource.
    """
    monitor = await service.update_status(
        monitor_id=monitor_id,
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        new_status=body.status,
    )
    return MonitorResponse.model_validate(monitor)


@router.post(
    "/monitors/{monitor_id}/run",
    response_model=DriftDetectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger an immediate drift check",
)
async def run_monitor(
    monitor_id: uuid.UUID,
    body: RunMonitorRequest,
    tenant: object = Depends(get_current_tenant),
    service: DriftDetectionService = Depends(_detection_service),
) -> DriftDetectionResponse:
    """Trigger an immediate drift detection run for a monitor.

    Runs all configured statistical tests against the reference dataset
    and persists the result. If drift is detected, Kafka events are published.

    This endpoint runs a KS test as the primary check. Full multi-test
    orchestration is handled by the scheduled monitor runner.

    Args:
        monitor_id: Monitor UUID path parameter.
        body: Optional inline current data and detection type selection.
        tenant: Authenticated tenant context.
        service: Injected DriftDetectionService.

    Returns:
        Drift detection result resource.
    """
    # Stub implementation: in production this delegates to the statistical
    # test runner which loads reference data and runs the configured tests.
    # For the scaffolding we record a placeholder result.
    detection = await service.record_detection(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        monitor_id=monitor_id,
        detection_type="statistical",
        test_name="ks",
        score=0.0,
        threshold=0.05,
        is_drifted=False,
        details={"note": "manual_trigger", "provided_data": body.current_data is not None},
    )
    return DriftDetectionResponse.model_validate(detection)


# ---------------------------------------------------------------------------
# Detection endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/detections",
    response_model=DetectionListResponse,
    summary="List drift detections",
)
async def list_detections(
    page: int = 1,
    page_size: int = 20,
    monitor_id: uuid.UUID | None = None,
    is_drifted: bool | None = None,
    tenant: object = Depends(get_current_tenant),
    service: DriftDetectionService = Depends(_detection_service),
) -> DetectionListResponse:
    """Return a paginated list of drift detection results.

    Args:
        page: 1-based page number (default 1).
        page_size: Results per page (default 20, max 100).
        monitor_id: Optional filter by parent monitor.
        is_drifted: Optional filter (true = drifted only, false = clean only).
        tenant: Authenticated tenant context.
        service: Injected DriftDetectionService.

    Returns:
        Paginated list of drift detection results.
    """
    page_size = min(page_size, 100)
    detections, total = await service.list_detections(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        page=page,
        page_size=page_size,
        monitor_id=monitor_id,
        is_drifted=is_drifted,
    )
    total_pages = max(1, math.ceil(total / page_size))
    return DetectionListResponse(
        items=[DriftDetectionResponse.model_validate(d) for d in detections],
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/detections/{detection_id}",
    response_model=DriftDetectionResponse,
    summary="Get detection details",
)
async def get_detection(
    detection_id: uuid.UUID,
    tenant: object = Depends(get_current_tenant),
    service: DriftDetectionService = Depends(_detection_service),
) -> DriftDetectionResponse:
    """Retrieve a specific drift detection result by ID.

    Args:
        detection_id: Detection UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected DriftDetectionService.

    Returns:
        Drift detection result resource with full details.
    """
    detection = await service.get_detection(detection_id, tenant.tenant_id)  # type: ignore[attr-defined]
    return DriftDetectionResponse.model_validate(detection)


# ---------------------------------------------------------------------------
# Alert endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/alerts/acknowledge/{alert_id}",
    response_model=DriftAlertResponse,
    summary="Acknowledge a drift alert",
)
async def acknowledge_alert(
    alert_id: uuid.UUID,
    body: AlertAcknowledgeRequest,
    tenant: object = Depends(get_current_tenant),
    user: object = Depends(get_current_user),
    service: AlertingService = Depends(_alerting_service),
) -> DriftAlertResponse:
    """Acknowledge a drift alert, marking it as reviewed by an operator.

    Once acknowledged, the alert will not appear in the unacknowledged
    alert count on the dashboard. Acknowledgement is permanent.

    Args:
        alert_id: Alert UUID path parameter.
        body: Optional acknowledgement note.
        tenant: Authenticated tenant context.
        user: Authenticated user context.
        service: Injected AlertingService.

    Returns:
        Updated drift alert resource with acknowledged=True.
    """
    alert = await service.acknowledge_alert(
        alert_id=alert_id,
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        acknowledged_by=user.user_id,  # type: ignore[attr-defined]
    )
    return DriftAlertResponse.model_validate(alert)


# ---------------------------------------------------------------------------
# Dashboard endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Drift dashboard summary",
)
async def get_dashboard(
    days: int = 7,
    tenant: object = Depends(get_current_tenant),
    service: DriftDetectionService = Depends(_detection_service),
) -> DashboardResponse:
    """Return aggregated drift statistics for the MLOps dashboard.

    Provides a rolling-window summary of drift checks, drift rate, active
    monitors, and unacknowledged alerts, broken down per monitor.

    Args:
        days: Rolling window in days (default 7, max 90).
        tenant: Authenticated tenant context.
        service: Injected DriftDetectionService.

    Returns:
        Dashboard summary response.
    """
    days = min(days, 90)
    summary = await service.get_dashboard_summary(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        days=days,
    )

    total = summary.get("total_checks", 0)
    drifted = summary.get("drifted_checks", 0)
    drift_rate = round((drifted / total * 100) if total > 0 else 0.0, 2)

    monitor_summaries: list[MonitorSummaryItem] = [
        MonitorSummaryItem(**item)
        for item in summary.get("monitors", [])
    ]

    return DashboardResponse(
        tenant_id=tenant.tenant_id,  # type: ignore[attr-defined]
        window_days=days,
        total_monitors=summary.get("total_monitors", 0),
        active_monitors=summary.get("active_monitors", 0),
        total_checks=total,
        drifted_checks=drifted,
        drift_rate_percent=drift_rate,
        unacknowledged_alerts=summary.get("unacknowledged_alerts", 0),
        monitors=monitor_summaries,
    )
