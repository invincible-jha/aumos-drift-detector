"""Pydantic request and response schemas for the Drift Detector API.

All schemas use strict validation. Input schemas validate request bodies;
response schemas serialise ORM models for API consumers.
"""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / pagination
# ---------------------------------------------------------------------------


class PaginationMeta(BaseModel):
    """Pagination metadata included in list responses."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1, description="Current 1-based page number")
    page_size: int = Field(ge=1, le=100, description="Results per page")
    total: int = Field(ge=0, description="Total number of matching records")
    total_pages: int = Field(ge=1, description="Total number of pages")


# ---------------------------------------------------------------------------
# DriftMonitor schemas
# ---------------------------------------------------------------------------


class MonitorCreateRequest(BaseModel):
    """Request body for creating a drift monitor."""

    model_config = ConfigDict(str_strip_whitespace=True)

    model_id: uuid.UUID = Field(description="UUID of the model to monitor")
    name: str = Field(min_length=1, max_length=255, description="Human-readable monitor name")
    feature_columns: list[str] = Field(
        min_length=1,
        description="List of feature column names to include in drift checks",
    )
    reference_data_uri: str = Field(
        min_length=5,
        max_length=1024,
        description="S3/MinIO URI to the reference (baseline) dataset",
    )
    schedule_cron: str | None = Field(
        default=None,
        max_length=100,
        description="Cron expression for scheduled runs (e.g. '0 * * * *' = hourly)",
    )
    thresholds: dict = Field(
        default_factory=dict,
        description=(
            "Per-test threshold overrides. Keys: ks, psi, chi2, adwin, ddm. "
            "Empty dict uses global service defaults."
        ),
    )

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: list[str]) -> list[str]:
        """Ensure all feature column names are non-empty strings.

        Args:
            value: List of feature column names.

        Returns:
            Validated list.

        Raises:
            ValueError: If any column name is blank.
        """
        for col in value:
            if not col.strip():
                raise ValueError("Feature column names must not be blank")
        return [c.strip() for c in value]


class MonitorStatusUpdateRequest(BaseModel):
    """Request body for updating a monitor's operational status."""

    status: Literal["active", "paused", "disabled"] = Field(
        description="New monitor status"
    )


class MonitorResponse(BaseModel):
    """API response for a single DriftMonitor."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    model_id: uuid.UUID
    name: str
    feature_columns: list
    reference_data_uri: str
    schedule_cron: str | None
    status: str
    thresholds: dict
    created_at: datetime
    updated_at: datetime


class MonitorListResponse(BaseModel):
    """API response for a paginated list of drift monitors."""

    model_config = ConfigDict(frozen=True)

    items: list[MonitorResponse]
    pagination: PaginationMeta


# ---------------------------------------------------------------------------
# DriftDetection schemas
# ---------------------------------------------------------------------------


class RunMonitorRequest(BaseModel):
    """Optional request body for triggering an immediate drift check.

    If current_data is provided, it is used instead of fetching live data.
    The outer keys should be feature names; values should be lists of samples.
    """

    current_data: dict | None = Field(
        default=None,
        description=(
            "Optional inline current dataset as column-oriented dict. "
            "If omitted, the service fetches the current window from the configured source."
        ),
    )
    detection_types: list[Literal["statistical", "concept"]] = Field(
        default=["statistical"],
        description="Which detection categories to run",
    )


class DriftDetectionResponse(BaseModel):
    """API response for a single drift detection result."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    monitor_id: uuid.UUID
    detection_type: str
    test_name: str
    score: float
    threshold: float
    is_drifted: bool
    details: dict
    detected_at: datetime


class DetectionListResponse(BaseModel):
    """API response for a paginated list of drift detections."""

    model_config = ConfigDict(frozen=True)

    items: list[DriftDetectionResponse]
    pagination: PaginationMeta


# ---------------------------------------------------------------------------
# DriftAlert schemas
# ---------------------------------------------------------------------------


class AlertAcknowledgeRequest(BaseModel):
    """Request body for acknowledging a drift alert."""

    note: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional operator note attached to the acknowledgement",
    )


class DriftAlertResponse(BaseModel):
    """API response for a single drift alert."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    detection_id: uuid.UUID
    severity: str
    channel: str
    message: str
    acknowledged: bool
    acknowledged_by: uuid.UUID | None
    acknowledged_at: datetime | None
    created_at: datetime


class AlertListResponse(BaseModel):
    """API response for a paginated list of drift alerts."""

    model_config = ConfigDict(frozen=True)

    items: list[DriftAlertResponse]
    pagination: PaginationMeta


# ---------------------------------------------------------------------------
# Dashboard schemas
# ---------------------------------------------------------------------------


class MonitorSummaryItem(BaseModel):
    """Summary statistics for a single monitor on the dashboard."""

    model_config = ConfigDict(frozen=True)

    monitor_id: uuid.UUID
    monitor_name: str
    model_id: uuid.UUID
    status: str
    total_checks: int
    drifted_checks: int
    last_check_at: datetime | None
    last_drift_at: datetime | None
    unacknowledged_alerts: int


class DashboardResponse(BaseModel):
    """Drift dashboard summary aggregated across all monitors for a tenant."""

    model_config = ConfigDict(frozen=True)

    tenant_id: uuid.UUID
    window_days: Annotated[int, Field(ge=1, le=90)]
    total_monitors: int
    active_monitors: int
    total_checks: int
    drifted_checks: int
    drift_rate_percent: float
    unacknowledged_alerts: int
    monitors: list[MonitorSummaryItem]


# ---------------------------------------------------------------------------
# Performance estimation schemas (GAP-165: CBPE/DLE)
# ---------------------------------------------------------------------------


class PerformanceEstimateRequest(BaseModel):
    """Request body for labelless performance estimation."""

    production_probabilities: list[list[float]] = Field(
        description="Per-sample class probabilities from the model in production",
    )
    production_features: list[list[float]] | None = Field(
        default=None,
        description="Raw feature vectors for DLE estimation (optional, required for DLE method)",
    )
    method: Literal["cbpe", "dle"] = Field(
        default="cbpe",
        description="Estimation method: 'cbpe' (confidence-based) or 'dle' (direct loss)",
    )
    metric: str = Field(
        default="accuracy",
        description="Performance metric to estimate: accuracy, f1, auroc, log_loss",
    )


class PerformanceEstimateResponse(BaseModel):
    """Response schema for a performance estimate."""

    model_config = ConfigDict(frozen=True)

    monitor_id: uuid.UUID = Field(description="Monitor that was evaluated")
    method: str = Field(description="Estimation method used")
    metric: str = Field(description="Metric estimated")
    estimated_value: float = Field(description="Estimated metric value")
    confidence_interval_95: dict = Field(description="95% bootstrap confidence interval")
    n_samples: int = Field(description="Number of production samples used")
    estimated_at: datetime = Field(description="Estimation timestamp")


# ---------------------------------------------------------------------------
# Multivariate drift schemas (GAP-169)
# ---------------------------------------------------------------------------


class MultivariateDriftRequest(BaseModel):
    """Request body for multivariate drift detection."""

    reference_data: list[list[float]] = Field(
        description="Reference feature matrix, shape (n_ref, n_features)",
    )
    production_data: list[list[float]] = Field(
        description="Production feature matrix, shape (n_prod, n_features)",
    )
    method: Literal["pca", "classifier"] = Field(
        default="pca",
        description="Multivariate drift method: 'pca' (reconstruction error) or 'classifier' (C2ST)",
    )
    threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Drift detection threshold")


class MultivariateDriftResponse(BaseModel):
    """Response for a multivariate drift check."""

    model_config = ConfigDict(frozen=True)

    method: str
    drift_detected: bool
    score: float
    threshold: float
    details: dict


# ---------------------------------------------------------------------------
# HTML report schemas (GAP-170)
# ---------------------------------------------------------------------------


class ReportGenerateRequest(BaseModel):
    """Request body for generating an interactive HTML drift report."""

    monitor_id: uuid.UUID = Field(description="Monitor whose history to include in the report")
    window_days: int = Field(default=30, ge=1, le=365, description="Lookback window in days")


class ReportGenerateResponse(BaseModel):
    """Response containing the generated report download URL."""

    model_config = ConfigDict(frozen=True)

    monitor_id: uuid.UUID
    report_uri: str = Field(description="S3/MinIO URI where the HTML report is stored")
    generated_at: datetime


# ---------------------------------------------------------------------------
# Custom plugin schemas (GAP-173)
# ---------------------------------------------------------------------------


class CustomPluginRunRequest(BaseModel):
    """Request body for running a custom drift test plugin."""

    plugin_code: str = Field(
        description="Python source code defining drift_test(reference, production) -> dict",
        min_length=10,
    )
    reference_data: list[float] = Field(description="Reference distribution samples")
    production_data: list[float] = Field(description="Production distribution samples")


class CustomPluginRunResponse(BaseModel):
    """Response from a custom plugin drift test run."""

    model_config = ConfigDict(frozen=True)

    drift_detected: bool
    score: float | None = None
    details: dict
