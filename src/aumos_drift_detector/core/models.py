"""SQLAlchemy ORM models for the AumOS Drift Detector.

All tables use the `drf_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel, Base


class DriftMonitor(AumOSModel):
    """Configuration for a scheduled drift monitor attached to a deployed model.

    A monitor defines which features to watch, the reference dataset URI,
    the cron schedule for periodic checks, and per-feature drift thresholds.

    Table: drf_monitors
    """

    __tablename__ = "drf_monitors"

    # Model being monitored (FK to aumos-model-registry conceptually — stored as UUID)
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="UUID of the model being monitored (from aumos-model-registry)",
    )

    # Human-readable name for this monitor
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # List of feature column names to include in drift checks
    feature_columns: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Array of feature column names to monitor",
    )

    # URI to the reference (baseline) dataset — S3/MinIO path
    reference_data_uri: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="S3/MinIO URI to the reference dataset (parquet or CSV)",
    )

    # Cron expression for scheduled drift checks (e.g. '0 * * * *' = hourly)
    schedule_cron: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Cron expression for scheduled runs; null = manual-only",
    )

    # Monitor state: active | paused | disabled
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        comment="active | paused | disabled",
    )

    # Per-monitor threshold overrides (JSON). Keys: ks, psi, chi2, adwin, ddm
    thresholds: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Per-monitor threshold overrides; empty dict uses global defaults",
    )

    detections: Mapped[list["DriftDetection"]] = relationship(
        "DriftDetection",
        back_populates="monitor",
        cascade="all, delete-orphan",
        order_by="desc(DriftDetection.detected_at)",
    )


class DriftDetection(Base):
    """A single drift detection result produced by running a monitor.

    Captures the test name, raw score, threshold used, drift verdict,
    and a detailed breakdown per feature or window.

    Table: drf_detections
    """

    __tablename__ = "drf_detections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    monitor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drf_monitors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # detection_type: statistical | concept
    detection_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="statistical | concept",
    )

    # Name of the test that produced this result
    test_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="ks | psi | chi2 | adwin | ddm | eddm",
    )

    # Aggregate drift score for this detection run
    score: Mapped[float] = mapped_column(nullable=False)

    # The threshold value that was compared against the score
    threshold: Mapped[float] = mapped_column(nullable=False)

    # True if score exceeded the threshold (i.e., drift detected)
    is_drifted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Per-feature breakdown, window details, p-values, etc.
    details: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Per-feature scores, p-values, and any additional test metadata",
    )

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    monitor: Mapped["DriftMonitor"] = relationship(
        "DriftMonitor", back_populates="detections"
    )

    alerts: Mapped[list["DriftAlert"]] = relationship(
        "DriftAlert",
        back_populates="detection",
        cascade="all, delete-orphan",
    )


class DriftAlert(Base):
    """An alert raised when a DriftDetection crosses severity thresholds.

    Alerts can be acknowledged by platform operators. Each alert is linked
    to the DriftDetection that triggered it.

    Table: drf_alerts
    """

    __tablename__ = "drf_alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    detection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drf_detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # severity: info | warning | critical
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="info | warning | critical",
    )

    # Notification channel: email | slack | pagerduty | webhook
    channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="internal",
        comment="email | slack | pagerduty | webhook | internal",
    )

    # Human-readable alert message
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Whether this alert has been acknowledged by an operator
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # UUID of the user who acknowledged the alert (null if not yet acknowledged)
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    detection: Mapped["DriftDetection"] = relationship(
        "DriftDetection", back_populates="alerts"
    )
