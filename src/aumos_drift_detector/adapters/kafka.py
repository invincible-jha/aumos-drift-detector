"""Kafka event publisher for drift detection events.

Publishes three event types:
- drift.detected      — any drift check result (drifted or not)
- drift.retraining_required — threshold crossed, retraining needed
- drift.alert_raised  — alert created for a detected drift
"""

import uuid
from datetime import UTC, datetime

from aumos_common.config import KafkaSettings
from aumos_common.events import EventPublisher
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Kafka topic names for drift events
TOPIC_DRIFT_DETECTED = "drift.detected"
TOPIC_RETRAINING_REQUIRED = "drift.retraining_required"
TOPIC_ALERT_RAISED = "drift.alert_raised"


class DriftEventPublisher:
    """Kafka publisher for AumOS drift detection events.

    Wraps aumos_common.EventPublisher and provides strongly-typed
    methods for each drift event type.
    """

    def __init__(self, kafka_settings: KafkaSettings) -> None:
        """Initialise with Kafka connection settings.

        Args:
            kafka_settings: Kafka bootstrap server and client configuration.
        """
        self._publisher = EventPublisher(kafka_settings)

    async def start(self) -> None:
        """Start the underlying Kafka producer connection.

        Should be called during application startup in the lifespan handler.
        """
        await self._publisher.start()
        logger.info("DriftEventPublisher started")

    async def stop(self) -> None:
        """Flush pending messages and close the Kafka producer.

        Should be called during application shutdown in the lifespan handler.
        """
        await self._publisher.stop()
        logger.info("DriftEventPublisher stopped")

    async def publish_drift_detected(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        detection_id: uuid.UUID,
        test_name: str,
        score: float,
        is_drifted: bool,
    ) -> None:
        """Publish a drift.detected event to Kafka.

        Published for every drift check, regardless of whether drift was found.
        Consumers that only care about positive detections should filter on
        the `is_drifted` field.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Monitor that produced this detection.
            detection_id: Detection record UUID.
            test_name: Name of the algorithm that ran.
            score: Raw drift score (p-value or PSI).
            is_drifted: True if drift was detected.
        """
        payload = {
            "event_type": "drift.detected",
            "tenant_id": str(tenant_id),
            "monitor_id": str(monitor_id),
            "detection_id": str(detection_id),
            "test_name": test_name,
            "score": score,
            "is_drifted": is_drifted,
            "occurred_at": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(TOPIC_DRIFT_DETECTED, payload)
        logger.debug(
            "Published drift.detected",
            detection_id=str(detection_id),
            is_drifted=is_drifted,
        )

    async def publish_retraining_required(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID,
        reason: str,
    ) -> None:
        """Publish a drift.retraining_required event to Kafka.

        Consumed by aumos-mlops-lifecycle to schedule a retraining job
        for the affected model.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: Monitor that flagged the need for retraining.
            model_id: Model UUID to retrain.
            detection_id: Detection that triggered this retraining request.
            reason: Human-readable description of which test failed.
        """
        payload = {
            "event_type": "drift.retraining_required",
            "tenant_id": str(tenant_id),
            "monitor_id": str(monitor_id),
            "model_id": str(model_id),
            "detection_id": str(detection_id),
            "reason": reason,
            "occurred_at": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(TOPIC_RETRAINING_REQUIRED, payload)
        logger.info(
            "Published drift.retraining_required",
            model_id=str(model_id),
            reason=reason,
        )

    async def publish_alert_raised(
        self,
        tenant_id: uuid.UUID,
        alert_id: uuid.UUID,
        severity: str,
        message: str,
    ) -> None:
        """Publish a drift.alert_raised event to Kafka.

        Consumed by notification services to send alerts via configured channels.

        Args:
            tenant_id: Owning tenant UUID.
            alert_id: Alert record UUID.
            severity: Alert severity level (info | warning | critical).
            message: Alert message text.
        """
        payload = {
            "event_type": "drift.alert_raised",
            "tenant_id": str(tenant_id),
            "alert_id": str(alert_id),
            "severity": severity,
            "message": message,
            "occurred_at": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(TOPIC_ALERT_RAISED, payload)
        logger.debug(
            "Published drift.alert_raised",
            alert_id=str(alert_id),
            severity=severity,
        )
