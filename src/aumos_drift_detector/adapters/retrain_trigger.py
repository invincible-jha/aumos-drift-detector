"""Automated retraining trigger adapter for drift-driven model retraining.

Evaluates drift detection results, performance degradation signals, and
periodic schedules to determine when a model should be retrained. Enforces
cooldown periods to prevent thrashing, dispatches Kafka events to the
aumos-mlops-lifecycle service, and maintains a full trigger history.

Example:
    >>> trigger = RetrainTrigger(publisher=my_kafka_publisher)
    >>> trigger.configure_policy(model_id=uuid4(), policy=TriggerPolicy(drift_score_threshold=0.3))
    >>> event = await trigger.evaluate_drift_trigger(tenant_id=..., monitor_id=..., model_id=...,
    ...     detection_id=..., drift_score=0.45, test_name="psi")
    >>> event.triggered
    True
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

from aumos_drift_detector.core.interfaces import IDriftEventPublisher

logger = get_logger(__name__)


class TriggerReason(str, Enum):
    """Classification of what caused a retraining trigger."""

    DRIFT_SCORE = "drift_score"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class TriggerOutcome(str, Enum):
    """Outcome of a trigger evaluation."""

    TRIGGERED = "triggered"
    SUPPRESSED_COOLDOWN = "suppressed_cooldown"
    SUPPRESSED_THRESHOLD = "suppressed_threshold"
    SUPPRESSED_DISABLED = "suppressed_disabled"


@dataclass
class TriggerPolicy:
    """Per-model trigger policy configuration.

    Attributes:
        drift_score_threshold: Minimum drift score to trigger retraining (0–1).
            For p-value tests (KS, chi2) this is the maximum p-value (e.g., 0.05).
            For PSI this is the minimum PSI (e.g., 0.25).
        performance_accuracy_threshold: Minimum accuracy below which retraining fires.
        performance_rmse_threshold: Maximum RMSE above which retraining fires.
        cooldown_hours: Minimum hours between successive retraining triggers for a model.
        scheduled_interval_hours: If > 0, trigger retraining every N hours regardless
            of drift/performance signals.
        enabled: Whether this policy is active.
        require_all_tests_drift: If True, ALL statistical tests must show drift before
            triggering; if False, ANY test crossing threshold triggers retraining.
    """

    drift_score_threshold: float = 0.25
    performance_accuracy_threshold: float | None = None
    performance_rmse_threshold: float | None = None
    cooldown_hours: int = 24
    scheduled_interval_hours: int = 0
    enabled: bool = True
    require_all_tests_drift: bool = False


@dataclass
class TriggerEvent:
    """Record of a retraining trigger evaluation.

    Attributes:
        event_id: Unique identifier for this trigger event.
        tenant_id: Owning tenant UUID.
        monitor_id: Drift monitor UUID.
        model_id: Target model UUID.
        detection_id: DriftDetection UUID that caused this evaluation (if drift-based).
        reason: What caused this trigger evaluation.
        outcome: Result of the evaluation.
        triggered: True if retraining was actually dispatched.
        drift_score: Drift score at time of evaluation.
        test_name: Name of the statistical test that fired.
        kafka_published: True if the Kafka event was successfully published.
        evaluated_at: UTC timestamp of this evaluation.
        message: Human-readable description.
    """

    event_id: uuid.UUID
    tenant_id: uuid.UUID
    monitor_id: uuid.UUID
    model_id: uuid.UUID
    detection_id: uuid.UUID | None
    reason: TriggerReason
    outcome: TriggerOutcome
    triggered: bool
    drift_score: float
    test_name: str
    kafka_published: bool
    evaluated_at: datetime
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all trigger event fields.
        """
        return {
            "event_id": str(self.event_id),
            "tenant_id": str(self.tenant_id),
            "monitor_id": str(self.monitor_id),
            "model_id": str(self.model_id),
            "detection_id": str(self.detection_id) if self.detection_id else None,
            "reason": self.reason.value,
            "outcome": self.outcome.value,
            "triggered": self.triggered,
            "drift_score": self.drift_score,
            "test_name": self.test_name,
            "kafka_published": self.kafka_published,
            "evaluated_at": self.evaluated_at.isoformat(),
            "message": self.message,
        }


class RetrainTrigger:
    """Automated retraining trigger with cooldown enforcement and Kafka dispatch.

    Evaluates three trigger signals:
    1. Drift-based: statistical test score crosses a per-model policy threshold.
    2. Performance-based: accuracy/RMSE metric crosses a policy threshold.
    3. Schedule-based: periodic retraining at a configured interval.

    Cooldown enforcement prevents retraining from being triggered more
    frequently than the configured cooldown_hours per model.

    Args:
        publisher: IDriftEventPublisher implementation for Kafka dispatch.
        default_policy: TriggerPolicy applied to models without explicit configuration.
    """

    def __init__(
        self,
        publisher: IDriftEventPublisher,
        default_policy: TriggerPolicy | None = None,
    ) -> None:
        """Initialise the retraining trigger.

        Args:
            publisher: Kafka event publisher implementation.
            default_policy: Fallback policy for unconfigured models.
        """
        self._publisher = publisher
        self._default_policy = default_policy or TriggerPolicy()
        # Per-model policy overrides
        self._policies: dict[uuid.UUID, TriggerPolicy] = {}
        # Last trigger time per model (for cooldown enforcement)
        self._last_triggered: dict[uuid.UUID, datetime] = {}
        # Full trigger history
        self._history: list[TriggerEvent] = []

    def configure_policy(self, model_id: uuid.UUID, policy: TriggerPolicy) -> None:
        """Set a trigger policy for a specific model.

        Args:
            model_id: UUID of the model to configure.
            policy: TriggerPolicy defining thresholds and cooldown for this model.
        """
        self._policies[model_id] = policy
        logger.info(
            "Trigger policy configured",
            model_id=str(model_id),
            drift_score_threshold=policy.drift_score_threshold,
            cooldown_hours=policy.cooldown_hours,
            enabled=policy.enabled,
        )

    def get_policy(self, model_id: uuid.UUID) -> TriggerPolicy:
        """Return the effective policy for a model (explicit or default).

        Args:
            model_id: UUID of the model.

        Returns:
            TriggerPolicy for this model.
        """
        return self._policies.get(model_id, self._default_policy)

    async def evaluate_drift_trigger(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID,
        drift_score: float,
        test_name: str,
    ) -> TriggerEvent:
        """Evaluate whether a drift detection result should trigger retraining.

        Checks the per-model policy threshold and cooldown. If criteria are met,
        publishes a drift.retraining_required Kafka event.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: UUID of the drift monitor.
            model_id: UUID of the model to potentially retrain.
            detection_id: UUID of the DriftDetection that fired.
            drift_score: The aggregate drift score from the statistical test.
            test_name: Name of the test that produced this score.

        Returns:
            TriggerEvent recording the evaluation result.
        """
        policy = self.get_policy(model_id)

        if not policy.enabled:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=detection_id,
                reason=TriggerReason.DRIFT_SCORE,
                outcome=TriggerOutcome.SUPPRESSED_DISABLED,
                triggered=False,
                drift_score=drift_score,
                test_name=test_name,
                kafka_published=False,
                message=f"Trigger disabled by policy for model {model_id}.",
            )

        # Check threshold (PSI and similar: higher = more drift; p-value: lower = more drift)
        is_p_value_test = test_name in ("ks", "chi2", "ks_2samp")
        if is_p_value_test:
            threshold_crossed = drift_score < policy.drift_score_threshold
        else:
            threshold_crossed = drift_score > policy.drift_score_threshold

        if not threshold_crossed:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=detection_id,
                reason=TriggerReason.DRIFT_SCORE,
                outcome=TriggerOutcome.SUPPRESSED_THRESHOLD,
                triggered=False,
                drift_score=drift_score,
                test_name=test_name,
                kafka_published=False,
                message=(
                    f"{test_name} score {drift_score:.4f} did not cross trigger threshold "
                    f"{policy.drift_score_threshold:.4f}."
                ),
            )

        # Check cooldown
        if not self._is_cooldown_clear(model_id, policy.cooldown_hours):
            last = self._last_triggered[model_id]
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=detection_id,
                reason=TriggerReason.DRIFT_SCORE,
                outcome=TriggerOutcome.SUPPRESSED_COOLDOWN,
                triggered=False,
                drift_score=drift_score,
                test_name=test_name,
                kafka_published=False,
                message=(
                    f"Cooldown active — last trigger at {last.isoformat()}, "
                    f"cooldown={policy.cooldown_hours}h."
                ),
            )

        # Trigger retraining
        reason_text = (
            f"{test_name} drift score {drift_score:.4f} crossed threshold "
            f"{policy.drift_score_threshold:.4f}"
        )
        kafka_ok = await self._publish_retraining_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=detection_id,
            reason=reason_text,
        )
        self._last_triggered[model_id] = datetime.now(tz=timezone.utc)

        return self._record_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=detection_id,
            reason=TriggerReason.DRIFT_SCORE,
            outcome=TriggerOutcome.TRIGGERED,
            triggered=True,
            drift_score=drift_score,
            test_name=test_name,
            kafka_published=kafka_ok,
            message=f"Retraining triggered: {reason_text}.",
        )

    async def evaluate_performance_trigger(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        accuracy: float | None = None,
        rmse: float | None = None,
    ) -> TriggerEvent:
        """Evaluate whether performance degradation warrants retraining.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: UUID of the drift monitor.
            model_id: UUID of the model.
            accuracy: Current accuracy value (None = not applicable).
            rmse: Current RMSE value (None = not applicable).

        Returns:
            TriggerEvent recording the evaluation result.
        """
        policy = self.get_policy(model_id)

        if not policy.enabled:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=None,
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                outcome=TriggerOutcome.SUPPRESSED_DISABLED,
                triggered=False,
                drift_score=0.0,
                test_name="performance",
                kafka_published=False,
                message="Trigger disabled by policy.",
            )

        threshold_crossed = False
        trigger_reason_text = ""

        if accuracy is not None and policy.performance_accuracy_threshold is not None:
            if accuracy < policy.performance_accuracy_threshold:
                threshold_crossed = True
                trigger_reason_text = (
                    f"Accuracy {accuracy:.4f} below threshold {policy.performance_accuracy_threshold:.4f}"
                )

        if rmse is not None and policy.performance_rmse_threshold is not None:
            if rmse > policy.performance_rmse_threshold:
                threshold_crossed = True
                trigger_reason_text = (
                    f"RMSE {rmse:.4f} above threshold {policy.performance_rmse_threshold:.4f}"
                )

        if not threshold_crossed:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=None,
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                outcome=TriggerOutcome.SUPPRESSED_THRESHOLD,
                triggered=False,
                drift_score=0.0,
                test_name="performance",
                kafka_published=False,
                message="Performance within acceptable range.",
            )

        if not self._is_cooldown_clear(model_id, policy.cooldown_hours):
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=None,
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                outcome=TriggerOutcome.SUPPRESSED_COOLDOWN,
                triggered=False,
                drift_score=0.0,
                test_name="performance",
                kafka_published=False,
                message="Cooldown active.",
            )

        kafka_ok = await self._publish_retraining_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=None,
            reason=trigger_reason_text,
        )
        self._last_triggered[model_id] = datetime.now(tz=timezone.utc)

        return self._record_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=None,
            reason=TriggerReason.PERFORMANCE_DEGRADATION,
            outcome=TriggerOutcome.TRIGGERED,
            triggered=True,
            drift_score=0.0,
            test_name="performance",
            kafka_published=kafka_ok,
            message=f"Retraining triggered by performance: {trigger_reason_text}.",
        )

    async def evaluate_scheduled_trigger(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
    ) -> TriggerEvent:
        """Evaluate whether the periodic schedule warrants a retraining trigger.

        Args:
            tenant_id: Owning tenant UUID.
            monitor_id: UUID of the monitor.
            model_id: UUID of the model.

        Returns:
            TriggerEvent recording the evaluation result.
        """
        policy = self.get_policy(model_id)

        if not policy.enabled or policy.scheduled_interval_hours <= 0:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=None,
                reason=TriggerReason.SCHEDULED,
                outcome=TriggerOutcome.SUPPRESSED_DISABLED,
                triggered=False,
                drift_score=0.0,
                test_name="schedule",
                kafka_published=False,
                message="Scheduled retraining not enabled in policy.",
            )

        last_trigger = self._last_triggered.get(model_id)
        now = datetime.now(tz=timezone.utc)
        interval = timedelta(hours=policy.scheduled_interval_hours)

        if last_trigger is not None and (now - last_trigger) < interval:
            return self._record_event(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=None,
                reason=TriggerReason.SCHEDULED,
                outcome=TriggerOutcome.SUPPRESSED_COOLDOWN,
                triggered=False,
                drift_score=0.0,
                test_name="schedule",
                kafka_published=False,
                message=f"Next scheduled trigger at {(last_trigger + interval).isoformat()}.",
            )

        kafka_ok = await self._publish_retraining_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=None,
            reason=f"Scheduled retraining (interval={policy.scheduled_interval_hours}h)",
        )
        self._last_triggered[model_id] = now

        return self._record_event(
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=None,
            reason=TriggerReason.SCHEDULED,
            outcome=TriggerOutcome.TRIGGERED,
            triggered=True,
            drift_score=0.0,
            test_name="schedule",
            kafka_published=kafka_ok,
            message=f"Scheduled retraining triggered (interval={policy.scheduled_interval_hours}h).",
        )

    def get_trigger_history(
        self,
        model_id: uuid.UUID | None = None,
        triggered_only: bool = False,
        limit: int = 100,
    ) -> list[TriggerEvent]:
        """Return trigger history with optional filters.

        Args:
            model_id: Filter by model UUID.
            triggered_only: If True, return only events where triggered=True.
            limit: Maximum number of results (newest-first).

        Returns:
            Filtered list of TriggerEvent, newest-first.
        """
        results = list(reversed(self._history))
        if model_id is not None:
            results = [e for e in results if e.model_id == model_id]
        if triggered_only:
            results = [e for e in results if e.triggered]
        return results[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_cooldown_clear(self, model_id: uuid.UUID, cooldown_hours: int) -> bool:
        """Check whether the cooldown period has elapsed for a model.

        Args:
            model_id: UUID of the model to check.
            cooldown_hours: Required cooldown duration in hours.

        Returns:
            True if cooldown has elapsed or model has never been triggered.
        """
        last = self._last_triggered.get(model_id)
        if last is None:
            return True
        return (datetime.now(tz=timezone.utc) - last) >= timedelta(hours=cooldown_hours)

    async def _publish_retraining_event(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID | None,
        reason: str,
    ) -> bool:
        """Publish a drift.retraining_required event to Kafka.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Monitor UUID.
            model_id: Target model UUID.
            detection_id: Detection UUID (may be None for performance/schedule triggers).
            reason: Human-readable trigger reason.

        Returns:
            True if published successfully, False on error.
        """
        try:
            await self._publisher.publish_retraining_required(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                model_id=model_id,
                detection_id=detection_id or uuid.uuid4(),
                reason=reason,
            )
            logger.info(
                "Retraining event published",
                model_id=str(model_id),
                monitor_id=str(monitor_id),
                reason=reason,
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to publish retraining event",
                model_id=str(model_id),
                error=str(exc),
            )
            return False

    def _record_event(
        self,
        tenant_id: uuid.UUID,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        detection_id: uuid.UUID | None,
        reason: TriggerReason,
        outcome: TriggerOutcome,
        triggered: bool,
        drift_score: float,
        test_name: str,
        kafka_published: bool,
        message: str,
    ) -> TriggerEvent:
        """Create and store a TriggerEvent in history.

        Args:
            tenant_id: Owning tenant.
            monitor_id: Monitor UUID.
            model_id: Model UUID.
            detection_id: Detection UUID (may be None).
            reason: Trigger reason classification.
            outcome: Trigger evaluation outcome.
            triggered: Whether retraining was dispatched.
            drift_score: Drift score at evaluation time.
            test_name: Statistical test or trigger type name.
            kafka_published: Whether the Kafka event was successfully published.
            message: Human-readable description.

        Returns:
            Newly created and stored TriggerEvent.
        """
        event = TriggerEvent(
            event_id=uuid.uuid4(),
            tenant_id=tenant_id,
            monitor_id=monitor_id,
            model_id=model_id,
            detection_id=detection_id,
            reason=reason,
            outcome=outcome,
            triggered=triggered,
            drift_score=drift_score,
            test_name=test_name,
            kafka_published=kafka_published,
            evaluated_at=datetime.now(tz=timezone.utc),
            message=message,
        )
        self._history.append(event)
        logger.debug(
            "Trigger event recorded",
            event_id=str(event.event_id),
            outcome=outcome.value,
            triggered=triggered,
        )
        return event
