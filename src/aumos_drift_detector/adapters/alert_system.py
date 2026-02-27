"""Configurable drift alerting system with multi-channel notification dispatch.

Implements rule-based alert evaluation, multi-channel dispatch (Slack, email,
PagerDuty), deduplication within configurable windows, severity-based escalation,
and a full alert history with acknowledgement tracking.

Example:
    >>> system = DriftAlertSystem()
    >>> system.add_rule(AlertRule(metric="psi_score", threshold=0.25, severity=AlertSeverity.WARNING, channel=AlertChannel.SLACK))
    >>> alerts = await system.evaluate_and_dispatch(monitor_id=uuid4(), metric_values={"psi_score": 0.31})
    >>> len(alerts)
    1
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import smtplib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels, ordered from least to most severe."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Notification channel types."""

    INTERNAL = "internal"
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"


class AlertStatus(str, Enum):
    """Lifecycle status of a dispatched alert."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


@dataclass
class AlertRule:
    """A configurable alert rule that triggers when a metric crosses a threshold.

    Attributes:
        rule_id: Unique identifier for this rule.
        metric: Metric name to evaluate (e.g., 'psi_score', 'ks_p_value').
        threshold: Threshold value. Alert fires when metric exceeds this.
        severity: Alert severity level.
        channel: Notification channel for dispatching.
        model_id: Optional UUID to restrict rule to a specific model.
        monitor_id: Optional UUID to restrict rule to a specific monitor.
        comparison: 'gt' (greater than) or 'lt' (less than) for threshold check.
        cooldown_minutes: Minimum minutes between repeated alerts for this rule.
        enabled: Whether this rule is currently active.
    """

    metric: str
    threshold: float
    severity: AlertSeverity
    channel: AlertChannel
    rule_id: uuid.UUID = field(default_factory=uuid.uuid4)
    model_id: uuid.UUID | None = None
    monitor_id: uuid.UUID | None = None
    comparison: str = "gt"
    cooldown_minutes: int = 60
    enabled: bool = True


@dataclass
class DispatchedAlert:
    """A dispatched alert instance with full lifecycle state.

    Attributes:
        alert_id: Unique identifier for this alert.
        rule_id: UUID of the AlertRule that triggered this alert.
        monitor_id: UUID of the drift monitor.
        metric: Metric name that triggered the rule.
        metric_value: Observed metric value.
        threshold: Rule threshold value.
        severity: Alert severity.
        channel: Notification channel used.
        status: Current alert lifecycle status.
        message: Human-readable alert message.
        raised_at: UTC timestamp when alert was raised.
        dispatched_at: UTC timestamp of dispatch (None if pending).
        acknowledged_at: UTC timestamp of acknowledgement (None if unacknowledged).
        acknowledged_by: UUID of acknowledging user (None if unacknowledged).
        escalated_at: UTC timestamp of escalation (None if not escalated).
        dedup_key: Fingerprint for deduplication.
    """

    alert_id: uuid.UUID
    rule_id: uuid.UUID
    monitor_id: uuid.UUID
    metric: str
    metric_value: float
    threshold: float
    severity: AlertSeverity
    channel: AlertChannel
    status: AlertStatus
    message: str
    raised_at: datetime
    dispatched_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: uuid.UUID | None = None
    escalated_at: datetime | None = None
    dedup_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all alert fields.
        """
        return {
            "alert_id": str(self.alert_id),
            "rule_id": str(self.rule_id),
            "monitor_id": str(self.monitor_id),
            "metric": self.metric,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "channel": self.channel.value,
            "status": self.status.value,
            "message": self.message,
            "raised_at": self.raised_at.isoformat(),
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": str(self.acknowledged_by) if self.acknowledged_by else None,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "dedup_key": self.dedup_key,
        }


@dataclass
class AlertChannelConfig:
    """Configuration for a notification channel.

    Attributes:
        slack_webhook_url: Slack incoming webhook URL.
        email_smtp_host: SMTP server hostname.
        email_smtp_port: SMTP server port.
        email_sender: Sender email address.
        email_recipients: List of recipient email addresses.
        email_smtp_username: SMTP authentication username.
        email_smtp_password: SMTP authentication password.
        pagerduty_routing_key: PagerDuty Events API v2 routing key.
        pagerduty_api_url: PagerDuty Events API URL.
        http_timeout_seconds: HTTP request timeout for webhook calls.
    """

    slack_webhook_url: str = ""
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_sender: str = "aumos-drift@noreply.example.com"
    email_recipients: list[str] = field(default_factory=list)
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    pagerduty_routing_key: str = ""
    pagerduty_api_url: str = "https://events.pagerduty.com/v2/enqueue"
    http_timeout_seconds: float = 10.0


class DriftAlertSystem:
    """Configurable drift alerting system with multi-channel dispatch.

    Manages alert rules, evaluates metric values against thresholds,
    dispatches notifications to configured channels, deduplicates alerts
    within cooldown windows, escalates by severity, and maintains a full
    history with acknowledgement support.

    Args:
        channel_config: Configuration for notification channels (Slack, email, PagerDuty).
        escalation_timeout_minutes: Minutes before an unacknowledged WARNING is
            escalated to CRITICAL and re-dispatched.
    """

    def __init__(
        self,
        channel_config: AlertChannelConfig | None = None,
        escalation_timeout_minutes: int = 120,
    ) -> None:
        """Initialise the alert system.

        Args:
            channel_config: Channel configuration; uses defaults if None.
            escalation_timeout_minutes: Minutes until unacknowledged alerts escalate.
        """
        self._config = channel_config or AlertChannelConfig()
        self._escalation_timeout = timedelta(minutes=escalation_timeout_minutes)
        self._rules: dict[uuid.UUID, AlertRule] = {}
        self._history: list[DispatchedAlert] = []
        # Dedup tracking: dedup_key → last dispatch time
        self._last_dispatched: dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule.

        Args:
            rule: AlertRule to add.
        """
        self._rules[rule.rule_id] = rule
        logger.info(
            "Alert rule registered",
            rule_id=str(rule.rule_id),
            metric=rule.metric,
            threshold=rule.threshold,
            severity=rule.severity.value,
        )

    def remove_rule(self, rule_id: uuid.UUID) -> bool:
        """Remove an alert rule by ID.

        Args:
            rule_id: UUID of the rule to remove.

        Returns:
            True if removed, False if rule_id not found.
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info("Alert rule removed", rule_id=str(rule_id))
            return True
        return False

    def update_rule(self, rule_id: uuid.UUID, **kwargs: Any) -> bool:
        """Update fields on an existing alert rule.

        Args:
            rule_id: UUID of the rule to update.
            **kwargs: Field name to new value mappings (e.g., threshold=0.3).

        Returns:
            True if updated, False if not found.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            return False
        for key, value in kwargs.items():
            if hasattr(rule, key):
                object.__setattr__(rule, key, value) if False else setattr(rule, key, value)
        return True

    async def evaluate_and_dispatch(
        self,
        monitor_id: uuid.UUID,
        metric_values: dict[str, float],
        model_id: uuid.UUID | None = None,
    ) -> list[DispatchedAlert]:
        """Evaluate all enabled rules against metric values and dispatch alerts.

        For each rule matching the monitor/model scope, checks if the metric
        value crosses the threshold. Deduplication suppresses alerts that fired
        within the cooldown window. Dispatches to the configured channel.

        Args:
            monitor_id: UUID of the drift monitor being evaluated.
            metric_values: Dict of metric_name to current value.
            model_id: Optional model UUID for rule scoping.

        Returns:
            List of DispatchedAlert for all newly fired (not suppressed) alerts.
        """
        dispatched: list[DispatchedAlert] = []
        now = datetime.now(tz=timezone.utc)

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.monitor_id is not None and rule.monitor_id != monitor_id:
                continue
            if rule.model_id is not None and rule.model_id != model_id:
                continue

            metric_value = metric_values.get(rule.metric)
            if metric_value is None:
                continue

            # Evaluate threshold
            crossed = (
                (rule.comparison == "gt" and metric_value > rule.threshold)
                or (rule.comparison == "lt" and metric_value < rule.threshold)
            )
            if not crossed:
                continue

            # Compute dedup key
            dedup_key = self._make_dedup_key(rule.rule_id, monitor_id, rule.metric)
            last_sent = self._last_dispatched.get(dedup_key)
            cooldown = timedelta(minutes=rule.cooldown_minutes)
            if last_sent and (now - last_sent) < cooldown:
                logger.debug(
                    "Alert suppressed (cooldown active)",
                    rule_id=str(rule.rule_id),
                    metric=rule.metric,
                    cooldown_minutes=rule.cooldown_minutes,
                )
                # Record as suppressed in history
                suppressed_alert = self._create_alert(
                    rule=rule,
                    monitor_id=monitor_id,
                    metric_value=metric_value,
                    dedup_key=dedup_key,
                    status=AlertStatus.SUPPRESSED,
                )
                self._history.append(suppressed_alert)
                continue

            alert = self._create_alert(
                rule=rule,
                monitor_id=monitor_id,
                metric_value=metric_value,
                dedup_key=dedup_key,
                status=AlertStatus.PENDING,
            )
            await self._dispatch_alert(alert)
            self._last_dispatched[dedup_key] = now
            self._history.append(alert)
            dispatched.append(alert)

        return dispatched

    async def escalate_overdue_alerts(self) -> list[DispatchedAlert]:
        """Escalate unacknowledged WARNING alerts that have passed the escalation timeout.

        Promoted alerts get CRITICAL severity and are re-dispatched via PagerDuty
        (or whatever channel is configured for critical alerts).

        Returns:
            List of alerts that were escalated.
        """
        now = datetime.now(tz=timezone.utc)
        escalated: list[DispatchedAlert] = []

        for alert in self._history:
            if alert.status not in (AlertStatus.DISPATCHED,):
                continue
            if alert.severity != AlertSeverity.WARNING:
                continue
            if alert.raised_at is None:
                continue
            if (now - alert.raised_at) < self._escalation_timeout:
                continue

            alert.status = AlertStatus.ESCALATED
            alert.severity = AlertSeverity.CRITICAL
            alert.escalated_at = now
            alert.message = f"[ESCALATED] {alert.message}"

            await self._dispatch_pagerduty(alert)
            escalated.append(alert)
            logger.warning(
                "Alert escalated to CRITICAL",
                alert_id=str(alert.alert_id),
                monitor_id=str(alert.monitor_id),
                metric=alert.metric,
            )

        return escalated

    def acknowledge_alert(
        self,
        alert_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> bool:
        """Mark an alert as acknowledged by an operator.

        Args:
            alert_id: UUID of the alert to acknowledge.
            acknowledged_by: UUID of the acknowledging user.

        Returns:
            True if found and acknowledged, False if not found or already acknowledged.
        """
        for alert in self._history:
            if alert.alert_id == alert_id:
                if alert.status == AlertStatus.ACKNOWLEDGED:
                    return False
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now(tz=timezone.utc)
                alert.acknowledged_by = acknowledged_by
                logger.info(
                    "Alert acknowledged",
                    alert_id=str(alert_id),
                    acknowledged_by=str(acknowledged_by),
                )
                return True
        return False

    def get_alert_history(
        self,
        monitor_id: uuid.UUID | None = None,
        severity: AlertSeverity | None = None,
        status: AlertStatus | None = None,
        limit: int = 100,
    ) -> list[DispatchedAlert]:
        """Return alert history filtered by optional criteria.

        Args:
            monitor_id: Filter by monitor UUID.
            severity: Filter by severity level.
            status: Filter by lifecycle status.
            limit: Maximum number of results to return (newest-first).

        Returns:
            Filtered list of DispatchedAlert, newest-first.
        """
        results = list(reversed(self._history))
        if monitor_id is not None:
            results = [a for a in results if a.monitor_id == monitor_id]
        if severity is not None:
            results = [a for a in results if a.severity == severity]
        if status is not None:
            results = [a for a in results if a.status == status]
        return results[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_alert(
        self,
        rule: AlertRule,
        monitor_id: uuid.UUID,
        metric_value: float,
        dedup_key: str,
        status: AlertStatus,
    ) -> DispatchedAlert:
        """Build a DispatchedAlert from a rule match.

        Args:
            rule: The AlertRule that was triggered.
            monitor_id: UUID of the monitor being evaluated.
            metric_value: Current metric value.
            dedup_key: Pre-computed deduplication fingerprint.
            status: Initial alert status.

        Returns:
            Newly constructed DispatchedAlert.
        """
        direction = "exceeded" if rule.comparison == "gt" else "fell below"
        message = (
            f"Drift alert [{rule.severity.value.upper()}]: {rule.metric} {direction} "
            f"threshold {rule.threshold:.4f} (observed {metric_value:.4f}) "
            f"on monitor {monitor_id}."
        )
        return DispatchedAlert(
            alert_id=uuid.uuid4(),
            rule_id=rule.rule_id,
            monitor_id=monitor_id,
            metric=rule.metric,
            metric_value=metric_value,
            threshold=rule.threshold,
            severity=rule.severity,
            channel=rule.channel,
            status=status,
            message=message,
            raised_at=datetime.now(tz=timezone.utc),
            dedup_key=dedup_key,
        )

    async def _dispatch_alert(self, alert: DispatchedAlert) -> None:
        """Route an alert to its configured notification channel.

        Args:
            alert: The DispatchedAlert to send.
        """
        try:
            if alert.channel == AlertChannel.SLACK:
                await self._dispatch_slack(alert)
            elif alert.channel == AlertChannel.EMAIL:
                await self._dispatch_email(alert)
            elif alert.channel == AlertChannel.PAGERDUTY:
                await self._dispatch_pagerduty(alert)
            else:
                # INTERNAL — log only
                logger.info("Internal alert dispatched", alert_id=str(alert.alert_id), message=alert.message)

            alert.status = AlertStatus.DISPATCHED
            alert.dispatched_at = datetime.now(tz=timezone.utc)
        except Exception as exc:
            logger.error(
                "Alert dispatch failed",
                alert_id=str(alert.alert_id),
                channel=alert.channel.value,
                error=str(exc),
            )

    async def _dispatch_slack(self, alert: DispatchedAlert) -> None:
        """POST a Slack notification to the configured webhook URL.

        Args:
            alert: Alert to dispatch.

        Raises:
            ValueError: If no Slack webhook URL is configured.
            httpx.HTTPError: If the HTTP request fails.
        """
        if not self._config.slack_webhook_url:
            raise ValueError("Slack webhook URL not configured in AlertChannelConfig")

        severity_emoji = {"info": ":information_source:", "warning": ":warning:", "critical": ":rotating_light:"}
        emoji = severity_emoji.get(alert.severity.value, ":bell:")
        payload = {
            "text": f"{emoji} *AumOS Drift Detector* — {alert.severity.value.upper()}",
            "attachments": [
                {
                    "color": {"info": "good", "warning": "warning", "critical": "danger"}.get(
                        alert.severity.value, "warning"
                    ),
                    "fields": [
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Observed Value", "value": f"{alert.metric_value:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                        {"title": "Monitor ID", "value": str(alert.monitor_id), "short": True},
                        {"title": "Alert ID", "value": str(alert.alert_id), "short": False},
                        {"title": "Message", "value": alert.message, "short": False},
                    ],
                    "ts": int(alert.raised_at.timestamp()),
                }
            ],
        }

        async with httpx.AsyncClient(timeout=self._config.http_timeout_seconds) as client:
            response = await client.post(
                self._config.slack_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.info("Slack alert dispatched", alert_id=str(alert.alert_id))

    async def _dispatch_email(self, alert: DispatchedAlert) -> None:
        """Send an email notification via SMTP.

        Runs in an executor thread since smtplib is synchronous.

        Args:
            alert: Alert to dispatch.

        Raises:
            ValueError: If no recipients are configured.
        """
        if not self._config.email_recipients:
            raise ValueError("No email recipients configured in AlertChannelConfig")

        subject = f"[AumOS Drift] {alert.severity.value.upper()} — {alert.metric}"
        body_text = (
            f"AumOS Drift Detector Alert\n\n"
            f"Severity: {alert.severity.value.upper()}\n"
            f"Metric: {alert.metric}\n"
            f"Observed Value: {alert.metric_value:.6f}\n"
            f"Threshold: {alert.threshold:.6f}\n"
            f"Monitor ID: {alert.monitor_id}\n"
            f"Alert ID: {alert.alert_id}\n"
            f"Raised At: {alert.raised_at.isoformat()}\n\n"
            f"Message:\n{alert.message}"
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._config.email_sender
        msg["To"] = ", ".join(self._config.email_recipients)
        msg.attach(MIMEText(body_text, "plain"))

        def _send_sync() -> None:
            with smtplib.SMTP(self._config.email_smtp_host, self._config.email_smtp_port) as smtp:
                smtp.ehlo()
                smtp.starttls()
                if self._config.email_smtp_username:
                    smtp.login(self._config.email_smtp_username, self._config.email_smtp_password)
                smtp.sendmail(
                    self._config.email_sender,
                    self._config.email_recipients,
                    msg.as_string(),
                )

        await asyncio.get_event_loop().run_in_executor(None, _send_sync)
        logger.info(
            "Email alert dispatched",
            alert_id=str(alert.alert_id),
            recipients=self._config.email_recipients,
        )

    async def _dispatch_pagerduty(self, alert: DispatchedAlert) -> None:
        """Create a PagerDuty incident via the Events API v2.

        Args:
            alert: Alert to dispatch.

        Raises:
            ValueError: If no PagerDuty routing key is configured.
            httpx.HTTPError: If the API request fails.
        """
        if not self._config.pagerduty_routing_key:
            raise ValueError("PagerDuty routing key not configured in AlertChannelConfig")

        payload = {
            "routing_key": self._config.pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": alert.dedup_key,
            "payload": {
                "summary": alert.message,
                "severity": alert.severity.value,
                "source": "aumos-drift-detector",
                "timestamp": alert.raised_at.isoformat(),
                "custom_details": {
                    "alert_id": str(alert.alert_id),
                    "monitor_id": str(alert.monitor_id),
                    "metric": alert.metric,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                },
            },
        }

        async with httpx.AsyncClient(timeout=self._config.http_timeout_seconds) as client:
            response = await client.post(
                self._config.pagerduty_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.info("PagerDuty incident created", alert_id=str(alert.alert_id))

    @staticmethod
    def _make_dedup_key(
        rule_id: uuid.UUID,
        monitor_id: uuid.UUID,
        metric: str,
    ) -> str:
        """Generate a stable deduplication fingerprint for an alert.

        Args:
            rule_id: UUID of the alert rule.
            monitor_id: UUID of the monitor.
            metric: Metric name.

        Returns:
            SHA-256 hex digest (first 16 characters).
        """
        raw = f"{rule_id}:{monitor_id}:{metric}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
