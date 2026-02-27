"""Automated drift report generation adapter.

Produces structured drift reports in JSON format with executive summaries,
per-feature assessments, performance impact analysis, recommended actions,
and visualisation data (heatmaps, time series). Supports scheduled report
distribution via email/Slack.

Example:
    >>> generator = DriftReportGenerator()
    >>> report = generator.generate_report(
    ...     monitor_id=uuid4(),
    ...     detection_results=[...],
    ...     feature_rankings=[...],
    ... )
    >>> report.executive_summary
    'Drift detected in 3 of 8 features. Recommend retraining within 24h.'
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class RecommendedAction(str, Enum):
    """Recommended action based on drift severity."""

    RETRAIN = "retrain"
    INVESTIGATE = "investigate"
    MONITOR = "monitor"
    IGNORE = "ignore"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    PDF_PLACEHOLDER = "pdf_placeholder"  # PDF requires a rendering pipeline outside this adapter


@dataclass
class FeatureDriftAssessment:
    """Per-feature drift assessment in a report.

    Attributes:
        feature_name: Name of the feature.
        drift_score: Raw drift score.
        threshold: Threshold value used for comparison.
        is_drifted: Whether drift was detected.
        test_name: Statistical test used.
        contribution_pct: Fraction of total drift attributed to this feature.
        importance_rank: Rank among all monitored features (1 = most drifted).
        recommended_action: Action recommended for this feature.
        mean_shift: Absolute shift in mean from reference to production.
        severity: 'info' | 'warning' | 'critical'.
    """

    feature_name: str
    drift_score: float
    threshold: float
    is_drifted: bool
    test_name: str
    contribution_pct: float
    importance_rank: int
    recommended_action: RecommendedAction
    mean_shift: float
    severity: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all assessment fields.
        """
        return {
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "test_name": self.test_name,
            "contribution_pct": self.contribution_pct,
            "importance_rank": self.importance_rank,
            "recommended_action": self.recommended_action.value,
            "mean_shift": self.mean_shift,
            "severity": self.severity,
        }


@dataclass
class PerformanceImpactAnalysis:
    """Analysis of performance impact from detected drift.

    Attributes:
        baseline_accuracy: Accuracy before drift (from baseline period).
        current_accuracy: Accuracy in the current production window.
        accuracy_delta: Absolute change (current - baseline).
        estimated_drift_contribution: Fraction of accuracy drop attributable to drift (0–1).
        risk_level: 'low' | 'medium' | 'high'.
        time_to_critical: Estimated hours until model becomes unreliable (None if stable).
    """

    baseline_accuracy: float | None
    current_accuracy: float | None
    accuracy_delta: float | None
    estimated_drift_contribution: float
    risk_level: str
    time_to_critical: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with performance impact fields.
        """
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "current_accuracy": self.current_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "estimated_drift_contribution": self.estimated_drift_contribution,
            "risk_level": self.risk_level,
            "time_to_critical": self.time_to_critical,
        }


@dataclass
class DriftHeatmapData:
    """Data for a drift score heatmap visualisation.

    The heatmap shows drift scores across features over time.

    Attributes:
        features: List of feature names (y-axis labels).
        time_labels: List of ISO timestamp strings (x-axis labels).
        scores_matrix: 2-D list [feature_index][time_index] of drift scores.
        threshold: Drift threshold value for colouring cells.
    """

    features: list[str]
    time_labels: list[str]
    scores_matrix: list[list[float]]
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with heatmap data.
        """
        return {
            "features": self.features,
            "time_labels": self.time_labels,
            "scores_matrix": self.scores_matrix,
            "threshold": self.threshold,
        }


@dataclass
class DriftReport:
    """A complete drift assessment report.

    Attributes:
        report_id: Unique identifier for this report.
        monitor_id: UUID of the drift monitor.
        model_id: UUID of the model being monitored.
        tenant_id: UUID of the owning tenant.
        generated_at: UTC timestamp of generation.
        period_start: Start of the assessment period.
        period_end: End of the assessment period (= generated_at).
        executive_summary: One-paragraph plain-text summary.
        total_features_monitored: Number of features monitored.
        drifted_feature_count: Number of features with detected drift.
        overall_drift_detected: True if any feature drifted.
        overall_recommended_action: Highest-priority action from all features.
        feature_assessments: Per-feature drift assessments.
        performance_impact: Performance impact analysis.
        heatmap_data: Drift heatmap visualisation data.
        time_series_data: Dict of feature_name to list of drift scores over time.
        top_contributing_features: Ranked list of top-3 drift contributors.
        tags: Arbitrary metadata.
    """

    report_id: uuid.UUID
    monitor_id: uuid.UUID
    model_id: uuid.UUID
    tenant_id: uuid.UUID
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    executive_summary: str
    total_features_monitored: int
    drifted_feature_count: int
    overall_drift_detected: bool
    overall_recommended_action: RecommendedAction
    feature_assessments: list[FeatureDriftAssessment]
    performance_impact: PerformanceImpactAnalysis
    heatmap_data: DriftHeatmapData | None
    time_series_data: dict[str, list[dict[str, Any]]]
    top_contributing_features: list[str]
    tags: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full report to a plain dict.

        Returns:
            Dict with all report fields.
        """
        return {
            "report_id": str(self.report_id),
            "monitor_id": str(self.monitor_id),
            "model_id": str(self.model_id),
            "tenant_id": str(self.tenant_id),
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "executive_summary": self.executive_summary,
            "total_features_monitored": self.total_features_monitored,
            "drifted_feature_count": self.drifted_feature_count,
            "overall_drift_detected": self.overall_drift_detected,
            "overall_recommended_action": self.overall_recommended_action.value,
            "feature_assessments": [fa.to_dict() for fa in self.feature_assessments],
            "performance_impact": self.performance_impact.to_dict(),
            "heatmap_data": self.heatmap_data.to_dict() if self.heatmap_data else None,
            "time_series_data": self.time_series_data,
            "top_contributing_features": self.top_contributing_features,
            "tags": self.tags,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string.

        Args:
            indent: JSON indentation spaces.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class ReportSchedule:
    """Configuration for scheduled report generation and distribution.

    Attributes:
        schedule_id: Unique identifier.
        monitor_id: UUID of the monitor to report on.
        cron_expression: Cron schedule for report generation.
        recipients_email: List of email addresses to send reports to.
        recipients_slack_webhook: Slack webhook URLs to post report summaries.
        report_format: Output format (JSON or PDF placeholder).
        include_heatmap: Whether to include the heatmap in the report.
        enabled: Whether this schedule is active.
        last_run_at: UTC timestamp of the last successful report run.
    """

    schedule_id: uuid.UUID
    monitor_id: uuid.UUID
    cron_expression: str
    recipients_email: list[str]
    recipients_slack_webhook: list[str]
    report_format: ReportFormat
    include_heatmap: bool = True
    enabled: bool = True
    last_run_at: datetime | None = None


class DriftReportGenerator:
    """Automated drift report generator with scheduling and distribution.

    Produces comprehensive drift reports that include:
    - Executive summaries for non-technical stakeholders
    - Per-feature drift assessments with severity and recommendations
    - Performance impact analysis linking drift to model degradation
    - Visualisation data (heatmaps, time series) for dashboards
    - JSON serialisation with optional PDF generation placeholder
    - Scheduled report distribution via email and Slack

    The generator maintains a report history and schedule registry.
    """

    def __init__(self) -> None:
        """Initialise the report generator."""
        self._report_history: list[DriftReport] = []
        self._schedules: dict[uuid.UUID, ReportSchedule] = {}

    def generate_report(
        self,
        monitor_id: uuid.UUID,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        feature_results: list[dict[str, Any]],
        period_start: datetime,
        baseline_accuracy: float | None = None,
        current_accuracy: float | None = None,
        historical_scores: dict[str, list[tuple[datetime, float]]] | None = None,
        tags: dict[str, str] | None = None,
    ) -> DriftReport:
        """Generate a complete drift assessment report.

        Args:
            monitor_id: UUID of the drift monitor.
            model_id: UUID of the model being monitored.
            tenant_id: UUID of the owning tenant.
            feature_results: List of feature result dicts. Each dict must contain:
                - feature_name (str)
                - drift_score (float)
                - threshold (float)
                - is_drifted (bool)
                - test_name (str)
                - contribution_pct (float, 0–1)
                - importance_rank (int)
                - mean_shift (float, optional)
            period_start: Start of the assessment period.
            baseline_accuracy: Optional baseline accuracy for impact analysis.
            current_accuracy: Optional current accuracy for impact analysis.
            historical_scores: Optional dict of feature_name to list of (timestamp, score)
                tuples for time series and heatmap data.
            tags: Optional report metadata tags.

        Returns:
            Generated DriftReport.
        """
        now = datetime.now(tz=timezone.utc)
        feature_assessments = self._build_feature_assessments(feature_results)
        performance_impact = self._build_performance_impact(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            drifted_count=sum(1 for fa in feature_assessments if fa.is_drifted),
            total_features=len(feature_assessments),
        )
        heatmap = self._build_heatmap(feature_results, historical_scores or {})
        time_series = self._build_time_series(historical_scores or {})

        drifted = [fa for fa in feature_assessments if fa.is_drifted]
        drifted_count = len(drifted)
        total = len(feature_assessments)
        overall_action = self._determine_overall_action(feature_assessments, performance_impact)
        top_features = [fa.feature_name for fa in sorted(feature_assessments, key=lambda x: x.importance_rank)[:3]]
        executive_summary = self._write_executive_summary(
            drifted_count=drifted_count,
            total_features=total,
            overall_action=overall_action,
            performance_impact=performance_impact,
            top_features=top_features,
            period_start=period_start,
            now=now,
        )

        report = DriftReport(
            report_id=uuid.uuid4(),
            monitor_id=monitor_id,
            model_id=model_id,
            tenant_id=tenant_id,
            generated_at=now,
            period_start=period_start,
            period_end=now,
            executive_summary=executive_summary,
            total_features_monitored=total,
            drifted_feature_count=drifted_count,
            overall_drift_detected=drifted_count > 0,
            overall_recommended_action=overall_action,
            feature_assessments=feature_assessments,
            performance_impact=performance_impact,
            heatmap_data=heatmap,
            time_series_data=time_series,
            top_contributing_features=top_features,
            tags=tags or {},
        )
        self._report_history.append(report)
        logger.info(
            "Drift report generated",
            report_id=str(report.report_id),
            monitor_id=str(monitor_id),
            drifted_features=drifted_count,
            total_features=total,
            recommended_action=overall_action.value,
        )
        return report

    def register_schedule(self, schedule: ReportSchedule) -> None:
        """Register a report generation schedule.

        Args:
            schedule: ReportSchedule configuration.
        """
        self._schedules[schedule.schedule_id] = schedule
        logger.info(
            "Report schedule registered",
            schedule_id=str(schedule.schedule_id),
            monitor_id=str(schedule.monitor_id),
            cron=schedule.cron_expression,
        )

    def remove_schedule(self, schedule_id: uuid.UUID) -> bool:
        """Remove a report schedule.

        Args:
            schedule_id: UUID of the schedule to remove.

        Returns:
            True if removed, False if not found.
        """
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    def list_schedules(self, monitor_id: uuid.UUID | None = None) -> list[ReportSchedule]:
        """List registered report schedules.

        Args:
            monitor_id: Optional filter by monitor.

        Returns:
            List of ReportSchedule.
        """
        schedules = list(self._schedules.values())
        if monitor_id is not None:
            schedules = [s for s in schedules if s.monitor_id == monitor_id]
        return schedules

    def get_report_history(
        self,
        monitor_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[DriftReport]:
        """Return report history with optional monitor filter.

        Args:
            monitor_id: Optional filter by monitor UUID.
            limit: Maximum number of reports to return (newest-first).

        Returns:
            List of DriftReport, newest-first.
        """
        reports = list(reversed(self._report_history))
        if monitor_id is not None:
            reports = [r for r in reports if r.monitor_id == monitor_id]
        return reports[:limit]

    def export_report_json(self, report_id: uuid.UUID) -> str:
        """Export a report as a formatted JSON string.

        Args:
            report_id: UUID of the report to export.

        Returns:
            JSON string.

        Raises:
            ValueError: If the report is not found.
        """
        report = next((r for r in self._report_history if r.report_id == report_id), None)
        if report is None:
            raise ValueError(f"Report {report_id} not found")
        return report.to_json()

    def generate_pdf_placeholder(self, report: DriftReport) -> str:
        """Return a structured text representation as a PDF placeholder.

        Full PDF generation requires an external library (e.g., reportlab or
        WeasyPrint) which is not bundled in this adapter. This method produces
        a structured text that a PDF pipeline can consume.

        Args:
            report: The DriftReport to render.

        Returns:
            Structured text string suitable for PDF rendering pipeline input.
        """
        lines: list[str] = [
            "=" * 80,
            "AUMOS DRIFT DETECTOR — DRIFT ASSESSMENT REPORT",
            "=" * 80,
            f"Report ID:    {report.report_id}",
            f"Monitor ID:   {report.monitor_id}",
            f"Model ID:     {report.model_id}",
            f"Generated At: {report.generated_at.isoformat()}",
            f"Period:       {report.period_start.isoformat()} → {report.period_end.isoformat()}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            report.executive_summary,
            "",
            "OVERALL RECOMMENDATION",
            "-" * 40,
            f"Action:  {report.overall_recommended_action.value.upper()}",
            f"Drifted: {report.drifted_feature_count} / {report.total_features_monitored} features",
            "",
            "PERFORMANCE IMPACT",
            "-" * 40,
            f"Risk Level:              {report.performance_impact.risk_level.upper()}",
            f"Baseline Accuracy:       {report.performance_impact.baseline_accuracy}",
            f"Current Accuracy:        {report.performance_impact.current_accuracy}",
            f"Accuracy Delta:          {report.performance_impact.accuracy_delta}",
            f"Drift Contribution:      {report.performance_impact.estimated_drift_contribution:.1%}",
            "",
            "PER-FEATURE ASSESSMENT",
            "-" * 40,
        ]
        for assessment in sorted(report.feature_assessments, key=lambda x: x.importance_rank):
            lines.append(
                f"  [{assessment.importance_rank:2d}] {assessment.feature_name:<30} "
                f"score={assessment.drift_score:.4f}  threshold={assessment.threshold:.4f}  "
                f"{'DRIFTED' if assessment.is_drifted else 'OK':<8}  "
                f"action={assessment.recommended_action.value}"
            )
        lines.extend(["", "=" * 80])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_assessments(
        self,
        feature_results: list[dict[str, Any]],
    ) -> list[FeatureDriftAssessment]:
        """Build FeatureDriftAssessment list from raw feature result dicts.

        Args:
            feature_results: List of feature result dicts.

        Returns:
            List of FeatureDriftAssessment ordered by importance_rank.
        """
        assessments: list[FeatureDriftAssessment] = []
        for result in feature_results:
            drift_score = float(result.get("drift_score", 0.0))
            threshold = float(result.get("threshold", 0.25))
            is_drifted = bool(result.get("is_drifted", False))
            contribution_pct = float(result.get("contribution_pct", 0.0))

            # Determine severity
            if not is_drifted:
                severity = "info"
            elif drift_score > threshold * 2:
                severity = "critical"
            elif drift_score > threshold * 1.5:
                severity = "warning"
            else:
                severity = "info"

            # Determine recommended action
            if not is_drifted:
                action = RecommendedAction.IGNORE
            elif severity == "critical":
                action = RecommendedAction.RETRAIN
            elif severity == "warning":
                action = RecommendedAction.INVESTIGATE
            else:
                action = RecommendedAction.MONITOR

            assessments.append(
                FeatureDriftAssessment(
                    feature_name=str(result.get("feature_name", "unknown")),
                    drift_score=drift_score,
                    threshold=threshold,
                    is_drifted=is_drifted,
                    test_name=str(result.get("test_name", "unknown")),
                    contribution_pct=contribution_pct,
                    importance_rank=int(result.get("importance_rank", 999)),
                    recommended_action=action,
                    mean_shift=float(result.get("mean_shift", 0.0)),
                    severity=severity,
                )
            )
        return sorted(assessments, key=lambda x: x.importance_rank)

    def _build_performance_impact(
        self,
        baseline_accuracy: float | None,
        current_accuracy: float | None,
        drifted_count: int,
        total_features: int,
    ) -> PerformanceImpactAnalysis:
        """Compute performance impact analysis from accuracy values and drift counts.

        Args:
            baseline_accuracy: Accuracy before drift.
            current_accuracy: Current accuracy in production.
            drifted_count: Number of features with drift.
            total_features: Total number of monitored features.

        Returns:
            PerformanceImpactAnalysis.
        """
        accuracy_delta: float | None = None
        if baseline_accuracy is not None and current_accuracy is not None:
            accuracy_delta = current_accuracy - baseline_accuracy

        drift_fraction = drifted_count / total_features if total_features > 0 else 0.0
        drift_contribution = drift_fraction * 0.8  # Heuristic: 80% of drop attributable to drift

        if accuracy_delta is not None:
            if accuracy_delta < -0.1:
                risk_level = "high"
            elif accuracy_delta < -0.05:
                risk_level = "medium"
            else:
                risk_level = "low"
        elif drift_fraction > 0.5:
            risk_level = "high"
        elif drift_fraction > 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Simple linear extrapolation for time-to-critical
        time_to_critical: float | None = None
        if accuracy_delta is not None and accuracy_delta < 0 and baseline_accuracy:
            current = current_accuracy or baseline_accuracy
            rate_per_hour = abs(accuracy_delta) / 24.0  # Assume delta over 24 hours
            target = baseline_accuracy * 0.85  # 15% drop = critical
            if rate_per_hour > 0 and current > target:
                time_to_critical = (current - target) / rate_per_hour

        return PerformanceImpactAnalysis(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            accuracy_delta=accuracy_delta,
            estimated_drift_contribution=drift_contribution,
            risk_level=risk_level,
            time_to_critical=time_to_critical,
        )

    def _build_heatmap(
        self,
        feature_results: list[dict[str, Any]],
        historical_scores: dict[str, list[tuple[datetime, float]]],
    ) -> DriftHeatmapData | None:
        """Build heatmap data from feature results and historical scores.

        Args:
            feature_results: Current detection results.
            historical_scores: Dict of feature_name to (timestamp, score) pairs.

        Returns:
            DriftHeatmapData or None if insufficient data.
        """
        if not feature_results:
            return None

        features = [str(r.get("feature_name", "unknown")) for r in feature_results]
        threshold = float(feature_results[0].get("threshold", 0.25)) if feature_results else 0.25

        if not historical_scores:
            # Single-column heatmap from current results only
            scores_matrix = [[float(r.get("drift_score", 0.0))] for r in feature_results]
            return DriftHeatmapData(
                features=features,
                time_labels=["now"],
                scores_matrix=scores_matrix,
                threshold=threshold,
            )

        # Build multi-column heatmap from historical data
        all_timestamps: list[datetime] = sorted(
            {ts for scores in historical_scores.values() for ts, _ in scores}
        )
        if not all_timestamps:
            return None

        time_labels = [ts.isoformat() for ts in all_timestamps]
        scores_matrix: list[list[float]] = []

        for feature_name in features:
            feature_history = {ts: score for ts, score in historical_scores.get(feature_name, [])}
            row = [float(feature_history.get(ts, 0.0)) for ts in all_timestamps]
            scores_matrix.append(row)

        return DriftHeatmapData(
            features=features,
            time_labels=time_labels,
            scores_matrix=scores_matrix,
            threshold=threshold,
        )

    @staticmethod
    def _build_time_series(
        historical_scores: dict[str, list[tuple[datetime, float]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Convert historical scores dict to serialisable time series format.

        Args:
            historical_scores: Dict of feature_name to (timestamp, score) pairs.

        Returns:
            Dict of feature_name to list of {timestamp, score} dicts.
        """
        return {
            feature_name: [
                {"timestamp": ts.isoformat(), "score": score}
                for ts, score in sorted(scores, key=lambda x: x[0])
            ]
            for feature_name, scores in historical_scores.items()
        }

    @staticmethod
    def _determine_overall_action(
        assessments: list[FeatureDriftAssessment],
        performance_impact: PerformanceImpactAnalysis,
    ) -> RecommendedAction:
        """Determine the overall recommended action from all feature assessments.

        Escalates to the highest-priority action found. If performance risk is
        high, always recommends RETRAIN.

        Args:
            assessments: Per-feature assessments.
            performance_impact: Performance impact analysis.

        Returns:
            Overall RecommendedAction.
        """
        if performance_impact.risk_level == "high":
            return RecommendedAction.RETRAIN

        action_priority = {
            RecommendedAction.RETRAIN: 3,
            RecommendedAction.INVESTIGATE: 2,
            RecommendedAction.MONITOR: 1,
            RecommendedAction.IGNORE: 0,
        }
        max_action = RecommendedAction.IGNORE
        max_priority = 0
        for assessment in assessments:
            priority = action_priority.get(assessment.recommended_action, 0)
            if priority > max_priority:
                max_priority = priority
                max_action = assessment.recommended_action
        return max_action

    @staticmethod
    def _write_executive_summary(
        drifted_count: int,
        total_features: int,
        overall_action: RecommendedAction,
        performance_impact: PerformanceImpactAnalysis,
        top_features: list[str],
        period_start: datetime,
        now: datetime,
    ) -> str:
        """Compose a plain-text executive summary paragraph.

        Args:
            drifted_count: Number of features with detected drift.
            total_features: Total features monitored.
            overall_action: Overall recommended action.
            performance_impact: Performance impact analysis.
            top_features: Top-3 contributing feature names.
            period_start: Assessment period start timestamp.
            now: Report generation timestamp.

        Returns:
            Executive summary string.
        """
        duration_hours = max(1, int((now - period_start).total_seconds() / 3600))
        drift_summary = (
            f"{drifted_count} of {total_features} monitored features"
            if total_features > 0
            else "0 features"
        )
        top_feat_str = ", ".join(top_features) if top_features else "none identified"

        if drifted_count == 0:
            status_text = "No data drift was detected during this assessment period."
            action_text = "No immediate action is required."
        else:
            status_text = (
                f"Data drift was detected in {drift_summary} over the past {duration_hours} hours. "
                f"The primary contributing features are: {top_feat_str}."
            )
            action_map = {
                RecommendedAction.RETRAIN: (
                    "Immediate model retraining is recommended to restore performance."
                ),
                RecommendedAction.INVESTIGATE: (
                    "Investigation of the drifted features is recommended before deciding on retraining."
                ),
                RecommendedAction.MONITOR: (
                    "Continued monitoring is recommended; drift is within acceptable bounds."
                ),
                RecommendedAction.IGNORE: "No action required at this time.",
            }
            action_text = action_map.get(overall_action, "Review the feature assessments below.")

        perf_text = ""
        if performance_impact.accuracy_delta is not None:
            direction = "decreased" if performance_impact.accuracy_delta < 0 else "improved"
            perf_text = (
                f" Model accuracy has {direction} by "
                f"{abs(performance_impact.accuracy_delta):.2%} "
                f"(risk level: {performance_impact.risk_level})."
            )

        return f"{status_text}{perf_text} {action_text}"
