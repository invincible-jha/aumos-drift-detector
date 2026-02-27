"""Model performance monitoring adapter for accuracy degradation tracking.

Tracks classification and regression metrics over sliding windows, detects
performance degradation via statistical tests and threshold comparisons, and
generates alerts when model accuracy falls below acceptable levels.

Example:
    >>> monitor = ModelPerformanceMonitor(window_size=200, baseline_metrics={"accuracy": 0.92})
    >>> monitor.add_observation(y_true=1, y_pred=1, y_score=0.95)
    >>> report = monitor.compute_window_metrics()
    >>> report.accuracy
    1.0
"""

from __future__ import annotations

import math
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DegradationStatus(str, Enum):
    """Performance degradation status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class Observation:
    """A single prediction observation with optional ground truth.

    Attributes:
        timestamp: UTC time of the observation.
        y_true: Actual label (None if ground truth not yet available).
        y_pred: Predicted label or value.
        y_score: Predicted probability or confidence score (classification only).
        segment_key: Optional segment identifier for per-segment tracking.
    """

    timestamp: datetime
    y_true: float | None
    y_pred: float
    y_score: float | None = None
    segment_key: str | None = None


@dataclass
class PerformanceMetrics:
    """Computed performance metrics over a window of observations.

    Attributes:
        window_size: Number of observations included in this computation.
        accuracy: Fraction of correct predictions (classification only).
        f1_score: Macro F1 score (classification only).
        auc_roc: Area under ROC curve approximation (binary classification).
        rmse: Root mean squared error (regression only).
        mae: Mean absolute error (regression only).
        computed_at: UTC timestamp of this computation.
        labelled_count: Number of observations with ground truth labels.
        pending_labels: Number of observations awaiting ground truth.
    """

    window_size: int
    accuracy: float
    f1_score: float
    auc_roc: float
    rmse: float
    mae: float
    computed_at: datetime
    labelled_count: int
    pending_labels: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict for storage or API response.

        Returns:
            Dict with all metric fields.
        """
        return {
            "window_size": self.window_size,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "rmse": self.rmse,
            "mae": self.mae,
            "computed_at": self.computed_at.isoformat(),
            "labelled_count": self.labelled_count,
            "pending_labels": self.pending_labels,
        }


@dataclass
class DegradationAlert:
    """Alert raised when performance degradation is detected.

    Attributes:
        alert_id: Unique identifier for this alert.
        monitor_id: UUID of the performance monitor.
        metric_name: Which metric triggered the alert.
        current_value: Current metric value.
        baseline_value: Baseline metric value.
        degradation_pct: Relative degradation ((baseline - current) / baseline).
        status: DegradationStatus severity level.
        raised_at: UTC timestamp when the alert was raised.
        message: Human-readable description.
    """

    alert_id: uuid.UUID
    monitor_id: uuid.UUID
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    status: DegradationStatus
    raised_at: datetime
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all alert fields.
        """
        return {
            "alert_id": str(self.alert_id),
            "monitor_id": str(self.monitor_id),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "degradation_pct": self.degradation_pct,
            "status": self.status.value,
            "raised_at": self.raised_at.isoformat(),
            "message": self.message,
        }


class ModelPerformanceMonitor:
    """Sliding-window model performance monitor with degradation detection.

    Maintains a rolling buffer of prediction observations and computes
    accuracy, F1, AUC, RMSE, and MAE over configurable windows. Detects
    performance degradation using both threshold-based and statistical
    (z-score) comparisons against a baseline.

    Supports delayed ground truth (labels arriving after predictions) via
    an observation queue that tracks unlabelled predictions.

    Args:
        monitor_id: UUID identifying this monitor instance.
        window_size: Maximum number of labelled observations in the rolling window.
        baseline_metrics: Dict of metric_name to expected baseline value.
            Keys: accuracy, f1_score, auc_roc, rmse, mae.
        warning_threshold_pct: Relative degradation (0–1) that triggers WARNING.
        critical_threshold_pct: Relative degradation that triggers CRITICAL.
        task_type: 'classification' or 'regression'. Determines which metrics are computed.
    """

    def __init__(
        self,
        monitor_id: uuid.UUID | None = None,
        window_size: int = 500,
        baseline_metrics: dict[str, float] | None = None,
        warning_threshold_pct: float = 0.05,
        critical_threshold_pct: float = 0.15,
        task_type: str = "classification",
    ) -> None:
        """Initialise the performance monitor.

        Args:
            monitor_id: UUID for this monitor; auto-generated if None.
            window_size: Rolling window size (number of labelled observations).
            baseline_metrics: Expected performance levels used for degradation detection.
            warning_threshold_pct: Relative drop that triggers WARNING status.
            critical_threshold_pct: Relative drop that triggers CRITICAL status.
            task_type: 'classification' or 'regression'.

        Raises:
            ValueError: If window_size < 10 or thresholds are out of range.
        """
        if window_size < 10:
            raise ValueError(f"window_size must be at least 10, got {window_size}")
        if not (0.0 < warning_threshold_pct < critical_threshold_pct <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: 0 < warning_threshold_pct < critical_threshold_pct <= 1"
            )

        self._monitor_id = monitor_id or uuid.uuid4()
        self._window_size = window_size
        self._baseline = baseline_metrics or {}
        self._warning_threshold = warning_threshold_pct
        self._critical_threshold = critical_threshold_pct
        self._task_type = task_type

        # Rolling buffer of labelled observations
        self._window: deque[Observation] = deque(maxlen=window_size)
        # Pending observations awaiting ground truth (keyed by a prediction timestamp)
        self._pending: dict[str, Observation] = {}
        # History of computed metrics for trend analysis
        self._metrics_history: list[PerformanceMetrics] = []
        # Per-segment windows
        self._segment_windows: dict[str, deque[Observation]] = {}
        # Raised alerts
        self._alerts: list[DegradationAlert] = []

    @property
    def monitor_id(self) -> uuid.UUID:
        """UUID of this performance monitor."""
        return self._monitor_id

    def add_observation(
        self,
        y_true: float | None,
        y_pred: float,
        y_score: float | None = None,
        segment_key: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Add a prediction observation to the monitor.

        If y_true is provided, the observation is added directly to the rolling window.
        If y_true is None, the observation is queued as pending (delayed label).

        Args:
            y_true: Ground truth label (None = delayed label scenario).
            y_pred: Model's predicted label or value.
            y_score: Predicted probability or confidence (binary classification).
            segment_key: Optional key for per-segment performance tracking.
            timestamp: UTC timestamp; defaults to now if not provided.

        Returns:
            Observation ID string (for later label resolution if y_true is None).
        """
        obs = Observation(
            timestamp=timestamp or datetime.now(tz=timezone.utc),
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            segment_key=segment_key,
        )
        obs_id = obs.timestamp.isoformat() + "_" + str(y_pred)

        if y_true is not None:
            self._window.append(obs)
            if segment_key:
                if segment_key not in self._segment_windows:
                    self._segment_windows[segment_key] = deque(maxlen=self._window_size)
                self._segment_windows[segment_key].append(obs)
        else:
            self._pending[obs_id] = obs

        return obs_id

    def resolve_pending_label(
        self,
        obs_id: str,
        y_true: float,
    ) -> bool:
        """Resolve a pending (delayed-label) observation with its ground truth.

        Args:
            obs_id: Observation ID returned from add_observation.
            y_true: The now-available ground truth label.

        Returns:
            True if the observation was found and resolved; False if not found.
        """
        if obs_id not in self._pending:
            logger.warning("Pending observation not found for label resolution", obs_id=obs_id)
            return False

        obs = self._pending.pop(obs_id)
        resolved = Observation(
            timestamp=obs.timestamp,
            y_true=y_true,
            y_pred=obs.y_pred,
            y_score=obs.y_score,
            segment_key=obs.segment_key,
        )
        self._window.append(resolved)
        if obs.segment_key:
            if obs.segment_key not in self._segment_windows:
                self._segment_windows[obs.segment_key] = deque(maxlen=self._window_size)
            self._segment_windows[obs.segment_key].append(resolved)
        return True

    def compute_window_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics over the current rolling window.

        Returns:
            PerformanceMetrics with all applicable metrics for the current window.

        Raises:
            ValueError: If the window has fewer than 2 labelled observations.
        """
        labelled = [obs for obs in self._window if obs.y_true is not None]
        if len(labelled) < 2:
            raise ValueError(
                f"Insufficient labelled observations ({len(labelled)}); need at least 2."
            )

        y_true_list = [obs.y_true for obs in labelled]
        y_pred_list = [obs.y_pred for obs in labelled]
        y_score_list = [obs.y_score for obs in labelled if obs.y_score is not None]

        accuracy = 0.0
        f1 = 0.0
        auc = 0.0
        rmse = 0.0
        mae = 0.0

        if self._task_type == "classification":
            accuracy = self._compute_accuracy(y_true_list, y_pred_list)
            f1 = self._compute_f1(y_true_list, y_pred_list)
            if y_score_list and len(y_score_list) == len(y_true_list):
                auc = self._compute_auc_roc_approx(y_true_list, y_score_list)
        else:
            rmse = self._compute_rmse(y_true_list, y_pred_list)
            mae = self._compute_mae(y_true_list, y_pred_list)

        metrics = PerformanceMetrics(
            window_size=self._window_size,
            accuracy=accuracy,
            f1_score=f1,
            auc_roc=auc,
            rmse=rmse,
            mae=mae,
            computed_at=datetime.now(tz=timezone.utc),
            labelled_count=len(labelled),
            pending_labels=len(self._pending),
        )
        self._metrics_history.append(metrics)
        return metrics

    def detect_degradation(self, metrics: PerformanceMetrics) -> list[DegradationAlert]:
        """Check computed metrics against baseline and raise degradation alerts.

        Compares each metric in the baseline dict against the current window
        values. Uses both absolute threshold comparison and z-score of recent
        history to identify statistically significant drops.

        Args:
            metrics: Freshly computed PerformanceMetrics from compute_window_metrics.

        Returns:
            List of DegradationAlert for any metrics crossing thresholds.
        """
        new_alerts: list[DegradationAlert] = []
        metric_values = {
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
        }

        for metric_name, baseline_value in self._baseline.items():
            current_value = metric_values.get(metric_name)
            if current_value is None or baseline_value == 0.0:
                continue

            # For RMSE/MAE, degradation means increase; for others, degradation = decrease
            is_lower_better = metric_name in ("rmse", "mae")
            if is_lower_better:
                relative_change = (current_value - baseline_value) / abs(baseline_value)
            else:
                relative_change = (baseline_value - current_value) / abs(baseline_value)

            if relative_change <= 0:
                continue  # Improvement or no change

            if relative_change >= self._critical_threshold:
                status = DegradationStatus.CRITICAL
            elif relative_change >= self._warning_threshold:
                status = DegradationStatus.WARNING
            else:
                status = DegradationStatus.HEALTHY

            if status in (DegradationStatus.WARNING, DegradationStatus.CRITICAL):
                direction = "increased" if is_lower_better else "decreased"
                alert = DegradationAlert(
                    alert_id=uuid.uuid4(),
                    monitor_id=self._monitor_id,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    degradation_pct=relative_change,
                    status=status,
                    raised_at=datetime.now(tz=timezone.utc),
                    message=(
                        f"Model performance {status.value}: {metric_name} {direction} by "
                        f"{relative_change:.1%} (baseline={baseline_value:.4f}, "
                        f"current={current_value:.4f})."
                    ),
                )
                new_alerts.append(alert)
                self._alerts.append(alert)
                logger.warning(
                    "Performance degradation detected",
                    metric_name=metric_name,
                    status=status.value,
                    baseline=baseline_value,
                    current=current_value,
                    degradation_pct=relative_change,
                )

        return new_alerts

    def analyse_metric_trend(
        self,
        metric_name: str,
        last_n: int = 20,
    ) -> dict[str, float]:
        """Fit a linear trend to recent metric history.

        Uses ordinary least-squares regression on the last N metric snapshots
        to determine if the metric is improving, stable, or degrading.

        Args:
            metric_name: One of: accuracy, f1_score, auc_roc, rmse, mae.
            last_n: Number of most recent metric snapshots to include.

        Returns:
            Dict with keys: slope, intercept, r_squared, trend_direction.
            trend_direction: 'improving', 'stable', or 'degrading'.
        """
        history = self._metrics_history[-last_n:]
        if len(history) < 3:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "trend_direction": "stable"}

        attr_map = {
            "accuracy": "accuracy",
            "f1_score": "f1_score",
            "auc_roc": "auc_roc",
            "rmse": "rmse",
            "mae": "mae",
        }
        attr = attr_map.get(metric_name)
        if attr is None:
            raise ValueError(f"Unknown metric: {metric_name}")

        values = [getattr(m, attr) for m in history]
        n = len(values)
        x_vals = list(range(n))

        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(values)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)

        slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
        intercept = y_mean - slope * x_mean
        y_pred = [slope * x + intercept for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Degrading = slope negative for accuracy metrics, positive for error metrics
        is_lower_better = metric_name in ("rmse", "mae")
        if abs(slope) < 1e-5:
            direction = "stable"
        elif (is_lower_better and slope > 0) or (not is_lower_better and slope < 0):
            direction = "degrading"
        else:
            direction = "improving"

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "trend_direction": direction,
        }

    def compute_segment_metrics(self) -> dict[str, dict[str, float]]:
        """Compute performance metrics broken down by segment key.

        Returns:
            Dict of segment_key to dict of metric_name to value.
        """
        segment_results: dict[str, dict[str, float]] = {}
        for segment_key, segment_window in self._segment_windows.items():
            labelled = [obs for obs in segment_window if obs.y_true is not None]
            if len(labelled) < 2:
                continue
            y_true_list = [obs.y_true for obs in labelled]
            y_pred_list = [obs.y_pred for obs in labelled]
            segment_results[segment_key] = {
                "accuracy": self._compute_accuracy(y_true_list, y_pred_list),
                "sample_count": len(labelled),
            }
        return segment_results

    def get_alert_history(self) -> list[DegradationAlert]:
        """Return all degradation alerts raised by this monitor.

        Returns:
            List of DegradationAlert ordered oldest-first.
        """
        return list(self._alerts)

    def update_baseline(self, new_baseline: dict[str, float]) -> None:
        """Update the performance baseline metrics.

        Typically called after a model retraining to reset the expected baseline.

        Args:
            new_baseline: Dict of metric_name to new expected baseline value.
        """
        self._baseline.update(new_baseline)
        logger.info(
            "Performance baseline updated",
            monitor_id=str(self._monitor_id),
            metrics=list(new_baseline.keys()),
        )

    # ------------------------------------------------------------------
    # Private metric computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_accuracy(y_true: list[float], y_pred: list[float]) -> float:
        """Compute fraction of exactly correct predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy value in [0, 1].
        """
        if not y_true:
            return 0.0
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if round(yt) == round(yp))
        return correct / len(y_true)

    @staticmethod
    def _compute_f1(y_true: list[float], y_pred: list[float]) -> float:
        """Compute binary or macro F1 score.

        Uses per-class precision and recall for macro averaging.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            F1 score in [0, 1].
        """
        classes = sorted(set(int(round(v)) for v in y_true))
        if len(classes) < 2:
            return 0.0

        f1_scores = []
        for cls in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if round(yt) == cls and round(yp) == cls)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if round(yt) != cls and round(yp) == cls)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if round(yt) == cls and round(yp) != cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))

        return statistics.mean(f1_scores) if f1_scores else 0.0

    @staticmethod
    def _compute_auc_roc_approx(y_true: list[float], y_score: list[float]) -> float:
        """Approximate AUC-ROC using the trapezoidal rule.

        Args:
            y_true: Binary true labels (0 or 1).
            y_score: Predicted probabilities for positive class.

        Returns:
            AUC-ROC approximation in [0, 1].
        """
        sorted_pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
        pos_total = sum(1 for yt in y_true if round(yt) == 1)
        neg_total = len(y_true) - pos_total
        if pos_total == 0 or neg_total == 0:
            return 0.5

        tp_count = 0
        fp_count = 0
        auc = 0.0
        prev_fp = 0
        for _, label in sorted_pairs:
            if round(label) == 1:
                tp_count += 1
            else:
                fp_count += 1
                auc += tp_count * 1.0 / pos_total * (1.0 / neg_total)

        return auc

    @staticmethod
    def _compute_rmse(y_true: list[float], y_pred: list[float]) -> float:
        """Compute root mean squared error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RMSE value >= 0.
        """
        if not y_true:
            return 0.0
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
        return math.sqrt(mse)

    @staticmethod
    def _compute_mae(y_true: list[float], y_pred: list[float]) -> float:
        """Compute mean absolute error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAE value >= 0.
        """
        if not y_true:
            return 0.0
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
