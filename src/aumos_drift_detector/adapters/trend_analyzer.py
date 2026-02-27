"""Historical drift trend analyser with change point detection and forecasting.

Maintains time-series data of drift scores per feature, detects whether trends
are increasing/stable/seasonal, identifies change points using CUSUM and
variance-based methods, correlates drift with performance degradation, and
generates simple linear forecasts of future drift.

Example:
    >>> analyzer = DriftTrendAnalyzer()
    >>> for day_offset in range(30):
    ...     score = 0.01 * day_offset + 0.05  # linear upward trend
    ...     analyzer.record_drift_score(monitor_id=uuid4(), feature_name="age", score=score)
    >>> trend = analyzer.analyse_feature_trend(monitor_id=..., feature_name="age")
    >>> trend.direction
    'increasing'
"""

from __future__ import annotations

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class DriftScoreEntry:
    """A single timestamped drift score for a feature.

    Attributes:
        recorded_at: UTC timestamp of this drift check.
        score: Drift score value (higher = more drift).
        test_name: Name of the statistical test that produced this score.
        is_drifted: Whether the score crossed the detection threshold.
    """

    recorded_at: datetime
    score: float
    test_name: str
    is_drifted: bool


@dataclass
class TrendAnalysis:
    """Result of trend analysis for a single feature.

    Attributes:
        feature_name: Name of the feature.
        monitor_id: UUID of the monitor.
        observation_count: Number of drift score observations.
        direction: 'increasing', 'stable', or 'decreasing'.
        slope: Linear regression slope (drift units per observation).
        r_squared: R-squared goodness-of-fit of the linear model.
        mean_score: Mean drift score over the analysis window.
        std_score: Standard deviation of drift scores.
        drift_rate_pct: Fraction of observations where drift was detected.
        change_points: Indices of detected change points in the time series.
        is_seasonal: Whether a seasonal pattern was detected.
        analysed_at: UTC timestamp of this analysis.
    """

    feature_name: str
    monitor_id: uuid.UUID
    observation_count: int
    direction: str
    slope: float
    r_squared: float
    mean_score: float
    std_score: float
    drift_rate_pct: float
    change_points: list[int]
    is_seasonal: bool
    analysed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all trend analysis fields.
        """
        return {
            "feature_name": self.feature_name,
            "monitor_id": str(self.monitor_id),
            "observation_count": self.observation_count,
            "direction": self.direction,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "drift_rate_pct": self.drift_rate_pct,
            "change_points": self.change_points,
            "is_seasonal": self.is_seasonal,
            "analysed_at": self.analysed_at.isoformat(),
        }


@dataclass
class ChangePoint:
    """A detected change point in a drift score time series.

    Attributes:
        index: Position in the score sequence where the change occurred.
        score_before: Mean drift score before the change point.
        score_after: Mean drift score after the change point.
        magnitude: Absolute shift in mean (score_after - score_before).
        detected_at: The timestamp of the entry at the change point index.
        cusum_statistic: CUSUM statistic value at detection.
    """

    index: int
    score_before: float
    score_after: float
    magnitude: float
    detected_at: datetime
    cusum_statistic: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with change point fields.
        """
        return {
            "index": self.index,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "magnitude": self.magnitude,
            "detected_at": self.detected_at.isoformat(),
            "cusum_statistic": self.cusum_statistic,
        }


@dataclass
class DriftForecast:
    """Simple linear extrapolation forecast of future drift scores.

    Attributes:
        feature_name: Feature being forecast.
        monitor_id: UUID of the monitor.
        horizon_steps: Number of future steps (observations) forecasted.
        forecasted_scores: Predicted drift score for each future step.
        confidence_interval_lower: Lower bound of 80% confidence interval.
        confidence_interval_upper: Upper bound of 80% confidence interval.
        predicted_drift_step: Step at which drift is predicted to occur (None = no drift predicted).
        forecast_generated_at: UTC timestamp of forecast generation.
    """

    feature_name: str
    monitor_id: uuid.UUID
    horizon_steps: int
    forecasted_scores: list[float]
    confidence_interval_lower: list[float]
    confidence_interval_upper: list[float]
    predicted_drift_step: int | None
    forecast_generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with forecast fields.
        """
        return {
            "feature_name": self.feature_name,
            "monitor_id": str(self.monitor_id),
            "horizon_steps": self.horizon_steps,
            "forecasted_scores": self.forecasted_scores,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "predicted_drift_step": self.predicted_drift_step,
            "forecast_generated_at": self.forecast_generated_at.isoformat(),
        }


class DriftTrendAnalyzer:
    """Historical drift trend analyser with change point detection and forecasting.

    Collects and analyses time-series drift scores per feature and monitor.
    Provides:
    - Trend direction detection (increasing, stable, decreasing)
    - CUSUM-based change point detection
    - Seasonal pattern identification (autocorrelation-based)
    - Performance-drift correlation analysis
    - Linear forecast with confidence intervals
    - Drift trend reports per monitor

    Args:
        max_history_per_feature: Maximum number of score entries to retain per
            (monitor_id, feature_name) key. Oldest entries are evicted.
    """

    def __init__(self, max_history_per_feature: int = 10_000) -> None:
        """Initialise the trend analyser.

        Args:
            max_history_per_feature: History buffer size per feature time series.
        """
        self._max_history = max_history_per_feature
        # Key: (monitor_id_str, feature_name) → list of DriftScoreEntry
        self._score_history: dict[tuple[str, str], list[DriftScoreEntry]] = {}
        # Key: monitor_id_str → list of (timestamp, performance_metric) for correlation
        self._performance_history: dict[str, list[tuple[datetime, float]]] = {}

    def record_drift_score(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        score: float,
        test_name: str = "unknown",
        is_drifted: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a drift score observation to the time series.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Name of the feature.
            score: Drift score value.
            test_name: Statistical test that produced this score.
            is_drifted: Whether the score crossed the threshold.
            timestamp: UTC timestamp; defaults to now.
        """
        key = (str(monitor_id), feature_name)
        entry = DriftScoreEntry(
            recorded_at=timestamp or datetime.now(tz=timezone.utc),
            score=score,
            test_name=test_name,
            is_drifted=is_drifted,
        )
        if key not in self._score_history:
            self._score_history[key] = []
        self._score_history[key].append(entry)

        # Evict oldest if over limit
        if len(self._score_history[key]) > self._max_history:
            self._score_history[key] = self._score_history[key][-self._max_history:]

    def record_performance_metric(
        self,
        monitor_id: uuid.UUID,
        metric_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a performance metric value for correlation analysis.

        Args:
            monitor_id: UUID of the drift monitor.
            metric_value: Performance metric value (e.g., accuracy).
            timestamp: UTC timestamp; defaults to now.
        """
        key = str(monitor_id)
        entry = (timestamp or datetime.now(tz=timezone.utc), metric_value)
        if key not in self._performance_history:
            self._performance_history[key] = []
        self._performance_history[key].append(entry)

        if len(self._performance_history[key]) > self._max_history:
            self._performance_history[key] = self._performance_history[key][-self._max_history:]

    def analyse_feature_trend(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        window_size: int | None = None,
    ) -> TrendAnalysis:
        """Compute trend analysis for a specific feature time series.

        Fits a linear regression to the drift scores and classifies the trend
        direction. Also runs CUSUM change point detection and autocorrelation
        seasonality check.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Name of the feature to analyse.
            window_size: Optional number of most-recent observations to analyse.
                None means use all available history.

        Returns:
            TrendAnalysis with direction, slope, r-squared, and change points.

        Raises:
            ValueError: If fewer than 3 observations are available.
        """
        key = (str(monitor_id), feature_name)
        history = self._score_history.get(key, [])
        if window_size is not None:
            history = history[-window_size:]
        if len(history) < 3:
            raise ValueError(
                f"Insufficient history for trend analysis: {len(history)} observations "
                f"(need at least 3) for monitor={monitor_id}, feature={feature_name}"
            )

        scores = [entry.score for entry in history]
        n = len(scores)
        x_vals = list(range(n))
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(scores)

        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
        intercept = y_mean - slope * x_mean

        y_pred = [slope * x + intercept for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(scores, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in scores)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Classify direction: slope relative to mean score magnitude
        relative_slope = slope / abs(y_mean) if y_mean != 0 else slope
        if abs(relative_slope) < 0.005:
            direction = "stable"
        elif relative_slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        change_points = self._detect_cusum_change_points(scores)
        is_seasonal = self._check_seasonality(scores)
        drift_rate = sum(1 for e in history if e.is_drifted) / n

        analysis = TrendAnalysis(
            feature_name=feature_name,
            monitor_id=monitor_id,
            observation_count=n,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            mean_score=y_mean,
            std_score=float(statistics.stdev(scores)) if n > 1 else 0.0,
            drift_rate_pct=drift_rate,
            change_points=[cp.index for cp in change_points],
            is_seasonal=is_seasonal,
            analysed_at=datetime.now(tz=timezone.utc),
        )
        logger.info(
            "Drift trend analysed",
            monitor_id=str(monitor_id),
            feature=feature_name,
            direction=direction,
            slope=slope,
            change_point_count=len(change_points),
        )
        return analysis

    def detect_change_points(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        cusum_threshold: float = 4.0,
    ) -> list[ChangePoint]:
        """Run CUSUM change point detection on a feature's drift score time series.

        CUSUM accumulates deviations from the mean. When the cumulative sum
        exceeds `cusum_threshold` standard deviations, a change point is flagged
        and the accumulator is reset.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to analyse.
            cusum_threshold: Number of standard deviations for CUSUM detection.

        Returns:
            List of ChangePoint ordered by index.

        Raises:
            ValueError: If fewer than 5 observations are available.
        """
        key = (str(monitor_id), feature_name)
        history = self._score_history.get(key, [])
        if len(history) < 5:
            raise ValueError(f"Need at least 5 observations for change point detection, got {len(history)}")

        scores = [e.score for e in history]
        return self._detect_cusum_change_points(scores, threshold=cusum_threshold, history=history)

    def correlate_drift_and_performance(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
    ) -> dict[str, float]:
        """Compute Pearson correlation between drift scores and performance metric.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature drift time series to correlate.

        Returns:
            Dict with keys: pearson_r, lag_0_r, lag_1_r, lag_2_r.
            lag_N_r is the correlation when drift is shifted N steps forward
            relative to performance (testing whether drift predicts performance drops).
        """
        key = (str(monitor_id), feature_name)
        drift_history = self._score_history.get(key, [])
        perf_history = self._performance_history.get(str(monitor_id), [])

        min_len = min(len(drift_history), len(perf_history))
        if min_len < 3:
            return {"pearson_r": 0.0, "lag_0_r": 0.0, "lag_1_r": 0.0, "lag_2_r": 0.0}

        drift_scores = [e.score for e in drift_history[-min_len:]]
        perf_values = [v for _, v in perf_history[-min_len:]]

        results = {}
        for lag in range(3):
            if lag == 0:
                x_vals = drift_scores
                y_vals = perf_values
            else:
                x_vals = drift_scores[:-lag]
                y_vals = perf_values[lag:]
            r = self._pearson_correlation(x_vals, y_vals)
            results[f"lag_{lag}_r"] = r

        results["pearson_r"] = results["lag_0_r"]
        return results

    def forecast_drift(
        self,
        monitor_id: uuid.UUID,
        feature_name: str,
        horizon_steps: int = 10,
        drift_threshold: float = 0.25,
    ) -> DriftForecast:
        """Generate a linear extrapolation forecast of future drift scores.

        Fits a linear regression to the full drift history and extrapolates
        `horizon_steps` observations into the future. Computes an 80% prediction
        interval based on residual standard error.

        Args:
            monitor_id: UUID of the drift monitor.
            feature_name: Feature to forecast.
            horizon_steps: Number of future observations to forecast.
            drift_threshold: Drift score above which drift is considered detected.

        Returns:
            DriftForecast with predicted scores and confidence intervals.

        Raises:
            ValueError: If fewer than 3 observations are available.
        """
        key = (str(monitor_id), feature_name)
        history = self._score_history.get(key, [])
        if len(history) < 3:
            raise ValueError(f"Need at least 3 observations for forecasting, got {len(history)}")

        scores = [e.score for e in history]
        n = len(scores)
        x_vals = list(range(n))
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(scores)

        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
        intercept = y_mean - slope * x_mean

        # Residual standard error
        y_pred_in_sample = [slope * x + intercept for x in x_vals]
        residuals = [y - yp for y, yp in zip(scores, y_pred_in_sample)]
        rse = math.sqrt(sum(r ** 2 for r in residuals) / max(1, n - 2))

        # 80% prediction interval multiplier (~1.282 for normal distribution)
        z_80 = 1.282

        forecasted: list[float] = []
        ci_lower: list[float] = []
        ci_upper: list[float] = []
        predicted_drift_step: int | None = None

        for step in range(horizon_steps):
            future_x = n + step
            predicted = slope * future_x + intercept
            # Prediction interval widens with distance from in-sample range
            leverage_factor = math.sqrt(1 + 1 / n + (future_x - x_mean) ** 2 / max(ss_xx, 1e-10))
            margin = z_80 * rse * leverage_factor

            forecasted.append(max(0.0, predicted))
            ci_lower.append(max(0.0, predicted - margin))
            ci_upper.append(max(0.0, predicted + margin))

            if predicted_drift_step is None and predicted > drift_threshold:
                predicted_drift_step = step

        forecast = DriftForecast(
            feature_name=feature_name,
            monitor_id=monitor_id,
            horizon_steps=horizon_steps,
            forecasted_scores=forecasted,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            predicted_drift_step=predicted_drift_step,
            forecast_generated_at=datetime.now(tz=timezone.utc),
        )
        logger.info(
            "Drift forecast generated",
            monitor_id=str(monitor_id),
            feature=feature_name,
            horizon_steps=horizon_steps,
            predicted_drift_step=predicted_drift_step,
        )
        return forecast

    def generate_trend_report(
        self,
        monitor_id: uuid.UUID,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive trend report for all features of a monitor.

        Args:
            monitor_id: UUID of the drift monitor.
            window_size: Optional window to pass to each feature's trend analysis.

        Returns:
            Dict with monitor_id, feature_trends list, and summary statistics.
        """
        feature_keys = [
            feature_name
            for (mid, feature_name) in self._score_history
            if mid == str(monitor_id)
        ]

        feature_trends: list[dict[str, Any]] = []
        increasing_count = 0
        stable_count = 0
        decreasing_count = 0

        for feature_name in feature_keys:
            try:
                trend = self.analyse_feature_trend(
                    monitor_id=monitor_id,
                    feature_name=feature_name,
                    window_size=window_size,
                )
                feature_trends.append(trend.to_dict())
                if trend.direction == "increasing":
                    increasing_count += 1
                elif trend.direction == "decreasing":
                    decreasing_count += 1
                else:
                    stable_count += 1
            except ValueError:
                # Insufficient history for this feature — skip
                pass

        report: dict[str, Any] = {
            "monitor_id": str(monitor_id),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "feature_count": len(feature_trends),
            "summary": {
                "increasing_features": increasing_count,
                "stable_features": stable_count,
                "decreasing_features": decreasing_count,
                "overall_trend": (
                    "increasing"
                    if increasing_count > stable_count + decreasing_count
                    else "stable"
                ),
            },
            "feature_trends": feature_trends,
        }
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_cusum_change_points(
        self,
        scores: list[float],
        threshold: float = 4.0,
        history: list[DriftScoreEntry] | None = None,
    ) -> list[ChangePoint]:
        """Internal CUSUM change point detection.

        Args:
            scores: List of drift score values.
            threshold: CUSUM threshold in standard deviation units.
            history: Optional list of DriftScoreEntry for timestamps.

        Returns:
            List of ChangePoint ordered by index.
        """
        if len(scores) < 5:
            return []

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 1.0
        std = std or 1.0

        cusum_pos = 0.0
        cusum_neg = 0.0
        change_points: list[ChangePoint] = []
        last_change_idx = 0

        for i, score in enumerate(scores):
            normalised = (score - mean) / std
            cusum_pos = max(0.0, cusum_pos + normalised - 0.5)
            cusum_neg = max(0.0, cusum_neg - normalised - 0.5)

            if cusum_pos > threshold or cusum_neg > threshold:
                if i - last_change_idx > 3:  # Minimum separation between change points
                    before_scores = scores[last_change_idx:i]
                    after_scores = scores[i:]
                    score_before = statistics.mean(before_scores) if before_scores else mean
                    score_after = statistics.mean(after_scores[:10]) if after_scores else mean
                    detected_at = (
                        history[i].recorded_at
                        if history and i < len(history)
                        else datetime.now(tz=timezone.utc)
                    )
                    change_points.append(
                        ChangePoint(
                            index=i,
                            score_before=score_before,
                            score_after=score_after,
                            magnitude=abs(score_after - score_before),
                            detected_at=detected_at,
                            cusum_statistic=max(cusum_pos, cusum_neg),
                        )
                    )
                    last_change_idx = i
                # Reset accumulators after detection
                cusum_pos = 0.0
                cusum_neg = 0.0

        return change_points

    @staticmethod
    def _check_seasonality(scores: list[float], period: int = 7) -> bool:
        """Check for seasonal patterns using autocorrelation at the given period.

        A feature is considered seasonal if autocorrelation at the target period
        is significantly higher than at shorter lags.

        Args:
            scores: List of drift scores.
            period: Period in observations to test for (default: 7 = weekly).

        Returns:
            True if seasonal pattern is detected.
        """
        n = len(scores)
        if n < period * 2:
            return False

        mean = statistics.mean(scores)
        variance = statistics.variance(scores) if n > 1 else 1.0
        if variance == 0:
            return False

        def autocorrelation(lag: int) -> float:
            if lag >= n:
                return 0.0
            cov = sum((scores[i] - mean) * (scores[i + lag] - mean) for i in range(n - lag)) / n
            return cov / variance

        acf_at_period = abs(autocorrelation(period))
        acf_nearby = max(abs(autocorrelation(period - 1)), abs(autocorrelation(period + 1)))
        return acf_at_period > 0.3 and acf_at_period > acf_nearby * 1.5

    @staticmethod
    def _pearson_correlation(x_vals: list[float], y_vals: list[float]) -> float:
        """Compute Pearson correlation coefficient.

        Args:
            x_vals: First variable values.
            y_vals: Second variable values.

        Returns:
            Pearson r in [-1, 1], or 0.0 if computation is not possible.
        """
        n = min(len(x_vals), len(y_vals))
        if n < 2:
            return 0.0
        x = x_vals[:n]
        y = y_vals[:n]
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
        if denom_x == 0 or denom_y == 0:
            return 0.0
        return num / (denom_x * denom_y)
