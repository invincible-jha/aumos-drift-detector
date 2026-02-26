"""DDM (Drift Detection Method) and EDDM (Enhanced DDM) concept drift detectors.

DDM tracks the online error rate of a classifier and signals drift when the
error rate increases significantly above the minimum observed since the last
reset. It maintains two levels:

    WARNING — error is rising (may be transient noise)
    DRIFT   — error has crossed the drift threshold (retraining likely needed)

EDDM (Enhanced DDM) improves on DDM by considering the distance between two
consecutive errors, which makes it more sensitive to gradual drift.

Reference (DDM):
    Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
    "Learning with drift detection". Proceedings of the 17th Brazilian
    Symposium on Artificial Intelligence (SBIA 2004), pp. 286-295.

Reference (EDDM):
    Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavalda, R.,
    & Morales-Bueno, R. (2006). "Early drift detection method".
    Proceedings of the ECML/PKDD International Workshop on Knowledge Discovery
    from Data Streams.

Example (DDM):
    >>> detector = DdmDetector(warning_level=2.0, drift_level=3.0)
    >>> for _ in range(100):
    ...     detector.update(0.0)   # correct predictions
    >>> detector.detect().value
    'normal'
    >>> for _ in range(50):
    ...     detector.update(1.0)   # all wrong — drift
    >>> detector.detect().value
    'drift'
"""

import math
from dataclasses import dataclass
from enum import Enum


class DriftLevel(str, Enum):
    """Current drift level reported by a concept drift detector.

    Attributes:
        NORMAL:  No drift detected. Continue monitoring.
        WARNING: Error rate rising above baseline. May be transient noise.
        DRIFT:   Significant drift detected. Consider retraining.
    """

    NORMAL = "normal"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass
class DdmState:
    """Serialisable snapshot of DDM / EDDM detector state.

    Attributes:
        level: Current drift level.
        n_samples: Total number of samples processed.
        error_rate: Current online error rate.
        min_error_rate: Minimum error rate observed since last reset.
        warning_level: Warning level multiplier.
        drift_level: Drift level multiplier.
    """

    level: DriftLevel
    n_samples: int
    error_rate: float
    min_error_rate: float
    warning_level: float
    drift_level: float

    def to_dict(self) -> dict:
        """Serialise to dict for Kafka event payload or JSONB storage.

        Returns:
            Dict representation of detector state.
        """
        return {
            "detector": "ddm",
            "level": self.level.value,
            "n_samples": self.n_samples,
            "error_rate": self.error_rate,
            "min_error_rate": self.min_error_rate,
            "warning_level": self.warning_level,
            "drift_level": self.drift_level,
        }


class DdmDetector:
    """Drift Detection Method (DDM) for streaming concept drift.

    DDM models the error rate as a Bernoulli process and applies
    a normal approximation to track when the error rate has
    increased significantly above its historical minimum.

    The drift condition is:
        p + s >= p_min + K * s_min

    where:
        p     = current online error rate
        s     = std dev of p (= sqrt(p*(1-p)/n))
        p_min = minimum p observed since last reset
        s_min = std dev at the time p_min was achieved
        K     = warning_level or drift_level multiplier

    Args:
        warning_level: Multiplier for the warning threshold (default 2.0).
        drift_level: Multiplier for the drift threshold (default 3.0).
        min_num_instances: Minimum samples before drift detection starts (default 30).
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        """Initialise DDM detector.

        Args:
            warning_level: Standard deviation multiplier for the warning level.
            drift_level: Standard deviation multiplier for the drift level.
            min_num_instances: Minimum samples before drift checks begin.

        Raises:
            ValueError: If warning_level >= drift_level.
        """
        if warning_level >= drift_level:
            raise ValueError(
                f"warning_level ({warning_level}) must be less than "
                f"drift_level ({drift_level})"
            )
        self._warning_level = warning_level
        self._drift_level = drift_level
        self._min_instances = min_num_instances
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialise / reset all internal state to post-reset values."""
        self._n: int = 0          # number of samples seen
        self._p: float = 1.0      # current error rate
        self._s: float = 0.0      # current std dev
        self._p_min: float = float("inf")  # minimum error rate seen
        self._s_min: float = float("inf")  # std dev at p_min
        self._level: DriftLevel = DriftLevel.NORMAL

    @property
    def warning_level(self) -> float:
        """Warning multiplier for this detector."""
        return self._warning_level

    @property
    def drift_level(self) -> float:
        """Drift multiplier for this detector."""
        return self._drift_level

    def update(self, error: float) -> None:
        """Add one binary error observation and update drift level.

        Args:
            error: 1.0 if the model made an error on this sample, 0.0 if correct.
                   Values between 0 and 1 are accepted for probabilistic errors.

        Example:
            >>> detector = DdmDetector()
            >>> detector.update(0.0)  # correct prediction
            >>> detector.update(1.0)  # incorrect prediction
        """
        self._n += 1

        # Incremental update of error rate (p) and its std dev (s)
        # p = running mean of error, s = sqrt(p*(1-p)/n)
        self._p += (error - self._p) / self._n
        self._s = math.sqrt(self._p * (1.0 - self._p) / self._n)

        if self._n < self._min_instances:
            return

        # Track minimum p + s
        if self._p + self._s <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s

        if math.isinf(self._p_min):
            return

        metric = self._p + self._s
        warning_bound = self._p_min + self._warning_level * self._s_min
        drift_bound = self._p_min + self._drift_level * self._s_min

        if metric >= drift_bound:
            self._level = DriftLevel.DRIFT
            self._reset_state()  # DDM resets after confirming drift
        elif metric >= warning_bound:
            self._level = DriftLevel.WARNING
        else:
            self._level = DriftLevel.NORMAL

    def detect(self) -> DriftLevel:
        """Return the current drift level.

        Returns:
            DriftLevel reflecting the most recent update outcome.
        """
        return self._level

    def reset(self) -> None:
        """Manually reset the detector (e.g., after external retraining).

        After reset, the detector requires `min_num_instances` samples
        before drift detection resumes.
        """
        self._reset_state()

    def get_state(self) -> DdmState:
        """Return a serialisable snapshot of the current state.

        Returns:
            DdmState with current error rate and drift level.
        """
        return DdmState(
            level=self._level,
            n_samples=self._n,
            error_rate=self._p,
            min_error_rate=self._p_min if not math.isinf(self._p_min) else 0.0,
            warning_level=self._warning_level,
            drift_level=self._drift_level,
        )


class EddmDetector:
    """Enhanced DDM (EDDM) concept drift detector.

    EDDM improves on DDM by tracking the distance between consecutive errors
    instead of the raw error rate. This makes it more sensitive to gradual
    drift where errors increase slowly over time.

    The EDDM metric is:
        p' = mean distance between errors
        s' = std dev of distance between errors
        Drift when: (p' + 2*s')^2 / (p'_max + 2*s'_max)^2 < threshold

    In this implementation the warning and drift thresholds are expressed as
    fractions of the maximum (p' + 2*s')^2 seen so far, following the
    original paper's recommendation of 0.95 (warning) and 0.90 (drift).

    Args:
        warning_level: Warning threshold as fraction of max metric (default 0.95).
        drift_level: Drift threshold as fraction of max metric (default 0.90).
        min_num_errors: Minimum errors before drift detection starts (default 30).
    """

    def __init__(
        self,
        warning_level: float = 0.95,
        drift_level: float = 0.90,
        min_num_errors: int = 30,
    ) -> None:
        """Initialise EDDM detector.

        Args:
            warning_level: Warning threshold (fraction of max). Must be > drift_level.
            drift_level: Drift threshold (fraction of max). Must be < warning_level.
            min_num_errors: Minimum number of errors before detection starts.

        Raises:
            ValueError: If drift_level >= warning_level.
        """
        if drift_level >= warning_level:
            raise ValueError(
                f"drift_level ({drift_level}) must be less than "
                f"warning_level ({warning_level})"
            )
        self._warning_level = warning_level
        self._drift_level = drift_level
        self._min_num_errors = min_num_errors
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._n: int = 0          # total samples
        self._n_errors: int = 0   # total errors
        self._last_error: int = 0 # sample index of the last error
        self._mean_distance: float = 0.0
        self._var_distance: float = 0.0   # variance of inter-error distances (M2)
        self._max_metric: float = 0.0    # max (mean + 2*std)^2 observed
        self._level: DriftLevel = DriftLevel.NORMAL

    @property
    def warning_level(self) -> float:
        """Warning threshold fraction for this detector."""
        return self._warning_level

    @property
    def drift_level(self) -> float:
        """Drift threshold fraction for this detector."""
        return self._drift_level

    def update(self, error: float) -> None:
        """Add one binary error observation and update drift level.

        Args:
            error: 1.0 if the model made an error on this sample, 0.0 if correct.

        Example:
            >>> detector = EddmDetector()
            >>> detector.update(0.0)  # correct prediction
            >>> detector.update(1.0)  # error — updates inter-error distance
        """
        self._n += 1
        is_error = error >= 0.5  # treat as binary

        if not is_error:
            return

        self._n_errors += 1
        distance = float(self._n - self._last_error)
        self._last_error = self._n

        # Welford's online algorithm for mean and variance of inter-error distances
        delta = distance - self._mean_distance
        self._mean_distance += delta / self._n_errors
        delta2 = distance - self._mean_distance
        self._var_distance += delta * delta2

        if self._n_errors < self._min_num_errors:
            return

        std_distance = math.sqrt(self._var_distance / self._n_errors) if self._n_errors > 1 else 0.0
        metric_sq = (self._mean_distance + 2.0 * std_distance) ** 2

        if metric_sq > self._max_metric:
            self._max_metric = metric_sq

        if self._max_metric == 0.0:
            return

        ratio = metric_sq / self._max_metric

        if ratio < self._drift_level:
            self._level = DriftLevel.DRIFT
            self._reset_state()
        elif ratio < self._warning_level:
            self._level = DriftLevel.WARNING
        else:
            self._level = DriftLevel.NORMAL

    def detect(self) -> DriftLevel:
        """Return the current drift level.

        Returns:
            DriftLevel reflecting the most recent update outcome.
        """
        return self._level

    def reset(self) -> None:
        """Manually reset the detector after external retraining.

        After reset the detector requires `min_num_errors` errors
        before drift detection resumes.
        """
        self._reset_state()

    def get_state(self) -> DdmState:
        """Return a serialisable snapshot of the current state.

        Returns:
            DdmState compatible snapshot (using error_rate field for mean_distance).
        """
        return DdmState(
            level=self._level,
            n_samples=self._n,
            error_rate=self._mean_distance,
            min_error_rate=0.0,
            warning_level=self._warning_level,
            drift_level=self._drift_level,
        )
