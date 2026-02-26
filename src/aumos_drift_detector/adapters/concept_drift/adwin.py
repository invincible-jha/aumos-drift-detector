"""ADWIN (ADaptive WINdowing) concept drift detector.

ADWIN is a streaming change-detection algorithm that maintains an adaptive
sliding window of recent values. It continuously tests whether the mean in
two sub-windows of the current window are significantly different. When a
statistically significant difference is found, the older portion of the
window is discarded (the window shrinks), signalling concept drift.

Key properties:
- No fixed window size — adapts to the rate of change automatically
- Provides a rigorous false positive bound controlled by `delta`
- Efficient O(log n) memory using an exponential histogram representation
- Works on any real-valued sequence (prediction error, accuracy, etc.)

Reference:
    Bifet, A. & Gavalda, R. (2007). "Learning from time-changing data with
    adaptive windowing". Proceedings of the Seventh SIAM International
    Conference on Data Mining (SDM 2007), pp. 443-448.

Example:
    >>> detector = AdwinDetector(delta=0.002)
    >>> import random
    >>> random.seed(42)
    >>> for _ in range(200):
    ...     detector.update(random.gauss(0, 1))  # stable period
    >>> detector.detect()
    <DriftLevel.NORMAL: 'normal'>
    >>> for _ in range(100):
    ...     detector.update(random.gauss(5, 1))  # concept drift
    >>> detector.detect()
    <DriftLevel.DRIFT: 'drift'>
"""

import math
from dataclasses import dataclass, field
from enum import Enum


class _DriftLevelLocal(str, Enum):
    """Local drift level enum (imported into parent package as DriftLevel)."""

    NORMAL = "normal"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass
class AdwinState:
    """Serialisable snapshot of ADWIN detector state.

    Attributes:
        drift_detected: True if drift was detected in the last update.
        window_size: Current number of elements in the adaptive window.
        window_mean: Mean of elements in the current window.
        total_updates: Total number of elements seen since last reset.
        delta: Confidence parameter in use.
    """

    drift_detected: bool
    window_size: int
    window_mean: float
    total_updates: int
    delta: float

    def to_dict(self) -> dict:
        """Serialise to dict for Kafka event payload or JSONB storage.

        Returns:
            Dict representation of detector state.
        """
        return {
            "detector": "adwin",
            "drift_detected": self.drift_detected,
            "window_size": self.window_size,
            "window_mean": self.window_mean,
            "total_updates": self.total_updates,
            "delta": self.delta,
        }


@dataclass
class _Bucket:
    """One entry in ADWIN's exponential histogram.

    Attributes:
        total: Sum of values in this bucket.
        variance: Variance accumulator for this bucket.
        size: Number of elements represented by this bucket.
    """

    total: float = 0.0
    variance: float = 0.0
    size: int = 0


class AdwinDetector:
    """ADaptive WINdowing concept drift detector.

    Maintains an exponential histogram (a list of buckets) where each bucket
    represents 2^i elements. When the window grows large enough that any
    sub-window split shows a statistically significant mean difference, the
    detector reports drift and drops the older portion of the window.

    This implementation uses the simplified version of the ADWIN algorithm
    that tracks sum and variance in each bucket without storing raw elements.

    Args:
        delta: Confidence parameter (false positive rate bound). Smaller delta
               means fewer false positives but slower detection. Default 0.002.
        max_buckets: Maximum number of buckets per row in the histogram.
                     Higher = more memory, finer resolution. Default 5.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5) -> None:
        """Initialise ADWIN detector.

        Args:
            delta: False positive bound (0 < delta < 1). Common values: 0.002, 0.05.
            max_buckets: Maximum buckets per row in the exponential histogram.

        Raises:
            ValueError: If delta is not in (0, 1).
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self._delta = delta
        self._max_buckets = max_buckets
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialise / reset all internal state variables."""
        # Exponential histogram: list of lists of _Bucket objects
        # _buckets[i] contains buckets representing 2^i elements each
        self._buckets: list[list[_Bucket]] = [[]]
        self._total: float = 0.0
        self._variance: float = 0.0
        self._width: int = 0  # total number of elements in the window
        self._n_buckets: int = 0  # total number of buckets
        self._drift_detected: bool = False
        self._total_updates: int = 0

    @property
    def delta(self) -> float:
        """Confidence parameter for this detector."""
        return self._delta

    @property
    def width(self) -> int:
        """Number of elements in the current adaptive window."""
        return self._width

    @property
    def mean(self) -> float:
        """Mean of elements in the current window. Returns 0.0 if empty."""
        if self._width == 0:
            return 0.0
        return self._total / self._width

    def update(self, value: float) -> None:
        """Add one element to the ADWIN window and check for drift.

        After inserting the new element, the detector compresses the histogram
        if needed and runs the drift test over all sub-window splits.

        Args:
            value: The next value in the stream (typically a prediction error
                   or accuracy measurement, range [0, 1] but not required).
        """
        self._total_updates += 1
        self._drift_detected = False

        # Insert new element as a singleton bucket at level 0
        new_bucket = _Bucket(total=value, variance=0.0, size=1)
        if not self._buckets:
            self._buckets.append([])
        self._buckets[0].append(new_bucket)
        self._n_buckets += 1

        # Update window statistics
        self._width += 1
        old_mean = self._total / self._width if self._width > 1 else value
        self._total += value
        new_mean = self._total / self._width
        # Welford's online variance update
        self._variance += (value - old_mean) * (value - new_mean)

        # Compress: merge buckets that overflow at any level
        self._compress_buckets()

        # Check for drift across all possible sub-window splits
        self._drift_detected = self._drift_test()

    def detect(self) -> "_DriftLevelLocal":
        """Return the current drift level based on the last update.

        Returns:
            DriftLevel.DRIFT if drift was detected on last update, else NORMAL.
        """
        if self._drift_detected:
            return _DriftLevelLocal.DRIFT
        return _DriftLevelLocal.NORMAL

    def reset(self) -> None:
        """Reset the detector to its initial state.

        Should be called after drift is detected and the model has been
        retrained or a new stable period is expected.
        """
        self._reset_state()

    def get_state(self) -> AdwinState:
        """Return a serialisable snapshot of the current detector state.

        Returns:
            AdwinState with current window statistics.
        """
        return AdwinState(
            drift_detected=self._drift_detected,
            window_size=self._width,
            window_mean=self.mean,
            total_updates=self._total_updates,
            delta=self._delta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_buckets(self) -> None:
        """Merge bucket pairs that overflow the max_buckets limit per level.

        When a level has more than max_buckets, the two oldest buckets are
        merged into a single bucket at the next level. This maintains the
        O(log n) space property.
        """
        level = 0
        while level < len(self._buckets):
            if len(self._buckets[level]) > self._max_buckets:
                if level + 1 >= len(self._buckets):
                    self._buckets.append([])
                # Merge the two oldest (first two) buckets at this level
                b0, b1 = self._buckets[level][0], self._buckets[level][1]
                self._buckets[level] = self._buckets[level][2:]
                merged_size = b0.size + b1.size
                merged_total = b0.total + b1.total
                mean0 = b0.total / b0.size if b0.size > 0 else 0.0
                mean1 = b1.total / b1.size if b1.size > 0 else 0.0
                merged_var = (
                    b0.variance + b1.variance
                    + (b0.size * b1.size / merged_size) * (mean0 - mean1) ** 2
                )
                self._buckets[level + 1].append(
                    _Bucket(total=merged_total, variance=merged_var, size=merged_size)
                )
                self._n_buckets -= 1
            level += 1

    def _drift_test(self) -> bool:
        """Test for drift by evaluating all sub-window splits.

        Iterates from the newest sub-window outward. For each split point,
        tests whether the mean difference exceeds the epsilon_cut threshold.

        Returns:
            True if drift is detected (and the old portion is dropped).
        """
        if self._width < 2:
            return False

        epsilon_cut = self._compute_epsilon_cut(self._width)

        # Accumulate sub-window stats from newest to oldest
        sub_total = 0.0
        sub_size = 0
        sub_variance = 0.0

        # Iterate over bucket levels from newest (level 0) to oldest
        for level in range(len(self._buckets)):
            for bucket in reversed(self._buckets[level]):
                sub_size += bucket.size
                sub_total += bucket.total
                sub_variance += bucket.variance

                complement_size = self._width - sub_size
                if complement_size <= 0:
                    continue

                complement_total = self._total - sub_total
                sub_mean = sub_total / sub_size
                comp_mean = complement_total / complement_size
                mean_diff = abs(sub_mean - comp_mean)

                if mean_diff >= epsilon_cut and sub_size >= 1:
                    # Drift detected — drop the older portion of the window
                    self._drop_old_buckets(complement_size)
                    self._total = sub_total
                    self._variance = sub_variance
                    self._width = sub_size
                    return True

        return False

    def _compute_epsilon_cut(self, window_size: int) -> float:
        """Compute the ADWIN epsilon_cut threshold for the given window size.

        epsilon_cut = sqrt((1 / (2 * m)) * ln(4 * W / delta))
        where m = harmonic mean of sub-window and complement sizes.

        Args:
            window_size: Total number of elements in the current window.

        Returns:
            Epsilon threshold for the mean difference test.
        """
        if window_size < 2:
            return float("inf")
        log_term = math.log(4 * window_size / self._delta)
        # Use the simplified form: epsilon = sqrt(ln(4W/delta) / (2 * W))
        return math.sqrt(log_term / (2.0 * window_size))

    def _drop_old_buckets(self, num_elements_to_drop: int) -> None:
        """Remove the oldest `num_elements_to_drop` elements from the window.

        Modifies self._buckets in-place, removing from the highest (oldest) levels.

        Args:
            num_elements_to_drop: Number of elements to discard.
        """
        remaining = num_elements_to_drop
        for level in range(len(self._buckets) - 1, -1, -1):
            while self._buckets[level] and remaining > 0:
                oldest = self._buckets[level][0]
                if oldest.size <= remaining:
                    remaining -= oldest.size
                    self._buckets[level].pop(0)
                    self._n_buckets -= 1
                else:
                    # Partially remove elements from the oldest bucket
                    frac = remaining / oldest.size
                    oldest.total -= frac * oldest.total
                    oldest.variance -= frac * oldest.variance
                    oldest.size -= remaining
                    remaining = 0
