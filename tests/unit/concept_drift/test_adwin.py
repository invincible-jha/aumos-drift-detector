"""Unit tests for the ADWIN concept drift detector."""

import pytest

from aumos_drift_detector.adapters.concept_drift.adwin import AdwinDetector, AdwinState
from aumos_drift_detector.adapters.concept_drift.ddm import DriftLevel


class TestAdwinDetector:
    """Tests for AdwinDetector with controlled error sequences."""

    def test_stable_stream_no_drift(self) -> None:
        """A stable stream of constant values must not trigger drift."""
        detector = AdwinDetector(delta=0.002)
        for _ in range(500):
            detector.update(0.1)
        assert detector.detect() == DriftLevel.NORMAL

    def test_abrupt_shift_triggers_drift(self) -> None:
        """An abrupt shift from 0.1 to 0.9 must eventually trigger drift."""
        detector = AdwinDetector(delta=0.002)
        # Stable period
        for _ in range(300):
            detector.update(0.1)
        # Drift period — abrupt change
        drift_detected = False
        for _ in range(500):
            detector.update(0.9)
            if detector.detect() == DriftLevel.DRIFT:
                drift_detected = True
                break
        assert drift_detected, "ADWIN should detect the abrupt mean shift"

    def test_window_shrinks_after_drift(self) -> None:
        """After drift, the window should be smaller than before the shift."""
        detector = AdwinDetector(delta=0.002)
        for _ in range(300):
            detector.update(0.1)
        width_before = detector.width

        for _ in range(500):
            detector.update(0.9)
            if detector.detect() == DriftLevel.DRIFT:
                break

        assert detector.width < width_before

    def test_reset_clears_state(self) -> None:
        """After reset, window size must be 0."""
        detector = AdwinDetector(delta=0.002)
        for i in range(100):
            detector.update(float(i % 2))
        detector.reset()
        assert detector.width == 0
        assert detector.mean == 0.0

    def test_invalid_delta_raises(self) -> None:
        """Delta must be strictly between 0 and 1."""
        with pytest.raises(ValueError, match="delta"):
            AdwinDetector(delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            AdwinDetector(delta=1.0)
        with pytest.raises(ValueError, match="delta"):
            AdwinDetector(delta=-0.1)

    def test_get_state_returns_adwin_state(self) -> None:
        """get_state() must return an AdwinState with correct delta."""
        detector = AdwinDetector(delta=0.01)
        for _ in range(10):
            detector.update(0.5)
        state = detector.get_state()
        assert isinstance(state, AdwinState)
        assert state.delta == 0.01
        assert state.window_size == detector.width

    def test_state_to_dict_contains_detector_key(self) -> None:
        """State dict must include the 'detector' key set to 'adwin'."""
        detector = AdwinDetector()
        detector.update(0.5)
        d = detector.get_state().to_dict()
        assert d["detector"] == "adwin"
        assert "drift_detected" in d
        assert "window_size" in d

    def test_mean_updates_incrementally(self) -> None:
        """Mean must track the average of values in the current window."""
        detector = AdwinDetector(delta=0.002)
        # All values are 0.5 — mean should be close to 0.5
        for _ in range(200):
            detector.update(0.5)
        assert abs(detector.mean - 0.5) < 0.01

    def test_single_update(self) -> None:
        """Single update must not crash and must return NORMAL."""
        detector = AdwinDetector()
        detector.update(0.5)
        assert detector.detect() == DriftLevel.NORMAL

    def test_total_updates_counts_correctly(self) -> None:
        """total_updates in state must equal the number of update() calls."""
        detector = AdwinDetector()
        for i in range(42):
            detector.update(float(i % 2) * 0.1)
        state = detector.get_state()
        assert state.total_updates == 42
