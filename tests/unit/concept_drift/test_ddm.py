"""Unit tests for DDM and EDDM concept drift detectors."""

import pytest

from aumos_drift_detector.adapters.concept_drift.ddm import (
    DdmDetector,
    DdmState,
    DriftLevel,
    EddmDetector,
)


class TestDdmDetector:
    """Tests for the Drift Detection Method (DDM) detector."""

    def test_stable_stream_no_drift(self) -> None:
        """A stable low-error stream must remain at NORMAL level."""
        detector = DdmDetector(warning_level=2.0, drift_level=3.0)
        for _ in range(200):
            detector.update(0.0)  # all correct
        assert detector.detect() == DriftLevel.NORMAL

    def test_high_error_stream_detects_drift(self) -> None:
        """All-wrong predictions after stable period must trigger drift."""
        detector = DdmDetector(warning_level=2.0, drift_level=3.0)
        # Stable low-error period
        for _ in range(100):
            detector.update(0.0)
        # All errors — should trigger drift
        drift_detected = False
        for _ in range(300):
            detector.update(1.0)
            if detector.detect() == DriftLevel.DRIFT:
                drift_detected = True
                break
        assert drift_detected

    def test_invalid_levels_raise(self) -> None:
        """warning_level >= drift_level must raise ValueError."""
        with pytest.raises(ValueError, match="warning_level"):
            DdmDetector(warning_level=3.0, drift_level=2.0)
        with pytest.raises(ValueError, match="warning_level"):
            DdmDetector(warning_level=3.0, drift_level=3.0)

    def test_detector_starts_at_normal(self) -> None:
        """Freshly created detector must report NORMAL before any updates."""
        detector = DdmDetector()
        assert detector.detect() == DriftLevel.NORMAL

    def test_min_instances_delays_detection(self) -> None:
        """Drift must not be flagged before min_num_instances samples."""
        detector = DdmDetector(min_num_instances=50)
        # Feed 49 errors — no drift yet
        for _ in range(49):
            detector.update(1.0)
        assert detector.detect() == DriftLevel.NORMAL

    def test_reset_clears_state(self) -> None:
        """Manual reset must bring detector back to NORMAL."""
        detector = DdmDetector()
        for _ in range(50):
            detector.update(1.0)
        detector.reset()
        assert detector.detect() == DriftLevel.NORMAL

    def test_get_state_type(self) -> None:
        """get_state() must return a DdmState instance."""
        detector = DdmDetector()
        for _ in range(10):
            detector.update(0.1)
        state = detector.get_state()
        assert isinstance(state, DdmState)

    def test_state_to_dict_keys(self) -> None:
        """State dict must include required keys."""
        detector = DdmDetector()
        for _ in range(10):
            detector.update(0.0)
        d = detector.get_state().to_dict()
        assert "detector" in d
        assert "level" in d
        assert "n_samples" in d
        assert "error_rate" in d

    def test_warning_before_drift(self) -> None:
        """Detector may pass through WARNING before reaching DRIFT."""
        detector = DdmDetector(warning_level=2.0, drift_level=3.0, min_num_instances=30)
        # Stable period
        for _ in range(100):
            detector.update(0.05)
        # Rising error — may see WARNING
        levels_seen = set()
        for _ in range(500):
            detector.update(0.5)
            levels_seen.add(detector.detect())
            if DriftLevel.DRIFT in levels_seen:
                break
        # Either warning or drift should have been observed
        assert DriftLevel.DRIFT in levels_seen or DriftLevel.WARNING in levels_seen


class TestEddmDetector:
    """Tests for the Enhanced Drift Detection Method (EDDM) detector."""

    def test_stable_stream_no_drift(self) -> None:
        """A stream with evenly spaced errors must remain NORMAL."""
        detector = EddmDetector()
        # Errors at regular intervals — uniform inter-error distance
        for i in range(500):
            detector.update(1.0 if i % 10 == 0 else 0.0)
        assert detector.detect() == DriftLevel.NORMAL

    def test_increasing_error_rate_detects_drift(self) -> None:
        """Errors clustering together (shrinking inter-error distance) must trigger drift."""
        detector = EddmDetector(min_num_errors=10)
        # Stable period with sparse errors
        for i in range(200):
            detector.update(1.0 if i % 20 == 0 else 0.0)
        # Drift: dense errors (inter-error distance shrinks to 1)
        drift_detected = False
        for _ in range(500):
            detector.update(1.0)
            if detector.detect() == DriftLevel.DRIFT:
                drift_detected = True
                break
        assert drift_detected

    def test_invalid_levels_raise(self) -> None:
        """drift_level >= warning_level must raise ValueError."""
        with pytest.raises(ValueError, match="drift_level"):
            EddmDetector(warning_level=0.90, drift_level=0.95)
        with pytest.raises(ValueError, match="drift_level"):
            EddmDetector(warning_level=0.90, drift_level=0.90)

    def test_detector_starts_at_normal(self) -> None:
        """Fresh EDDM detector must be NORMAL before updates."""
        detector = EddmDetector()
        assert detector.detect() == DriftLevel.NORMAL

    def test_reset_restores_normal(self) -> None:
        """reset() must restore detector to NORMAL."""
        detector = EddmDetector()
        for _ in range(100):
            detector.update(1.0)
        detector.reset()
        assert detector.detect() == DriftLevel.NORMAL

    def test_get_state_returns_ddm_state(self) -> None:
        """get_state() must return a DdmState instance."""
        detector = EddmDetector()
        for i in range(20):
            detector.update(1.0 if i % 5 == 0 else 0.0)
        state = detector.get_state()
        assert isinstance(state, DdmState)
        assert state.level in DriftLevel.__members__.values()

    def test_update_with_only_correct_predictions(self) -> None:
        """No errors in the stream must not produce any drift signal."""
        detector = EddmDetector()
        for _ in range(500):
            detector.update(0.0)  # all correct — never reaches min_num_errors
        assert detector.detect() == DriftLevel.NORMAL
