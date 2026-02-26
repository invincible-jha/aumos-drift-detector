"""Unit tests for the Kolmogorov-Smirnov drift test.

Tests use deterministic distributions to verify correct p-value and drift verdict.
"""

import numpy as np
import pytest

from aumos_drift_detector.adapters.statistical_tests.ks_test import (
    KolmogorovSmirnovResult,
    KolmogorovSmirnovTest,
)


class TestKolmogorovSmirnovTest:
    """Tests for KolmogorovSmirnovTest.run() with known distribution pairs."""

    def test_identical_distributions_no_drift(self) -> None:
        """Identical distributions must not be flagged as drifted."""
        rng = np.random.default_rng(seed=42)
        reference = rng.normal(0, 1, 10000)
        production = rng.normal(0, 1, 10000)
        result = KolmogorovSmirnovTest.run(reference, production, threshold=0.05)
        # With 10k samples from same dist, p-value should be large
        assert not result.is_drifted
        assert result.p_value >= 0.05

    def test_heavily_shifted_distribution_detects_drift(self) -> None:
        """A large mean shift must be detected as drift."""
        rng = np.random.default_rng(seed=0)
        reference = rng.normal(0, 1, 1000)
        production = rng.normal(10, 1, 1000)  # 10-sigma shift
        result = KolmogorovSmirnovTest.run(reference, production, threshold=0.05)
        assert result.is_drifted
        assert result.p_value < 0.05
        assert result.statistic > 0.5  # large KS statistic

    def test_result_contains_correct_sample_sizes(self) -> None:
        """Result must record reference and production sample sizes."""
        ref = np.arange(100, dtype=float)
        prod = np.arange(200, dtype=float)
        result = KolmogorovSmirnovTest.run(ref, prod, threshold=0.05)
        assert result.reference_size == 100
        assert result.production_size == 200

    def test_feature_name_propagated(self) -> None:
        """Feature name must appear in the result."""
        ref = np.linspace(0, 1, 100)
        prod = np.linspace(0, 1, 100)
        result = KolmogorovSmirnovTest.run(ref, prod, feature_name="age")
        assert result.feature_name == "age"

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict() must include all required keys for JSONB storage."""
        ref = np.linspace(0, 1, 50)
        prod = np.linspace(0, 1, 50)
        result = KolmogorovSmirnovTest.run(ref, prod)
        d = result.to_dict()
        for key in ("test", "feature", "statistic", "p_value", "threshold", "is_drifted"):
            assert key in d

    def test_empty_reference_raises_value_error(self) -> None:
        """Empty reference array must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            KolmogorovSmirnovTest.run(np.array([]), np.array([1.0, 2.0]))

    def test_empty_production_raises_value_error(self) -> None:
        """Empty production array must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            KolmogorovSmirnovTest.run(np.array([1.0, 2.0]), np.array([]))

    def test_nan_values_removed_before_test(self) -> None:
        """NaN values must be silently removed before the test runs."""
        ref = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        prod = np.array([1.0, 2.0, 3.0])
        result = KolmogorovSmirnovTest.run(ref, prod)
        assert result.reference_size == 3
        assert result.production_size == 3

    def test_multivariate_returns_result_per_feature(self) -> None:
        """run_multivariate must return one result per feature key."""
        rng = np.random.default_rng(seed=7)
        reference = {
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.normal(5, 2, 500),
        }
        production = {
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.normal(5, 2, 500),
        }
        results = KolmogorovSmirnovTest.run_multivariate(reference, production)
        assert set(results.keys()) == {"feature_a", "feature_b"}
        assert all(isinstance(v, KolmogorovSmirnovResult) for v in results.values())

    def test_multivariate_mismatched_keys_raises(self) -> None:
        """Mismatched feature keys between reference and production must raise."""
        ref = {"a": np.array([1.0, 2.0])}
        prod = {"b": np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="same feature columns"):
            KolmogorovSmirnovTest.run_multivariate(ref, prod)

    def test_statistic_is_bounded(self) -> None:
        """KS statistic must be in [0, 1]."""
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        prod = np.array([10.0, 11.0, 12.0, 13.0])
        result = KolmogorovSmirnovTest.run(ref, prod)
        assert 0.0 <= result.statistic <= 1.0

    def test_custom_threshold_affects_verdict(self) -> None:
        """A very strict threshold (0.001) should change drift verdict."""
        rng = np.random.default_rng(seed=99)
        ref = rng.normal(0, 1, 500)
        prod = rng.normal(0.3, 1, 500)  # tiny shift — p-value may be borderline

        result_strict = KolmogorovSmirnovTest.run(ref, prod, threshold=0.999)
        assert result_strict.is_drifted  # almost always drifted at 99.9% threshold

        result_lenient = KolmogorovSmirnovTest.run(ref, prod, threshold=0.0001)
        assert not result_lenient.is_drifted  # tiny shift, very lenient threshold
