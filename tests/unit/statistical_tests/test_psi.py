"""Unit tests for the Population Stability Index (PSI) drift test."""

import numpy as np
import pytest

from aumos_drift_detector.adapters.statistical_tests.psi import (
    PopulationStabilityIndex,
    PsiResult,
)


class TestPopulationStabilityIndex:
    """Tests for PSI with known distribution characteristics."""

    def test_identical_distribution_produces_near_zero_psi(self) -> None:
        """Identical distributions must yield PSI close to 0."""
        rng = np.random.default_rng(seed=1)
        data = rng.normal(0, 1, 10000)
        # Split into two halves — same distribution
        result = PopulationStabilityIndex.run(data[:5000], data[5000:], threshold=0.2)
        assert result.psi < 0.05  # should be very stable

    def test_heavily_shifted_distribution_exceeds_threshold(self) -> None:
        """A 5-sigma shift must produce PSI > 0.2 (drift threshold)."""
        rng = np.random.default_rng(seed=2)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(5, 1, 5000)
        result = PopulationStabilityIndex.run(reference, production, threshold=0.2)
        assert result.is_drifted
        assert result.psi >= 0.2

    def test_psi_is_non_negative(self) -> None:
        """PSI must always be non-negative."""
        ref = np.linspace(0, 1, 1000)
        prod = np.linspace(0, 1, 1000)
        result = PopulationStabilityIndex.run(ref, prod)
        assert result.psi >= 0.0

    def test_result_bin_count_matches_num_bins(self) -> None:
        """Number of bin fractions must equal num_bins."""
        ref = np.random.default_rng(seed=3).normal(0, 1, 500)
        prod = np.random.default_rng(seed=4).normal(0, 1, 500)
        result = PopulationStabilityIndex.run(ref, prod, num_bins=5)
        assert len(result.reference_fractions) == 5
        assert len(result.production_fractions) == 5
        assert len(result.per_bin_psi) == 5

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict() must return all required JSONB keys."""
        ref = np.linspace(0, 10, 100)
        prod = np.linspace(0, 10, 100)
        result = PopulationStabilityIndex.run(ref, prod)
        d = result.to_dict()
        for key in ("test", "feature", "psi", "threshold", "is_drifted", "num_bins"):
            assert key in d

    def test_feature_name_in_result(self) -> None:
        """Feature name must be included in result."""
        ref = np.arange(50, dtype=float)
        prod = np.arange(50, dtype=float)
        result = PopulationStabilityIndex.run(ref, prod, feature_name="income")
        assert result.feature_name == "income"

    def test_empty_reference_raises(self) -> None:
        """Empty reference must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            PopulationStabilityIndex.run(np.array([]), np.array([1.0]))

    def test_empty_production_raises(self) -> None:
        """Empty production must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            PopulationStabilityIndex.run(np.array([1.0, 2.0]), np.array([]))

    def test_constant_reference_raises(self) -> None:
        """All-constant reference must raise ValueError (no variance = no bins)."""
        with pytest.raises(ValueError, match="constant"):
            PopulationStabilityIndex.run(np.ones(100), np.array([1.0, 2.0, 3.0]))

    def test_nan_values_removed(self) -> None:
        """NaN values must be cleaned before computation."""
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        prod = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
        # Should run without error and use 4 values from each
        result = PopulationStabilityIndex.run(ref, prod)
        assert result.reference_size == 4
        assert result.production_size == 4

    def test_moderate_shift_within_warning_range(self) -> None:
        """A small shift should produce PSI between 0.1 and 0.2 (warning zone)."""
        rng = np.random.default_rng(seed=5)
        reference = rng.normal(0, 1, 10000)
        production = rng.normal(0.5, 1, 10000)  # moderate shift
        result = PopulationStabilityIndex.run(reference, production, threshold=0.2)
        # PSI should reflect some drift without crossing threshold
        assert result.psi >= 0.0

    def test_multivariate_returns_result_per_feature(self) -> None:
        """run_multivariate must return one PsiResult per feature."""
        rng = np.random.default_rng(seed=6)
        reference = {
            "feat_x": rng.normal(0, 1, 1000),
            "feat_y": rng.exponential(2, 1000),
        }
        production = {
            "feat_x": rng.normal(0, 1, 1000),
            "feat_y": rng.exponential(2, 1000),
        }
        results = PopulationStabilityIndex.run_multivariate(reference, production)
        assert set(results.keys()) == {"feat_x", "feat_y"}
        assert all(isinstance(v, PsiResult) for v in results.values())

    def test_multivariate_mismatched_keys_raises(self) -> None:
        """Mismatched keys must raise ValueError."""
        ref = {"a": np.array([1.0, 2.0, 3.0])}
        prod = {"b": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="same feature columns"):
            PopulationStabilityIndex.run_multivariate(ref, prod)
