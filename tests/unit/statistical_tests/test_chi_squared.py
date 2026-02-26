"""Unit tests for the Chi-squared categorical drift test."""

import pytest

from aumos_drift_detector.adapters.statistical_tests.chi_squared import (
    ChiSquaredResult,
    ChiSquaredTest,
)


class TestChiSquaredTest:
    """Tests for ChiSquaredTest.run() with known category distributions."""

    def test_identical_distribution_no_drift(self) -> None:
        """Identical category frequencies must not be flagged as drift."""
        ref = {"A": 500, "B": 300, "C": 200}
        prod = {"A": 500, "B": 300, "C": 200}
        result = ChiSquaredTest.run(ref, prod, threshold=0.05)
        assert not result.is_drifted
        # Identical distributions → chi2 statistic = 0, p_value = 1.0
        assert result.statistic == pytest.approx(0.0, abs=1e-6)

    def test_heavily_shifted_distribution_detects_drift(self) -> None:
        """A large shift in category proportions must be detected."""
        ref = {"cat": 900, "dog": 100}
        prod = {"cat": 100, "dog": 900}  # complete inversion
        result = ChiSquaredTest.run(ref, prod, threshold=0.05)
        assert result.is_drifted
        assert result.p_value < 0.05

    def test_stable_distribution_within_noise(self) -> None:
        """Small random variation around expected proportions must not trigger drift."""
        ref = {"A": 600, "B": 400}
        prod = {"A": 610, "B": 390}  # trivial deviation
        result = ChiSquaredTest.run(ref, prod, threshold=0.05)
        assert not result.is_drifted

    def test_degrees_of_freedom_equals_categories_minus_one(self) -> None:
        """Degrees of freedom must equal number of categories minus 1."""
        ref = {"X": 100, "Y": 100, "Z": 100}
        prod = {"X": 100, "Y": 100, "Z": 100}
        result = ChiSquaredTest.run(ref, prod)
        assert result.degrees_of_freedom == 2  # 3 categories - 1

    def test_categories_list_is_sorted(self) -> None:
        """Category list in result must be alphabetically sorted."""
        ref = {"Z": 100, "A": 200, "M": 150}
        prod = {"Z": 100, "A": 200, "M": 150}
        result = ChiSquaredTest.run(ref, prod)
        assert result.categories == ["A", "M", "Z"]

    def test_new_production_category_included(self) -> None:
        """A category present in production but not in reference must be included."""
        ref = {"cat": 500, "dog": 500}
        prod = {"cat": 400, "dog": 400, "bird": 200}  # new category
        result = ChiSquaredTest.run(ref, prod)
        assert "bird" in result.categories
        # bird has expected count of 0 * production_total — should be detected as drift
        assert result.is_drifted

    def test_feature_name_in_result(self) -> None:
        """Feature name must be propagated to the result."""
        ref = {"red": 100, "blue": 100}
        prod = {"red": 100, "blue": 100}
        result = ChiSquaredTest.run(ref, prod, feature_name="color")
        assert result.feature_name == "color"

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict() must include all required JSONB storage keys."""
        ref = {"A": 50, "B": 50}
        prod = {"A": 50, "B": 50}
        d = ChiSquaredTest.run(ref, prod).to_dict()
        for key in ("test", "feature", "statistic", "p_value", "threshold", "is_drifted",
                    "degrees_of_freedom", "categories"):
            assert key in d

    def test_empty_reference_raises(self) -> None:
        """Empty reference must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            ChiSquaredTest.run({}, {"A": 10})

    def test_zero_reference_total_raises(self) -> None:
        """Zero total reference count must raise ValueError."""
        with pytest.raises(ValueError, match="total count is 0"):
            ChiSquaredTest.run({"A": 0, "B": 0}, {"A": 10})

    def test_zero_production_total_raises(self) -> None:
        """Zero total production count must raise ValueError."""
        with pytest.raises(ValueError, match="total count is 0"):
            ChiSquaredTest.run({"A": 100}, {"A": 0})

    def test_counts_from_array(self) -> None:
        """counts_from_array must return correct per-category counts."""
        values = ["cat", "dog", "cat", "bird", "dog", "cat"]
        counts = ChiSquaredTest.counts_from_array(values)
        assert counts["cat"] == 3
        assert counts["dog"] == 2
        assert counts["bird"] == 1

    def test_statistic_is_non_negative(self) -> None:
        """Chi-squared statistic must always be non-negative."""
        ref = {"A": 200, "B": 300}
        prod = {"A": 250, "B": 250}
        result = ChiSquaredTest.run(ref, prod)
        assert result.statistic >= 0.0

    def test_p_value_bounded(self) -> None:
        """p-value must be in [0, 1]."""
        ref = {"X": 1000, "Y": 1000}
        prod = {"X": 999, "Y": 1001}
        result = ChiSquaredTest.run(ref, prod)
        assert 0.0 <= result.p_value <= 1.0
