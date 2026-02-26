"""Chi-squared test for categorical feature drift.

The chi-squared goodness-of-fit test compares the observed category frequencies
in the production dataset against the expected frequencies derived from the
reference dataset. A small p-value indicates that the production distribution
is significantly different from the reference (drift detected).

The test statistic is:
    chi2 = sum((observed_i - expected_i)^2 / expected_i)

where expected_i = reference_fraction_i * production_total.

Reference:
    Pearson, K. (1900). "On the criterion that a given system of deviations from
    the probable in the case of a correlated system of variables is such that it
    can be reasonably supposed to have arisen from random sampling".
    Philosophical Magazine, 50, 157-175.

    scipy.stats.chisquare:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

Example:
    >>> ref_counts = {"cat": 500, "dog": 300, "bird": 200}
    >>> prod_counts = {"cat": 490, "dog": 310, "bird": 200}  # stable
    >>> result = ChiSquaredTest.run(ref_counts, prod_counts, threshold=0.05)
    >>> result.is_drifted
    False
    >>> drifted_counts = {"cat": 100, "dog": 700, "bird": 200}  # shifted
    >>> result2 = ChiSquaredTest.run(ref_counts, drifted_counts, threshold=0.05)
    >>> result2.is_drifted
    True
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ChiSquaredResult:
    """Result of a chi-squared categorical drift test.

    Attributes:
        statistic: Chi-squared test statistic (non-negative).
        p_value: p-value from the chi-squared distribution.
        threshold: Significance level used to determine drift.
        is_drifted: True if p_value < threshold.
        degrees_of_freedom: Number of categories minus 1.
        categories: Ordered list of category labels tested.
        reference_counts: Observed counts per category in reference.
        production_counts: Observed counts per category in production.
        expected_counts: Expected production counts derived from reference fractions.
        feature_name: Name of the feature tested.
    """

    statistic: float
    p_value: float
    threshold: float
    is_drifted: bool
    degrees_of_freedom: int
    categories: list[str]
    reference_counts: list[int]
    production_counts: list[int]
    expected_counts: list[float]
    feature_name: str = "unknown"

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSONB storage.

        Returns:
            Dict representation of this result.
        """
        return {
            "test": "chi2",
            "feature": self.feature_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "degrees_of_freedom": self.degrees_of_freedom,
            "categories": self.categories,
            "reference_counts": self.reference_counts,
            "production_counts": self.production_counts,
            "expected_counts": self.expected_counts,
        }


class ChiSquaredTest:
    """Chi-squared test for categorical feature drift.

    Stateless class — use the `run` class method directly.
    """

    @classmethod
    def run(
        cls,
        reference_counts: dict[str, int],
        production_counts: dict[str, int],
        threshold: float = 0.05,
        feature_name: str = "unknown",
    ) -> ChiSquaredResult:
        """Run a chi-squared goodness-of-fit test for categorical drift.

        The null hypothesis is that the production category frequencies match
        the reference proportions. A significant result (p_value < threshold)
        indicates the category distribution has shifted.

        Categories present in reference but absent in production are assigned 0.
        Categories present in production but absent in reference are appended
        to the reference with a count of 0 (they contribute to the statistic
        as "unexpected" categories).

        Args:
            reference_counts: Dict mapping category labels to their counts in reference.
            production_counts: Dict mapping category labels to their counts in production.
            threshold: Significance level (default 0.05 = 5%). Drift if p_value < threshold.
            feature_name: Optional label for the feature being tested.

        Returns:
            ChiSquaredResult with test statistic, p-value, and drift verdict.

        Raises:
            ValueError: If reference_counts is empty, or reference total is 0.

        Example:
            >>> ref = {"A": 600, "B": 300, "C": 100}
            >>> prod = {"A": 100, "B": 700, "C": 200}  # heavy shift
            >>> ChiSquaredTest.run(ref, prod).is_drifted
            True
        """
        if not reference_counts:
            raise ValueError("reference_counts must not be empty")

        ref_total = sum(reference_counts.values())
        if ref_total == 0:
            raise ValueError("Reference total count is 0 — cannot compute proportions")

        # Build a unified category list (reference + any new categories in production)
        all_categories = sorted(
            set(reference_counts.keys()) | set(production_counts.keys())
        )

        ref_array = np.array(
            [reference_counts.get(cat, 0) for cat in all_categories], dtype=float
        )
        prod_array = np.array(
            [production_counts.get(cat, 0) for cat in all_categories], dtype=float
        )

        prod_total = float(prod_array.sum())
        if prod_total == 0:
            raise ValueError("Production total count is 0 — cannot perform test")

        # Expected production counts = ref_proportion * prod_total
        ref_proportions = ref_array / ref_total
        expected = ref_proportions * prod_total

        # Run scipy chi-squared test (observed vs. expected)
        chi2_result = stats.chisquare(f_obs=prod_array, f_exp=expected)

        degrees_of_freedom = len(all_categories) - 1

        return ChiSquaredResult(
            statistic=float(chi2_result.statistic),
            p_value=float(chi2_result.pvalue),
            threshold=threshold,
            is_drifted=bool(chi2_result.pvalue < threshold),
            degrees_of_freedom=degrees_of_freedom,
            categories=all_categories,
            reference_counts=[int(reference_counts.get(c, 0)) for c in all_categories],
            production_counts=[int(production_counts.get(c, 0)) for c in all_categories],
            expected_counts=expected.tolist(),
            feature_name=feature_name,
        )

    @classmethod
    def counts_from_array(cls, values: np.ndarray) -> dict[str, int]:
        """Build a category-count dict from a 1-D array of category labels.

        Convenience method to convert raw value arrays into the dict format
        expected by `run`.

        Args:
            values: 1-D array (or list) of categorical values (strings or ints).

        Returns:
            Dict mapping each unique value to its count.

        Example:
            >>> ChiSquaredTest.counts_from_array(["A", "B", "A", "C", "B"])
            {'A': 2, 'B': 2, 'C': 1}
        """
        unique, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
        return {str(u): int(c) for u, c in zip(unique, counts)}
