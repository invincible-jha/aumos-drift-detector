"""Kolmogorov-Smirnov two-sample drift test.

The KS test measures whether two samples come from the same continuous
distribution by comparing their empirical CDFs. The test statistic D is
the maximum absolute difference between the two ECDFs. A small p-value
indicates that the distributions are likely different (drift detected).

Reference:
    Kolmogorov (1933), "Sulla determinazione empirica di una legge di
    distribuzione". Giornale dell'Istituto Italiano degli Attuari, 4, 83-91.

    scipy.stats.ks_2samp documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

Example:
    >>> import numpy as np
    >>> from aumos_drift_detector.adapters.statistical_tests.ks_test import KolmogorovSmirnovTest
    >>> reference = np.random.normal(0, 1, 1000)
    >>> production = np.random.normal(0, 1, 1000)   # same distribution
    >>> result = KolmogorovSmirnovTest.run(reference, production, threshold=0.05)
    >>> result.is_drifted  # False (p-value should be large)
    False
    >>> shifted = np.random.normal(2, 1, 1000)       # shifted distribution
    >>> result2 = KolmogorovSmirnovTest.run(reference, shifted, threshold=0.05)
    >>> result2.is_drifted  # True (p-value < 0.05)
    True
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class KolmogorovSmirnovResult:
    """Result of a Kolmogorov-Smirnov two-sample test.

    Attributes:
        statistic: KS test statistic D (max ECDF difference, range [0, 1]).
        p_value: Two-sided p-value. Small p-value indicates drift.
        threshold: The significance level used to determine drift.
        is_drifted: True if p_value < threshold.
        reference_size: Number of samples in the reference dataset.
        production_size: Number of samples in the production dataset.
        feature_name: Name of the feature tested (optional label).
    """

    statistic: float
    p_value: float
    threshold: float
    is_drifted: bool
    reference_size: int
    production_size: int
    feature_name: str = "unknown"

    def to_dict(self) -> dict:
        """Serialise result to a plain dict for JSONB storage.

        Returns:
            Dict representation of this result.
        """
        return {
            "test": "ks",
            "feature": self.feature_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "reference_size": self.reference_size,
            "production_size": self.production_size,
        }


class KolmogorovSmirnovTest:
    """Two-sample Kolmogorov-Smirnov drift test for continuous features.

    This class provides a single class-method `run` to perform the test.
    It is stateless — no instance creation is required.
    """

    @classmethod
    def run(
        cls,
        reference: np.ndarray,
        production: np.ndarray,
        threshold: float = 0.05,
        feature_name: str = "unknown",
    ) -> KolmogorovSmirnovResult:
        """Run the two-sample KS test between reference and production data.

        The test null hypothesis is that both samples come from the same
        continuous distribution. If the p-value is below `threshold`, we
        reject the null hypothesis and report drift.

        Args:
            reference: 1-D array of reference (baseline) feature values.
            production: 1-D array of current (production) feature values.
            threshold: Significance level. Drift is flagged when p_value < threshold.
                       Common choices: 0.05 (5%) or 0.01 (1%).
            feature_name: Optional label for the feature being tested.

        Returns:
            KolmogorovSmirnovResult containing the statistic, p-value, and verdict.

        Raises:
            ValueError: If either array is empty or contains non-finite values.

        Example:
            >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> prod = np.array([5.0, 6.0, 7.0, 8.0, 9.0])  # shifted
            >>> result = KolmogorovSmirnovTest.run(ref, prod, threshold=0.05)
            >>> result.is_drifted
            True
        """
        reference = np.asarray(reference, dtype=float)
        production = np.asarray(production, dtype=float)

        if reference.size == 0:
            raise ValueError("Reference dataset must not be empty")
        if production.size == 0:
            raise ValueError("Production dataset must not be empty")

        # Remove NaN values before testing
        reference = reference[np.isfinite(reference)]
        production = production[np.isfinite(production)]

        if reference.size == 0:
            raise ValueError("Reference dataset contains only NaN/inf values")
        if production.size == 0:
            raise ValueError("Production dataset contains only NaN/inf values")

        ks_result = stats.ks_2samp(reference, production, alternative="two-sided")

        return KolmogorovSmirnovResult(
            statistic=float(ks_result.statistic),
            p_value=float(ks_result.pvalue),
            threshold=threshold,
            is_drifted=bool(ks_result.pvalue < threshold),
            reference_size=int(reference.size),
            production_size=int(production.size),
            feature_name=feature_name,
        )

    @classmethod
    def run_multivariate(
        cls,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        threshold: float = 0.05,
    ) -> dict[str, KolmogorovSmirnovResult]:
        """Run KS tests across multiple features.

        Args:
            reference: Dict of feature_name → reference array.
            production: Dict of feature_name → production array.
            threshold: Significance level applied to all features.

        Returns:
            Dict of feature_name → KolmogorovSmirnovResult.

        Raises:
            ValueError: If reference and production have different feature keys.
        """
        if set(reference.keys()) != set(production.keys()):
            raise ValueError(
                "Reference and production must have the same feature columns. "
                f"Reference: {sorted(reference.keys())}, "
                f"Production: {sorted(production.keys())}"
            )

        return {
            feature: cls.run(
                reference=reference[feature],
                production=production[feature],
                threshold=threshold,
                feature_name=feature,
            )
            for feature in reference
        }
