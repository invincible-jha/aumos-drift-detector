"""Population Stability Index (PSI) drift test.

PSI measures how much a feature's distribution has shifted between two
populations (typically reference vs. current). It was originally developed
for credit scoring to compare scorecards over time but is now widely used
in MLOps monitoring.

PSI is computed by:
1. Binning the reference distribution into `num_bins` equal-frequency buckets
2. Computing the fraction of production samples that fall into each bin
3. PSI = sum_i((prod_frac_i - ref_frac_i) * ln(prod_frac_i / ref_frac_i))

Interpretation:
    PSI < 0.1     → No significant drift (stable)
    0.1 ≤ PSI < 0.2 → Moderate drift (investigate)
    PSI ≥ 0.2     → Significant drift (retraining likely needed)

Reference:
    Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
    Intelligent Credit Scoring. Wiley.

Example:
    >>> import numpy as np
    >>> ref = np.random.normal(0, 1, 10000)
    >>> prod_stable = np.random.normal(0, 1, 5000)
    >>> result = PopulationStabilityIndex.run(ref, prod_stable)
    >>> result.psi < 0.1   # stable
    True
    >>> prod_shifted = np.random.normal(2, 1, 5000)
    >>> result2 = PopulationStabilityIndex.run(ref, prod_shifted)
    >>> result2.psi > 0.2  # significant drift
    True
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PsiResult:
    """Result of a Population Stability Index calculation.

    Attributes:
        psi: PSI score (non-negative). Higher = more drift.
        threshold: PSI threshold for declaring drift.
        is_drifted: True if psi >= threshold.
        num_bins: Number of bins used for the calculation.
        bin_edges: Bin boundary values.
        reference_fractions: Reference fraction per bin.
        production_fractions: Production fraction per bin.
        per_bin_psi: Contribution to total PSI from each bin.
        feature_name: Name of the feature tested.
        reference_size: Number of reference samples.
        production_size: Number of production samples.
    """

    psi: float
    threshold: float
    is_drifted: bool
    num_bins: int
    bin_edges: list[float]
    reference_fractions: list[float]
    production_fractions: list[float]
    per_bin_psi: list[float]
    feature_name: str = "unknown"
    reference_size: int = 0
    production_size: int = 0

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSONB storage.

        Returns:
            Dict representation of this result.
        """
        return {
            "test": "psi",
            "feature": self.feature_name,
            "psi": self.psi,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "num_bins": self.num_bins,
            "bin_edges": self.bin_edges,
            "reference_fractions": self.reference_fractions,
            "production_fractions": self.production_fractions,
            "per_bin_psi": self.per_bin_psi,
            "reference_size": self.reference_size,
            "production_size": self.production_size,
        }


# Minimum fraction per bin to avoid log(0) — standard practice is 0.0001
_EPSILON = 1e-4


class PopulationStabilityIndex:
    """Population Stability Index for detecting distribution shift.

    Stateless class — use the `run` class method directly.
    """

    @classmethod
    def run(
        cls,
        reference: np.ndarray,
        production: np.ndarray,
        threshold: float = 0.2,
        num_bins: int = 10,
        feature_name: str = "unknown",
    ) -> PsiResult:
        """Compute the PSI between reference and production distributions.

        Bin edges are derived from the reference distribution using equal-frequency
        (quantile-based) binning. Both arrays are then histogrammed against those
        same edges. A small epsilon is applied to avoid division by zero.

        Args:
            reference: 1-D array of reference feature values.
            production: 1-D array of current (production) feature values.
            threshold: PSI threshold for declaring drift (default 0.2).
            num_bins: Number of bins (default 10). More bins = finer resolution
                      but noisier estimate with small samples.
            feature_name: Optional label for the feature being tested.

        Returns:
            PsiResult with PSI score, per-bin breakdown, and drift verdict.

        Raises:
            ValueError: If either array is empty, or all values are identical.

        Example:
            >>> import numpy as np
            >>> ref = np.linspace(0, 10, 1000)
            >>> prod = np.linspace(5, 15, 1000)  # shifted by 5
            >>> result = PopulationStabilityIndex.run(ref, prod, threshold=0.2)
            >>> result.is_drifted
            True
        """
        reference = np.asarray(reference, dtype=float)
        production = np.asarray(production, dtype=float)

        reference = reference[np.isfinite(reference)]
        production = production[np.isfinite(production)]

        if reference.size == 0:
            raise ValueError("Reference dataset is empty after removing NaN/inf")
        if production.size == 0:
            raise ValueError("Production dataset is empty after removing NaN/inf")
        if np.unique(reference).size == 1:
            raise ValueError(
                "Reference distribution is constant — PSI requires variance"
            )

        # Compute equal-frequency bin edges from reference
        # Prepend -inf and append +inf to catch all production values
        quantiles = np.linspace(0, 100, num_bins + 1)
        raw_edges = np.percentile(reference, quantiles)
        bin_edges = np.concatenate([[-np.inf], raw_edges[1:-1], [np.inf]])

        # Count samples per bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # Convert to fractions with epsilon to avoid log(0)
        ref_fractions = np.maximum(ref_counts / reference.size, _EPSILON)
        prod_fractions = np.maximum(prod_counts / production.size, _EPSILON)

        # PSI per bin = (prod - ref) * ln(prod / ref)
        per_bin_psi = (prod_fractions - ref_fractions) * np.log(prod_fractions / ref_fractions)
        psi_total = float(np.sum(per_bin_psi))

        # Bin edges for serialisation (exclude ±inf)
        serialisable_edges = [
            float(e) if np.isfinite(e) else (float("inf") if e > 0 else float("-inf"))
            for e in raw_edges
        ]

        return PsiResult(
            psi=psi_total,
            threshold=threshold,
            is_drifted=bool(psi_total >= threshold),
            num_bins=num_bins,
            bin_edges=serialisable_edges,
            reference_fractions=ref_fractions.tolist(),
            production_fractions=prod_fractions.tolist(),
            per_bin_psi=per_bin_psi.tolist(),
            feature_name=feature_name,
            reference_size=int(reference.size),
            production_size=int(production.size),
        )

    @classmethod
    def run_multivariate(
        cls,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        threshold: float = 0.2,
        num_bins: int = 10,
    ) -> dict[str, PsiResult]:
        """Compute PSI across multiple features.

        Args:
            reference: Dict of feature_name → reference array.
            production: Dict of feature_name → production array.
            threshold: PSI threshold applied to all features.
            num_bins: Number of histogram bins per feature.

        Returns:
            Dict of feature_name → PsiResult.
        """
        if set(reference.keys()) != set(production.keys()):
            raise ValueError("Reference and production must have the same feature columns")

        return {
            feature: cls.run(
                reference=reference[feature],
                production=production[feature],
                threshold=threshold,
                num_bins=num_bins,
                feature_name=feature,
            )
            for feature in reference
        }
