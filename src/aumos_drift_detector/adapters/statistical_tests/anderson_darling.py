"""Anderson-Darling two-sample test for aumos-drift-detector.

The Anderson-Darling test is more sensitive to differences in the tails
of distributions compared to the Kolmogorov-Smirnov test. It uses the
empirical CDF weighted by tail sensitivity.

GAP-166: Extended Statistical Tests
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


def anderson_darling_test(
    reference: list[float],
    production: list[float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Perform the two-sample Anderson-Darling test for distribution equality.

    Uses scipy.stats.anderson_ksamp for the k-sample variant, which accepts
    exactly two samples for a two-sample comparison.

    Args:
        reference: Reference distribution samples.
        production: Production distribution samples.
        threshold: Significance level p-value threshold. Drift is detected
            when p_value < threshold.

    Returns:
        Dictionary with statistic, p_value, drift_detected flag, and metadata.
    """
    try:
        from scipy.stats import anderson_ksamp  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Anderson-Darling test. Install with: pip install scipy"
        ) from exc

    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    result = anderson_ksamp([ref_arr, prod_arr])
    statistic = float(result.statistic)
    # anderson_ksamp provides significance_level, not p-value directly
    # significance_level is the minimum significance level where H0 would be rejected
    significance_level = float(result.significance_level) / 100.0  # convert % to fraction
    drift_detected = significance_level < threshold

    logger.debug(
        "anderson_darling_test",
        statistic=statistic,
        significance_level=significance_level,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "anderson_darling",
        "statistic": statistic,
        "p_value": significance_level,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_reference": len(ref_arr),
        "n_production": len(prod_arr),
    }
