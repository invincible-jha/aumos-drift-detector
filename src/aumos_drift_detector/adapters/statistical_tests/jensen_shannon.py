"""Jensen-Shannon divergence drift test for aumos-drift-detector.

Jensen-Shannon divergence is a symmetric, smoothed version of KL divergence
bounded in [0, 1] (when using base-2 logarithm and dividing by log(2)).
Suitable for comparing discrete or binned continuous distributions.

GAP-166: Extended Statistical Tests
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_BINS = 20


def jensen_shannon_test(
    reference: list[float],
    production: list[float],
    threshold: float = 0.1,
    n_bins: int = _DEFAULT_BINS,
) -> dict[str, Any]:
    """Compute the Jensen-Shannon divergence between two continuous distributions.

    Bins both samples into a shared histogram, then computes JS divergence.
    The score is in [0, 1]: 0 = identical distributions, 1 = fully disjoint.

    Args:
        reference: Reference distribution samples.
        production: Production distribution samples.
        threshold: Drift detection threshold for JS divergence.
        n_bins: Number of histogram bins for discretization.

    Returns:
        Dictionary with score, drift_detected flag, and metadata.
    """
    try:
        from scipy.spatial.distance import jensenshannon  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Jensen-Shannon test. Install with: pip install scipy"
        ) from exc

    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    # Build shared bin edges from the combined range
    combined_min = min(ref_arr.min(), prod_arr.min())
    combined_max = max(ref_arr.max(), prod_arr.max())
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    ref_hist, _ = np.histogram(ref_arr, bins=bin_edges, density=False)
    prod_hist, _ = np.histogram(prod_arr, bins=bin_edges, density=False)

    # Normalize to probability distributions
    ref_dist = ref_hist.astype(float) + 1e-10  # smoothing
    prod_dist = prod_hist.astype(float) + 1e-10
    ref_dist /= ref_dist.sum()
    prod_dist /= prod_dist.sum()

    js_score = float(jensenshannon(ref_dist, prod_dist))
    drift_detected = js_score > threshold

    logger.debug(
        "jensen_shannon_test",
        score=js_score,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "jensen_shannon",
        "score": js_score,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_bins": n_bins,
        "n_reference": len(ref_arr),
        "n_production": len(prod_arr),
    }
