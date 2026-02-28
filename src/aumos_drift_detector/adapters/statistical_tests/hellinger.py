"""Hellinger distance drift test for aumos-drift-detector.

Hellinger distance is a symmetric measure of the difference between two
probability distributions, bounded in [0, 1]. Closely related to Bhattacharyya
coefficient, and more stable than KL divergence for zero-probability bins.

GAP-166: Extended Statistical Tests
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_BINS = 20


def hellinger_test(
    reference: list[float],
    production: list[float],
    threshold: float = 0.1,
    n_bins: int = _DEFAULT_BINS,
) -> dict[str, Any]:
    """Compute the Hellinger distance between two continuous distributions.

    Bins both samples, normalizes to probability distributions, then computes
    the Hellinger distance H(P, Q) = (1/√2) * ||√P - √Q||₂.

    Score is in [0, 1]: 0 = identical, 1 = disjoint distributions.

    Args:
        reference: Reference distribution samples.
        production: Production distribution samples.
        threshold: Drift detection threshold.
        n_bins: Number of histogram bins.

    Returns:
        Dictionary with score, drift_detected flag, and metadata.
    """
    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    combined_min = min(ref_arr.min(), prod_arr.min())
    combined_max = max(ref_arr.max(), prod_arr.max())
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    ref_hist, _ = np.histogram(ref_arr, bins=bin_edges, density=False)
    prod_hist, _ = np.histogram(prod_arr, bins=bin_edges, density=False)

    # Add small smoothing to avoid sqrt(0) edge cases
    ref_dist = (ref_hist.astype(float) + 1e-10)
    prod_dist = (prod_hist.astype(float) + 1e-10)
    ref_dist /= ref_dist.sum()
    prod_dist /= prod_dist.sum()

    hellinger = float((1.0 / np.sqrt(2.0)) * np.sqrt(((np.sqrt(ref_dist) - np.sqrt(prod_dist)) ** 2).sum()))
    drift_detected = hellinger > threshold

    logger.debug(
        "hellinger_test",
        score=hellinger,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "hellinger",
        "score": hellinger,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_bins": n_bins,
        "n_reference": len(ref_arr),
        "n_production": len(prod_arr),
    }
