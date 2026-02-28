"""Wasserstein-1 distance (Earth Mover's Distance) drift test.

Measures the minimum cost to transform one distribution into another.
Normalized by IQR to make it scale-invariant across features.

GAP-166: Extended Statistical Tests
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


def wasserstein_test(
    reference: list[float],
    production: list[float],
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute the IQR-normalized Wasserstein-1 distance between two samples.

    Uses scipy.stats.wasserstein_distance for the underlying computation.
    Normalizes by the reference IQR to make the score comparable across
    features with different scales.

    Args:
        reference: Reference (baseline) distribution samples.
        production: Production distribution samples.
        threshold: Drift detection threshold for the normalized score.

    Returns:
        Dictionary with score, drift_detected flag, and metadata.
    """
    try:
        from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Wasserstein test. Install with: pip install scipy"
        ) from exc

    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    raw_distance = float(wasserstein_distance(ref_arr, prod_arr))

    # IQR normalization: avoids division by zero by falling back to std
    q75, q25 = np.percentile(ref_arr, [75, 25])
    iqr = float(q75 - q25)
    if iqr < 1e-10:
        iqr = float(ref_arr.std()) or 1.0

    normalized_score = raw_distance / iqr
    drift_detected = normalized_score > threshold

    logger.debug(
        "wasserstein_test",
        raw_distance=raw_distance,
        normalized_score=normalized_score,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "wasserstein",
        "score": normalized_score,
        "raw_distance": raw_distance,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_reference": len(ref_arr),
        "n_production": len(prod_arr),
    }
