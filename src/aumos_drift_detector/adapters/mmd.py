"""Maximum Mean Discrepancy (MMD) embedding drift detector.

MMD with RBF kernel measures the distance between distributions in a
reproducing kernel Hilbert space. Particularly effective for detecting
drift in high-dimensional embeddings (e.g. sentence embeddings, image features).

GAP-167: MMD Embedding Drift
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


def _rbf_kernel(
    X: "np.ndarray[Any, Any]",
    Y: "np.ndarray[Any, Any]",
    bandwidth: float,
) -> "np.ndarray[Any, Any]":
    """Compute the RBF (Gaussian) kernel matrix between X and Y.

    Args:
        X: Matrix of shape (n, d).
        Y: Matrix of shape (m, d).
        bandwidth: RBF bandwidth parameter sigma^2.

    Returns:
        Kernel matrix of shape (n, m).
    """
    diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
    sq_distances = (diff ** 2).sum(axis=-1)  # (n, m)
    return np.exp(-sq_distances / (2.0 * bandwidth))


def _median_bandwidth(
    X: "np.ndarray[Any, Any]",
    Y: "np.ndarray[Any, Any]",
) -> float:
    """Estimate bandwidth using the median heuristic over combined pairwise distances.

    Args:
        X: First sample matrix.
        Y: Second sample matrix.

    Returns:
        Estimated bandwidth (sigma^2).
    """
    combined = np.vstack([X, Y])
    n = len(combined)
    # Sample up to 500 points to keep cost manageable
    if n > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=500, replace=False)
        combined = combined[idx]
    diffs = combined[:, None, :] - combined[None, :, :]
    sq_dists = (diffs ** 2).sum(axis=-1)
    median_sq = float(np.median(sq_dists[sq_dists > 0]))
    return max(median_sq, 1e-6)


def mmd_test(
    reference_embeddings: list[list[float]],
    production_embeddings: list[list[float]],
    threshold: float = 0.05,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    """Compute the unbiased MMD^2 statistic between two embedding sets.

    Uses the unbiased U-statistic estimator for MMD^2 with an RBF kernel.
    Bandwidth is estimated via median heuristic if not provided.

    Args:
        reference_embeddings: Reference embedding vectors, shape (n, d).
        production_embeddings: Production embedding vectors, shape (m, d).
        threshold: Drift detection threshold for MMD^2 score.
        bandwidth: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        Dictionary with mmd_squared score, drift_detected flag, and metadata.
    """
    X = np.asarray(reference_embeddings, dtype=float)
    Y = np.asarray(production_embeddings, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    bw = bandwidth if bandwidth is not None else _median_bandwidth(X, Y)

    n = len(X)
    m = len(Y)

    # Unbiased MMD^2 estimator
    K_XX = _rbf_kernel(X, X, bw)
    K_YY = _rbf_kernel(Y, Y, bw)
    K_XY = _rbf_kernel(X, Y, bw)

    # Remove diagonal for unbiased estimate
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd_sq = (K_XX.sum() / (n * (n - 1))) + (K_YY.sum() / (m * (m - 1))) - 2.0 * K_XY.mean()
    mmd_sq = float(mmd_sq)
    drift_detected = mmd_sq > threshold

    logger.debug(
        "mmd_test",
        mmd_squared=mmd_sq,
        bandwidth=bw,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "mmd",
        "mmd_squared": mmd_sq,
        "bandwidth": bw,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_reference": n,
        "n_production": m,
        "embedding_dim": X.shape[1],
    }
