"""Multivariate drift detection for aumos-drift-detector.

Implements two complementary multivariate drift methods:
  1. PCA reconstruction error — detects geometric distribution shift
  2. Classifier two-sample test (C2ST) — trains a binary classifier to
     distinguish reference from production samples

GAP-169: Multivariate Drift Detection
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


def pca_reconstruction_drift(
    reference: list[list[float]],
    production: list[list[float]],
    threshold: float = 0.15,
    n_components: int = 10,
) -> dict[str, Any]:
    """Detect multivariate drift via PCA reconstruction error.

    Fits a PCA on the reference data, then measures the reconstruction error
    for production data. High reconstruction error indicates the production
    distribution has moved outside the reference manifold.

    Args:
        reference: Reference feature matrix, shape (n_ref, d).
        production: Production feature matrix, shape (n_prod, d).
        threshold: Drift detection threshold for normalized reconstruction error.
        n_components: Number of PCA components to retain.

    Returns:
        Dictionary with reconstruction_error, drift_detected, and metadata.
    """
    try:
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for PCA drift. Install with: pip install scikit-learn"
        ) from exc

    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_arr)
    prod_scaled = scaler.transform(prod_arr)

    n_comp = min(n_components, ref_scaled.shape[1], len(ref_scaled) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(ref_scaled)

    # Reconstruction error: project then reconstruct
    ref_reconstructed = pca.inverse_transform(pca.transform(ref_scaled))
    prod_reconstructed = pca.inverse_transform(pca.transform(prod_scaled))

    ref_error = float(np.mean((ref_scaled - ref_reconstructed) ** 2))
    prod_error = float(np.mean((prod_scaled - prod_reconstructed) ** 2))

    # Normalize by reference error to get a relative score
    normalized_error = (prod_error - ref_error) / (ref_error + 1e-10)
    drift_detected = normalized_error > threshold

    logger.debug(
        "pca_reconstruction_drift",
        ref_error=ref_error,
        prod_error=prod_error,
        normalized_error=normalized_error,
        drift_detected=drift_detected,
    )
    return {
        "test": "pca_reconstruction",
        "reconstruction_error": prod_error,
        "reference_reconstruction_error": ref_error,
        "normalized_error": normalized_error,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "n_components": n_comp,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def classifier_two_sample_test(
    reference: list[list[float]],
    production: list[list[float]],
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Classifier two-sample test (C2ST) for multivariate drift detection.

    Trains a gradient boosting classifier to distinguish reference (label=0)
    from production (label=1) samples. An AUROC significantly above 0.5
    indicates the distributions are different (easy to discriminate).

    Args:
        reference: Reference feature matrix.
        production: Production feature matrix.
        threshold: AUROC threshold above which drift is detected (default 0.6).

    Returns:
        Dictionary with auroc, drift_detected, and feature importances.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
        from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
        from sklearn.model_selection import cross_val_predict  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for C2ST. Install with: pip install scikit-learn"
        ) from exc

    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    X = np.vstack([ref_arr, prod_arr])
    y = np.concatenate([np.zeros(len(ref_arr)), np.ones(len(prod_arr))])

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auroc = float(roc_auc_score(y, proba))
    drift_detected = auroc > threshold

    # Fit on full data for feature importances
    clf.fit(X, y)
    importances = clf.feature_importances_.tolist()

    logger.debug(
        "classifier_two_sample_test",
        auroc=auroc,
        threshold=threshold,
        drift_detected=drift_detected,
    )
    return {
        "test": "classifier_two_sample",
        "auroc": auroc,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "feature_importances": importances,
        "n_reference": len(ref_arr),
        "n_production": len(prod_arr),
    }
