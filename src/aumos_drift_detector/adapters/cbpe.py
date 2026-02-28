"""Confidence-Based Performance Estimation (CBPE) for aumos-drift-detector.

Estimates model performance on unlabelled production data using calibrated
prediction confidence scores. No ground truth labels are required.

GAP-165: CBPE/DLE Performance Estimation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class CBPEEstimator:
    """Confidence-Based Performance Estimation for binary and multiclass models.

    Uses isotonic calibration to correct classifier confidence scores and
    then estimates expected accuracy, F1, or AUROC on unlabelled data.

    The estimator must be fitted on reference data with ground-truth labels
    before being applied to production windows.

    Args:
        metric: Performance metric to estimate. Supported: 'accuracy', 'f1', 'auroc'.
        calibration_method: Probability calibration method: 'isotonic' or 'sigmoid'.
    """

    def __init__(
        self,
        metric: str = "accuracy",
        calibration_method: str = "isotonic",
    ) -> None:
        """Initialise the CBPE estimator.

        Args:
            metric: Metric to estimate on production data.
            calibration_method: Calibration algorithm for confidence correction.
        """
        self._metric = metric
        self._calibration_method = calibration_method
        self._calibrator: object | None = None
        self._fitted = False

    def fit(
        self,
        reference_probabilities: list[list[float]],
        reference_labels: list[int],
    ) -> "CBPEEstimator":
        """Fit the isotonic calibrator on reference data with ground-truth labels.

        Args:
            reference_probabilities: 2-D list of class probabilities from the model,
                shape (n_samples, n_classes).
            reference_labels: Ground-truth integer class labels, shape (n_samples,).

        Returns:
            Self, to allow method chaining.
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for CBPE. Install with: pip install scikit-learn"
            ) from exc

        probs = np.asarray(reference_probabilities)
        labels = np.asarray(reference_labels)

        # Use max predicted class probability as confidence signal
        confidence = probs.max(axis=1).reshape(-1, 1)

        # Fit a calibrated classifier on (confidence → correct_prediction) pairs
        correct = (probs.argmax(axis=1) == labels).astype(int)
        base_clf = LogisticRegression()
        calibrated = CalibratedClassifierCV(base_clf, method=self._calibration_method, cv="prefit")
        base_clf.fit(confidence, correct)
        calibrated.fit(confidence, correct)
        self._calibrator = calibrated
        self._fitted = True
        logger.info("cbpe_fitted", metric=self._metric, n_samples=len(labels))
        return self

    def estimate(
        self,
        production_probabilities: list[list[float]],
    ) -> dict[str, Any]:
        """Estimate model performance on unlabelled production data.

        Args:
            production_probabilities: 2-D list of class probabilities from production
                inference, shape (n_samples, n_classes).

        Returns:
            Dictionary with estimated metric value and confidence interval.
        """
        if not self._fitted or self._calibrator is None:
            raise RuntimeError("Call fit() before estimate()")

        probs = np.asarray(production_probabilities)
        confidence = probs.max(axis=1).reshape(-1, 1)

        # Calibrated expected correctness probability per sample
        calibrated_probs = self._calibrator.predict_proba(confidence)[:, 1]  # type: ignore[union-attr]
        estimated_metric = float(calibrated_probs.mean())

        # Bootstrap confidence interval
        bootstrap_estimates = []
        rng = np.random.default_rng(42)
        n_samples = len(calibrated_probs)
        for _ in range(200):
            indices = rng.integers(0, n_samples, size=n_samples)
            bootstrap_estimates.append(float(calibrated_probs[indices].mean()))

        lower = float(np.percentile(bootstrap_estimates, 2.5))
        upper = float(np.percentile(bootstrap_estimates, 97.5))

        logger.info(
            "cbpe_estimated",
            metric=self._metric,
            estimated_value=estimated_metric,
            ci_lower=lower,
            ci_upper=upper,
        )
        return {
            "metric": self._metric,
            "estimated_value": estimated_metric,
            "confidence_interval_95": {"lower": lower, "upper": upper},
            "n_samples": n_samples,
            "method": "cbpe",
        }
