"""Direct Loss Estimation (DLE) for aumos-drift-detector.

Trains a gradient-boosted error predictor on reference data with ground-truth
labels, then applies it to unlabelled production data to directly estimate
the expected model loss without requiring labels.

GAP-165: CBPE/DLE Performance Estimation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DLEEstimator:
    """Direct Loss Estimation using a gradient-boosted error predictor.

    Trains a secondary model to predict per-sample loss from raw features,
    then uses it to estimate aggregate loss on unlabelled production windows.

    Args:
        loss_function: Loss to estimate — 'log_loss', 'absolute_error', or 'squared_error'.
        n_estimators: Number of gradient boosting estimators.
        max_depth: Maximum tree depth for the error predictor.
    """

    def __init__(
        self,
        loss_function: str = "log_loss",
        n_estimators: int = 100,
        max_depth: int = 3,
    ) -> None:
        """Initialise the DLE estimator.

        Args:
            loss_function: Loss metric to estimate.
            n_estimators: Number of boosting rounds.
            max_depth: Tree depth limit for the error predictor.
        """
        self._loss_function = loss_function
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._error_predictor: object | None = None
        self._fitted = False

    def fit(
        self,
        reference_features: list[list[float]],
        reference_labels: list[int],
        reference_predictions: list[list[float]],
    ) -> "DLEEstimator":
        """Train the error predictor on reference data.

        Computes per-sample loss on the reference set, then trains a
        gradient-boosted model to predict that loss from raw features.

        Args:
            reference_features: Input feature matrix, shape (n_samples, n_features).
            reference_labels: Ground-truth labels, shape (n_samples,).
            reference_predictions: Model output probabilities, shape (n_samples, n_classes).

        Returns:
            Self, for method chaining.
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for DLE. Install with: pip install scikit-learn"
            ) from exc

        features = np.asarray(reference_features)
        labels = np.asarray(reference_labels)
        predictions = np.asarray(reference_predictions)

        # Compute per-sample loss
        per_sample_loss = self._compute_loss(predictions, labels)

        error_predictor = GradientBoostingRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=42,
        )
        error_predictor.fit(features, per_sample_loss)
        self._error_predictor = error_predictor
        self._fitted = True
        logger.info(
            "dle_fitted",
            loss_function=self._loss_function,
            n_samples=len(labels),
        )
        return self

    def estimate(
        self,
        production_features: list[list[float]],
    ) -> dict[str, Any]:
        """Estimate expected loss on unlabelled production data.

        Args:
            production_features: Input features from production, shape (n_samples, n_features).

        Returns:
            Dictionary with estimated loss value and variance.
        """
        if not self._fitted or self._error_predictor is None:
            raise RuntimeError("Call fit() before estimate()")

        features = np.asarray(production_features)
        predicted_losses = self._error_predictor.predict(features)  # type: ignore[union-attr]
        estimated_loss = float(predicted_losses.mean())
        loss_std = float(predicted_losses.std())

        logger.info(
            "dle_estimated",
            loss_function=self._loss_function,
            estimated_loss=estimated_loss,
            n_samples=len(features),
        )
        return {
            "loss_function": self._loss_function,
            "estimated_loss": estimated_loss,
            "loss_std": loss_std,
            "n_samples": len(features),
            "method": "dle",
        }

    def _compute_loss(self, predictions: "np.ndarray[Any, Any]", labels: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
        """Compute per-sample loss from predictions and labels.

        Args:
            predictions: Predicted probabilities, shape (n_samples, n_classes).
            labels: Ground-truth labels, shape (n_samples,).

        Returns:
            Per-sample loss array.
        """
        if self._loss_function == "log_loss":
            # Clip to avoid log(0)
            n_samples = len(labels)
            clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
            sample_losses = np.zeros(n_samples)
            for i in range(n_samples):
                label = int(labels[i])
                sample_losses[i] = -np.log(clipped[i, label])
            return sample_losses
        elif self._loss_function == "absolute_error":
            predicted_class = predictions.argmax(axis=1)
            return np.abs(predicted_class - labels).astype(float)
        else:
            # squared_error
            predicted_class = predictions.argmax(axis=1)
            return ((predicted_class - labels) ** 2).astype(float)
