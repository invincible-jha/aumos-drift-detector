"""SHAP/LIME-based feature importance adapter for drift explanation.

Provides root-cause analysis for detected drift by identifying which features
contributed most to the distributional shift, using SHAP values and LIME
explanations to attribute drift scores to individual feature dimensions.

Example:
    >>> import numpy as np
    >>> explainer = DriftFeatureImportance()
    >>> reference = {"age": np.random.normal(35, 10, 500), "income": np.random.normal(50000, 15000, 500)}
    >>> production = {"age": np.random.normal(42, 10, 200), "income": np.random.normal(50000, 15000, 200)}
    >>> ranking = explainer.rank_features_by_drift(reference, production, drift_scores={"age": 0.82, "income": 0.03})
    >>> ranking[0].feature_name
    'age'
"""

from __future__ import annotations

import math
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureImportanceResult:
    """Result of feature-level drift importance analysis.

    Attributes:
        feature_name: Name of the feature.
        drift_score: Raw drift score (PSI, KS statistic, chi2 p-value, etc.).
        importance_rank: Ordinal rank (1 = most important contributor to drift).
        shap_mean_abs: Mean absolute SHAP value across analysed samples.
        lime_weight: LIME linear coefficient for this feature.
        contribution_pct: Fraction of total drift attributable to this feature (0–1).
        reference_mean: Mean of feature in reference distribution.
        production_mean: Mean of feature in production distribution.
        mean_shift: Absolute shift in mean (production_mean - reference_mean).
        reference_std: Standard deviation in reference distribution.
        production_std: Standard deviation in production distribution.
    """

    feature_name: str
    drift_score: float
    importance_rank: int
    shap_mean_abs: float
    lime_weight: float
    contribution_pct: float
    reference_mean: float
    production_mean: float
    mean_shift: float
    reference_std: float
    production_std: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSONB storage or API response.

        Returns:
            Dict with all importance fields.
        """
        return {
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "importance_rank": self.importance_rank,
            "shap_mean_abs": self.shap_mean_abs,
            "lime_weight": self.lime_weight,
            "contribution_pct": self.contribution_pct,
            "reference_mean": self.reference_mean,
            "production_mean": self.production_mean,
            "mean_shift": self.mean_shift,
            "reference_std": self.reference_std,
            "production_std": self.production_std,
        }


@dataclass
class WaterfallChartData:
    """Data for a SHAP waterfall chart visualisation.

    Attributes:
        base_value: Starting expected value (mean drift score across all features).
        feature_contributions: List of (feature_name, signed_contribution) tuples.
        final_value: Total drift score (base_value + sum of contributions).
    """

    base_value: float
    feature_contributions: list[tuple[str, float]]
    final_value: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict.

        Returns:
            Dict suitable for JSON serialisation.
        """
        return {
            "base_value": self.base_value,
            "feature_contributions": [
                {"feature": name, "contribution": value}
                for name, value in self.feature_contributions
            ],
            "final_value": self.final_value,
        }


@dataclass
class HistoricalImportanceEntry:
    """One timestamped snapshot of feature importance state.

    Attributes:
        recorded_at: UTC timestamp when this entry was captured.
        monitor_id: UUID of the drift monitor that produced this snapshot.
        feature_rankings: Ordered list of feature importance results.
    """

    recorded_at: datetime
    monitor_id: uuid.UUID
    feature_rankings: list[FeatureImportanceResult]


class DriftFeatureImportance:
    """SHAP/LIME-based drift attribution and feature importance analyser.

    Computes which features contributed most to detected drift by:
    - Approximating SHAP values from distribution differences
    - Computing LIME linear weights from local neighbourhoods
    - Ranking features by their drift contribution percentage
    - Generating waterfall and beeswarm visualisation data
    - Tracking historical importance over time

    This adapter uses scipy and numpy for pure-Python SHAP approximation
    without requiring a training model object, making it distribution-based
    rather than model-based. For model-based SHAP, pass predictions alongside
    feature arrays.
    """

    def __init__(self, history_max_entries: int = 1000) -> None:
        """Initialise the feature importance adapter.

        Args:
            history_max_entries: Maximum number of historical snapshots to retain
                in memory before oldest entries are evicted.
        """
        self._history_max_entries = history_max_entries
        self._history: list[HistoricalImportanceEntry] = []

    def compute_shap_approximation(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Approximate SHAP values using Shapley-style marginal contributions.

        For distribution drift, the SHAP approximation treats each feature's
        contribution as the marginal change in KL divergence when that feature
        is included vs excluded. We approximate using standardised mean shift
        weighted by distributional overlap loss.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.

        Returns:
            Dict of feature_name to mean absolute SHAP approximation value.

        Raises:
            ValueError: If reference and production have mismatched features.
        """
        if set(reference.keys()) != set(production.keys()):
            raise ValueError(
                "Reference and production must have identical feature sets. "
                f"Reference: {sorted(reference.keys())}, "
                f"Production: {sorted(production.keys())}"
            )

        shap_values: dict[str, float] = {}
        for feature_name, ref_array in reference.items():
            prod_array = production[feature_name]
            ref_clean = ref_array[np.isfinite(ref_array)]
            prod_clean = prod_array[np.isfinite(prod_array)]

            if ref_clean.size == 0 or prod_clean.size == 0:
                shap_values[feature_name] = 0.0
                continue

            ref_mean = float(np.mean(ref_clean))
            ref_std = float(np.std(ref_clean)) or 1.0
            prod_mean = float(np.mean(prod_clean))
            prod_std = float(np.std(prod_clean)) or 1.0

            # Standardised mean shift (effect size, Cohen's d approximation)
            pooled_std = math.sqrt((ref_std**2 + prod_std**2) / 2.0)
            mean_shift_effect = abs(prod_mean - ref_mean) / pooled_std if pooled_std > 0 else 0.0

            # Variance ratio contribution (log-ratio of variances)
            variance_ratio = prod_std / ref_std
            variance_effect = abs(math.log(variance_ratio)) if variance_ratio > 0 else 0.0

            # Combined SHAP approximation
            shap_values[feature_name] = (mean_shift_effect + 0.5 * variance_effect)

        logger.debug(
            "SHAP approximation computed",
            feature_count=len(shap_values),
            top_feature=max(shap_values, key=lambda k: shap_values[k]) if shap_values else None,
        )
        return shap_values

    def compute_lime_weights(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        neighbourhood_size: int = 50,
    ) -> dict[str, float]:
        """Compute LIME linear weights for each feature's drift contribution.

        Implements a simplified LIME that fits a linear model in a local
        neighbourhood around the production distribution centroid, weighted
        by proximity to the reference distribution.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.
            neighbourhood_size: Number of samples to use in the local
                neighbourhood for linear fitting. Default 50.

        Returns:
            Dict of feature_name to LIME linear weight.
        """
        lime_weights: dict[str, float] = {}
        for feature_name, ref_array in reference.items():
            prod_array = production.get(feature_name, np.array([]))
            ref_clean = ref_array[np.isfinite(ref_array)]
            prod_clean = prod_array[np.isfinite(prod_array)]

            if ref_clean.size < 2 or prod_clean.size < 2:
                lime_weights[feature_name] = 0.0
                continue

            # Sample neighbourhood around production centroid
            sample_count = min(neighbourhood_size, prod_clean.size)
            rng = np.random.default_rng(seed=42)
            neighbourhood = rng.choice(prod_clean, size=sample_count, replace=False)

            # Compute kernel weights (Gaussian kernel centred at production mean)
            prod_mean = float(np.mean(prod_clean))
            prod_std = float(np.std(prod_clean)) or 1.0
            kernel_width = prod_std
            kernel_weights = np.exp(-0.5 * ((neighbourhood - prod_mean) / kernel_width) ** 2)

            # Weighted mean of neighbourhood vs reference mean — the LIME coefficient
            ref_mean = float(np.mean(ref_clean))
            weighted_mean = float(np.average(neighbourhood, weights=kernel_weights))
            lime_weights[feature_name] = abs(weighted_mean - ref_mean) / (prod_std or 1.0)

        return lime_weights

    def rank_features_by_drift(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        drift_scores: dict[str, float],
    ) -> list[FeatureImportanceResult]:
        """Rank features by their contribution to drift, from most to least.

        Combines SHAP approximation, LIME weights, and raw drift scores to
        produce a ranked list with contribution percentages.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.
            drift_scores: Pre-computed drift scores per feature (e.g., KS statistic
                or PSI value). Higher = more drift.

        Returns:
            List of FeatureImportanceResult ordered by importance_rank (1 = highest).
        """
        shap_values = self.compute_shap_approximation(reference, production)
        lime_weights = self.compute_lime_weights(reference, production)

        total_drift = sum(drift_scores.values()) or 1.0

        results: list[FeatureImportanceResult] = []
        for feature_name, drift_score in drift_scores.items():
            ref_array = reference.get(feature_name, np.array([]))
            prod_array = production.get(feature_name, np.array([]))
            ref_clean = ref_array[np.isfinite(ref_array)] if ref_array.size > 0 else np.array([0.0])
            prod_clean = prod_array[np.isfinite(prod_array)] if prod_array.size > 0 else np.array([0.0])

            ref_mean = float(np.mean(ref_clean))
            prod_mean = float(np.mean(prod_clean))
            ref_std = float(np.std(ref_clean))
            prod_std = float(np.std(prod_clean))

            results.append(
                FeatureImportanceResult(
                    feature_name=feature_name,
                    drift_score=drift_score,
                    importance_rank=0,  # assigned after sorting
                    shap_mean_abs=shap_values.get(feature_name, 0.0),
                    lime_weight=lime_weights.get(feature_name, 0.0),
                    contribution_pct=drift_score / total_drift,
                    reference_mean=ref_mean,
                    production_mean=prod_mean,
                    mean_shift=prod_mean - ref_mean,
                    reference_std=ref_std,
                    production_std=prod_std,
                )
            )

        # Sort by combined score: SHAP + drift_score (normalised)
        max_shap = max((r.shap_mean_abs for r in results), default=1.0) or 1.0
        max_drift = max((r.drift_score for r in results), default=1.0) or 1.0
        results.sort(
            key=lambda r: (r.shap_mean_abs / max_shap + r.drift_score / max_drift),
            reverse=True,
        )
        for rank, result in enumerate(results, start=1):
            object.__setattr__(result, "importance_rank", rank) if hasattr(result, "__dataclass_fields__") else None
            results[rank - 1] = FeatureImportanceResult(
                feature_name=result.feature_name,
                drift_score=result.drift_score,
                importance_rank=rank,
                shap_mean_abs=result.shap_mean_abs,
                lime_weight=result.lime_weight,
                contribution_pct=result.contribution_pct,
                reference_mean=result.reference_mean,
                production_mean=result.production_mean,
                mean_shift=result.mean_shift,
                reference_std=result.reference_std,
                production_std=result.production_std,
            )

        logger.info(
            "Feature drift ranking computed",
            feature_count=len(results),
            top_feature=results[0].feature_name if results else None,
            top_contribution_pct=results[0].contribution_pct if results else None,
        )
        return results

    def generate_waterfall_data(
        self,
        feature_rankings: list[FeatureImportanceResult],
    ) -> WaterfallChartData:
        """Generate SHAP waterfall chart data for visualisation.

        The waterfall chart shows how each feature's contribution builds up
        from a baseline (mean drift) to the final aggregate drift score.

        Args:
            feature_rankings: Ranked feature importance results (from rank_features_by_drift).

        Returns:
            WaterfallChartData with base value, per-feature contributions, and final value.
        """
        total_drift = sum(r.drift_score for r in feature_rankings)
        mean_drift = total_drift / len(feature_rankings) if feature_rankings else 0.0
        base_value = mean_drift

        contributions: list[tuple[str, float]] = []
        running_total = base_value
        for result in sorted(feature_rankings, key=lambda r: r.importance_rank):
            contribution = result.drift_score - mean_drift
            contributions.append((result.feature_name, contribution))
            running_total += contribution

        return WaterfallChartData(
            base_value=base_value,
            feature_contributions=contributions,
            final_value=running_total,
        )

    def generate_beeswarm_data(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
        feature_rankings: list[FeatureImportanceResult],
        samples_per_feature: int = 100,
    ) -> dict[str, list[dict[str, float]]]:
        """Generate beeswarm plot data (sample-level SHAP values per feature).

        Each sample gets a SHAP value approximated by its deviation from the
        reference distribution mean, weighted by the feature's importance rank.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.
            feature_rankings: Ranked feature importance results.
            samples_per_feature: Number of production samples to include per feature.

        Returns:
            Dict of feature_name to list of {value, shap_value} dicts for plotting.
        """
        beeswarm_data: dict[str, list[dict[str, float]]] = {}
        rank_map = {r.feature_name: r for r in feature_rankings}

        for feature_name, prod_array in production.items():
            result = rank_map.get(feature_name)
            if result is None:
                continue

            prod_clean = prod_array[np.isfinite(prod_array)]
            if prod_clean.size == 0:
                beeswarm_data[feature_name] = []
                continue

            sample_count = min(samples_per_feature, prod_clean.size)
            rng = np.random.default_rng(seed=feature_name.__hash__() % (2**31))
            sampled = rng.choice(prod_clean, size=sample_count, replace=False)

            ref_mean = result.reference_mean
            ref_std = result.reference_std or 1.0
            importance_weight = result.shap_mean_abs

            points: list[dict[str, float]] = []
            for sample_value in sampled:
                # Approximate per-sample SHAP: deviation from reference normalised by std
                per_sample_shap = importance_weight * (sample_value - ref_mean) / ref_std
                points.append({"value": float(sample_value), "shap_value": float(per_sample_shap)})

            beeswarm_data[feature_name] = points

        return beeswarm_data

    def record_historical_importance(
        self,
        monitor_id: uuid.UUID,
        feature_rankings: list[FeatureImportanceResult],
    ) -> None:
        """Append a timestamped feature importance snapshot to history.

        Evicts oldest entries when the history buffer is full.

        Args:
            monitor_id: UUID of the drift monitor that produced this snapshot.
            feature_rankings: Ranked feature importance results to record.
        """
        entry = HistoricalImportanceEntry(
            recorded_at=datetime.now(tz=timezone.utc),
            monitor_id=monitor_id,
            feature_rankings=list(feature_rankings),
        )
        self._history.append(entry)
        if len(self._history) > self._history_max_entries:
            self._history.pop(0)
        logger.debug(
            "Historical importance entry recorded",
            monitor_id=str(monitor_id),
            history_size=len(self._history),
        )

    def get_historical_importance(
        self,
        monitor_id: uuid.UUID,
        limit: int = 50,
    ) -> list[HistoricalImportanceEntry]:
        """Retrieve historical importance snapshots for a monitor.

        Args:
            monitor_id: UUID of the drift monitor to filter by.
            limit: Maximum number of most-recent entries to return.

        Returns:
            List of HistoricalImportanceEntry ordered newest-first.
        """
        matching = [e for e in self._history if e.monitor_id == monitor_id]
        return list(reversed(matching[-limit:]))

    def analyse_cross_feature_interactions(
        self,
        reference: dict[str, np.ndarray],
        production: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Pearson correlation shift between reference and production.

        Identifies feature pairs where the correlation structure changed significantly,
        suggesting interaction drift rather than marginal drift.

        Args:
            reference: Dict of feature_name to reference samples array.
            production: Dict of feature_name to production samples array.

        Returns:
            Dict of feature_a to Dict of feature_b to correlation_delta.
            Positive delta means correlation increased in production.
        """
        feature_names = sorted(set(reference.keys()) & set(production.keys()))
        interactions: dict[str, dict[str, float]] = defaultdict(dict)

        for i, feature_a in enumerate(feature_names):
            for feature_b in feature_names[i + 1:]:
                ref_a = reference[feature_a][np.isfinite(reference[feature_a])]
                ref_b = reference[feature_b][np.isfinite(reference[feature_b])]
                prod_a = production[feature_a][np.isfinite(production[feature_a])]
                prod_b = production[feature_b][np.isfinite(production[feature_b])]

                min_ref = min(len(ref_a), len(ref_b))
                min_prod = min(len(prod_a), len(prod_b))

                if min_ref < 3 or min_prod < 3:
                    continue

                ref_corr = float(np.corrcoef(ref_a[:min_ref], ref_b[:min_ref])[0, 1])
                prod_corr = float(np.corrcoef(prod_a[:min_prod], prod_b[:min_prod])[0, 1])

                if not (math.isfinite(ref_corr) and math.isfinite(prod_corr)):
                    continue

                delta = prod_corr - ref_corr
                interactions[feature_a][feature_b] = delta
                interactions[feature_b][feature_a] = delta

        return dict(interactions)
