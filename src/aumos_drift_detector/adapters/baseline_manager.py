"""Baseline (reference distribution) management adapter.

Captures, versions, and manages reference distributions for drift detection.
Supports multi-window baselines (7d, 30d, 90d), statistical summary storage,
baseline rotation policies, and import/export for reproducibility.

Example:
    >>> import numpy as np
    >>> manager = BaselineManager()
    >>> data = {"age": np.random.normal(35, 10, 1000), "income": np.random.normal(50000, 15000, 1000)}
    >>> baseline = manager.capture_baseline(model_id=uuid4(), model_version="v1.2", feature_data=data)
    >>> baseline.version
    'v1.2'
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Standard multi-window sizes in days
STANDARD_WINDOW_DAYS = (7, 30, 90)


@dataclass
class FeatureStatistics:
    """Statistical summary of a single feature's distribution.

    Attributes:
        feature_name: Name of the feature.
        count: Number of non-null observations.
        mean: Arithmetic mean.
        std: Standard deviation.
        min_value: Minimum value.
        max_value: Maximum value.
        percentile_25: 25th percentile.
        median: 50th percentile.
        percentile_75: 75th percentile.
        percentile_95: 95th percentile.
        percentile_99: 99th percentile.
        null_fraction: Fraction of null/NaN values in the original array.
        histogram_counts: Bin counts for the distribution histogram.
        histogram_edges: Bin edge values for the histogram.
        unique_count: Number of unique values (useful for categorical features).
    """

    feature_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    percentile_25: float
    median: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    null_fraction: float
    histogram_counts: list[int]
    histogram_edges: list[float]
    unique_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all feature statistics.
        """
        return {
            "feature_name": self.feature_name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentile_25": self.percentile_25,
            "median": self.median,
            "percentile_75": self.percentile_75,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "null_fraction": self.null_fraction,
            "histogram_counts": self.histogram_counts,
            "histogram_edges": self.histogram_edges,
            "unique_count": self.unique_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureStatistics":
        """Deserialise from a plain dict.

        Args:
            data: Dict produced by to_dict.

        Returns:
            Reconstructed FeatureStatistics.
        """
        return cls(**data)


@dataclass
class BaselineVersion:
    """A versioned snapshot of a model's reference distribution.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        model_id: UUID of the model this baseline belongs to.
        model_version: Version string of the model (e.g., 'v1.2').
        feature_stats: Per-feature statistical summaries.
        captured_at: UTC timestamp when this baseline was captured.
        sample_count: Total number of samples used to build this baseline.
        data_uri: Optional S3/MinIO URI of the source dataset.
        tags: Arbitrary key-value metadata tags.
        fingerprint: SHA-256 hash of the baseline statistics (for change detection).
        window_days: Window size in days that this baseline represents (0 = full).
        is_active: Whether this is the currently active baseline for the model.
    """

    baseline_id: uuid.UUID
    model_id: uuid.UUID
    model_version: str
    feature_stats: dict[str, FeatureStatistics]
    captured_at: datetime
    sample_count: int
    data_uri: str
    tags: dict[str, str]
    fingerprint: str
    window_days: int
    is_active: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict for export or storage.

        Returns:
            Dict with all baseline fields.
        """
        return {
            "baseline_id": str(self.baseline_id),
            "model_id": str(self.model_id),
            "model_version": self.model_version,
            "feature_stats": {name: stats.to_dict() for name, stats in self.feature_stats.items()},
            "captured_at": self.captured_at.isoformat(),
            "sample_count": self.sample_count,
            "data_uri": self.data_uri,
            "tags": self.tags,
            "fingerprint": self.fingerprint,
            "window_days": self.window_days,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineVersion":
        """Deserialise from a plain dict (e.g., from JSON import).

        Args:
            data: Dict produced by to_dict.

        Returns:
            Reconstructed BaselineVersion.
        """
        return cls(
            baseline_id=uuid.UUID(data["baseline_id"]),
            model_id=uuid.UUID(data["model_id"]),
            model_version=data["model_version"],
            feature_stats={
                name: FeatureStatistics.from_dict(stats)
                for name, stats in data["feature_stats"].items()
            },
            captured_at=datetime.fromisoformat(data["captured_at"]),
            sample_count=data["sample_count"],
            data_uri=data.get("data_uri", ""),
            tags=data.get("tags", {}),
            fingerprint=data["fingerprint"],
            window_days=data.get("window_days", 0),
            is_active=data.get("is_active", False),
        )


@dataclass
class BaselineComparison:
    """Result of comparing two baseline versions.

    Attributes:
        old_baseline_id: UUID of the older baseline.
        new_baseline_id: UUID of the newer baseline.
        features_compared: Number of features compared.
        features_changed: Features with statistically significant mean shifts.
        mean_shift_by_feature: Dict of feature_name to (old_mean, new_mean, pct_change).
        std_shift_by_feature: Dict of feature_name to (old_std, new_std, pct_change).
        distribution_change_score: Aggregate distributional change score (0–1).
        generated_at: UTC timestamp of this comparison.
    """

    old_baseline_id: uuid.UUID
    new_baseline_id: uuid.UUID
    features_compared: int
    features_changed: list[str]
    mean_shift_by_feature: dict[str, tuple[float, float, float]]
    std_shift_by_feature: dict[str, tuple[float, float, float]]
    distribution_change_score: float
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict.

        Returns:
            Dict with all comparison fields.
        """
        return {
            "old_baseline_id": str(self.old_baseline_id),
            "new_baseline_id": str(self.new_baseline_id),
            "features_compared": self.features_compared,
            "features_changed": self.features_changed,
            "mean_shift_by_feature": {
                k: {"old": v[0], "new": v[1], "pct_change": v[2]}
                for k, v in self.mean_shift_by_feature.items()
            },
            "std_shift_by_feature": {
                k: {"old": v[0], "new": v[1], "pct_change": v[2]}
                for k, v in self.std_shift_by_feature.items()
            },
            "distribution_change_score": self.distribution_change_score,
            "generated_at": self.generated_at.isoformat(),
        }


class BaselineManager:
    """Manages reference distribution baselines for drift detection.

    Provides:
    - Baseline capture from feature arrays with full statistical profiling
    - Versioning tied to model versions
    - Multi-window baseline variants (7d, 30d, 90d)
    - Comparison between baseline versions
    - Active baseline rotation with configurable retention policy
    - JSON export and import for portability

    All baselines are stored in-memory in this adapter. In production,
    the caller is responsible for persisting BaselineVersion.to_dict()
    output to the database or object store.
    """

    def __init__(self, max_versions_per_model: int = 10) -> None:
        """Initialise the baseline manager.

        Args:
            max_versions_per_model: Maximum number of baseline versions to retain
                per model before oldest non-active versions are evicted.
        """
        self._max_versions = max_versions_per_model
        # model_id → list of BaselineVersion (ordered oldest to newest)
        self._baselines: dict[uuid.UUID, list[BaselineVersion]] = {}

    def capture_baseline(
        self,
        model_id: uuid.UUID,
        model_version: str,
        feature_data: dict[str, np.ndarray],
        data_uri: str = "",
        tags: dict[str, str] | None = None,
        window_days: int = 0,
        activate: bool = True,
    ) -> BaselineVersion:
        """Capture a new baseline version from feature data arrays.

        Computes full statistical profiles (mean, std, percentiles, histogram)
        for each feature and stores them as a versioned baseline snapshot.

        Args:
            model_id: UUID of the model this baseline represents.
            model_version: Version string (e.g., 'v1.2', 'staging-2026-02-01').
            feature_data: Dict of feature_name to 1-D numpy array of values.
            data_uri: Optional S3/MinIO URI of the source dataset.
            tags: Optional arbitrary metadata tags.
            window_days: Window size this baseline covers (0 = full training set).
            activate: If True, mark this as the active baseline and deactivate others.

        Returns:
            Newly created BaselineVersion.

        Raises:
            ValueError: If feature_data is empty or any array contains no finite values.
        """
        if not feature_data:
            raise ValueError("feature_data must contain at least one feature")

        feature_stats: dict[str, FeatureStatistics] = {}
        total_samples = 0

        for feature_name, raw_array in feature_data.items():
            array = np.asarray(raw_array, dtype=float)
            null_count = int(np.sum(~np.isfinite(array)))
            null_fraction = null_count / len(array) if len(array) > 0 else 0.0
            clean = array[np.isfinite(array)]

            if clean.size == 0:
                raise ValueError(f"Feature '{feature_name}' contains no finite values")

            total_samples = max(total_samples, int(clean.size))
            counts, edges = np.histogram(clean, bins=20)
            stats = FeatureStatistics(
                feature_name=feature_name,
                count=int(clean.size),
                mean=float(np.mean(clean)),
                std=float(np.std(clean)),
                min_value=float(np.min(clean)),
                max_value=float(np.max(clean)),
                percentile_25=float(np.percentile(clean, 25)),
                median=float(np.percentile(clean, 50)),
                percentile_75=float(np.percentile(clean, 75)),
                percentile_95=float(np.percentile(clean, 95)),
                percentile_99=float(np.percentile(clean, 99)),
                null_fraction=null_fraction,
                histogram_counts=counts.tolist(),
                histogram_edges=edges.tolist(),
                unique_count=int(len(np.unique(clean))),
            )
            feature_stats[feature_name] = stats

        fingerprint = self._compute_fingerprint(feature_stats)
        baseline = BaselineVersion(
            baseline_id=uuid.uuid4(),
            model_id=model_id,
            model_version=model_version,
            feature_stats=feature_stats,
            captured_at=datetime.now(tz=timezone.utc),
            sample_count=total_samples,
            data_uri=data_uri,
            tags=tags or {},
            fingerprint=fingerprint,
            window_days=window_days,
            is_active=activate,
        )

        if activate:
            # Deactivate all existing baselines for this model
            for existing in self._baselines.get(model_id, []):
                existing.is_active = False

        if model_id not in self._baselines:
            self._baselines[model_id] = []
        self._baselines[model_id].append(baseline)
        self._evict_old_versions(model_id)

        logger.info(
            "Baseline captured",
            baseline_id=str(baseline.baseline_id),
            model_id=str(model_id),
            model_version=model_version,
            feature_count=len(feature_stats),
            sample_count=total_samples,
            window_days=window_days,
            is_active=activate,
        )
        return baseline

    def capture_multi_window_baselines(
        self,
        model_id: uuid.UUID,
        model_version: str,
        feature_data: dict[str, np.ndarray],
        window_days_list: tuple[int, ...] = STANDARD_WINDOW_DAYS,
        data_uri: str = "",
    ) -> list[BaselineVersion]:
        """Capture multiple windowed baselines from the same dataset.

        For each window size (e.g., 7d, 30d, 90d), samples the most recent
        N days of data (by row index, assuming data is chronologically sorted)
        and computes a baseline.

        Args:
            model_id: UUID of the model.
            model_version: Model version string.
            feature_data: Full feature dataset (chronologically sorted rows).
            window_days_list: Tuple of window sizes in days to capture.
            data_uri: Optional source URI.

        Returns:
            List of BaselineVersion, one per window size.
        """
        total_rows = min(arr.size for arr in feature_data.values()) if feature_data else 0
        baselines: list[BaselineVersion] = []

        for window_days in window_days_list:
            # Approximate rows per day and slice the tail
            rows_per_day = max(1, total_rows // 90)  # Assume 90d is full dataset
            rows_for_window = min(total_rows, window_days * rows_per_day)
            window_data = {
                name: arr[-rows_for_window:] for name, arr in feature_data.items()
            }
            baseline = self.capture_baseline(
                model_id=model_id,
                model_version=model_version,
                feature_data=window_data,
                data_uri=data_uri,
                tags={"window_days": str(window_days)},
                window_days=window_days,
                activate=False,  # Only the full baseline should be active
            )
            baselines.append(baseline)

        logger.info(
            "Multi-window baselines captured",
            model_id=str(model_id),
            windows=list(window_days_list),
        )
        return baselines

    def get_active_baseline(self, model_id: uuid.UUID) -> BaselineVersion | None:
        """Return the currently active baseline for a model.

        Args:
            model_id: UUID of the model.

        Returns:
            Active BaselineVersion or None if no baselines exist.
        """
        versions = self._baselines.get(model_id, [])
        for v in reversed(versions):
            if v.is_active:
                return v
        return None

    def get_baseline_by_id(self, baseline_id: uuid.UUID) -> BaselineVersion | None:
        """Retrieve a specific baseline by its UUID.

        Args:
            baseline_id: UUID to look up.

        Returns:
            BaselineVersion or None if not found.
        """
        for versions in self._baselines.values():
            for v in versions:
                if v.baseline_id == baseline_id:
                    return v
        return None

    def list_baselines(
        self,
        model_id: uuid.UUID,
        include_inactive: bool = True,
    ) -> list[BaselineVersion]:
        """List all baselines for a model.

        Args:
            model_id: UUID of the model.
            include_inactive: If False, return only the active baseline.

        Returns:
            List of BaselineVersion ordered newest-first.
        """
        versions = list(reversed(self._baselines.get(model_id, [])))
        if not include_inactive:
            versions = [v for v in versions if v.is_active]
        return versions

    def compare_baselines(
        self,
        old_baseline_id: uuid.UUID,
        new_baseline_id: uuid.UUID,
        significance_threshold: float = 0.1,
    ) -> BaselineComparison:
        """Compare two baseline versions and quantify distributional shift.

        Args:
            old_baseline_id: UUID of the older (reference) baseline.
            new_baseline_id: UUID of the newer baseline.
            significance_threshold: Fractional mean shift above which a feature is
                flagged as "changed" (e.g., 0.1 = 10% shift).

        Returns:
            BaselineComparison with per-feature shift analysis.

        Raises:
            ValueError: If either baseline ID is not found.
        """
        old = self.get_baseline_by_id(old_baseline_id)
        new = self.get_baseline_by_id(new_baseline_id)
        if old is None:
            raise ValueError(f"Baseline {old_baseline_id} not found")
        if new is None:
            raise ValueError(f"Baseline {new_baseline_id} not found")

        common_features = set(old.feature_stats.keys()) & set(new.feature_stats.keys())
        features_changed: list[str] = []
        mean_shifts: dict[str, tuple[float, float, float]] = {}
        std_shifts: dict[str, tuple[float, float, float]] = {}
        total_change = 0.0

        for feature_name in common_features:
            old_stats = old.feature_stats[feature_name]
            new_stats = new.feature_stats[feature_name]

            old_mean = old_stats.mean
            new_mean = new_stats.mean
            mean_pct = abs(new_mean - old_mean) / abs(old_mean) if old_mean != 0 else 0.0
            mean_shifts[feature_name] = (old_mean, new_mean, mean_pct)

            old_std = old_stats.std
            new_std = new_stats.std
            std_pct = abs(new_std - old_std) / abs(old_std) if old_std != 0 else 0.0
            std_shifts[feature_name] = (old_std, new_std, std_pct)

            combined_change = (mean_pct + 0.5 * std_pct) / 1.5
            total_change += combined_change

            if mean_pct > significance_threshold or std_pct > significance_threshold:
                features_changed.append(feature_name)

        change_score = min(1.0, total_change / len(common_features)) if common_features else 0.0

        comparison = BaselineComparison(
            old_baseline_id=old_baseline_id,
            new_baseline_id=new_baseline_id,
            features_compared=len(common_features),
            features_changed=sorted(features_changed),
            mean_shift_by_feature=mean_shifts,
            std_shift_by_feature=std_shifts,
            distribution_change_score=change_score,
            generated_at=datetime.now(tz=timezone.utc),
        )
        logger.info(
            "Baseline comparison completed",
            old_baseline_id=str(old_baseline_id),
            new_baseline_id=str(new_baseline_id),
            features_changed=len(features_changed),
            change_score=change_score,
        )
        return comparison

    def activate_baseline(
        self,
        baseline_id: uuid.UUID,
    ) -> bool:
        """Set a specific baseline as active and deactivate all others for that model.

        Args:
            baseline_id: UUID of the baseline to activate.

        Returns:
            True if found and activated, False if not found.
        """
        target = self.get_baseline_by_id(baseline_id)
        if target is None:
            return False

        for v in self._baselines.get(target.model_id, []):
            v.is_active = v.baseline_id == baseline_id

        logger.info("Baseline activated", baseline_id=str(baseline_id), model_id=str(target.model_id))
        return True

    def export_baseline(self, baseline_id: uuid.UUID) -> str:
        """Export a baseline version as a JSON string.

        Args:
            baseline_id: UUID of the baseline to export.

        Returns:
            JSON string representation of the baseline.

        Raises:
            ValueError: If the baseline is not found.
        """
        baseline = self.get_baseline_by_id(baseline_id)
        if baseline is None:
            raise ValueError(f"Baseline {baseline_id} not found")
        return json.dumps(baseline.to_dict(), indent=2)

    def import_baseline(self, json_str: str, activate: bool = False) -> BaselineVersion:
        """Import a baseline from a JSON string (previously exported).

        Args:
            json_str: JSON string produced by export_baseline.
            activate: If True, activate this baseline after import.

        Returns:
            Imported BaselineVersion.

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
        """
        try:
            data = json.loads(json_str)
            baseline = BaselineVersion.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid baseline JSON: {exc}") from exc

        model_id = baseline.model_id
        if model_id not in self._baselines:
            self._baselines[model_id] = []

        if activate:
            for v in self._baselines[model_id]:
                v.is_active = False
            baseline.is_active = True

        self._baselines[model_id].append(baseline)
        logger.info(
            "Baseline imported",
            baseline_id=str(baseline.baseline_id),
            model_id=str(model_id),
            model_version=baseline.model_version,
        )
        return baseline

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_old_versions(self, model_id: uuid.UUID) -> None:
        """Remove oldest non-active baselines when the cap is exceeded.

        Args:
            model_id: UUID of the model whose baselines to trim.
        """
        versions = self._baselines.get(model_id, [])
        if len(versions) <= self._max_versions:
            return

        # Keep active versions and the most recent non-active ones
        inactive = [v for v in versions if not v.is_active]
        active = [v for v in versions if v.is_active]
        evict_count = len(versions) - self._max_versions
        inactive = inactive[evict_count:]  # Drop oldest inactive first
        self._baselines[model_id] = inactive + active
        logger.debug(
            "Old baseline versions evicted",
            model_id=str(model_id),
            evicted_count=evict_count,
        )

    @staticmethod
    def _compute_fingerprint(feature_stats: dict[str, FeatureStatistics]) -> str:
        """Compute a SHA-256 fingerprint from feature statistics.

        Args:
            feature_stats: Dict of feature_name to FeatureStatistics.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        canonical = json.dumps(
            {name: stats.to_dict() for name, stats in sorted(feature_stats.items())},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
