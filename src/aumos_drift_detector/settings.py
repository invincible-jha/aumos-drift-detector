"""Drift Detector service settings extending the AumOS base configuration."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS Drift Detector service.

    Extends base AumOS settings with drift-detection-specific configuration
    for statistical thresholds, concept drift detectors, and retraining triggers.

    All environment variables use the AUMOS_DRIFT_ prefix.
    """

    service_name: str = "aumos-drift-detector"

    # -------------------------------------------------------------------------
    # Statistical drift thresholds (global defaults; per-monitor values override)
    # -------------------------------------------------------------------------

    # Kolmogorov-Smirnov: drift if p_value < threshold
    drift_threshold_ks: float = 0.05

    # Population Stability Index: drift if psi > threshold
    drift_threshold_psi: float = 0.2

    # PSI warning level (alert issued but not yet "drifted")
    drift_warning_threshold_psi: float = 0.1

    # Chi-squared: drift if p_value < threshold
    drift_threshold_chi2: float = 0.05

    # -------------------------------------------------------------------------
    # ADWIN concept drift detector
    # -------------------------------------------------------------------------

    # Confidence parameter: smaller delta = less sensitive to change
    adwin_delta: float = 0.002

    # Maximum number of sub-windows stored in ADWIN's exponential histogram
    adwin_max_buckets: int = 5

    # -------------------------------------------------------------------------
    # DDM concept drift detector
    # -------------------------------------------------------------------------

    # Number of standard deviations above baseline that triggers a warning
    ddm_warning_level: float = 2.0

    # Number of standard deviations above baseline that signals drift
    ddm_drift_level: float = 3.0

    # Minimum number of samples before drift detection starts
    ddm_min_num_instances: int = 30

    # -------------------------------------------------------------------------
    # Retraining trigger
    # -------------------------------------------------------------------------

    retraining_trigger_enabled: bool = True
    retraining_topic: str = "drift.retraining_required"

    # Minimum seconds between retraining triggers for the same monitor
    retraining_cooldown_seconds: int = 3600

    # -------------------------------------------------------------------------
    # Reference data storage (MinIO / S3-compatible)
    # -------------------------------------------------------------------------

    reference_data_bucket: str = "aumos-drift-reference"

    model_config = SettingsConfigDict(env_prefix="AUMOS_DRIFT_")
