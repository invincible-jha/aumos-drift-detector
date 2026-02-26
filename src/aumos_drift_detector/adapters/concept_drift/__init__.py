"""Concept drift detectors for streaming prediction error monitoring.

Available detectors:
- AdwinDetector  — ADaptive WINdowing (ADWIN) algorithm
- DdmDetector    — Drift Detection Method (DDM)
- EddmDetector   — Enhanced DDM (EDDM)

All detectors are stateful objects with a consistent interface:
    detector.update(error: float) -> None
    detector.detect() -> DriftLevel
    detector.reset() -> None
"""

from aumos_drift_detector.adapters.concept_drift.adwin import (
    AdwinDetector,
    AdwinState,
)
from aumos_drift_detector.adapters.concept_drift.ddm import (
    DdmDetector,
    DdmState,
    DriftLevel,
    EddmDetector,
)

__all__ = [
    "AdwinDetector",
    "AdwinState",
    "DdmDetector",
    "DdmState",
    "DriftLevel",
    "EddmDetector",
]
