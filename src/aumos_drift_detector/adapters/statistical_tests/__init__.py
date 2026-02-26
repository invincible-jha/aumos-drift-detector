"""Statistical drift tests — pure functions using scipy and numpy.

Available tests:
- KolmogorovSmirnovTest  — two-sample KS test for continuous features
- PopulationStabilityIndex — PSI with configurable binning
- ChiSquaredTest — chi-squared test for categorical features
"""

from aumos_drift_detector.adapters.statistical_tests.chi_squared import (
    ChiSquaredResult,
    ChiSquaredTest,
)
from aumos_drift_detector.adapters.statistical_tests.ks_test import (
    KolmogorovSmirnovResult,
    KolmogorovSmirnovTest,
)
from aumos_drift_detector.adapters.statistical_tests.psi import (
    PopulationStabilityIndex,
    PsiResult,
)

__all__ = [
    "KolmogorovSmirnovTest",
    "KolmogorovSmirnovResult",
    "PopulationStabilityIndex",
    "PsiResult",
    "ChiSquaredTest",
    "ChiSquaredResult",
]
