"""
Statistical validation checks for StatGuard.
"""

from .base import (
    StatisticalCheck,
    ColumnCheck,
    DataFrameCheck,
    CheckResult,
    normalize_violations,
    create_violation,
)

from .sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
    CovariateBalanceCheck,
    StatisticalPowerCheck,
    EffectSizeCheck,
)

from .distribution import (
    ZeroVarianceCheck,
    NearZeroVarianceCheck,
    SkewnessCheck,
    KurtosisCheck,
    NormalityCheck,
    HeteroscedasticityCheck,
    RangeCheck,
)

from .unit_integrity import (
    UnitIntegrityCheck,
    DuplicateRowsCheck,
    MissingDataCheck,
    DataTypeCheck,
    ConstantColumnCheck,
)

from .outliers import (
    OutlierCheck,
    ExtremeValueCheck,
    WinsorizationCheck,
)

from .correlation import (
    CorrelationCheck,
    MulticollinearityCheck,
    TargetCorrelationCheck,
    CorrelationWithGroupCheck,
)

from .cardinality import (
    CardinalityCheck,
    EmptyCategoryCheck,
    CategoricalBalanceCheck,
    HighCardinalityIDCheck,
)

from .missing_data import (
    MissingPatternCheck,
    MissingTargetCheck,
    MissingByFeatureCheck,
    CompleteCaseAnalysisCheck,
)

__all__ = [
    # Base classes
    "StatisticalCheck",
    "ColumnCheck",
    "DataFrameCheck",
    "CheckResult",
    "normalize_violations",
    "create_violation",
    
    # Sample size checks
    "MinimumSampleSizeCheck",
    "BalancedGroupsCheck",
    "CovariateBalanceCheck",
    "StatisticalPowerCheck",
    "EffectSizeCheck",
    
    # Distribution checks
    "ZeroVarianceCheck",
    "NearZeroVarianceCheck",
    "SkewnessCheck",
    "KurtosisCheck",
    "NormalityCheck",
    "HeteroscedasticityCheck",
    "RangeCheck",
    
    # Unit integrity checks
    "UnitIntegrityCheck",
    "DuplicateRowsCheck",
    "MissingDataCheck",
    "DataTypeCheck",
    "ConstantColumnCheck",
    
    # Outlier checks
    "OutlierCheck",
    "ExtremeValueCheck",
    "WinsorizationCheck",
    
    # Correlation checks
    "CorrelationCheck",
    "MulticollinearityCheck",
    "TargetCorrelationCheck",
    "CorrelationWithGroupCheck",
    
    # Cardinality checks
    "CardinalityCheck",
    "EmptyCategoryCheck",
    "CategoricalBalanceCheck",
    "HighCardinalityIDCheck",
    
    # Missing data checks
    "MissingPatternCheck",
    "MissingTargetCheck",
    "MissingByFeatureCheck",
    "CompleteCaseAnalysisCheck",
]

# Default check registry
DEFAULT_CHECKS = [
    MinimumSampleSizeCheck(),
    BalancedGroupsCheck(),
    CovariateBalanceCheck(),
    ZeroVarianceCheck,
    NearZeroVarianceCheck(),
    SkewnessCheck(),
    KurtosisCheck(),
    NormalityCheck(),
    UnitIntegrityCheck(),
    DuplicateRowsCheck(),
    MissingDataCheck(),
    ConstantColumnCheck(),
    OutlierCheck(),
    CorrelationCheck(),
    CardinalityCheck(),
    MissingTargetCheck(),
]
