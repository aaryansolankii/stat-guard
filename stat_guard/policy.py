"""
Policy configuration for StatGuard validation.

Policies define thresholds and parameters for all validation checks.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationPolicy:
    """
    Comprehensive validation policy configuration.
    
    This class encapsulates all thresholds and parameters used
    across different validation checks.
    """
    
    # Sample Size & Power
    min_sample_size: int = 30
    min_sample_size_per_group: int = 15
    max_imbalance_ratio: float = 2.0
    max_smd: float = 0.25
    min_power: float = 0.80
    
    # Distribution & Normality
    max_skewness: float = 2.0
    max_kurtosis: float = 7.0
    normality_alpha: float = 0.05
    min_shapiro_sample: int = 20
    max_shapiro_sample: int = 5000
    
    # Variance
    variance_threshold: float = 1e-10
    near_zero_variance_ratio: float = 0.95
    
    # Missing Data
    max_missing_pct: float = 0.05
    max_missing_pct_column: float = 0.20
    flag_missing_pattern: bool = True
    
    # Outliers
    outlier_method: str = "iqr"  # "iqr", "zscore", "mad"
    outlier_threshold: float = 3.0
    max_outlier_pct: float = 0.05
    flag_outlier_clusters: bool = True
    
    # Correlation
    max_correlation: float = 0.95
    vif_threshold: float = 5.0
    correlation_method: str = "pearson"
    
    # Cardinality
    max_cardinality_ratio: float = 0.95  # unique/count
    min_cardinality_ratio: float = 0.01
    rare_category_threshold: int = 5
    
    # Duplicates
    check_duplicate_rows: bool = True
    check_duplicate_units: bool = True
    
    # Data Quality
    flag_constant_columns: bool = True
    flag_high_cardinality: bool = True
    flag_suspicious_values: bool = True
    
    # Time Series
    check_timestamp_regularity: bool = True
    max_timestamp_gap_pct: float = 0.10
    
    # Profiling
    compute_correlations: bool = True
    compute_quantiles: bool = True
    max_unique_for_histogram: int = 50
    histogram_bins: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationPolicy":
        """Create policy from dictionary."""
        valid_keys = {k for k in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# Predefined policies
DEFAULT_POLICY = ValidationPolicy()

STRICT_POLICY = ValidationPolicy(
    min_sample_size=50,
    min_sample_size_per_group=25,
    max_imbalance_ratio=1.5,
    max_smd=0.10,
    max_skewness=1.5,
    max_kurtosis=5.0,
    max_missing_pct=0.02,
    max_missing_pct_column=0.10,
    max_correlation=0.90,
    vif_threshold=4.0,
    outlier_threshold=2.5,
    max_outlier_pct=0.02,
)

LENIENT_POLICY = ValidationPolicy(
    min_sample_size=10,
    min_sample_size_per_group=5,
    max_imbalance_ratio=5.0,
    max_smd=0.50,
    max_skewness=3.0,
    max_kurtosis=10.0,
    max_missing_pct=0.20,
    max_missing_pct_column=0.50,
    max_correlation=0.99,
    vif_threshold=10.0,
)

EXPERIMENT_POLICY = ValidationPolicy(
    min_sample_size=100,
    min_sample_size_per_group=50,
    max_imbalance_ratio=1.2,
    max_smd=0.10,
    max_missing_pct=0.01,
    check_duplicate_units=True,
    flag_constant_columns=True,
)

TIME_SERIES_POLICY = ValidationPolicy(
    check_timestamp_regularity=True,
    max_timestamp_gap_pct=0.05,
    normality_alpha=0.01,
)

# Policy registry
POLICIES: Dict[str, ValidationPolicy] = {
    "default": DEFAULT_POLICY,
    "strict": STRICT_POLICY,
    "lenient": LENIENT_POLICY,
    "experiment": EXPERIMENT_POLICY,
    "time_series": TIME_SERIES_POLICY,
}


def create_policy(
    base: str = "default",
    **overrides
) -> ValidationPolicy:
    """
    Create a custom policy based on an existing one.
    
    Args:
        base: Base policy name ("default", "strict", "lenient", etc.)
        **overrides: Policy parameters to override
        
    Returns:
        Custom ValidationPolicy instance
        
    Example:
        >>> policy = create_policy("default", min_sample_size=100, max_missing_pct=0.10)
    """
    if base not in POLICIES:
        raise ValueError(f"Unknown policy '{base}'. Available: {list(POLICIES.keys())}")
    
    base_policy = POLICIES[base]
    policy_dict = base_policy.to_dict()
    policy_dict.update(overrides)
    
    return ValidationPolicy.from_dict(policy_dict)


def register_policy(name: str, policy: ValidationPolicy) -> None:
    """
    Register a custom policy for reuse.
    
    Args:
        name: Policy name
        policy: ValidationPolicy instance
    """
    POLICIES[name] = policy


def get_policy(name: str) -> ValidationPolicy:
    """
    Get a policy by name.
    
    Args:
        name: Policy name
        
    Returns:
        ValidationPolicy instance
    """
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICIES.keys())}")
    return POLICIES[name]
