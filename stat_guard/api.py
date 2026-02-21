"""
Main API for StatGuard.

Provides high-level functions for data validation and profiling.
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd

from .engine import ValidationEngine, DataValidator, ValidationError
from .report import ValidationReport
from .policy import POLICIES, create_policy, ValidationPolicy
from .profilers.data_profiler import DataProfiler, DatasetProfile


# Global engine instance
_engine = ValidationEngine()
_validator = DataValidator()
_profiler = DataProfiler()


def validate(
    data: pd.DataFrame,
    *,
    target_col: str,
    group_col: Optional[str] = None,
    unit_col: Optional[str] = None,
    policy: Union[str, ValidationPolicy] = "default",
    fail_fast: bool = False,
    verbose: bool = False,
) -> ValidationReport:
    """
    Validate data for statistical analysis.
    
    This is the main entry point for data validation. It runs all
    configured checks and returns a detailed report.
    
    Args:
        data: Input DataFrame to validate
        target_col: The numeric metric column being analyzed
        group_col: Optional column defining experimental groups
        unit_col: Optional column with unit identifiers
        policy: Validation policy ("default", "strict", "lenient", etc.)
        fail_fast: Stop on first error if True
        verbose: Print progress information
        
    Returns:
        ValidationReport with all violations and statistics
        
    Example:
        >>> import pandas as pd
        >>> import stat_guard as sg
        >>> 
        >>> data = pd.DataFrame({
        ...     "metric": [1, 2, 3, 4, 5],
        ...     "group": ["A", "A", "B", "B", "B"]
        ... })
        >>> 
        >>> report = sg.validate(data, target_col="metric", group_col="group")
        >>> print(report.is_valid)
        True
    """
    if verbose:
        _engine.verbose = True
    
    return _engine.validate(
        data=data,
        target_col=target_col,
        group_col=group_col,
        unit_col=unit_col,
        policy=policy,
        fail_fast=fail_fast,
    )


def profile(
    data: pd.DataFrame,
    target_col: Optional[str] = None,
    compute_correlations: bool = True,
) -> DatasetProfile:
    """
    Generate comprehensive data profile.
    
    Similar to ydata-profiling, this provides detailed statistics
    about the dataset including distributions, correlations, and
    data quality indicators.
    
    Args:
        data: Input DataFrame to profile
        target_col: Optional target column for focused analysis
        compute_correlations: Whether to compute correlation matrix
        
    Returns:
        DatasetProfile with comprehensive statistics
        
    Example:
        >>> profile = sg.profile(data)
        >>> print(profile.n_rows, profile.n_columns)
        >>> print(profile.to_dict())
    """
    profiler = DataProfiler(compute_correlations=compute_correlations)
    return profiler.profile(data, target_col=target_col)


def quick_check(
    data: pd.DataFrame,
    target_col: str,
) -> bool:
    """
    Quick validity check without full report.
    
    Args:
        data: Input DataFrame
        target_col: Target column to check
        
    Returns:
        True if data passes all critical checks
        
    Example:
        >>> if sg.quick_check(data, "metric"):
        ...     print("Data is valid!")
    """
    return _validator.is_valid(data, target_col)


def compare(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    target_col: str,
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare two datasets for drift and differences.
    
    Useful for detecting data drift between training and test sets,
    or before/after comparisons.
    
    Args:
        data1: First dataset (e.g., training data)
        data2: Second dataset (e.g., test data)
        target_col: Target column to compare
        group_col: Optional grouping column
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> comparison = sg.compare(train_data, test_data, "target")
        >>> print(comparison["drift_detected"])
    """
    from scipy import stats
    
    results = {
        "data1_shape": data1.shape,
        "data2_shape": data2.shape,
        "target_col": target_col,
    }
    
    # Basic statistics comparison
    x1 = data1[target_col].dropna()
    x2 = data2[target_col].dropna()
    
    results["data1_stats"] = {
        "mean": x1.mean(),
        "std": x1.std(),
        "count": len(x1),
    }
    
    results["data2_stats"] = {
        "mean": x2.mean(),
        "std": x2.std(),
        "count": len(x2),
    }
    
    # Statistical tests
    try:
        # Kolmogorov-Smirnov test for distribution differences
        ks_stat, ks_pvalue = stats.ks_2samp(x1, x2)
        results["ks_test"] = {
            "statistic": ks_stat,
            "p_value": ks_pvalue,
            "significant": ks_pvalue < 0.05
        }
        
        # T-test for mean differences
        t_stat, t_pvalue = stats.ttest_ind(x1, x2)
        results["t_test"] = {
            "statistic": t_stat,
            "p_value": t_pvalue,
            "significant": t_pvalue < 0.05
        }
        
        # Levene's test for variance differences
        w_stat, w_pvalue = stats.levene(x1, x2)
        results["levene_test"] = {
            "statistic": w_stat,
            "p_value": w_pvalue,
            "significant": w_pvalue < 0.05
        }
        
        # Overall drift detection
        results["drift_detected"] = any([
            results["ks_test"]["significant"],
            results["t_test"]["significant"],
            results["levene_test"]["significant"]
        ])
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def register_validator(check) -> None:
    """
    Register a custom validation check.
    
    Args:
        check: Custom check instance with `name` and `run()` method
        
    Example:
        >>> class MyCheck:
        ...     name = "My Custom Check"
        ...     def run(self, data, target_col, **kwargs):
        ...         # Return Violation or list of Violations
        ...         pass
        >>> 
        >>> sg.register_validator(MyCheck())
    """
    _engine.register(check)


def validate_multiple(
    data: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str] = None,
    unit_col: Optional[str] = None,
    policy: str = "default",
) -> Dict[str, ValidationReport]:
    """
    Validate multiple target columns at once.
    
    Args:
        data: Input DataFrame
        target_cols: List of target columns to validate
        group_col: Optional grouping column
        unit_col: Optional unit identifier column
        policy: Validation policy
        
    Returns:
        Dictionary mapping column names to validation reports
        
    Example:
        >>> reports = sg.validate_multiple(data, ["col1", "col2", "col3"])
        >>> for col, report in reports.items():
        ...     print(f"{col}: {report.is_valid}")
    """
    return _engine.validate_multiple(
        data=data,
        target_cols=target_cols,
        group_col=group_col,
        unit_col=unit_col,
        policy=policy,
    )


def get_available_policies() -> List[str]:
    """
    Get list of available policy names.
    
    Returns:
        List of policy names
    """
    return list(POLICIES.keys())


def create_custom_policy(
    base: str = "default",
    **overrides
) -> ValidationPolicy:
    """
    Create a custom validation policy.
    
    Args:
        base: Base policy name
        **overrides: Policy parameters to override
        
    Returns:
        Custom ValidationPolicy
        
    Example:
        >>> policy = sg.create_custom_policy("default", min_sample_size=100)
        >>> report = sg.validate(data, target_col="x", policy=policy)
    """
    return create_policy(base, **overrides)


def list_checks() -> List[str]:
    """
    List all registered validation checks.
    
    Returns:
        List of check names
    """
    return _engine.list_checks()


# Convenience functions for common use cases

def check_experiment(
    data: pd.DataFrame,
    metric_col: str,
    treatment_col: str,
    user_id_col: Optional[str] = None,
) -> ValidationReport:
    """
    Specialized check for A/B tests and experiments.
    
    Uses the "experiment" policy which has stricter thresholds
    for experimental data.
    
    Args:
        data: Experiment data
        metric_col: Metric being measured
        treatment_col: Treatment/group assignment column
        user_id_col: Optional user identifier column
        
    Returns:
        ValidationReport
    """
    return validate(
        data=data,
        target_col=metric_col,
        group_col=treatment_col,
        unit_col=user_id_col,
        policy="experiment",
    )


def check_time_series(
    data: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
) -> ValidationReport:
    """
    Specialized check for time series data.
    
    Args:
        data: Time series data
        target_col: Target column
        timestamp_col: Timestamp column
        
    Returns:
        ValidationReport
    """
    return validate(
        data=data,
        target_col=target_col,
        policy="time_series",
    )


__all__ = [
    # Core functions
    "validate",
    "profile",
    "quick_check",
    "compare",
    "register_validator",
    "validate_multiple",
    
    # Policy functions
    "get_available_policies",
    "create_custom_policy",
    
    # Utility functions
    "list_checks",
    "check_experiment",
    "check_time_series",
    
    # Classes
    "ValidationReport",
    "DatasetProfile",
    "ValidationPolicy",
    "ValidationError",
]
