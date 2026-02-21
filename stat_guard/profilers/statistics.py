"""
Statistical computation utilities for StatGuard.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats


def compute_statistics(
    series: pd.Series,
    include_quantiles: bool = True,
    include_shape: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a series.
    
    Args:
        series: Input data series
        include_quantiles: Whether to include quantile statistics
        include_shape: Whether to include skewness and kurtosis
        
    Returns:
        Dictionary of statistics
    """
    values = series.dropna()
    
    if len(values) == 0:
        return {"count": 0}
    
    result = {
        "count": len(values),
        "missing": series.isna().sum(),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
    }
    
    if include_quantiles:
        result.update({
            "q05": values.quantile(0.05),
            "q25": values.quantile(0.25),
            "q50": values.quantile(0.50),
            "q75": values.quantile(0.75),
            "q95": values.quantile(0.95),
            "iqr": values.quantile(0.75) - values.quantile(0.25),
        })
    
    if include_shape and len(values) >= 8:
        result.update({
            "skewness": stats.skew(values),
            "kurtosis": stats.kurtosis(values),
        })
    
    # Add normality test for larger samples
    if len(values) >= 20 and len(values) <= 5000:
        sample = values.sample(min(len(values), 500), random_state=42)

            # FIX: prevent divide-by-zero warning
        sample_clean = sample.dropna()

        # FINAL FIX — must check uniqueness AND variance
        if len(sample_clean) >= 3:
            if sample_clean.nunique() <= 1:
                pass  # constant data → skip
            elif sample_clean.var() < 1e-12:
                pass  # near-constant data → skip
            else:
                try:
                    _, p_value = stats.shapiro(sample_clean)
                    result["normality_pvalue"] = p_value
                    result["is_normal"] = p_value > 0.05
                except Exception:
                    pass    
    return result


def compute_group_statistics(
    data: pd.DataFrame,
    target_col: str,
    group_col: str,
    include_tests: bool = True
) -> Dict[str, Any]:
    """
    Compute statistics grouped by a categorical variable.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        group_col: Grouping column name
        include_tests: Whether to include statistical tests
        
    Returns:
        Dictionary with group statistics and comparisons
    """
    groups = data.groupby(group_col)[target_col]
    
    # Compute statistics for each group
    group_stats = {}
    for name, group in groups:
        group_stats[str(name)] = compute_statistics(group)
    
    result = {
        "groups": group_stats,
        "n_groups": len(group_stats),
    }
    
    # Add comparison statistics
    if include_tests and len(group_stats) >= 2:
        group_values = [g.dropna() for _, g in groups if len(g.dropna()) > 0]
        
        if len(group_values) >= 2:
            # ANOVA
            if all(g.std() > 1e-12 for g in group_values):
                try:
                    f_stat, p_value = stats.f_oneway(*group_values)
                    result["anova"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                except Exception:
                    pass
            
            # Levene's test for equal variances
            if all(g.std() > 1e-12 for g in group_values):
                try:
                    w_stat, p_value = stats.levene(*group_values)
                    result["levene"] = {
                        "w_statistic": w_stat,
                        "p_value": p_value,
                        "equal_variances": p_value > 0.05
                    }
                except Exception:
                    pass    
    return result


def compute_effect_size(
    group1: pd.Series,
    group2: pd.Series
) -> Dict[str, float]:
    """
    Compute effect size measures between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Dictionary of effect size measures
    """
    x1 = group1.dropna()
    x2 = group2.dropna()
    
    if len(x1) == 0 or len(x2) == 0:
        return {}
    
    # Cohen's d
    pooled_std = np.sqrt((x1.var() + x2.var()) / 2)
    cohens_d = abs(x1.mean() - x2.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Hedges' g (corrected for small samples)
    n1, n2 = len(x1), len(x2)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction
    
    # Glass's delta (uses control group std)
    glass_delta = abs(x1.mean() - x2.mean()) / x2.std() if x2.std() > 0 else 0
    
    return {
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "glass_delta": glass_delta,
        "interpretation": _interpret_effect_size(cohens_d)
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_confidence_interval(
    series: pd.Series,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute confidence interval for the mean.
    
    Args:
        series: Input data
        confidence: Confidence level (default 0.95)
        
    Returns:
        Dictionary with CI bounds
    """
    values = series.dropna()
    
    if len(values) < 2:
        return {}
    
    mean = values.mean()
    std = values.std()

    if std < 1e-12:
        return {"mean": values.mean()}

    sem = stats.sem(values)    
    
    try:
        ci_low, ci_high = stats.t.interval(
            confidence,
            len(values) - 1,
            loc=mean,
            scale=sem
        )
        return {
            "mean": mean,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "confidence": confidence,
            "margin_of_error": (ci_high - ci_low) / 2
        }
    except Exception:
        return {"mean": mean}


def compute_outlier_statistics(
    series: pd.Series,
    method: str = "iqr"
) -> Dict[str, Any]:
    """
    Compute outlier statistics for a series.
    
    Args:
        series: Input data
        method: Outlier detection method ("iqr", "zscore", "mad")
        
    Returns:
        Dictionary with outlier information
    """
    values = series.dropna()
    
    if len(values) < 10:
        return {"outlier_count": 0}
    
    if method == "iqr":
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (values < lower) | (values > upper)
    
    elif method == "zscore":
        std = values.std()

        if std < 1e-12:
            return {
                "outlier_count": 0,
                "outlier_percentage": 0,
                "method": method
            }

        z_scores = np.abs(stats.zscore(values))
        outlier_mask = z_scores > 3
        lower = upper = None
    
    elif method == "mad":
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return {"outlier_count": 0}
        modified_z = 0.6745 * (values - median) / mad
        outlier_mask = np.abs(modified_z) > 3.5
        lower = upper = None
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_count = outlier_mask.sum()
    outlier_pct = outlier_count / len(values) * 100
    
    result = {
        "outlier_count": int(outlier_count),
        "outlier_percentage": outlier_pct,
        "method": method,
    }
    
    if lower is not None:
        result["bounds"] = {"lower": lower, "upper": upper}
    
    if outlier_count > 0:
        outlier_values = values[outlier_mask]
        result["outlier_range"] = {
            "min": outlier_values.min(),
            "max": outlier_values.max()
        }
    
    return result
