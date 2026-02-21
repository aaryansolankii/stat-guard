"""
Distribution and statistical assumption validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats

from .base import StatisticalCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class ZeroVarianceCheck(StatisticalCheck):
    """Detects zero or near-zero variance in metrics."""

    @property
    def name(self) -> str:
        return "Zero Variance"
    
    @property
    def description(self) -> str:
        return "Checks for columns with no or minimal variation"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        variance_threshold: float = 1e-10,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        for g, vals in groups.items():
            if len(vals) < 2:
                continue
                
            variance = vals.var()
            n_unique = vals.nunique()
            
            if n_unique <= 1 or variance < variance_threshold:
                violations.append(create_violation(
                    code=ViolationCodes.ZERO_VARIANCE,
                    severity=Severity.ERROR,
                    message=f"Zero or near-zero variance in group '{g}' (var={variance:.2e})",
                    suggestion="Metric has no variability - check data collection or choose different metric",
                    context={
                        "group": g,
                        "variance": variance,
                        "unique_values": n_unique
                    },
                    check_name=self.name
                ))
        
        return violations


class NearZeroVarianceCheck(StatisticalCheck):
    """
    Detects near-zero variance (high percentage of same value).
    
    This is often a sign of data quality issues or imputed values.
    """

    @property
    def name(self) -> str:
        return "Near-Zero Variance"
    
    @property
    def description(self) -> str:
        return "Detects columns where one value dominates"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        near_zero_variance_ratio: float = 0.95,
        **kwargs
    ) -> List[Violation]:
        violations = []
        values = data[target_col].dropna()
        
        if len(values) == 0:
            return []
        
        # Calculate frequency of most common value
        value_counts = values.value_counts()
        most_common_freq = value_counts.iloc[0] / len(values)
        
        if most_common_freq > near_zero_variance_ratio:
            most_common_value = value_counts.index[0]
            violations.append(create_violation(
                code=ViolationCodes.NEAR_ZERO_VARIANCE,
                severity=Severity.WARNING,
                message=f"Near-zero variance: {most_common_freq:.1%} of values are {most_common_value}",
                suggestion="Check for imputed values, data quality issues, or consider removing this variable",
                context={
                    "dominant_value": most_common_value,
                    "frequency": most_common_freq,
                    "threshold": near_zero_variance_ratio
                },
                check_name=self.name
            ))
        
        return violations


class SkewnessCheck(StatisticalCheck):
    """Detects high skewness in distributions."""

    @property
    def name(self) -> str:
        return "Skewness"
    
    @property
    def description(self) -> str:
        return "Checks for asymmetric distributions"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_skewness: float = 2.0,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        for g, vals in groups.items():
            if len(vals) < 10:
                continue
            
            s = stats.skew(vals)
            
            if abs(s) > max_skewness:
                severity = Severity.ERROR if abs(s) > 4 else Severity.WARNING
                violations.append(create_violation(
                    code=ViolationCodes.HIGH_SKEWNESS,
                    severity=severity,
                    message=f"High skewness ({s:.2f}) in group '{g}'",
                    suggestion="Mean may be misleading; consider median, log-transform, or non-parametric tests",
                    context={
                        "group": g,
                        "skewness": s,
                        "threshold": max_skewness,
                        "direction": "right" if s > 0 else "left"
                    },
                    check_name=self.name
                ))
        
        return violations


class KurtosisCheck(StatisticalCheck):
    """Detects extreme kurtosis (heavy tails or light tails)."""

    @property
    def name(self) -> str:
        return "Kurtosis"
    
    @property
    def description(self) -> str:
        return "Checks for unusual tail behavior in distributions"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_kurtosis: float = 7.0,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        for g, vals in groups.items():
            if len(vals) < 20:
                continue
            
            k = stats.kurtosis(vals)
            
            if abs(k) > max_kurtosis:
                violations.append(create_violation(
                    code=ViolationCodes.HIGH_KURTOSIS,
                    severity=Severity.WARNING,
                    message=f"High kurtosis ({k:.2f}) in group '{g}'",
                    suggestion="Distribution has heavy tails; consider robust methods or outlier treatment",
                    context={
                        "group": g,
                        "kurtosis": k,
                        "threshold": max_kurtosis,
                        "interpretation": "heavy_tailed" if k > 0 else "light_tailed"
                    },
                    check_name=self.name
                ))
        
        return violations


class NormalityCheck(StatisticalCheck):
    """
    Tests for normality using Shapiro-Wilk test.
    
    For large samples (>5000), uses a subsample for performance.
    """

    @property
    def name(self) -> str:
        return "Normality"
    
    @property
    def description(self) -> str:
        return "Tests if data follows a normal distribution"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        normality_alpha: float = 0.05,
        min_shapiro_sample: int = 20,
        max_shapiro_sample: int = 5000,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        for g, vals in groups.items():

            vals_clean = pd.Series(vals).dropna()

            # FINAL HARD GUARD
            if len(vals_clean) < min_shapiro_sample:
                continue

            # HARD GUARD 1: constant values
            if vals_clean.nunique() <= 1:
                continue

            # HARD GUARD 2: near constant values
            if vals_clean.std() < 1e-8:
                continue

            # HARD GUARD 3: minimum unique threshold
            if vals_clean.nunique() < 4:
                continue
            
            # Sample AFTER guards
            if len(vals_clean) > max_shapiro_sample:
                vals_clean = vals_clean.sample(max_shapiro_sample, random_state=42)

            try:
                _, p_value = stats.shapiro(vals_clean)
            except Exception:
                continue            

            if p_value < normality_alpha:
                violations.append(create_violation(
                    code=ViolationCodes.NON_NORMAL,
                    severity=Severity.WARNING,
                    message=f"Non-normal distribution detected in group '{g}' (p={p_value:.4f})",
                    suggestion="Consider non-parametric tests or data transformation",
                    context={
                        "group": g,
                        "p_value": p_value,
                        "alpha": normality_alpha,
                        "sample_size": len(vals)
                    },
                    check_name=self.name
                ))
        
        return violations


class HeteroscedasticityCheck(StatisticalCheck):
    """
    Tests for heteroscedasticity (unequal variances) between groups.
    
    Uses Levene's test which is robust to non-normality.
    """

    @property
    def name(self) -> str:
        return "Heteroscedasticity"
    
    @property
    def description(self) -> str:
        return "Tests for unequal variances between groups"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> List[Violation]:
        if group_col is None:
            return []
        
        groups = self._groups(data, target_col, group_col)
        
        if len(groups) < 2:
            return []
        
        group_values = list(groups.values())
        
        if any(len(v) < 3 for v in group_values):
            return []
        
        try:
            _, p_value = stats.levene(*group_values)
        except Exception:
            return []
        
        if p_value < alpha:
            variances = {g: v.var() for g, v in groups.items()}
            return [create_violation(
                code=ViolationCodes.HETEROSCEDASTICITY,
                severity=Severity.WARNING,
                message=f"Heteroscedasticity detected (p={p_value:.4f})",
                suggestion="Consider Welch's t-test or variance-stabilizing transformation",
                context={
                    "p_value": p_value,
                    "group_variances": variances,
                    "variance_ratio": max(variances.values()) / min(variances.values())
                },
                check_name=self.name
            )]
        
        return []


class RangeCheck(StatisticalCheck):
    """
    Validates that values fall within expected ranges.
    
    Useful for detecting data entry errors or impossible values.
    """

    @property
    def name(self) -> str:
        return "Range Validation"
    
    @property
    def description(self) -> str:
        return "Checks if values are within expected ranges"
    
    @property
    def category(self) -> str:
        return "distribution"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        values = data[target_col].dropna()
        
        if min_value is not None:
            below_min = values < min_value
            if below_min.any():
                violations.append(create_violation(
                    code=ViolationCodes.SUSPICIOUS_PATTERN,
                    severity=Severity.ERROR,
                    message=f"{below_min.sum()} values below minimum ({min_value})",
                    suggestion="Check for data entry errors",
                    context={
                        "count": int(below_min.sum()),
                        "min_allowed": min_value,
                        "actual_min": values.min()
                    },
                    check_name=self.name
                ))
        
        if max_value is not None:
            above_max = values > max_value
            if above_max.any():
                violations.append(create_violation(
                    code=ViolationCodes.SUSPICIOUS_PATTERN,
                    severity=Severity.ERROR,
                    message=f"{above_max.sum()} values above maximum ({max_value})",
                    suggestion="Check for data entry errors",
                    context={
                        "count": int(above_max.sum()),
                        "max_allowed": max_value,
                        "actual_max": values.max()
                    },
                    check_name=self.name
                ))
        
        return violations
