"""
Outlier detection and validation checks.
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from .base import StatisticalCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class OutlierCheck(StatisticalCheck):
    """
    Detects outliers using multiple methods.
    
    Supports IQR, Z-score, and MAD (Median Absolute Deviation) methods.
    """

    @property
    def name(self) -> str:
        return "Outlier Detection"
    
    @property
    def description(self) -> str:
        return "Detects extreme values that may indicate errors or anomalies"
    
    @property
    def category(self) -> str:
        return "outliers"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        max_outlier_pct: float = 0.05,
        flag_outlier_clusters: bool = True,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        for g, vals in groups.items():
            if len(vals) < 10:
                continue
            
            # Detect outliers
            outlier_mask = self._detect_outliers(
                vals, method=outlier_method, threshold=outlier_threshold
            )
            
            outlier_pct = outlier_mask.mean()
            
            if outlier_pct > max_outlier_pct:
                severity = Severity.ERROR if outlier_pct > 0.15 else Severity.WARNING
                outlier_values = vals[outlier_mask]
                
                violations.append(create_violation(
                    code=ViolationCodes.EXTREME_OUTLIERS,
                    severity=severity,
                    message=f"High outlier percentage ({outlier_pct:.1%}) in group '{g}'",
                    suggestion="Review outliers for data errors or consider robust methods",
                    context={
                        "group": g,
                        "outlier_count": int(outlier_mask.sum()),
                        "outlier_percentage": outlier_pct * 100,
                        "threshold_percentage": max_outlier_pct * 100,
                        "method": outlier_method,
                        "outlier_range": {
                            "min": outlier_values.min(),
                            "max": outlier_values.max()
                        }
                    },
                    check_name=self.name
                ))
            elif outlier_pct > 0:
                # Moderate outliers - info level
                violations.append(create_violation(
                    code=ViolationCodes.MODERATE_OUTLIERS,
                    severity=Severity.INFO,
                    message=f"Moderate outliers detected ({outlier_pct:.1%}) in group '{g}'",
                    suggestion="Review if outliers are valid data points",
                    context={
                        "group": g,
                        "outlier_count": int(outlier_mask.sum()),
                        "outlier_percentage": outlier_pct * 100
                    },
                    check_name=self.name
                ))
            
            # Check for outlier clusters
            if flag_outlier_clusters and outlier_pct > 0:
                cluster_violations = self._check_outlier_clusters(vals, outlier_mask, g)
                violations.extend(cluster_violations)
        
        return violations
    
    def _detect_outliers(
        self,
        values: pd.Series,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.Series:
        """Detect outliers using specified method."""
        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (values < lower) | (values > upper)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            return z_scores > threshold
        
        elif method == "mad":
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad == 0:
                return pd.Series(False, index=values.index)
            modified_z = 0.6745 * (values - median) / mad
            return np.abs(modified_z) > threshold
        
        else:
            raise ValueError(f"Unknown outlier method: {method}")
    
    def _check_outlier_clusters(
        self,
        values: pd.Series,
        outlier_mask: pd.Series,
        group_name: str
    ) -> List[Violation]:
        """Check if outliers form suspicious clusters."""
        violations = []
        outlier_values = values[outlier_mask]
        
        if len(outlier_values) < 5:
            return violations
        
        # Check if outliers are all on one side
        lower_outliers = (outlier_values < values.median()).sum()
        upper_outliers = len(outlier_values) - lower_outliers
        
        if lower_outliers == 0 or upper_outliers == 0:
            side = "upper" if upper_outliers > 0 else "lower"
            violations.append(create_violation(
                code=ViolationCodes.OUTLIER_CLUSTER,
                severity=Severity.WARNING,
                message=f"All outliers are on the {side} side in group '{group_name}'",
                suggestion="Check for one-sided data collection issues or censoring",
                context={
                    "group": group_name,
                    "side": side,
                    "outlier_count": len(outlier_values)
                },
                check_name=self.name
            ))
        
        return violations


class ExtremeValueCheck(StatisticalCheck):
    """
    Detects extreme values based on domain-specific thresholds.
    
    Useful for catching impossible values (e.g., negative ages).
    """

    @property
    def name(self) -> str:
        return "Extreme Values"
    
    @property
    def description(self) -> str:
        return "Validates values against domain-specific thresholds"
    
    @property
    def category(self) -> str:
        return "outliers"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        values = data[target_col].dropna()
        
        if lower_bound is not None:
            below_bound = values < lower_bound
            if below_bound.any():
                violations.append(create_violation(
                    code=ViolationCodes.EXTREME_OUTLIERS,
                    severity=Severity.ERROR,
                    message=f"{below_bound.sum()} values below lower bound ({lower_bound})",
                    suggestion="Check for data entry errors or incorrect units",
                    context={
                        "count": int(below_bound.sum()),
                        "lower_bound": lower_bound,
                        "min_value": values.min(),
                        "examples": values[below_bound].head(5).tolist()
                    },
                    check_name=self.name
                ))
        
        if upper_bound is not None:
            above_bound = values > upper_bound
            if above_bound.any():
                violations.append(create_violation(
                    code=ViolationCodes.EXTREME_OUTLIERS,
                    severity=Severity.ERROR,
                    message=f"{above_bound.sum()} values above upper bound ({upper_bound})",
                    suggestion="Check for data entry errors or incorrect units",
                    context={
                        "count": int(above_bound.sum()),
                        "upper_bound": upper_bound,
                        "max_value": values.max(),
                        "examples": values[above_bound].head(5).tolist()
                    },
                    check_name=self.name
                ))
        
        return violations


class WinsorizationCheck(StatisticalCheck):
    """
    Recommends winsorization if extreme values are present.
    
    This is an informational check that suggests data treatment.
    """

    @property
    def name(self) -> str:
        return "Winsorization Recommendation"
    
    @property
    def description(self) -> str:
        return "Suggests winsorization for datasets with extreme values"
    
    @property
    def category(self) -> str:
        return "outliers"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        winsorize_threshold: float = 0.01,
        **kwargs
    ) -> List[Violation]:
        violations = []
        values = data[target_col].dropna()
        
        if len(values) < 100:
            return violations
        
        # Check if extreme percentiles have extreme values
        lower_pctile = values.quantile(winsorize_threshold)
        upper_pctile = values.quantile(1 - winsorize_threshold)
        
        iqr = values.quantile(0.75) - values.quantile(0.25)
        
        if iqr > 0:
            lower_extreme = (values < lower_pctile).sum()
            upper_extreme = (values > upper_pctile).sum()
            
            if lower_extreme > 0 or upper_extreme > 0:
                violations.append(create_violation(
                    code=ViolationCodes.MODERATE_OUTLIERS,
                    severity=Severity.INFO,
                    message=f"Consider winsorization at {winsorize_threshold:.1%} level",
                    suggestion="Winsorization can reduce the impact of extreme values",
                    context={
                        "lower_extreme_count": int(lower_extreme),
                        "upper_extreme_count": int(upper_extreme),
                        "winsorize_threshold": winsorize_threshold,
                        "lower_value": lower_pctile,
                        "upper_value": upper_pctile
                    },
                    check_name=self.name
                ))
        
        return violations
