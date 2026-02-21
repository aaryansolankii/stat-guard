"""
Correlation and multicollinearity validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import StatisticalCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class CorrelationCheck(StatisticalCheck):
    """
    Detects high correlations between numeric columns.
    
    High correlations can indicate redundancy or multicollinearity issues.
    """

    @property
    def name(self) -> str:
        return "Correlation Analysis"
    
    @property
    def description(self) -> str:
        return "Detects high correlations between variables"
    
    @property
    def category(self) -> str:
        return "correlation"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_correlation: float = 0.95,
        correlation_method: str = "pearson",
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # FIX: remove constant columns
        safe_cols = []
        for col in numeric_cols:
            if data[col].std() > 1e-12:
                safe_cols.append(col)

        if len(safe_cols) < 2:
            return violations
        
        # Compute correlation matrix
        try:
            corr_matrix = data[safe_cols].corr(method=correlation_method)
        except Exception:
            return violations
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > max_correlation:
                    high_corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        # Report violations
        for pair in high_corr_pairs:
            severity = Severity.ERROR if abs(pair["correlation"]) > 0.99 else Severity.WARNING
            code = ViolationCodes.PERFECT_CORRELATION if abs(pair["correlation"]) > 0.99 else ViolationCodes.HIGH_CORRELATION
            
            violations.append(create_violation(
                code=code,
                severity=severity,
                message=f"High correlation ({pair['correlation']:.3f}) between '{pair['col1']}' and '{pair['col2']}'",
                suggestion="Consider removing one variable or using dimensionality reduction",
                context={
                    "column1": pair["col1"],
                    "column2": pair["col2"],
                    "correlation": pair["correlation"],
                    "method": correlation_method
                },
                check_name=self.name
            ))
        
        return violations


class MulticollinearityCheck(StatisticalCheck):
    """
    Detects multicollinearity using Variance Inflation Factor (VIF).
    
    VIF > 5 or 10 indicates problematic multicollinearity.
    """

    @property
    def name(self) -> str:
        return "Multicollinearity (VIF)"
    
    @property
    def description(self) -> str:
        return "Detects multicollinearity using Variance Inflation Factor"
    
    @property
    def category(self) -> str:
        return "correlation"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        vif_threshold: float = 5.0,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            return violations
        
        # Get numeric columns (excluding target)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if len(numeric_cols) < 2:
            return violations
        
        # Prepare data (handle missing values)
        X = data[numeric_cols].dropna()

        # FIX: remove constant columns
        X = X.loc[:, X.std() > 1e-12]

        if X.shape[1] < 2:
            return violations
                
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        except Exception:
            return violations
        
        # Find high VIF values
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]
        
        if len(high_vif) > 0:
            for _, row in high_vif.iterrows():
                severity = Severity.ERROR if row["VIF"] > 10 else Severity.WARNING
                violations.append(create_violation(
                    code=ViolationCodes.MULTICOLLINEARITY,
                    severity=severity,
                    message=f"High VIF ({row['VIF']:.2f}) for '{row['feature']}'",
                    suggestion="Consider removing or combining correlated predictors",
                    context={
                        "feature": row["feature"],
                        "vif": row["VIF"],
                        "threshold": vif_threshold
                    },
                    check_name=self.name
                ))
        
        return violations


class TargetCorrelationCheck(StatisticalCheck):
    """
    Analyzes correlations between features and target variable.
    
    Identifies features with very low predictive power.
    """

    @property
    def name(self) -> str:
        return "Target Correlation"
    
    @property
    def description(self) -> str:
        return "Analyzes feature-target correlations"
    
    @property
    def category(self) -> str:
        return "correlation"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        min_target_correlation: float = 0.01,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Get numeric columns (excluding target)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if len(numeric_cols) == 0:
            return violations
        
        low_corr_features = []
        
        for col in numeric_cols:
            try:
                if data[col].std() < 1e-12 or data[target_col].std() < 1e-12:
                    continue

                corr = data[col].corr(data[target_col])
                if pd.isna(corr):
                    continue
                if abs(corr) < min_target_correlation:
                    low_corr_features.append({
                        "feature": col,
                        "correlation": corr
                    })
            except Exception:
                continue
        
        if len(low_corr_features) > 0:
            violations.append(create_violation(
                code=ViolationCodes.HIGH_CORRELATION,
                severity=Severity.INFO,
                message=f"{len(low_corr_features)} features have very low correlation with target",
                suggestion="Consider feature selection or engineering",
                context={
                    "low_correlation_features": low_corr_features,
                    "threshold": min_target_correlation
                },
                check_name=self.name
            ))
        
        return violations


class CorrelationWithGroupCheck(StatisticalCheck):
    """
    Checks if correlations differ significantly between groups.
    
    This can indicate moderation effects or data quality issues.
    """

    @property
    def name(self) -> str:
        return "Group Correlation Differences"
    
    @property
    def description(self) -> str:
        return "Checks for correlation differences between groups"
    
    @property
    def category(self) -> str:
        return "correlation"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_correlation_diff: float = 0.3,
        **kwargs
    ) -> List[Violation]:
        if group_col is None:
            return []
        
        violations = []
        
        # Get numeric columns (excluding target and group)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in [target_col, group_col]]
        
        groups = data[group_col].unique()
        
        if len(groups) < 2:
            return []
        
        for col in numeric_cols:
            correlations = {}
            
            for g in groups:
                group_data = data[data[group_col] == g]
                if len(group_data) > 5:
                    try:
                        if group_data[col].std() < 1e-12 or group_data[target_col].std() < 1e-12:
                            continue

                        corr = group_data[col].corr(group_data[target_col])
                        if not pd.isna(corr):
                            correlations[g] = corr
                    except Exception:
                        continue
            
            if len(correlations) >= 2:
                corr_values = list(correlations.values())
                max_diff = max(corr_values) - min(corr_values)
                
                if max_diff > max_correlation_diff:
                    violations.append(create_violation(
                        code=ViolationCodes.HIGH_CORRELATION,
                        severity=Severity.INFO,
                        message=f"Large correlation difference ({max_diff:.3f}) for '{col}' between groups",
                        suggestion="Consider group-specific models or interaction terms",
                        context={
                            "feature": col,
                            "correlations_by_group": correlations,
                            "max_difference": max_diff
                        },
                        check_name=self.name
                    ))
        
        return violations
