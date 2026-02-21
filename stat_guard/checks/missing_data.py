"""
Missing data pattern analysis and validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats

from .base import StatisticalCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class MissingPatternCheck(StatisticalCheck):
    """
    Analyzes patterns in missing data.
    
    Detects:
    - Missing completely at random (MCAR)
    - Missing not at random (MNAR) patterns
    - Systematic missingness by group
    """

    @property
    def name(self) -> str:
        return "Missing Pattern Analysis"
    
    @property
    def description(self) -> str:
        return "Analyzes patterns and mechanisms of missing data"
    
    @property
    def category(self) -> str:
        return "missing_data"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Check for systematic missingness by group
        if group_col is not None:
            group_missing = data.groupby(group_col, observed=True).apply(
                lambda x: x.isna().mean().mean(),
                include_groups=False
            )
            
            if group_missing.std() > 0.05:  # Significant variation
                violations.append(create_violation(
                    code=ViolationCodes.MISSING_NOT_AT_RANDOM,
                    severity=Severity.WARNING,
                    message="Missing data rates vary significantly across groups",
                    suggestion="Investigate if missingness is related to group assignment",
                    context={
                        "missing_by_group": group_missing.to_dict(),
                        "std": group_missing.std()
                    },
                    check_name=self.name
                ))
        
        # Check for column-wise missing patterns
        col_missing = data.isna().mean()
        high_missing = col_missing[col_missing > 0.1]
        
        if len(high_missing) > 0:
            # Check if missing patterns are correlated
            missing_matrix = data[high_missing.index].isna()
            
            if len(high_missing) >= 2:
                # Check for columns that are always missing together
                for i in range(len(high_missing)):
                    for j in range(i + 1, len(high_missing)):
                        col1 = high_missing.index[i]
                        col2 = high_missing.index[j]
                        
                        # Check if missing patterns are identical
                        if (missing_matrix[col1] == missing_matrix[col2]).all():
                            violations.append(create_violation(
                                code=ViolationCodes.MISSING_PATTERN,
                                severity=Severity.INFO,
                                message=f"Columns '{col1}' and '{col2}' have identical missing patterns",
                                suggestion="These columns may be derived from the same source",
                                context={
                                    "column1": col1,
                                    "column2": col2
                                },
                                check_name=self.name
                            ))
        
        return violations


class MissingTargetCheck(StatisticalCheck):
    """
    Specifically checks for missing values in the target column.
    
    Missing target values cannot be used for analysis.
    """

    @property
    def name(self) -> str:
        return "Target Missingness"
    
    @property
    def description(self) -> str:
        return "Checks for missing values in the target variable"
    
    @property
    def category(self) -> str:
        return "missing_data"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        target_missing = data[target_col].isna()
        missing_count = target_missing.sum()
        
        if missing_count > 0:
            missing_pct = missing_count / len(data)
            
            severity = Severity.ERROR if missing_pct > 0.1 else Severity.WARNING
            
            violations.append(create_violation(
                code=ViolationCodes.EXCESSIVE_MISSING,
                severity=severity,
                message=f"Target column has {missing_count} missing values ({missing_pct:.1%})",
                suggestion="Remove rows with missing targets or use imputation carefully",
                context={
                    "missing_count": int(missing_count),
                    "missing_percentage": missing_pct * 100,
                    "total_rows": len(data)
                },
                check_name=self.name
            ))
        
        return violations


class MissingByFeatureCheck(StatisticalCheck):
    """
    Analyzes if missingness in one feature is related to values in another.
    
    This can indicate MNAR (Missing Not At Random) mechanisms.
    """

    @property
    def name(self) -> str:
        return "Missing-Feature Relationship"
    
    @property
    def description(self) -> str:
        return "Checks if missingness is related to other feature values"
    
    @property
    def category(self) -> str:
        return "missing_data"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in data.columns:
            if col == target_col:
                continue
            
            missing_mask = data[col].isna()
            
            if missing_mask.sum() < 5 or missing_mask.sum() > len(data) * 0.9:
                continue
            
            # Check if missingness is related to target values
            if target_col in numeric_cols:
                target_when_missing = data.loc[missing_mask, target_col].mean()
                target_when_present = data.loc[~missing_mask, target_col].mean()
                
                # Simple t-test for difference
                try:
                    missing_values = data.loc[missing_mask, target_col].dropna()
                    present_values = data.loc[~missing_mask, target_col].dropna()
                    
                    if len(missing_values) > 5 and len(present_values) > 5:
                        _, p_value = stats.ttest_ind(missing_values, present_values)
                        
                        if p_value < 0.05:
                            violations.append(create_violation(
                                code=ViolationCodes.MISSING_NOT_AT_RANDOM,
                                severity=Severity.INFO,
                                message=f"Missingness in '{col}' is related to target values (p={p_value:.4f})",
                                suggestion="Consider MNAR mechanisms in your analysis",
                                context={
                                    "column": col,
                                    "p_value": p_value,
                                    "target_mean_when_missing": target_when_missing,
                                    "target_mean_when_present": target_when_present
                                },
                                check_name=self.name
                            ))
                except Exception:
                    continue
        
        return violations


class CompleteCaseAnalysisCheck(StatisticalCheck):
    """
    Evaluates the feasibility of complete case analysis.
    
    Warns if too many cases would be excluded.
    """

    @property
    def name(self) -> str:
        return "Complete Case Analysis"
    
    @property
    def description(self) -> str:
        return "Evaluates impact of listwise deletion"
    
    @property
    def category(self) -> str:
        return "missing_data"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Calculate complete cases
        complete_cases = data.dropna().shape[0]
        total_cases = len(data)
        complete_case_ratio = complete_cases / total_cases
        
        cases_lost = total_cases - complete_cases
        
        if complete_case_ratio < 0.7:
            violations.append(create_violation(
                code=ViolationCodes.COMPLETE_CASE_RATIO_LOW,
                severity=Severity.ERROR,
                message=f"Complete case analysis would lose {cases_lost} cases ({(1-complete_case_ratio):.1%})",
                suggestion="Use multiple imputation or full information maximum likelihood",
                context={
                    "complete_cases": complete_cases,
                    "total_cases": total_cases,
                    "cases_lost": cases_lost,
                    "retention_rate": complete_case_ratio
                },
                check_name=self.name
            ))
        elif complete_case_ratio < 0.9:
            violations.append(create_violation(
                code=ViolationCodes.COMPLETE_CASE_RATIO_LOW,
                severity=Severity.WARNING,
                message=f"Complete case analysis would lose {cases_lost} cases ({(1-complete_case_ratio):.1%})",
                suggestion="Consider imputation methods to retain more data",
                context={
                    "complete_cases": complete_cases,
                    "total_cases": total_cases,
                    "cases_lost": cases_lost,
                    "retention_rate": complete_case_ratio
                },
                check_name=self.name
            ))
        
        return violations
