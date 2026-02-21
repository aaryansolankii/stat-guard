"""
Cardinality and categorical variable validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import StatisticalCheck, ColumnCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class CardinalityCheck(ColumnCheck):
    """
    Validates cardinality (number of unique values) in categorical columns.
    
    Flags:
    - High cardinality (potential ID columns)
    - Low cardinality (near-constant columns)
    - Rare categories
    """

    @property
    def name(self) -> str:
        return "Cardinality"
    
    @property
    def description(self) -> str:
        return "Checks for unusual cardinality in categorical variables"

    def run_column(
        self,
        series: pd.Series,
        col_name: str,
        max_cardinality_ratio: float = 0.95,
        min_cardinality_ratio: float = 0.01,
        rare_category_threshold: int = 5,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        n_total = len(series)
        n_unique = series.nunique(dropna=False)
        cardinality_ratio = n_unique / n_total
        
        # Check for high cardinality (potential ID column)
        if cardinality_ratio > max_cardinality_ratio:
            violations.append(create_violation(
                code=ViolationCodes.HIGH_CARDINALITY,
                severity=Severity.WARNING,
                message=f"High cardinality in '{col_name}' ({n_unique}/{n_total} = {cardinality_ratio:.1%})",
                suggestion="Column may be an identifier; consider excluding from analysis",
                context={
                    "column": col_name,
                    "unique_values": n_unique,
                    "total_rows": n_total,
                    "cardinality_ratio": cardinality_ratio,
                    "threshold": max_cardinality_ratio
                },
                check_name=self.name
            ))
        
        # Check for low cardinality
        if cardinality_ratio < min_cardinality_ratio and n_unique > 1:
            violations.append(create_violation(
                code=ViolationCodes.LOW_CARDINALITY,
                severity=Severity.INFO,
                message=f"Low cardinality in '{col_name}' ({n_unique} unique values)",
                suggestion="Consider if this variable provides enough information",
                context={
                    "column": col_name,
                    "unique_values": n_unique,
                    "cardinality_ratio": cardinality_ratio
                },
                check_name=self.name
            ))
        
        # Check for rare categories
        if n_unique > 1:
            value_counts = series.value_counts()
            rare_categories = value_counts[value_counts < rare_category_threshold]
            
            if len(rare_categories) > 0:
                violations.append(create_violation(
                    code=ViolationCodes.RARE_CATEGORIES,
                    severity=Severity.WARNING,
                    message=f"{len(rare_categories)} rare categories in '{col_name}' (<{rare_category_threshold} occurrences)",
                    suggestion="Consider combining rare categories or using regularization",
                    context={
                        "column": col_name,
                        "rare_category_count": len(rare_categories),
                        "threshold": rare_category_threshold,
                        "rare_categories": rare_categories.to_dict()
                    },
                    check_name=self.name
                ))
        
        return violations


class EmptyCategoryCheck(ColumnCheck):
    """
    Detects empty or all-null categories.
    """

    @property
    def name(self) -> str:
        return "Empty Categories"
    
    @property
    def description(self) -> str:
        return "Detects categories with no valid data"

    def run_column(
        self,
        series: pd.Series,
        col_name: str,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Check for all-null categories in group columns
        null_count = series.isna().sum()
        
        if null_count > 0 and null_count == len(series):
            violations.append(create_violation(
                code=ViolationCodes.EMPTY_CATEGORIES,
                severity=Severity.ERROR,
                message=f"Column '{col_name}' is entirely null",
                suggestion="Remove this column from analysis",
                context={
                    "column": col_name,
                    "null_count": null_count
                },
                check_name=self.name
            ))
        
        return violations


class CategoricalBalanceCheck(StatisticalCheck):
    """
    Checks balance of categorical variables across groups.
    
    Important for ensuring representative samples.
    """

    @property
    def name(self) -> str:
        return "Categorical Balance"
    
    @property
    def description(self) -> str:
        return "Checks if categorical distributions are balanced across groups"
    
    @property
    def category(self) -> str:
        return "cardinality"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_imbalance_ratio: float = 2.0,
        **kwargs
    ) -> List[Violation]:
        if group_col is None:
            return []
        
        violations = []
        
        # Get categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in [target_col, group_col]]
        
        for col in cat_cols:
            # Create crosstab
            try:
                crosstab = pd.crosstab(data[col], data[group_col])
            except Exception:
                continue
            
            # Check if any category is severely imbalanced across groups
            for category in crosstab.index:
                row = crosstab.loc[category]
                if row.min() > 0:  # Avoid division by zero
                    ratio = row.max() / row.min()
                    if ratio > max_imbalance_ratio:
                        violations.append(create_violation(
                            code=ViolationCodes.UNBALANCED_GROUPS,
                            severity=Severity.WARNING,
                            message=f"Category '{category}' in '{col}' is imbalanced across groups (ratio={ratio:.2f})",
                            suggestion="Check for sampling bias or stratification issues",
                            context={
                                "column": col,
                                "category": category,
                                "imbalance_ratio": ratio,
                                "distribution": row.to_dict()
                            },
                            check_name=self.name
                        ))
        
        return violations


class HighCardinalityIDCheck(StatisticalCheck):
    """
    Specifically checks for potential ID columns that should be excluded.
    
    ID columns have unique values for each row and provide no analytical value.
    """

    @property
    def name(self) -> str:
        return "ID Column Detection"
    
    @property
    def description(self) -> str:
        return "Detects potential ID columns that should be excluded from analysis"
    
    @property
    def category(self) -> str:
        return "cardinality"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        for col in data.columns:
            if col in [target_col, group_col, unit_col]:
                continue
            
            n_unique = data[col].nunique(dropna=False)
            n_total = len(data)
            
            # Check if column is likely an ID
            if n_unique == n_total:
                # Additional check: are values sequential or random?
                sample_values = data[col].dropna().head(100)
                
                # Check for common ID patterns
                is_likely_id = (
                    col.lower().endswith(('_id', 'id', '_key', 'key')) or
                    col.lower() in ['uuid', 'guid', 'index', 'row_number'] or
                    self._is_sequential(sample_values)
                )
                
                if is_likely_id:
                    violations.append(create_violation(
                        code=ViolationCodes.HIGH_CARDINALITY,
                        severity=Severity.WARNING,
                        message=f"Column '{col}' appears to be an ID column (all unique values)",
                        suggestion="Exclude ID columns from statistical analysis",
                        context={
                            "column": col,
                            "unique_values": n_unique,
                            "pattern": "sequential" if self._is_sequential(sample_values) else "random"
                        },
                        check_name=self.name
                    ))
        
        return violations
    
    def _is_sequential(self, values: pd.Series) -> bool:
        """Check if values appear to be sequential numbers."""
        try:
            numeric_values = pd.to_numeric(values, errors='coerce').dropna()
            if len(numeric_values) < 2:
                return False
            
            diffs = numeric_values.diff().dropna()
            return (diffs == 1).all() or (diffs.nunique() == 1 and diffs.iloc[0] == 1)
        except Exception:
            return False
