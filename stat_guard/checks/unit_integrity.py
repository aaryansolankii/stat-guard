"""
Unit integrity and data quality validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import StatisticalCheck, create_violation
from ..violations import Violation, Severity, ViolationCodes


class UnitIntegrityCheck(StatisticalCheck):
    """
    Validates unit-level data integrity.
    
    Checks for:
    - Missing unit identifiers
    - Duplicate unit identifiers
    - Units appearing in multiple groups (leakage)
    """

    @property
    def name(self) -> str:
        return "Unit Integrity"
    
    @property
    def description(self) -> str:
        return "Validates unit identifier consistency and prevents cross-group leakage"
    
    @property
    def category(self) -> str:
        return "integrity"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        if unit_col is None:
            return []
        
        violations = []
        
        # Check for missing unit IDs
        missing_mask = data[unit_col].isna()
        if missing_mask.any():
            violations.append(create_violation(
                code=ViolationCodes.MISSING_UNIT_ID,
                severity=Severity.ERROR,
                message=f"{missing_mask.sum()} missing unit identifiers detected",
                suggestion="Remove or fix null unit IDs before analysis",
                context={
                    "count": int(missing_mask.sum()),
                    "percentage": missing_mask.mean() * 100
                },
                check_name=self.name
            ))
        
        # Check for duplicate unit IDs
        duplicated = data[unit_col].duplicated(keep=False)
        if duplicated.any():
            dup_values = data.loc[duplicated, unit_col].unique()
            violations.append(create_violation(
                code=ViolationCodes.DUPLICATE_OBSERVATIONS,
                severity=Severity.ERROR,
                message=f"{len(dup_values)} duplicate unit identifiers detected",
                suggestion="Each unit must appear exactly once; aggregate or deduplicate",
                context={
                    "unique_duplicates": len(dup_values),
                    "total_duplicates": int(duplicated.sum()),
                    "examples": dup_values[:10].tolist()
                },
                check_name=self.name
            ))
        
        # Check for cross-group leakage
        if group_col is not None:
            leakage = (
                data.groupby(unit_col)[group_col]
                .nunique()
                .gt(1)
            )
            
            if leakage.any():
                leaking_units = leakage[leakage].index.tolist()
                violations.append(create_violation(
                    code=ViolationCodes.UNIT_LEAKAGE,
                    severity=Severity.ERROR,
                    message=f"{len(leaking_units)} units appear in multiple groups",
                    suggestion="Fix group assignment to prevent unit-level leakage",
                    context={
                        "leaking_units": leaking_units[:20],
                        "total_leaking": len(leaking_units)
                    },
                    check_name=self.name
                ))
        
        return violations


class DuplicateRowsCheck(StatisticalCheck):
    """
    Detects completely duplicate rows in the dataset.
    
    Duplicate rows can inflate sample size and bias results.
    """

    @property
    def name(self) -> str:
        return "Duplicate Rows"
    
    @property
    def description(self) -> str:
        return "Detects completely identical rows"
    
    @property
    def category(self) -> str:
        return "integrity"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Check for duplicate rows
        duplicated = data.duplicated(keep=False)
        
        if duplicated.any():
            n_duplicates = duplicated.sum()
            n_unique_duplicates = data[duplicated].drop_duplicates().shape[0]
            
            violations.append(create_violation(
                code=ViolationCodes.DUPLICATE_ROWS,
                severity=Severity.ERROR,
                message=f"{n_duplicates} duplicate rows detected ({n_unique_duplicates} unique patterns)",
                suggestion="Remove duplicate rows before analysis",
                context={
                    "total_duplicates": int(n_duplicates),
                    "unique_patterns": int(n_unique_duplicates),
                    "percentage": (n_duplicates / len(data)) * 100
                },
                check_name=self.name
            ))
        
        return violations


class MissingDataCheck(StatisticalCheck):
    """
    Comprehensive missing data analysis.
    
    Checks for:
    - Overall missing percentage
    - Column-level missing percentages
    - Missing patterns
    """

    @property
    def name(self) -> str:
        return "Missing Data"
    
    @property
    def description(self) -> str:
        return "Analyzes missing data patterns and completeness"
    
    @property
    def category(self) -> str:
        return "integrity"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_missing_pct: float = 0.05,
        max_missing_pct_column: float = 0.20,
        flag_missing_pattern: bool = True,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Overall missing percentage
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        overall_missing_pct = missing_cells / total_cells
        
        if overall_missing_pct > max_missing_pct:
            violations.append(create_violation(
                code=ViolationCodes.EXCESSIVE_MISSING,
                severity=Severity.ERROR,
                message=f"Overall missing data ({overall_missing_pct:.1%}) exceeds threshold ({max_missing_pct:.1%})",
                suggestion="Investigate missing data mechanism and consider imputation",
                context={
                    "overall_missing_pct": overall_missing_pct,
                    "threshold": max_missing_pct,
                    "total_missing": int(missing_cells)
                },
                check_name=self.name
            ))
        
        # Column-level missing
        col_missing = data.isna().mean()
        high_missing_cols = col_missing[col_missing > max_missing_pct_column]
        
        if len(high_missing_cols) > 0:
            violations.append(create_violation(
                code=ViolationCodes.EXCESSIVE_MISSING,
                severity=Severity.WARNING,
                message=f"{len(high_missing_cols)} columns have >{max_missing_pct_column:.0%} missing values",
                suggestion="Consider removing high-missing columns or using advanced imputation",
                context={
                    "columns": high_missing_cols.to_dict(),
                    "threshold": max_missing_pct_column
                },
                check_name=self.name
            ))
        
        # Check complete case ratio
        complete_cases = data.dropna().shape[0]
        complete_case_ratio = complete_cases / len(data)
        
        if complete_case_ratio < 0.5:
            violations.append(create_violation(
                code=ViolationCodes.COMPLETE_CASE_RATIO_LOW,
                severity=Severity.WARNING,
                message=f"Only {complete_case_ratio:.1%} of rows are complete cases",
                suggestion="Consider using methods that handle missing data (e.g., multiple imputation)",
                context={
                    "complete_cases": complete_cases,
                    "total_rows": len(data),
                    "ratio": complete_case_ratio
                },
                check_name=self.name
            ))
        
        return violations


class DataTypeCheck(StatisticalCheck):
    """
    Validates data types and detects suspicious type conversions.
    
    Flags columns that may have been incorrectly typed.
    """

    @property
    def name(self) -> str:
        return "Data Type Consistency"
    
    @property
    def description(self) -> str:
        return "Checks for consistent data types and suspicious conversions"
    
    @property
    def category(self) -> str:
        return "integrity"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Check target column type
        target_series = data[target_col]
        
        # Check if numeric column has many non-numeric strings
        if target_series.dtype == object:
            try:
                pd.to_numeric(target_series, errors='raise')
            except Exception:
                non_numeric = target_series.apply(
                    lambda x: not pd.isna(x) and not isinstance(x, (int, float))
                ).sum()
                
                if non_numeric > 0:
                    violations.append(create_violation(
                        code=ViolationCodes.INCONSISTENT_DATA_TYPES,
                        severity=Severity.ERROR,
                        message=f"Target column contains {non_numeric} non-numeric values",
                        suggestion="Convert to numeric or exclude non-numeric values",
                        context={
                            "non_numeric_count": int(non_numeric),
                            "dtype": str(target_series.dtype)
                        },
                        check_name=self.name
                    ))
        
        return violations


class ConstantColumnCheck(StatisticalCheck):
    """
    Detects columns with constant values.
    
    Constant columns provide no information and should be removed.
    """

    @property
    def name(self) -> str:
        return "Constant Columns"
    
    @property
    def description(self) -> str:
        return "Detects columns with only one unique value"
    
    @property
    def category(self) -> str:
        return "integrity"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **kwargs
    ) -> List[Violation]:
        violations = []
        
        # Check target column
        n_unique = data[target_col].nunique(dropna=False)
        
        if n_unique == 1:
            constant_value = data[target_col].iloc[0]
            violations.append(create_violation(
                code=ViolationCodes.CONSTANT_COLUMN,
                severity=Severity.ERROR,
                message=f"Target column is constant (value: {constant_value})",
                suggestion="Remove constant columns as they provide no information",
                context={
                    "constant_value": constant_value,
                    "column": target_col
                },
                check_name=self.name
            ))
        
        return violations
