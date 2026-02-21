"""
Base classes for statistical validation checks.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from ..violations import Violation, Severity


class StatisticalCheck(ABC):
    """
    Abstract base class for all statistical validation checks.
    
    All checks must inherit from this class and implement the
    `name` property and `run` method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the check."""
        pass
    
    @property
    def description(self) -> str:
        """Brief description of what this check validates."""
        return "No description available"
    
    @property
    def category(self) -> str:
        """Category of the check (e.g., 'sample_size', 'distribution')."""
        return "general"

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        **policy
    ) -> Union[Violation, List[Violation], None]:
        """
        Execute the validation check.
        
        Args:
            data: Input DataFrame
            target_col: Column being analyzed
            group_col: Optional grouping column
            unit_col: Optional unit identifier column
            **policy: Policy parameters
            
        Returns:
            Violation(s) if issues found, None or empty list if check passes
        """
        pass

    def _groups(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Split data into groups for analysis.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            group_col: Optional grouping column
            
        Returns:
            Dictionary mapping group names to target values
        """
        if group_col is None:
            return {"all": data[target_col].dropna()}
        return {
            str(k): v[target_col].dropna()
            for k, v in data.groupby(group_col, observed=True)
        }
    
    def _safe_compute(
        self,
        func,
        default=None,
        error_msg: str = "Computation failed"
    ) -> Any:
        """
        Safely execute a computation with error handling.
        
        Args:
            func: Function to execute
            default: Default value on failure
            error_msg: Error message for logging
            
        Returns:
            Result of func or default value
        """
        try:
            return func()
        except Exception:
            return default


class ColumnCheck(ABC):
    """
    Base class for checks that operate on individual columns.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def run_column(
        self,
        series: pd.Series,
        col_name: str,
        **policy
    ) -> Union[Violation, List[Violation], None]:
        """
        Run check on a single column.
        
        Args:
            series: Column data
            col_name: Column name
            **policy: Policy parameters
            
        Returns:
            Violation(s) if issues found
        """
        pass
    
    def run_dataframe(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **policy
    ) -> List[Violation]:
        """
        Run check on all or specified columns of a DataFrame.
        
        Args:
            data: Input DataFrame
            columns: Optional list of columns to check
            **policy: Policy parameters
            
        Returns:
            List of all violations found
        """
        violations = []
        cols = columns or data.columns.tolist()
        
        for col in cols:
            if col in data.columns:
                result = self.run_column(data[col], col, **policy)
                if result:
                    if isinstance(result, list):
                        violations.extend(result)
                    else:
                        violations.append(result)
        
        return violations


class DataFrameCheck(ABC):
    """
    Base class for checks that operate on entire DataFrames.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        **policy
    ) -> Union[Violation, List[Violation], None]:
        """
        Run check on entire DataFrame.
        
        Args:
            data: Input DataFrame
            **policy: Policy parameters
            
        Returns:
            Violation(s) if issues found
        """
        pass


class CheckResult:
    """
    Standardized result container for checks.
    
    Provides a consistent interface for handling check results
    whether they pass, fail, or encounter errors.
    """
    
    def __init__(
        self,
        check_name: str,
        passed: bool,
        violations: Optional[List[Violation]] = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.check_name = check_name
        self.passed = passed
        self.violations = violations or []
        self.error = error
        self.duration_ms = duration_ms
        self.metadata = metadata or {}
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def has_errors(self) -> bool:
        return any(v.severity == Severity.ERROR for v in self.violations)
    
    @property
    def has_critical(self) -> bool:
        return any(v.severity == Severity.CRITICAL for v in self.violations)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


def normalize_violations(
    result: Union[Violation, List[Violation], None]
) -> List[Violation]:
    """
    Normalize various violation return types to a list.
    
    Args:
        result: Single violation, list of violations, or None
        
    Returns:
        Normalized list of violations
    """
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def create_violation(
    code: str,
    severity: Severity,
    message: str,
    suggestion: str,
    context: Optional[Dict[str, Any]] = None,
    check_name: str = ""
) -> Violation:
    """
    Factory function for creating violations.
    
    Args:
        code: Violation code
        severity: Severity level
        message: Description message
        suggestion: Recommended action
        context: Additional context
        check_name: Name of the check
        
    Returns:
        Violation instance
    """
    return Violation(
        code=code,
        severity=severity,
        message=message,
        suggestion=suggestion,
        context=context,
        check_name=check_name
    )
