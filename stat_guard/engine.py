"""
Validation engine for StatGuard.

Orchestrates all validation checks and generates reports.
"""

import time
from typing import Optional, Iterable, List, Dict, Any, Callable
import pandas as pd

from .policy import POLICIES, ValidationPolicy
from .report import ValidationReport
from .violations import Violation
from .checks.base import normalize_violations

# Import all checks
from .checks.sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
    CovariateBalanceCheck,
    StatisticalPowerCheck,
    EffectSizeCheck,
)
from .checks.distribution import (
    ZeroVarianceCheck,
    NearZeroVarianceCheck,
    SkewnessCheck,
    KurtosisCheck,
    NormalityCheck,
    HeteroscedasticityCheck,
    RangeCheck,
)
from .checks.unit_integrity import (
    UnitIntegrityCheck,
    DuplicateRowsCheck,
    MissingDataCheck,
    DataTypeCheck,
    ConstantColumnCheck,
)
from .checks.outliers import (
    OutlierCheck,
    ExtremeValueCheck,
    WinsorizationCheck,
)
from .checks.correlation import (
    CorrelationCheck,
    MulticollinearityCheck,
    TargetCorrelationCheck,
)
from .checks.cardinality import (
    CardinalityCheck,
    CategoricalBalanceCheck,
    HighCardinalityIDCheck,
)
from .checks.missing_data import (
    MissingPatternCheck,
    MissingTargetCheck,
    CompleteCaseAnalysisCheck,
)


class ValidationEngine:
    """
    Core engine that orchestrates statistical validation checks.
    
    The engine manages all validation checks, applies policies,
    and generates comprehensive reports.
    """

    # Default check registry
    DEFAULT_CHECKS = [
        # Sample size checks
        MinimumSampleSizeCheck(),
        BalancedGroupsCheck(),
        CovariateBalanceCheck(),
        StatisticalPowerCheck(),
        EffectSizeCheck(),
        
        # Distribution checks
        ZeroVarianceCheck(),
        NearZeroVarianceCheck(),
        SkewnessCheck(),
        KurtosisCheck(),
        NormalityCheck(),
        HeteroscedasticityCheck(),
        
        # Unit integrity checks
        UnitIntegrityCheck(),
        DuplicateRowsCheck(),
        MissingDataCheck(),
        DataTypeCheck(),
        ConstantColumnCheck(),
        
        # Outlier checks
        OutlierCheck(),
        ExtremeValueCheck(),
        WinsorizationCheck(),
        
        # Correlation checks
        CorrelationCheck(),
        MulticollinearityCheck(),
        TargetCorrelationCheck(),
        
        # Cardinality checks
        CardinalityCheck(),
        CategoricalBalanceCheck(),
        HighCardinalityIDCheck(),
        
        # Missing data checks
        MissingPatternCheck(),
        MissingTargetCheck(),
        CompleteCaseAnalysisCheck(),
    ]

    def __init__(
        self,
        checks: Optional[List] = None,
        verbose: bool = False
    ):
        """
        Initialize the validation engine.
        
        Args:
            checks: List of check instances (uses defaults if None)
            verbose: Whether to print progress
        """
        self.checks = checks or self.DEFAULT_CHECKS.copy()
        self.custom_checks: List = []
        self.verbose = verbose
        self._check_timings: Dict[str, float] = {}

    def register(self, check) -> "ValidationEngine":
        """
        Register a custom check.
        
        Args:
            check: Check instance to register
            
        Returns:
            Self for method chaining
        """
        self.custom_checks.append(check)
        return self

    def unregister(self, check_name: str) -> "ValidationEngine":
        """
        Unregister a check by name.
        
        Args:
            check_name: Name of check to remove
            
        Returns:
            Self for method chaining
        """
        self.checks = [c for c in self.checks if c.name != check_name]
        self.custom_checks = [c for c in self.custom_checks if c.name != check_name]
        return self

    def validate(
        self,
        data: pd.DataFrame,
        *,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        policy: str = "default",
        fail_fast: bool = False,
        include_summary_stats: bool = True,
    ) -> ValidationReport:
        """
        Run all validation checks and generate a report.
        
        Args:
            data: Input DataFrame
            target_col: Column being analyzed
            group_col: Optional grouping column
            unit_col: Optional unit identifier column
            policy: Policy name or ValidationPolicy instance
            fail_fast: Stop on first error
            include_summary_stats: Whether to compute summary statistics
            
        Returns:
            ValidationReport with all results
        """
        # Get policy configuration
        if isinstance(policy, str):
            if policy not in POLICIES:
                raise ValueError(
                    f"Unknown policy '{policy}'. "
                    f"Available: {list(POLICIES.keys())}"
                )
            cfg = POLICIES[policy].to_dict()
        else:
            cfg = policy.to_dict()
        
        # Initialize report
        report = ValidationReport()
        report.set_metadata(
            data_shape=data.shape,
            target_col=target_col,
            group_col=group_col,
            unit_col=unit_col,
            policy=policy if isinstance(policy, str) else "custom"
        )
        
        if include_summary_stats:
            summary_stats = self._compute_summary_stats(
                data, target_col, group_col
            )
            report.set_summary_stats(summary_stats)
        
        # Run all checks
        all_checks = self.checks + self.custom_checks
        
        for check in all_checks:
            check_start = time.time()
            
            if self.verbose:
                print(f"Running check: {check.name}...")
            
            try:
                result = check.run(
                    data=data,
                    target_col=target_col,
                    group_col=group_col,
                    unit_col=unit_col,
                    **cfg
                )
                
                violations = normalize_violations(result)
                
                for violation in violations:
                    report.add_violation(check.name, violation)
                    
                    if fail_fast and violation.severity.name in ["CRITICAL", "ERROR"]:
                        report.mark_check_complete(check.name, len(violations) == 0)
                        report.finalize()
                        return report
                
                report.mark_check_complete(check.name, len(violations) == 0)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Check failed: {e}")
                report.mark_check_complete(check.name, False)
            
            self._check_timings[check.name] = time.time() - check_start
        
        report.finalize()
        return report

    def validate_multiple(
        self,
        data: pd.DataFrame,
        target_cols: List[str],
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        policy: str = "default",
    ) -> Dict[str, ValidationReport]:
        """
        Validate multiple target columns.
        
        Args:
            data: Input DataFrame
            target_cols: List of target columns to validate
            group_col: Optional grouping column
            unit_col: Optional unit identifier column
            policy: Policy name
            
        Returns:
            Dictionary mapping column names to reports
        """
        reports = {}
        for col in target_cols:
            if col in data.columns:
                reports[col] = self.validate(
                    data=data,
                    target_col=col,
                    group_col=group_col,
                    unit_col=unit_col,
                    policy=policy
                )
        return reports

    def _compute_summary_stats(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute summary statistics for the target column."""
        from .profilers.statistics import compute_statistics, compute_group_statistics
        
        stats = {
            "target": compute_statistics(data[target_col])
        }
        
        if group_col and group_col in data.columns:
            stats["by_group"] = compute_group_statistics(
                data, target_col, group_col
            )
        
        return stats

    def get_check_timings(self) -> Dict[str, float]:
        """Get timing information for each check."""
        return self._check_timings.copy()

    def list_checks(self) -> List[str]:
        """Get list of all registered check names."""
        return [c.name for c in self.checks + self.custom_checks]

    def reset(self) -> "ValidationEngine":
        """Reset to default checks."""
        self.checks = self.DEFAULT_CHECKS.copy()
        self.custom_checks = []
        self._check_timings = {}
        return self


class DataValidator:
    """
    High-level validator with convenient methods.
    
    This class provides a simplified interface for common
    validation tasks.
    """

    def __init__(self, policy: str = "default"):
        """
        Initialize validator.
        
        Args:
            policy: Default policy to use
        """
        self.engine = ValidationEngine()
        self.default_policy = policy

    def check(
        self,
        data: pd.DataFrame,
        target_col: str,
        **kwargs
    ) -> ValidationReport:
        """
        Quick validation check.
        
        Args:
            data: Input DataFrame
            target_col: Target column
            **kwargs: Additional arguments for validate()
            
        Returns:
            ValidationReport
        """
        policy = kwargs.pop('policy', self.default_policy)
        return self.engine.validate(
            data=data,
            target_col=target_col,
            policy=policy,
            **kwargs
        )

    def is_valid(
        self,
        data: pd.DataFrame,
        target_col: str,
        **kwargs
    ) -> bool:
        """
        Quick validity check.
        
        Args:
            data: Input DataFrame
            target_col: Target column
            **kwargs: Additional arguments
            
        Returns:
            True if valid, False otherwise
        """
        report = self.check(data, target_col, **kwargs)
        return report.is_valid

    def assert_valid(
        self,
        data: pd.DataFrame,
        target_col: str,
        **kwargs
    ) -> None:
        """
        Assert that data is valid.
        
        Raises:
            ValidationError: If validation fails
        """
        report = self.check(data, target_col, **kwargs)
        if not report.is_valid:
            raise ValidationError(
                f"Validation failed with {len(report.errors)} errors"
            )


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass
