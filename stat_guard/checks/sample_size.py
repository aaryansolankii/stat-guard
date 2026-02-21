"""
Sample size and statistical power validation checks.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats

from .base import StatisticalCheck, normalize_violations, create_violation
from ..violations import Violation, Severity, ViolationCodes


class MinimumSampleSizeCheck(StatisticalCheck):
    """Validates that sample sizes meet minimum requirements."""

    @property
    def name(self) -> str:
        return "Minimum Sample Size"
    
    @property
    def description(self) -> str:
        return "Checks if groups have sufficient observations for reliable analysis"
    
    @property
    def category(self) -> str:
        return "sample_size"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        min_sample_size: int = 30,
        min_sample_size_per_group: int = 15,
        **kwargs
    ) -> List[Violation]:
        violations = []
        groups = self._groups(data, target_col, group_col)
        
        # Check overall sample size
        total_size = len(data[target_col].dropna())
        if total_size < min_sample_size:
            violations.append(create_violation(
                code=ViolationCodes.SAMPLE_TOO_SMALL,
                severity=Severity.ERROR,
                message=f"Total sample size ({total_size}) below minimum ({min_sample_size})",
                suggestion="Collect more data or use non-parametric methods",
                context={"actual": total_size, "required": min_sample_size},
                check_name=self.name
            ))
        
        # Check per-group sample sizes
        small_groups = {
            g: len(v) for g, v in groups.items()
            if len(v) < min_sample_size_per_group
        }
        
        if small_groups:
            violations.append(create_violation(
                code=ViolationCodes.SAMPLE_TOO_SMALL,
                severity=Severity.WARNING,
                message=f"Some groups have fewer than {min_sample_size_per_group} observations",
                suggestion="Consider combining groups or collecting more data",
                context=small_groups,
                check_name=self.name
            ))
        
        return violations


class BalancedGroupsCheck(StatisticalCheck):
    """Validates that groups are reasonably balanced in size."""

    @property
    def name(self) -> str:
        return "Balanced Groups"
    
    @property
    def description(self) -> str:
        return "Checks for significant imbalance between group sizes"
    
    @property
    def category(self) -> str:
        return "sample_size"

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
        groups = self._groups(data, target_col, group_col)
        
        if len(groups) < 2:
            return []
        
        sizes = [len(v) for v in groups.values()]
        
        # Check for empty groups
        if min(sizes) == 0:
            empty_groups = [g for g, v in groups.items() if len(v) == 0]
            violations.append(create_violation(
                code=ViolationCodes.UNBALANCED_GROUPS,
                severity=Severity.ERROR,
                message=f"Empty groups detected: {empty_groups}",
                suggestion="Fix group assignment or filtering",
                context={"empty_groups": empty_groups},
                check_name=self.name
            ))
            return violations
        
        # Check imbalance ratio
        ratio = max(sizes) / min(sizes)
        if ratio > max_imbalance_ratio:
            violations.append(create_violation(
                code=ViolationCodes.UNBALANCED_GROUPS,
                severity=Severity.WARNING,
                message=f"Group imbalance ratio {ratio:.2f} exceeds threshold {max_imbalance_ratio}",
                suggestion="Consider rebalancing, stratification, or weighted analysis",
                context={
                    "ratio": ratio,
                    "threshold": max_imbalance_ratio,
                    "group_sizes": {g: len(v) for g, v in groups.items()}
                },
                check_name=self.name
            ))
        
        return violations


class CovariateBalanceCheck(StatisticalCheck):
    """
    Validates covariate balance between groups using Standardized Mean Difference (SMD).
    
    SMD < 0.25 is generally considered acceptable.
    SMD < 0.10 is considered well-balanced.
    """

    @property
    def name(self) -> str:
        return "Covariate Balance (SMD)"
    
    @property
    def description(self) -> str:
        return "Checks for covariate imbalance between groups using standardized mean difference"
    
    @property
    def category(self) -> str:
        return "sample_size"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        max_smd: float = 0.25,
        **kwargs
    ) -> List[Violation]:
        if group_col is None:
            return []
        
        groups = self._groups(data, target_col, group_col)
        
        if len(groups) != 2:
            return []
        
        (g1_name, x1), (g2_name, x2) = list(groups.items())
        
        # Calculate pooled standard deviation
        var1 = x1.var()
        var2 = x2.var()
        
        if pd.isna(var1) or pd.isna(var2):
            return []
        
        pooled_std = np.sqrt((var1 + var2) / 2)
        
        if pooled_std == 0:
            return []
        
        mean1 = x1.mean()
        mean2 = x2.mean()
        smd = abs(mean1 - mean2) / pooled_std
        
        if smd > max_smd:
            severity = Severity.ERROR if smd > 0.5 else Severity.WARNING
            return [create_violation(
                code=ViolationCodes.COVARIATE_IMBALANCE,
                severity=severity,
                message=f"SMD imbalance detected ({smd:.3f}) between groups '{g1_name}' and '{g2_name}'",
                suggestion="Consider stratification, matching, or rebalancing",
                context={
                    "smd": smd,
                    "threshold": max_smd,
                    "group1_mean": mean1,
                    "group2_mean": mean2,
                    "group1": g1_name,
                    "group2": g2_name,
                },
                check_name=self.name
            )]
        
        return []


class StatisticalPowerCheck(StatisticalCheck):
    """
    Estimates statistical power for detecting a meaningful effect.
    
    Uses power analysis to determine if the sample size is sufficient
    to detect a specified effect size with desired power.
    """

    @property
    def name(self) -> str:
        return "Statistical Power"
    
    @property
    def description(self) -> str:
        return "Estimates if sample size provides adequate statistical power"
    
    @property
    def category(self) -> str:
        return "sample_size"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        min_power: float = 0.80,
        effect_size: Optional[float] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> List[Violation]:
        try:
            from statsmodels.stats.power import tt_ind_solve_power
        except ImportError:
            return []
        
        if group_col is None:
            return []
        
        groups = self._groups(data, target_col, group_col)
        
        if len(groups) != 2:
            return []
        
        (g1_name, x1), (g2_name, x2) = list(groups.items())
        n1, n2 = len(x1), len(x2)
        
        # Use Cohen's d if effect size not specified
        if effect_size is None:
            pooled_std = np.sqrt((x1.var() + x2.var()) / 2)
            if pooled_std > 0:
                effect_size = abs(x1.mean() - x2.mean()) / pooled_std
            else:
                effect_size = 0.2  # Default small effect
        
        # Average sample size
        nobs = (n1 + n2) / 2
        
        try:
            power = tt_ind_solve_power(
                effect_size=effect_size,
                nobs1=nobs,
                alpha=alpha,
                ratio=n2/n1 if n1 > 0 else 1
            )
        except Exception:
            return []
        
        if power < min_power:
            return [create_violation(
                code=ViolationCodes.INSUFFICIENT_POWER,
                severity=Severity.WARNING,
                message=f"Statistical power ({power:.2f}) below threshold ({min_power})",
                suggestion="Increase sample size or accept higher Type II error rate",
                context={
                    "power": power,
                    "required_power": min_power,
                    "effect_size": effect_size,
                    "sample_size": int(nobs),
                },
                check_name=self.name
            )]
        
        return []


class EffectSizeCheck(StatisticalCheck):
    """
    Checks if observed effect size is meaningful.
    
    Flags very small effect sizes that may not be practically significant.
    """

    @property
    def name(self) -> str:
        return "Effect Size"
    
    @property
    def description(self) -> str:
        return "Evaluates if observed effect size is practically meaningful"
    
    @property
    def category(self) -> str:
        return "sample_size"

    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        min_effect_size: float = 0.1,
        **kwargs
    ) -> List[Violation]:
        if group_col is None:
            return []
        
        groups = self._groups(data, target_col, group_col)
        
        if len(groups) != 2:
            return []
        
        (g1_name, x1), (g2_name, x2) = list(groups.items())
        
        pooled_std = np.sqrt((x1.var() + x2.var()) / 2)
        
        if pooled_std == 0:
            return []
        
        cohens_d = abs(x1.mean() - x2.mean()) / pooled_std
        
        if cohens_d < min_effect_size:
            return [create_violation(
                code=ViolationCodes.COVARIATE_IMBALANCE,
                severity=Severity.INFO,
                message=f"Effect size (Cohen's d = {cohens_d:.3f}) is very small",
                suggestion="Consider if this effect is practically meaningful",
                context={
                    "cohens_d": cohens_d,
                    "threshold": min_effect_size,
                    "interpretation": self._interpret_cohens_d(cohens_d)
                },
                check_name=self.name
            )]
        
        return []
    
    def _interpret_cohens_d(self, d: float) -> str:
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
