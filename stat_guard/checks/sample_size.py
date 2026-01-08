from typing import Optional
import pandas as pd
from .base import StatisticalCheck
from ..violations import Violation, Severity, ViolationCodes
import numpy as np


class MinimumSampleSizeCheck(StatisticalCheck):

    @property
    def name(self):
        return "Minimum Sample Size"

    def run(self, data, target_col, group_col, min_sample_size, **_):
        groups = self._groups(data, target_col, group_col)
        small = {g: len(v) for g, v in groups.items() if len(v) < min_sample_size}

        if small:
            return Violation(
                code=ViolationCodes.SAMPLE_TOO_SMALL,
                severity=Severity.ERROR,
                message=f"Sample size below {min_sample_size}",
                suggestion="Collect more data or use non-parametric methods",
                context=small,
            )
        return None


class BalancedGroupsCheck(StatisticalCheck):

    @property
    def name(self):
        return "Balanced Groups"

    def run(self, data, target_col, group_col, max_imbalance_ratio, **_):
        if group_col is None:
            return None

        groups = self._groups(data, target_col, group_col)
        sizes = [len(v) for v in groups.values()]
        if min(sizes) == 0:
            return Violation(
                code=ViolationCodes.UNBALANCED_GROUPS,
                severity=Severity.ERROR,
                message="One or more groups have zero observations",
                suggestion="Fix group assignment or filtering",
            )

        ratio = max(sizes) / min(sizes)
        if ratio > max_imbalance_ratio:
            return Violation(
                code=ViolationCodes.UNBALANCED_GROUPS,
                severity=Severity.WARNING,
                message=f"Group imbalance ratio {ratio:.2f}",
                suggestion="Consider rebalancing or stratification",
                context={"ratio": ratio},
            )
        return None
    
class CovariateBalanceCheck(StatisticalCheck):

    @property
    def name(self):
        return "Covariate Balance (SMD)"

    def run(self, data, target_col, group_col, max_smd=0.25, **_):
        if group_col is None:
            return []

        groups = self._groups(data, target_col, group_col)
        if len(groups) != 2:
            return []

        (g1, x1), (g2, x2) = groups.items()
        pooled_std = np.sqrt((x1.var() + x2.var()) / 2)

        if pooled_std == 0:
            return []

        smd = abs(x1.mean() - x2.mean()) / pooled_std

        if smd > max_smd:
            return [
                Violation(
                    code=ViolationCodes.UNBALANCED_GROUPS,
                    severity=Severity.WARNING,
                    message=f"SMD imbalance detected ({smd:.2f})",
                    suggestion="Consider stratification or rebalancing",
                    context={"smd": smd},
                )
            ]

        return []
