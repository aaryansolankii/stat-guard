import pandas as pd
from typing import Optional, Iterable

from .policy import POLICIES
from .report import ValidationReport
from .checks.sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
    CovariateBalanceCheck,
)

from .checks.distribution import (
    ZeroVarianceCheck,
    SkewnessCheck,
    NormalityCheck,
)

from .checks.unit_integrity import UnitIntegrityCheck


class ValidationEngine:
    """Core engine that orchestrates statistical validation checks."""

    def __init__(self):
        self.checks = [
            MinimumSampleSizeCheck(),
            BalancedGroupsCheck(),
            CovariateBalanceCheck(),
            ZeroVarianceCheck(),
            SkewnessCheck(),
            NormalityCheck(),
        ]
        self.checks.insert(0, UnitIntegrityCheck())

        self.custom_checks = []

    def register(self, check):
        self.custom_checks.append(check)

    def _normalize(self, result) -> Iterable:
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]

    def validate(
        self,
        data: pd.DataFrame,
        *,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        policy: str = "default",
        fail_fast: bool = False,
    ) -> ValidationReport:

        if policy not in POLICIES:
            raise ValueError(f"Unknown policy '{policy}'")

        cfg = POLICIES[policy]
        report = ValidationReport()

        for check in [*self.checks, *self.custom_checks]:
            results = self._normalize(
                check.run(
                    data=data,
                    target_col=target_col,
                    group_col=group_col,
                    unit_col=unit_col,
                    **cfg,
                )
            )
            

            for violation in results:
                report.add_violation(check.name, violation)

                if fail_fast and violation.severity.name == "ERROR":
                    return report

        return report
