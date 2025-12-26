import pandas as pd
from typing import Optional

from .policy import POLICIES
from .report import ValidationReport
from .checks.sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
)
from .checks.distribution import (
    ZeroVarianceCheck,
    SkewnessCheck,
    NormalityCheck,
)


class ValidationEngine:
    """Core engine that orchestrates statistical validation checks."""

    def __init__(self):
        self.checks = [
            MinimumSampleSizeCheck(),
            BalancedGroupsCheck(),
            ZeroVarianceCheck(),
            SkewnessCheck(),
            NormalityCheck(),
        ]

    def validate(
        self,
        data: pd.DataFrame,
        *,
        target_col: str,
        group_col: Optional[str] = None,
        policy: str = "default",
        fail_fast: bool = False,
    ) -> ValidationReport:
        """Run validation checks using the specified policy."""

        if policy not in POLICIES:
            raise ValueError(f"Unknown policy '{policy}'")

        cfg = POLICIES[policy]
        report = ValidationReport()

        for check in self.checks:
            violation = check.run(
                data=data,
                target_col=target_col,
                group_col=group_col,
                **cfg,
            )

            if violation:
                report.add_violation(violation)

                if fail_fast and violation.severity.name == "ERROR":
                    break

        return report
