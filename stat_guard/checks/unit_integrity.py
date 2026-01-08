from .base import StatisticalCheck
from ..violations import Violation, Severity, ViolationCodes


class UnitIntegrityCheck(StatisticalCheck):

    @property
    def name(self):
        return "Unit Integrity"

    def run(self, data, target_col, group_col, unit_col=None, **_):
        if unit_col is None:
            return []

        violations = []

        if data[unit_col].isna().any():
            violations.append(
                Violation(
                    code=ViolationCodes.MISSING_DATA,
                    severity=Severity.ERROR,
                    message="Missing unit identifiers detected",
                    suggestion="Remove or fix null unit IDs",
                )
            )

        duplicated = data[unit_col].duplicated()
        if duplicated.any():
            violations.append(
                Violation(
                    code=ViolationCodes.DUPLICATE_OBSERVATIONS,
                    severity=Severity.ERROR,
                    message="Duplicate unit identifiers detected",
                    suggestion="Each unit must appear exactly once",
                    context={"count": int(duplicated.sum())},
                )
            )

        if group_col is not None:
            leakage = (
                data.groupby(unit_col)[group_col]
                .nunique()
                .gt(1)
            )

            if leakage.any():
                violations.append(
                    Violation(
                        code=ViolationCodes.UNBALANCED_GROUPS,
                        severity=Severity.ERROR,
                        message="Units appear in multiple groups",
                        suggestion="Fix group assignment leakage",
                        context={"units": leakage[leakage].index.tolist()},
                    )
                )

        return violations
