import pandas as pd

from stat_guard.api import validate, register_validator
from stat_guard.violations import Violation, Severity


class DummyCustomValidator:
    name = "Dummy Custom Validator"

    def run(self, data, target_col, group_col, **_):
        return [
            Violation(
                code="SG999",
                severity=Severity.WARNING,
                message="Custom validator executed",
                suggestion="This is a test warning",
            )
        ]


def test_custom_validator_is_executed():
    register_validator(DummyCustomValidator())

    data = pd.DataFrame({
        "metric": [1, 2, 3, 4, 5],
        "group": ["A"] * 5,
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
    )

    assert any(v.code == "SG999" for v in report.warnings)
