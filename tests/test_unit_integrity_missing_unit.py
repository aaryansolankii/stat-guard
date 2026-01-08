import pandas as pd

from stat_guard.api import validate
from stat_guard.violations import Severity


def test_missing_unit_identifier_error():
    data = pd.DataFrame({
        "user_id": [1, 2, None, 4],
        "metric": [10, 12, 11, 13],
        "group": ["A", "A", "B", "B"],
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
        unit_col="user_id",
    )

    assert not report.is_valid
    assert any(v.severity == Severity.ERROR for v in report.errors)
