import pandas as pd

from stat_guard.api import validate


def test_multiple_validation_failures_are_all_reported():
    data = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "metric": [5, 5, 5, 5],
        "group": ["A", "A", "A", "A"],
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
        unit_col="user_id",
    )

    assert not report.is_valid
    assert len(report.errors) >= 2
