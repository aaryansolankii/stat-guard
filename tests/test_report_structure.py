import pandas as pd

from stat_guard.api import validate


def test_report_grouped_by_check_name():
    data = pd.DataFrame({
        "metric": [1, 1, 1],
        "group": ["A", "A", "A"],
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
    )

    structured = report.as_dict()

    assert isinstance(structured, dict)
    assert len(structured) > 0
