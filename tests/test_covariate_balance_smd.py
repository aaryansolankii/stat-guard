import numpy as np
import pandas as pd

from stat_guard.api import validate
from stat_guard.violations import Severity


def test_smd_balance_warning_triggered():
    np.random.seed(0)

    data = pd.DataFrame({
        "metric": np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(2.0, 1, 100),
        ]),
        "group": ["A"] * 100 + ["B"] * 100,
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
    )

    assert any(v.severity == Severity.WARNING for v in report.warnings)
