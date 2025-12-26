import pandas as pd
import numpy as np

from stat_guard.checks.sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
)
from stat_guard.checks.distribution import (
    ZeroVarianceCheck,
    SkewnessCheck,
    NormalityCheck,
)
from stat_guard.violations import Severity, ViolationCodes


def test_minimum_sample_size_error():
    data = pd.DataFrame({
        "metric": [1, 2, 3],
        "group": ["A", "A", "A"],
    })

    check = MinimumSampleSizeCheck()
    v = check.run(
        data=data,
        target_col="metric",
        group_col="group",
        min_sample_size=10,
    )

    assert v is not None
    assert v.code == ViolationCodes.SAMPLE_TOO_SMALL
    assert v.severity == Severity.ERROR


def test_balanced_groups_warning():
    data = pd.DataFrame({
        "metric": np.random.randn(150),
        "group": ["A"] * 50 + ["B"] * 100,
    })

    check = BalancedGroupsCheck()
    v = check.run(
        data=data,
        target_col="metric",
        group_col="group",
        max_imbalance_ratio=1.5,
    )

    assert v is not None
    assert v.code == ViolationCodes.UNBALANCED_GROUPS
    assert v.severity == Severity.WARNING


def test_zero_variance_error():
    data = pd.DataFrame({
        "metric": [5, 5, 5, 5],
        "group": ["A"] * 4,
    })

    check = ZeroVarianceCheck()
    v = check.run(
        data=data,
        target_col="metric",
        group_col="group",
    )

    assert v is not None
    assert v.code == ViolationCodes.ZERO_VARIANCE
    assert v.severity == Severity.ERROR


def test_skewness_warning():
    data = pd.DataFrame({
        "metric": np.random.exponential(scale=1.0, size=200),
    })

    check = SkewnessCheck()
    v = check.run(
        data=data,
        target_col="metric",
        group_col=None,
        max_skewness=1.0,
    )

    assert v is not None
    assert v.code == ViolationCodes.HIGH_SKEWNESS
    assert v.severity == Severity.WARNING


def test_normality_warning():
    data = pd.DataFrame({
        "metric": np.random.exponential(scale=1.0, size=300),
    })

    check = NormalityCheck()
    v = check.run(
        data=data,
        target_col="metric",
        group_col=None,
    )

    assert v is not None
    assert v.code == ViolationCodes.NON_NORMAL
    assert v.severity == Severity.WARNING
