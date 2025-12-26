import pandas as pd
import numpy as np

from stat_guard.api import validate
from stat_guard.violations import Severity


def test_valid_experiment_passes():
    np.random.seed(0)

    data = pd.DataFrame({
        "metric": np.concatenate([
            np.random.normal(100, 10, 100),
            np.random.normal(105, 10, 100),
        ]),
        "group": ["A"] * 100 + ["B"] * 100,
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
    )

    assert report.is_valid
    assert len(report.errors) == 0


def test_small_sample_fails():
    data = pd.DataFrame({
        "metric": [1, 2, 3],
        "group": ["A", "A", "A"],
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
    )

    assert not report.is_valid
    assert any(v.severity == Severity.ERROR for v in report.violations)


def test_fail_fast_stops_early():
    data = pd.DataFrame({
        "metric": [5, 5, 5, 5],
        "group": ["A"] * 4,
    })

    report = validate(
        data,
        target_col="metric",
        group_col="group",
        fail_fast=True,
    )

    assert not report.is_valid
    assert len(report.violations) == 1  # fail-fast enforced


def test_strict_policy_is_harder():
    data = pd.DataFrame({
        "metric": list(range(40)),
        "group": ["A"] * 40,
    })

    report_default = validate(
        data,
        target_col="metric",
        group_col="group",
        policy="default",
    )

    report_strict = validate(
        data,
        target_col="metric",
        group_col="group",
        policy="strict",
    )

    assert report_default.is_valid
    assert not report_strict.is_valid


def test_single_group_supported():
    data = pd.DataFrame({
        "metric": np.random.normal(50, 5, 100),
    })

    report = validate(
        data,
        target_col="metric",
    )

    assert report.is_valid
