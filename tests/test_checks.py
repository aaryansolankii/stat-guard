"""
Tests for individual validation checks.
"""

import pytest
import pandas as pd
import numpy as np

from stat_guard.checks.sample_size import (
    MinimumSampleSizeCheck,
    BalancedGroupsCheck,
    CovariateBalanceCheck,
)
from stat_guard.checks.distribution import (
    ZeroVarianceCheck,
    SkewnessCheck,
    NormalityCheck,
)
from stat_guard.violations import Severity, ViolationCodes


class TestMinimumSampleSizeCheck:
    """Tests for MinimumSampleSizeCheck."""
    
    def test_small_sample_error(self):
        data = pd.DataFrame({
            "metric": [1, 2, 3],
            "group": ["A", "A", "A"],
        })
        
        check = MinimumSampleSizeCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
            min_sample_size=10,
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.SAMPLE_TOO_SMALL for v in violations)
        assert any(v.severity == Severity.ERROR for v in violations)
    
    def test_adequate_sample_passes(self):
        data = pd.DataFrame({
            "metric": range(100),
            "group": ["A"] * 100,
        })
        
        check = MinimumSampleSizeCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
            min_sample_size=30,
        )
        
        assert len(violations) == 0


class TestBalancedGroupsCheck:
    """Tests for BalancedGroupsCheck."""
    
    def test_imbalanced_groups_warning(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "metric": np.random.randn(150),
            "group": ["A"] * 50 + ["B"] * 100,
        })
        
        check = BalancedGroupsCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
            max_imbalance_ratio=1.5,
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.UNBALANCED_GROUPS for v in violations)
    
    def test_balanced_groups_passes(self):
        data = pd.DataFrame({
            "metric": range(100),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        
        check = BalancedGroupsCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
            max_imbalance_ratio=2.0,
        )
        
        assert len(violations) == 0


class TestCovariateBalanceCheck:
    """Tests for CovariateBalanceCheck."""
    
    def test_large_smd_detected(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "metric": np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(2.0, 1, 100),
            ]),
            "group": ["A"] * 100 + ["B"] * 100,
        })
        
        check = CovariateBalanceCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
            max_smd=0.25,
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.COVARIATE_IMBALANCE for v in violations)


class TestZeroVarianceCheck:
    """Tests for ZeroVarianceCheck."""
    
    def test_zero_variance_error(self):
        data = pd.DataFrame({
            "metric": [5, 5, 5, 5],
            "group": ["A"] * 4,
        })
        
        check = ZeroVarianceCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col="group",
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.ZERO_VARIANCE for v in violations)
        assert any(v.severity == Severity.ERROR for v in violations)


class TestSkewnessCheck:
    """Tests for SkewnessCheck."""
    
    def test_high_skewness_warning(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "metric": np.random.exponential(scale=1.0, size=200),
        })
        
        check = SkewnessCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col=None,
            max_skewness=1.0,
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.HIGH_SKEWNESS for v in violations)


class TestNormalityCheck:
    """Tests for NormalityCheck."""
    
    def test_non_normal_warning(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "metric": np.random.exponential(scale=1.0, size=300),
        })
        
        check = NormalityCheck()
        violations = check.run(
            data=data,
            target_col="metric",
            group_col=None,
            normality_alpha=0.05,
        )
        
        assert len(violations) > 0
        assert any(v.code == ViolationCodes.NON_NORMAL for v in violations)
