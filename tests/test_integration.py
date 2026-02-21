"""
Integration tests for StatGuard.
"""

import pytest
import pandas as pd
import numpy as np

from stat_guard.api import validate, profile, compare
from stat_guard.violations import Severity


class TestValidate:
    """Integration tests for validate function."""
    
    def test_valid_experiment_passes(self):
        np.random.seed(42)

        data = pd.DataFrame({
            "metric": np.random.normal(100, 10, 200),
            "group": ["A"] * 100 + ["B"] * 100,
        })

        report = validate(
            data,
            target_col="metric",
            group_col="group",
        )

        assert report.is_valid
        assert len(report.errors) == 0
        
    def test_small_sample_fails(self):
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
    
    def test_fail_fast_stops_early(self):
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
        # With fail_fast, should stop after first error
        assert len(report.errors) >= 1
    
    def test_strict_policy_is_harder(self):
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
    
    def test_single_group_supported(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "metric": np.random.normal(50, 5, 100),
        })
        
        report = validate(
            data,
            target_col="metric",
        )
        
        assert report.is_valid


class TestProfile:
    """Integration tests for profile function."""
    
    def test_profile_returns_dataset_profile(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "numeric": np.random.normal(100, 15, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
        })
        
        result = profile(data)
        
        assert result.n_rows == 100
        assert result.n_columns == 2
        assert "numeric" in result.columns
        assert "categorical" in result.columns
    
    def test_profile_computes_statistics(self):
        np.random.seed(42)
        data = pd.DataFrame({
            "x": np.random.normal(50, 10, 100),
        })
        
        result = profile(data)
        
        col_profile = result.columns["x"]
        assert col_profile.mean is not None
        assert col_profile.std is not None
        assert col_profile.q25 is not None
        assert col_profile.q50 is not None
        assert col_profile.q75 is not None


class TestCompare:
    """Integration tests for compare function."""
    
    def test_detects_drift(self):
        np.random.seed(42)
        train = pd.DataFrame({
            "target": np.random.normal(0, 1, 1000),
        })
        test = pd.DataFrame({
            "target": np.random.normal(2, 1, 1000),  # Shifted mean
        })
        
        result = compare(train, test, target_col="target")
        
        assert result["drift_detected"] is True
        assert "ks_test" in result
        assert "t_test" in result
    
    def test_no_drift_for_similar(self):
        np.random.seed(42)
        train = pd.DataFrame({
            "target": np.random.normal(0, 1, 1000),
        })
        test = pd.DataFrame({
            "target": np.random.normal(0, 1, 1000),
        })
        
        result = compare(train, test, target_col="target")
        
        assert result["drift_detected"] is False


class TestReportFormats:
    """Tests for different report output formats."""
    
    def test_html_report_generation(self):
        data = pd.DataFrame({
            "metric": [1, 2, 3],
            "group": ["A", "A", "A"],
        })
        
        report = validate(data, target_col="metric", group_col="group")
        html = report.to_html()
        
        assert "<!DOCTYPE html>" in html
        assert "StatGuard" in html
    
    def test_json_report_generation(self):
        data = pd.DataFrame({
            "metric": [1, 2, 3],
            "group": ["A", "A", "A"],
        })
        
        report = validate(data, target_col="metric", group_col="group")
        json_str = report.to_json()
        
        assert "is_valid" in json_str
        assert "violations" in json_str
    
    def test_markdown_report_generation(self):
        data = pd.DataFrame({
            "metric": [1, 2, 3],
            "group": ["A", "A", "A"],
        })
        
        report = validate(data, target_col="metric", group_col="group")
        md = report.to_markdown()
        
        assert "#" in md  # Markdown header
        assert "Validation" in md
