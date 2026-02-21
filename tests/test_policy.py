"""
Tests for policy configuration.
"""

import pytest
from stat_guard.policy import (
    ValidationPolicy,
    create_policy,
    get_policy,
    POLICIES,
)


class TestValidationPolicy:
    """Tests for ValidationPolicy class."""
    
    def test_default_values(self):
        policy = ValidationPolicy()
        
        assert policy.min_sample_size == 30
        assert policy.max_imbalance_ratio == 2.0
        assert policy.max_missing_pct == 0.05
    
    def test_custom_values(self):
        policy = ValidationPolicy(min_sample_size=100, max_skewness=1.5)
        
        assert policy.min_sample_size == 100
        assert policy.max_skewness == 1.5
        # Default values should still work
        assert policy.max_imbalance_ratio == 2.0
    
    def test_to_dict(self):
        policy = ValidationPolicy(min_sample_size=50)
        d = policy.to_dict()
        
        assert d["min_sample_size"] == 50
        assert "max_skewness" in d
    
    def test_from_dict(self):
        d = {"min_sample_size": 75, "max_skewness": 1.0}
        policy = ValidationPolicy.from_dict(d)
        
        assert policy.min_sample_size == 75
        assert policy.max_skewness == 1.0


class TestCreatePolicy:
    """Tests for create_policy function."""
    
    def test_create_from_default(self):
        policy = create_policy("default", min_sample_size=100)
        
        assert policy.min_sample_size == 100
        # Other values from default
        assert policy.max_imbalance_ratio == 2.0
    
    def test_create_from_strict(self):
        policy = create_policy("strict")
        
        assert policy.min_sample_size == 50
        assert policy.max_imbalance_ratio == 1.5
    
    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            create_policy("unknown_policy")


class TestGetPolicy:
    """Tests for get_policy function."""
    
    def test_get_default_policy(self):
        policy = get_policy("default")
        
        assert isinstance(policy, ValidationPolicy)
        assert policy.min_sample_size == 30
    
    def test_get_strict_policy(self):
        policy = get_policy("strict")
        
        assert policy.min_sample_size == 50
    
    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            get_policy("unknown")


class TestPredefinedPolicies:
    """Tests for predefined policy values."""
    
    def test_default_policy_values(self):
        policy = POLICIES["default"]
        
        assert policy.min_sample_size == 30
        assert policy.max_imbalance_ratio == 2.0
        assert policy.max_missing_pct == 0.05
        assert policy.max_skewness == 2.0
    
    def test_strict_policy_values(self):
        policy = POLICIES["strict"]
        
        assert policy.min_sample_size == 50
        assert policy.max_imbalance_ratio == 1.5
        assert policy.max_missing_pct == 0.02
        assert policy.max_skewness == 1.5
    
    def test_lenient_policy_values(self):
        policy = POLICIES["lenient"]
        
        assert policy.min_sample_size == 10
        assert policy.max_imbalance_ratio == 5.0
    
    def test_experiment_policy_values(self):
        policy = POLICIES["experiment"]
        
        assert policy.min_sample_size == 100
        assert policy.max_imbalance_ratio == 1.2
