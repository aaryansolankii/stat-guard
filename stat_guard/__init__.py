"""
stat-guard: Prevent statistically invalid analyses from being shipped.

An opinionated, production-focused library that validates statistical
assumptions before analysis.
"""

# Export main API functions
from .api import (
    validate,
    profile,
    quick_check,
    compare,
    register_validator,
    validate_multiple,
    get_available_policies,
    create_custom_policy,
    list_checks,
    check_experiment,
    check_time_series,
)

# Export key classes
from .report import ValidationReport
from .profilers.data_profiler import DatasetProfile
from .policy import ValidationPolicy
from .engine import ValidationError

__version__ = "0.2.0"

__all__ = [
    "validate",
    "profile",
    "quick_check",
    "compare",
    "register_validator",
    "validate_multiple",
    "get_available_policies",
    "create_custom_policy",
    "list_checks",
    "check_experiment",
    "check_time_series",
    "ValidationReport",
    "DatasetProfile",
    "ValidationPolicy",
    "ValidationError",
]