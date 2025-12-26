"""
stat-guard: Prevent statistically invalid analyses from being shipped.

An opinionated, production-focused library that validates statistical
assumptions before analysis.
"""

from .api import validate

__version__ = "0.1.0"

__all__ = ["validate"]
