"""
Violation definitions and severity levels for StatGuard.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class Severity(Enum):
    """Severity levels for violations."""
    CRITICAL = "CRITICAL"  # Data is fundamentally unusable
    ERROR = "ERROR"        # Analysis should not proceed
    WARNING = "WARNING"    # Proceed with caution
    INFO = "INFO"          # Informational only


@dataclass
class Violation:
    """
    A single validation violation with detailed context.
    
    Attributes:
        code: Unique violation code (e.g., "SG101")
        severity: Severity level
        message: Human-readable description
        suggestion: Recommended action
        context: Additional diagnostic information
        timestamp: When the violation was detected
        check_name: Name of the check that found this violation
    """
    code: str
    severity: Severity
    message: str
    suggestion: str
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    check_name: str = ""
    
    def __post_init__(self):
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)
    
    def __str__(self) -> str:
        icon = {
            Severity.CRITICAL: "🔴",
            Severity.ERROR: "❌",
            Severity.WARNING: "⚠️",
            Severity.INFO: "ℹ️"
        }.get(self.severity, "•")
        
        msg = f"{icon} [{self.severity.value}] {self.code}\n  {self.message}\n  → {self.suggestion}"
        if self.context:
            msg += f"\n  Context: {self.context}"
        return msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "check_name": self.check_name,
        }


class ViolationCodes:
    """
    Standard violation codes for StatGuard.
    
    Code ranges:
    - SG1xx: Sample size and power issues
    - SG2xx: Distribution and statistical assumptions
    - SG3xx: Data quality and integrity
    - SG4xx: Correlation and multicollinearity
    - SG5xx: Outliers and anomalies
    - SG6xx: Missing data issues
    - SG7xx: Categorical and cardinality issues
    - SG8xx: Time series specific
    - SG9xx: Custom violations
    """
    
    # Sample Size & Power (SG1xx)
    SAMPLE_TOO_SMALL = "SG101"
    INSUFFICIENT_POWER = "SG102"
    UNBALANCED_GROUPS = "SG103"
    COVARIATE_IMBALANCE = "SG104"
    
    # Distribution & Assumptions (SG2xx)
    ZERO_VARIANCE = "SG201"
    NEAR_ZERO_VARIANCE = "SG202"
    HIGH_SKEWNESS = "SG203"
    HIGH_KURTOSIS = "SG204"
    NON_NORMAL = "SG205"
    HETEROSCEDASTICITY = "SG206"
    
    # Data Quality & Integrity (SG3xx)
    DUPLICATE_OBSERVATIONS = "SG301"
    DUPLICATE_ROWS = "SG302"
    UNIT_LEAKAGE = "SG303"
    MISSING_UNIT_ID = "SG304"
    INCONSISTENT_DATA_TYPES = "SG305"
    CONSTANT_COLUMN = "SG306"
    SUSPICIOUS_PATTERN = "SG307"
    
    # Correlation & Multicollinearity (SG4xx)
    HIGH_CORRELATION = "SG401"
    MULTICOLLINEARITY = "SG402"
    PERFECT_CORRELATION = "SG403"
    
    # Outliers & Anomalies (SG5xx)
    EXTREME_OUTLIERS = "SG501"
    MODERATE_OUTLIERS = "SG502"
    OUTLIER_CLUSTER = "SG503"
    
    # Missing Data (SG6xx)
    EXCESSIVE_MISSING = "SG601"
    MISSING_NOT_AT_RANDOM = "SG602"
    MISSING_PATTERN = "SG603"
    COMPLETE_CASE_RATIO_LOW = "SG604"
    
    # Categorical & Cardinality (SG7xx)
    HIGH_CARDINALITY = "SG701"
    LOW_CARDINALITY = "SG702"
    RARE_CATEGORIES = "SG703"
    EMPTY_CATEGORIES = "SG704"
    
    # Time Series (SG8xx)
    IRREGULAR_TIMESTAMPS = "SG801"
    MISSING_TIMESTAMPS = "SG802"
    NON_STATIONARY = "SG803"
    SEASONALITY_DETECTED = "SG804"
    
    # Data Drift (SG9xx)
    DATA_DRIFT_DETECTED = "SG901"
    CONCEPT_DRIFT_DETECTED = "SG902"
    DISTRIBUTION_SHIFT = "SG903"


@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    duration_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
        }
