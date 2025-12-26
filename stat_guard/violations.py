from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class Violation:
    code: str
    severity: Severity
    message: str
    suggestion: str
    context: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        msg = f"[{self.severity.value}] {self.code}\n  {self.message}\n  → {self.suggestion}"
        if self.context:
            msg += f"\n  Context: {self.context}"
        return msg


class ViolationCodes:
    SAMPLE_TOO_SMALL = "SG101"
    UNBALANCED_GROUPS = "SG102"
    ZERO_VARIANCE = "SG201"
    HIGH_SKEWNESS = "SG202"
    NON_NORMAL = "SG203"
    MISSING_DATA = "SG301"
    DUPLICATE_OBSERVATIONS = "SG302"
