from typing import List
from .violations import Violation, Severity


class ValidationReport:
    """Collects and summarizes validation results."""

    def __init__(self):
        self.violations: List[Violation] = []

    def add_violation(self, violation: Violation):
        self.violations.append(violation)

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        if self.is_valid:
            return "✓ Validation passed (no statistical errors detected)"

        lines = ["✗ Validation failed:"]
        for v in self.violations:
            lines.append(str(v))
        return "\n".join(lines)
