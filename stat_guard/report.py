from collections import defaultdict
from typing import Dict, List
from .violations import Violation, Severity


class ValidationReport:
    """Structured, multi-check validation report."""

    def __init__(self):
        self._by_check: Dict[str, List[Violation]] = defaultdict(list)

    def add_violation(self, check_name: str, violation: Violation):
        self._by_check[check_name].append(violation)

    @property
    def violations(self) -> List[Violation]:
        return [v for vs in self._by_check.values() for v in vs]

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def as_dict(self):
        return {
            check: [v.__dict__ for v in violations]
            for check, violations in self._by_check.items()
        }

    def __str__(self):
        if self.is_valid:
            return "✓ Validation passed (no statistical errors detected)"

        lines = ["✗ Validation failed:"]
        for check, violations in self._by_check.items():
            lines.append(f"\n[{check}]")
            for v in violations:
                lines.append(str(v))
        return "\n".join(lines)
