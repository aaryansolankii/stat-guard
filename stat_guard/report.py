"""
Validation report for StatGuard.

Provides structured reporting with multiple output formats.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime

from .violations import Violation, Severity, ValidationSummary


class ValidationReport:
    """
    Structured, multi-check validation report with multiple output formats.
    
    This class collects violations from all checks and provides
    methods to analyze and export results.
    """

    def __init__(self):
        self._by_check: Dict[str, List[Violation]] = defaultdict(list)
        self._metadata: Dict[str, Any] = {}
        self._summary_stats: Dict[str, Any] = {}
        self._start_time: datetime = datetime.now()
        self._end_time: Optional[datetime] = None
        self._check_results: Dict[str, bool] = {}

    def add_violation(self, check_name: str, violation: Violation):
        """Add a violation to the report."""
        violation.check_name = check_name
        self._by_check[check_name].append(violation)

    def set_metadata(
        self,
        data_shape: tuple,
        target_col: str,
        group_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        policy: str = "default"
    ):
        """Set report metadata."""
        self._metadata = {
            'rows': data_shape[0],
            'columns': data_shape[1],
            'target_col': target_col,
            'group_col': group_col,
            'unit_col': unit_col,
            'policy': policy,
        }

    def set_summary_stats(self, stats: Dict[str, Any]):
        """Set summary statistics."""
        self._summary_stats = stats

    def mark_check_complete(self, check_name: str, passed: bool):
        """Mark a check as complete."""
        self._check_results[check_name] = passed

    def finalize(self):
        """Finalize the report with timing information."""
        self._end_time = datetime.now()

    @property
    def violations(self) -> List[Violation]:
        """Get all violations."""
        return [v for vs in self._by_check.values() for v in vs]

    @property
    def critical(self) -> List[Violation]:
        """Get critical violations."""
        return [v for v in self.violations if v.severity == Severity.CRITICAL]

    @property
    def errors(self) -> List[Violation]:
        """Get error violations."""
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        """Get warning violations."""
        return [v for v in self.violations if v.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Violation]:
        """Get info violations."""
        return [v for v in self.violations if v.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no critical or error violations)."""
        return len(self.critical) == 0 and len(self.errors) == 0

    @property
    def can_proceed(self) -> bool:
        """Check if analysis can proceed (no critical violations)."""
        return len(self.critical) == 0

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_checks = len(self._check_results)
        passed_checks = sum(1 for p in self._check_results.values() if p)
        
        duration = 0.0
        if self._end_time:
            duration = (self._end_time - self._start_time).total_seconds()
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'critical_count': len(self.critical),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.infos),
            'duration_seconds': duration,
            'is_valid': self.is_valid,
        }

    def get_violations_by_severity(self, severity: Severity) -> List[Violation]:
        """Get violations filtered by severity."""
        return [v for v in self.violations if v.severity == severity]

    def get_violations_by_check(self, check_name: str) -> List[Violation]:
        """Get violations for a specific check."""
        return self._by_check.get(check_name, [])

    def has_violation_code(self, code: str) -> bool:
        """Check if a specific violation code exists."""
        return any(v.code == code for v in self.violations)

    def as_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'metadata': self._metadata,
            'summary': self.summary,
            'summary_stats': self._summary_stats,
            'violations': [v.to_dict() for v in self.violations],
            'violations_by_check': {
                check: [v.to_dict() for v in violations]
                for check, violations in self._by_check.items()
            },
            'check_results': self._check_results,
        }

    def to_html(
        self,
        title: str = "StatGuard Validation Report",
        include_plots: bool = True,
        theme: str = "light"
    ) -> str:
        """Generate HTML report."""
        from .reporters.html_reporter import HTMLReporter
        reporter = HTMLReporter(self)
        return reporter.generate(title=title, include_plots=include_plots, theme=theme)

    def save_html(
        self,
        filepath: str,
        title: str = "StatGuard Validation Report",
        include_plots: bool = True,
        theme: str = "light"
    ) -> None:
        """Save HTML report to file."""
        html = self.to_html(title=title, include_plots=include_plots, theme=theme)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Generate JSON report."""
        from .reporters.json_reporter import JSONReporter
        reporter = JSONReporter(self)
        return reporter.generate(indent=indent)

    def save_json(self, filepath: str, indent: Optional[int] = 2) -> None:
        """Save JSON report to file."""
        json_str = self.to_json(indent=indent)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)

    def to_markdown(self, title: str = "StatGuard Validation Report") -> str:
        """Generate Markdown report."""
        from .reporters.markdown_reporter import MarkdownReporter
        reporter = MarkdownReporter(self)
        return reporter.generate(title=title)

    def save_markdown(self, filepath: str, title: str = "StatGuard Validation Report") -> None:
        """Save Markdown report to file."""
        markdown = self.to_markdown(title=title)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)

    def __str__(self) -> str:
        """String representation of the report."""
        if self.is_valid:
            return "✅ Validation passed (no statistical errors detected)"

        lines = ["❌ Validation failed:"]
        lines.append(f"   Critical: {len(self.critical)}")
        lines.append(f"   Errors: {len(self.errors)}")
        lines.append(f"   Warnings: {len(self.warnings)}")
        lines.append("")
        
        for check, violations in self._by_check.items():
            if violations:
                lines.append(f"\n[{check}]")
                for v in violations:
                    lines.append(f"  {v}")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ValidationReport(valid={self.is_valid}, violations={len(self.violations)})"

    def print_summary(self):
        """Print a concise summary to console."""
        summary = self.summary
        print("=" * 50)
        print("StatGuard Validation Summary")
        print("=" * 50)
        print(f"Status: {'✅ PASSED' if self.is_valid else '❌ FAILED'}")
        print(f"Checks Run: {summary['total_checks']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['duration_seconds']:.2f}s")
        print("-" * 50)
        print(f"Critical: {summary['critical_count']}")
        print(f"Errors: {summary['error_count']}")
        print(f"Warnings: {summary['warning_count']}")
        print(f"Info: {summary['info_count']}")
        print("=" * 50)
