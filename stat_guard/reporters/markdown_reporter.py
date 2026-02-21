from __future__ import annotations
"""
Markdown report generator for StatGuard.
"""

from typing import Optional
from datetime import datetime
from stat_guard.report import ValidationReport

class MarkdownReporter:
    """Generate Markdown validation reports."""
    
    def __init__(self, report: "ValidationReport"):
        self.report = report
    
    def generate(
        self,
        title: str = "StatGuard Validation Report",
        include_toc: bool = True
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            title: Report title
            include_toc: Whether to include table of contents
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Status
        status = "✅ PASSED" if self.report.is_valid else "❌ FAILED"
        lines.append(f"## Validation Status: {status}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        summary = self.report.summary
        lines.append(f"- **Total Checks:** {summary.get('total_checks', 0)}")
        lines.append(f"- **Passed:** {summary.get('passed_checks', 0)}")
        lines.append(f"- **Failed:** {summary.get('failed_checks', 0)}")
        lines.append(f"- **Success Rate:** {summary.get('success_rate', 0):.1%}")
        lines.append("")
        
        # Severity counts
        lines.append("### Issues by Severity")
        lines.append("")
        lines.append(f"- 🔴 Critical: {summary.get('critical_count', 0)}")
        lines.append(f"- ❌ Errors: {summary.get('error_count', 0)}")
        lines.append(f"- ⚠️ Warnings: {summary.get('warning_count', 0)}")
        lines.append(f"- ℹ️ Info: {summary.get('info_count', 0)}")
        lines.append("")
        
        # Violations
        lines.append("## Violations")
        lines.append("")
        
        if not self.report.violations:
            lines.append("✅ No violations detected!")
            lines.append("")
        else:
            for check_name, violations in self.report._by_check.items():
                lines.append(f"### {check_name}")
                lines.append("")
                for v in violations:
                    icon = {
                        "CRITICAL": "🔴",
                        "ERROR": "❌",
                        "WARNING": "⚠️",
                        "INFO": "ℹ️"
                    }.get(v.severity.value, "•")
                    
                    lines.append(f"#### {icon} {v.code}")
                    lines.append("")
                    lines.append(f"**Severity:** {v.severity.value}")
                    lines.append("")
                    lines.append(f"**Message:** {v.message}")
                    lines.append("")
                    lines.append(f"**Suggestion:** {v.suggestion}")
                    lines.append("")
                    if v.context:
                        lines.append("**Context:**")
                        lines.append("```json")
                        lines.append(str(v.context))
                        lines.append("```")
                        lines.append("")
        
        # Metadata
        metadata = self.report._metadata
        if metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value if value is not None else 'N/A'}")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(
        self,
        filepath: str,
        title: str = "StatGuard Validation Report",
        include_toc: bool = True
    ) -> None:
        """
        Save Markdown report to file.
        
        Args:
            filepath: Output file path
            title: Report title
            include_toc: Whether to include table of contents
        """
        markdown = self.generate(title=title, include_toc=include_toc)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
