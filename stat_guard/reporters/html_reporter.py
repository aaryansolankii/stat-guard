"""
HTML report generator for StatGuard.

Creates interactive, visually appealing reports similar to ydata-profiling.
"""
from __future__ import annotations
from stat_guard.report import ValidationReport
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO

import pandas as pd
import numpy as np


class HTMLReporter:
    """Generate professional HTML validation reports."""
    
    def __init__(self, report: "ValidationReport"):
        self.report = report
    
    def generate(
        self,
        title: str = "StatGuard Validation Report",
        include_plots: bool = True,
        theme: str = "light"
    ) -> str:
        """
        Generate HTML report.
        
        Args:
            title: Report title
            include_plots: Whether to include visualizations
            theme: Color theme ("light" or "dark")
            
        Returns:
            HTML string
        """
        css = self._get_css(theme)
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"<title>{title}</title>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<style>{css}</style>",
            "</head>",
            "<body>",
            self._generate_header(title),
            self._generate_summary(),
            self._generate_violations_section(),
            self._generate_metadata_section(),
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _get_css(self, theme: str) -> str:
        """Get CSS styles."""
        if theme == "dark":
            return self._get_dark_css()
        return self._get_light_css()
    
    def _get_light_css(self) -> str:
        """Light theme CSS."""
        return """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-danger: #fff5f5;
            --bg-warning: #fffbeb;
            --bg-success: #f0fdf4;
            --bg-info: #eff6ff;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --color-danger: #dc2626;
            --color-warning: #d97706;
            --color-success: #059669;
            --color-info: #2563eb;
            --border-color: #e5e7eb;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-success {
            background: var(--bg-success);
            color: var(--color-success);
        }
        
        .status-error {
            background: var(--bg-danger);
            color: var(--color-danger);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        
        .summary-card {
            background: var(--bg-primary);
            padding: 20px;
            border-radius: 10px;
            box-shadow: var(--shadow);
        }
        
        .summary-card .label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .summary-card .value {
            font-size: 2rem;
            font-weight: 700;
        }
        
        .summary-card.critical .value { color: var(--color-danger); }
        .summary-card.error .value { color: var(--color-danger); }
        .summary-card.warning .value { color: var(--color-warning); }
        .summary-card.success .value { color: var(--color-success); }
        .summary-card.info .value { color: var(--color-info); }
        
        .section {
            background: var(--bg-primary);
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .section-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .section-header h2 {
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-content {
            padding: 24px;
        }
        
        .violation {
            border-left: 4px solid;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
            background: var(--bg-secondary);
        }
        
        .violation.critical {
            border-color: var(--color-danger);
            background: var(--bg-danger);
        }
        
        .violation.error {
            border-color: var(--color-danger);
            background: var(--bg-danger);
        }
        
        .violation.warning {
            border-color: var(--color-warning);
            background: var(--bg-warning);
        }
        
        .violation.info {
            border-color: var(--color-info);
            background: var(--bg-info);
        }
        
        .violation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .violation-code {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.875rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(0,0,0,0.1);
        }
        
        .violation-message {
            font-weight: 600;
            margin-bottom: 6px;
        }
        
        .violation-suggestion {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        
        .violation-context {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.8rem;
            background: rgba(0,0,0,0.05);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
        }
        
        .check-group {
            margin-bottom: 24px;
        }
        
        .check-group-header {
            font-size: 1rem;
            font-weight: 600;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .metadata-table th,
        .metadata-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .metadata-table th {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 16px;
        }
        
        @media (max-width: 768px) {
            body { padding: 10px; }
            .header h1 { font-size: 1.5rem; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
        }
        """
    
    def _get_dark_css(self) -> str:
        """Dark theme CSS."""
        return """
        :root {
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --bg-danger: rgba(220, 38, 38, 0.2);
            --bg-warning: rgba(217, 119, 6, 0.2);
            --bg-success: rgba(5, 150, 105, 0.2);
            --bg-info: rgba(37, 99, 235, 0.2);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --color-danger: #f87171;
            --color-warning: #fbbf24;
            --color-success: #34d399;
            --color-info: #60a5fa;
            --border-color: #374151;
            --shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        .header {
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-success {
            background: var(--bg-success);
            color: var(--color-success);
        }
        
        .status-error {
            background: var(--bg-danger);
            color: var(--color-danger);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        
        .summary-card {
            background: var(--bg-primary);
            padding: 20px;
            border-radius: 10px;
            box-shadow: var(--shadow);
        }
        
        .summary-card .label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .summary-card .value {
            font-size: 2rem;
            font-weight: 700;
        }
        
        .summary-card.critical .value { color: var(--color-danger); }
        .summary-card.error .value { color: var(--color-danger); }
        .summary-card.warning .value { color: var(--color-warning); }
        .summary-card.success .value { color: var(--color-success); }
        .summary-card.info .value { color: var(--color-info); }
        
        .section {
            background: var(--bg-primary);
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .section-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .section-header h2 {
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-content { padding: 24px; }
        
        .violation {
            border-left: 4px solid;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
            background: rgba(255,255,255,0.05);
        }
        
        .violation.critical { border-color: var(--color-danger); background: var(--bg-danger); }
        .violation.error { border-color: var(--color-danger); background: var(--bg-danger); }
        .violation.warning { border-color: var(--color-warning); background: var(--bg-warning); }
        .violation.info { border-color: var(--color-info); background: var(--bg-info); }
        
        .violation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .violation-code {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.875rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
        }
        
        .violation-message { font-weight: 600; margin-bottom: 6px; }
        .violation-suggestion { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 8px; }
        
        .violation-context {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.8rem;
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
        }
        
        .check-group { margin-bottom: 24px; }
        
        .check-group-header {
            font-size: 1rem;
            font-weight: 600;
            padding: 12px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .metadata-table th,
        .metadata-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .metadata-table th {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .empty-state-icon { font-size: 3rem; margin-bottom: 16px; }
        
        @media (max-width: 768px) {
            body { padding: 10px; }
            .header h1 { font-size: 1.5rem; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
        }
        """
    
    def _generate_header(self, title: str) -> str:
        """Generate report header."""
        is_valid = self.report.is_valid
        status_class = "status-success" if is_valid else "status-error"
        status_icon = "✓" if is_valid else "✗"
        status_text = "Validation Passed" if is_valid else "Validation Failed"
        
        return f"""
        <div class="container">
            <div class="header">
                <h1>
                    <span>🛡️</span>
                    {title}
                </h1>
                <p class="subtitle">
                    <span class="status-badge {status_class}">{status_icon} {status_text}</span>
                </p>
            </div>
        """
    
    def _generate_summary(self) -> str:
        """Generate summary section."""
        summary = self.report.summary
        
        return f"""
        <div class="container">
            <div class="summary-grid">
                <div class="summary-card critical">
                    <div class="label">Critical</div>
                    <div class="value">{summary.get('critical_count', 0)}</div>
                </div>
                <div class="summary-card error">
                    <div class="label">Errors</div>
                    <div class="value">{summary.get('error_count', 0)}</div>
                </div>
                <div class="summary-card warning">
                    <div class="label">Warnings</div>
                    <div class="value">{summary.get('warning_count', 0)}</div>
                </div>
                <div class="summary-card info">
                    <div class="label">Info</div>
                    <div class="value">{summary.get('info_count', 0)}</div>
                </div>
                <div class="summary-card success">
                    <div class="label">Checks Run</div>
                    <div class="value">{summary.get('total_checks', 0)}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Success Rate</div>
                    <div class="value">{summary.get('success_rate', 0):.0%}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_violations_section(self) -> str:
        """Generate violations section."""
        violations_by_check = self.report._by_check
        
        if not violations_by_check:
            return """
            <div class="container">
                <div class="section">
                    <div class="section-header">
                        <h2><span>📋</span> Violations</h2>
                    </div>
                    <div class="section-content">
                        <div class="empty-state">
                            <div class="empty-state-icon">🎉</div>
                            <p>No violations detected! Your data passed all checks.</p>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        check_groups = []
        for check_name, violations in violations_by_check.items():
            violation_items = []
            for v in violations:
                severity_class = v.severity.value.lower()
                context_html = ""
                if v.context:
                    context_str = str(v.context).replace("<", "&lt;").replace(">", "&gt;")
                    context_html = f'<div class="violation-context">{context_str}</div>'
                
                violation_items.append(f"""
                    <div class="violation {severity_class}">
                        <div class="violation-header">
                            <span class="violation-code">{v.code}</span>
                            <span>{v.severity.value}</span>
                        </div>
                        <div class="violation-message">{v.message}</div>
                        <div class="violation-suggestion">→ {v.suggestion}</div>
                        {context_html}
                    </div>
                """)
            
            check_groups.append(f"""
                <div class="check-group">
                    <div class="check-group-header">{check_name}</div>
                    {''.join(violation_items)}
                </div>
            """)
        
        return f"""
        <div class="container">
            <div class="section">
                <div class="section-header">
                    <h2><span>📋</span> Violations</h2>
                </div>
                <div class="section-content">
                    {''.join(check_groups)}
                </div>
            </div>
        </div>
        """
    
    def _generate_metadata_section(self) -> str:
        """Generate metadata section."""
        metadata = self.report._metadata
        
        if not metadata:
            return ""
        
        rows = []
        for key, value in metadata.items():
            rows.append(f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value if value is not None else '-'}</td>
                </tr>
            """)
        
        return f"""
        <div class="container">
            <div class="section">
                <div class="section-header">
                    <h2><span>ℹ️</span> Metadata</h2>
                </div>
                <div class="section-content">
                    <table class="metadata-table">
                        <thead>
                            <tr>
                                <th>Property</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
