"""
JSON report generator for StatGuard.
"""
from __future__ import annotations
from stat_guard.report import ValidationReport
import json
from typing import Dict, Any, Optional
from datetime import datetime


class JSONReporter:
    """Generate JSON validation reports."""
    
    def __init__(self, report: "ValidationReport"):
        self.report = report
    
    def generate(
        self,
        indent: Optional[int] = 2,
        include_metadata: bool = True
    ) -> str:
        """
        Generate JSON report.
        
        Args:
            indent: JSON indentation (None for compact)
            include_metadata: Whether to include metadata
            
        Returns:
            JSON string
        """
        data = self.report.as_dict()
        
        if not include_metadata:
            data.pop('metadata', None)
        
        # Add report generation timestamp
        data['generated_at'] = datetime.now().isoformat()
        data['version'] = '1.0.0'
        
        return json.dumps(data, indent=indent, default=str)
    
    def save(
        self,
        filepath: str,
        indent: Optional[int] = 2,
        include_metadata: bool = True
    ) -> None:
        """
        Save JSON report to file.
        
        Args:
            filepath: Output file path
            indent: JSON indentation
            include_metadata: Whether to include metadata
        """
        json_str = self.generate(indent=indent, include_metadata=include_metadata)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)


def export_to_json(
    data: Dict[str, Any],
    filepath: str,
    indent: int = 2
) -> None:
    """
    Export arbitrary data to JSON file.
    
    Args:
        data: Data to export
        filepath: Output file path
        indent: JSON indentation
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)
