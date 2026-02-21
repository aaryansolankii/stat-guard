"""
Command-line interface for StatGuard.

Provides a CLI for validating data files from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .api import validate, profile, compare, get_available_policies
from . import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="statguard",
        description="StatGuard: Statistical Data Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  statguard validate data.csv --target metric --group treatment
  statguard profile data.csv --output profile.html
  statguard compare train.csv test.csv --target metric
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate data for statistical analysis"
    )
    validate_parser.add_argument(
        "file",
        help="Input data file (CSV, Parquet, Excel)"
    )
    validate_parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target column to validate"
    )
    validate_parser.add_argument(
        "--group", "-g",
        help="Grouping column"
    )
    validate_parser.add_argument(
        "--unit", "-u",
        help="Unit identifier column"
    )
    validate_parser.add_argument(
        "--policy", "-p",
        default="default",
        choices=get_available_policies(),
        help="Validation policy"
    )
    validate_parser.add_argument(
        "--output", "-o",
        help="Output file for report (HTML, JSON, MD)"
    )
    validate_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error"
    )
    validate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Generate data profile"
    )
    profile_parser.add_argument(
        "file",
        help="Input data file"
    )
    profile_parser.add_argument(
        "--target", "-t",
        help="Target column for focused analysis"
    )
    profile_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file (HTML or JSON)"
    )
    profile_parser.add_argument(
        "--no-correlations",
        action="store_true",
        help="Skip correlation computation"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two datasets"
    )
    compare_parser.add_argument(
        "file1",
        help="First data file"
    )
    compare_parser.add_argument(
        "file2",
        help="Second data file"
    )
    compare_parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target column to compare"
    )
    compare_parser.add_argument(
        "--group", "-g",
        help="Grouping column"
    )
    compare_parser.add_argument(
        "--output", "-o",
        help="Output file (JSON)"
    )
    
    return parser


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from various formats."""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        return pd.read_csv(filepath)
    elif suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    elif suffix == ".json":
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_report(report, output_path: str) -> None:
    """Save report in appropriate format."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    
    if suffix == ".html":
        report.save_html(output_path)
    elif suffix == ".json":
        report.save_json(output_path)
    elif suffix in [".md", ".markdown"]:
        report.save_markdown(output_path)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def handle_validate(args) -> int:
    """Handle validate command."""
    try:
        data = load_data(args.file)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        print(f"Validating column: {args.target}")
    
    try:
        report = validate(
            data=data,
            target_col=args.target,
            group_col=args.group,
            unit_col=args.unit,
            policy=args.policy,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    report.print_summary()
    
    # Save report if output specified
    if args.output:
        try:
            save_report(report, args.output)
            print(f"\nReport saved to: {args.output}")
        except Exception as e:
            print(f"Error saving report: {e}", file=sys.stderr)
            return 1
    
    return 0 if report.is_valid else 1


def handle_profile(args) -> int:
    """Handle profile command."""
    try:
        data = load_data(args.file)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return 1
    
    print(f"Profiling {len(data)} rows, {len(data.columns)} columns...")
    
    try:
        dataset_profile = profile(
            data=data,
            target_col=args.target,
            compute_correlations=not args.no_correlations,
        )
    except Exception as e:
        print(f"Profiling error: {e}", file=sys.stderr)
        return 1
    
    # Save profile
    try:
        path = Path(args.output)
        if path.suffix.lower() == ".json":
            import json
            with open(args.output, 'w') as f:
                json.dump(dataset_profile.to_dict(), f, indent=2, default=str)
        else:
            # For HTML, we'd need a profile reporter
            print("HTML output not yet implemented for profiles")
            return 1
        print(f"Profile saved to: {args.output}")
    except Exception as e:
        print(f"Error saving profile: {e}", file=sys.stderr)
        return 1
    
    return 0


def handle_compare(args) -> int:
    """Handle compare command."""
    try:
        data1 = load_data(args.file1)
        data2 = load_data(args.file2)
    except Exception as e:
        print(f"Error loading files: {e}", file=sys.stderr)
        return 1
    
    print(f"Comparing datasets...")
    print(f"  File 1: {len(data1)} rows")
    print(f"  File 2: {len(data2)} rows")
    
    try:
        result = compare(
            data1=data1,
            data2=data2,
            target_col=args.target,
            group_col=args.group,
        )
    except Exception as e:
        print(f"Comparison error: {e}", file=sys.stderr)
        return 1
    
    # Print results
    print("\nComparison Results:")
    print(f"  Drift Detected: {result.get('drift_detected', 'N/A')}")
    
    if 'ks_test' in result:
        ks = result['ks_test']
        print(f"  KS Test: statistic={ks['statistic']:.4f}, p-value={ks['p_value']:.4f}")
    
    if 't_test' in result:
        tt = result['t_test']
        print(f"  T-Test: statistic={tt['statistic']:.4f}, p-value={tt['p_value']:.4f}")
    
    # Save if output specified
    if args.output:
        try:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nComparison saved to: {args.output}")
        except Exception as e:
            print(f"Error saving comparison: {e}", file=sys.stderr)
            return 1
    
    return 0 if not result.get('drift_detected', False) else 1


def main(args: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command is None:
        parser.print_help()
        return 1
    
    if parsed_args.command == "validate":
        return handle_validate(parsed_args)
    elif parsed_args.command == "profile":
        return handle_profile(parsed_args)
    elif parsed_args.command == "compare":
        return handle_compare(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
