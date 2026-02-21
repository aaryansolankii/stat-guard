"""
Basic usage examples for StatGuard.
"""

import pandas as pd
import numpy as np
import stat_guard as sg

# Set random seed for reproducibility
np.random.seed(42)


print("=" * 60)
print("StatGuard Basic Usage Examples")
print("=" * 60)


# Example 1: Basic Validation
print("\n1. Basic Validation")
print("-" * 40)

data = pd.DataFrame({
    "user_id": range(100),
    "metric": np.random.normal(100, 15, 100),
    "group": ["A"] * 50 + ["B"] * 50
})

report = sg.validate(data, target_col="metric", group_col="group", unit_col="user_id")

print(f"Valid: {report.is_valid}")
print(f"Errors: {len(report.errors)}")
print(f"Warnings: {len(report.warnings)}")


# Example 2: Data Profiling
print("\n2. Data Profiling")
print("-" * 40)

profile = sg.profile(data)

print(f"Rows: {profile.n_rows}")
print(f"Columns: {profile.n_columns}")
print(f"Missing: {profile.missing_cell_pct:.1%}")

for col_name, col_profile in profile.columns.items():
    if col_profile.is_numeric:
        print(f"  {col_name}: μ={col_profile.mean:.2f}, σ={col_profile.std:.2f}")


# Example 3: A/B Test Validation
print("\n3. A/B Test Validation")
print("-" * 40)

experiment_data = pd.DataFrame({
    "user_id": range(200),
    "conversion": np.concatenate([
        np.random.binomial(1, 0.10, 100),  # Control: 10% conversion
        np.random.binomial(1, 0.15, 100),  # Treatment: 15% conversion
    ]),
    "treatment": ["control"] * 100 + ["treatment"] * 100
})

report = sg.check_experiment(
    experiment_data,
    metric_col="conversion",
    treatment_col="treatment",
    user_id_col="user_id"
)

print(f"Experiment valid: {report.is_valid}")
for v in report.warnings:
    print(f"  Warning: {v.message}")


# Example 4: Data Drift Detection
print("\n4. Data Drift Detection")
print("-" * 40)

train_data = pd.DataFrame({
    "feature": np.random.normal(0, 1, 1000),
    "target": np.random.normal(0, 1, 1000)
})

test_data = pd.DataFrame({
    "feature": np.random.normal(0.5, 1.2, 1000),  # Slight drift
    "target": np.random.normal(0, 1, 1000)
})

comparison = sg.compare(train_data, test_data, target_col="target")

print(f"Drift detected: {comparison['drift_detected']}")
if 'ks_test' in comparison:
    print(f"KS test p-value: {comparison['ks_test']['p_value']:.4f}")


# Example 5: Custom Policy
print("\n5. Custom Policy")
print("-" * 40)

custom_policy = sg.create_custom_policy(
    base="default",
    min_sample_size=50,
    max_missing_pct=0.10
)

small_data = pd.DataFrame({
    "value": range(40),
    "group": ["A"] * 20 + ["B"] * 20
})

report = sg.validate(
    small_data,
    target_col="value",
    group_col="group",
    policy=custom_policy
)

print(f"Valid with custom policy: {report.is_valid}")


# Example 6: Multiple Column Validation
print("\n6. Multiple Column Validation")
print("-" * 40)

multi_data = pd.DataFrame({
    "col1": np.random.normal(0, 1, 100),
    "col2": np.random.normal(10, 2, 100),
    "col3": np.random.exponential(1, 100),
    "group": ["A"] * 50 + ["B"] * 50
})

reports = sg.validate_multiple(
    multi_data,
    target_cols=["col1", "col2", "col3"],
    group_col="group"
)

for col, rep in reports.items():
    status = "✅" if rep.is_valid else "❌"
    print(f"  {col}: {status}")


# Example 7: Report Export
print("\n7. Report Export")
print("-" * 40)

problem_data = pd.DataFrame({
    "metric": [5, 5, 5, 5],  # Zero variance
    "group": ["A"] * 4
})

report = sg.validate(problem_data, target_col="metric", group_col="group")

# Save different formats
report.save_json("report.json")
report.save_markdown("report.md")
report.save_html("report.html")

print("Reports saved: report.json, report.md, report.html")


print("\n" + "=" * 60)
print("Examples completed!")
print("=" * 60)
