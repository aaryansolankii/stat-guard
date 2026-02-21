# StatGuard 🛡️

**Production-Grade Statistical Data Validation & Profiling**

[![PyPI version](https://badge.fury.io/py/stat-guard.svg)](https://badge.fury.io/py/stat-guard)
[![Python versions](https://img.shields.io/pypi/pyversions/stat-guard.svg)](https://pypi.org/project/stat-guard/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

StatGuard is a comprehensive Python library for validating data integrity and statistical assumptions before running experimental or analytical workflows. Inspired by [ydata-profiling](https://github.com/ydataai/ydata-profiling), it provides detailed data profiling alongside rigorous statistical validation.

> **If StatGuard reports an error, the data should not be analyzed.**

---

## 🌟 Features

### Statistical Validation
- **Sample Size & Power**: Minimum sample size, group balance, covariate balance (SMD), statistical power estimation
- **Distribution Checks**: Zero variance, skewness, kurtosis, normality tests, heteroscedasticity
- **Data Integrity**: Unit integrity, duplicate detection, missing data patterns, constant columns
- **Outlier Detection**: IQR, Z-score, MAD methods with configurable thresholds
- **Correlation Analysis**: High correlation detection, multicollinearity (VIF), feature-target correlations
- **Cardinality Checks**: High/low cardinality detection, rare categories, ID column detection

### Data Profiling
- **Comprehensive Statistics**: Mean, std, quantiles, skewness, kurtosis
- **Missing Data Analysis**: Patterns, completeness ratios, MNAR detection
- **Correlation Matrices**: Pearson, Spearman, Kendall methods
- **Data Quality Scores**: Overall data health metrics

### Reporting
- **Multiple Formats**: HTML (interactive), JSON, Markdown
- **Beautiful Dashboards**: Light and dark themes
- **Detailed Context**: Every violation includes suggestions and diagnostic info

### Flexibility
- **Policy-Based**: Predefined policies (default, strict, lenient, experiment, time_series)
- **Custom Policies**: Create your own validation thresholds
- **Custom Checks**: Register your own validation logic
- **CLI Interface**: Command-line tool for automation

---

## 📦 Installation

```bash
pip install stat-guard
```

For enhanced statistics:

```bash
pip install stat-guard[stats]
```

---

## 🚀 Quick Start

### Basic Validation

```python
import pandas as pd
import stat_guard as sg

# Load your data
data = pd.DataFrame({
    "user_id": range(100),
    "metric": np.random.normal(100, 15, 100),
    "group": ["A"] * 50 + ["B"] * 50
})

# Validate
report = sg.validate(data, target_col="metric", group_col="group", unit_col="user_id")

# Check results
if report.is_valid:
    print("✅ Data is valid for analysis!")
else:
    print("❌ Validation failed:")
    print(report)

# Save HTML report
report.save_html("validation_report.html")
```

### Data Profiling

```python
# Generate comprehensive profile
profile = sg.profile(data)

# Access statistics
print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
print(f"Missing: {profile.missing_cell_pct:.1%}")

# Check column profiles
for col_name, col_profile in profile.columns.items():
    print(f"{col_name}: {col_profile.mean:.2f} ± {col_profile.std:.2f}")
```

### A/B Test Validation

```python
# Specialized validation for experiments
report = sg.check_experiment(
    data=experiment_data,
    metric_col="conversion_rate",
    treatment_col="treatment",
    user_id_col="user_id"
)

# Check SMD balance
for v in report.warnings:
    if "SMD" in v.message:
        print(f"Imbalance detected: {v.context}")
```

### Data Drift Detection

```python
# Compare training and test sets
comparison = sg.compare(train_data, test_data, target_col="target")

if comparison["drift_detected"]:
    print("⚠️ Data drift detected!")
    print(f"KS test p-value: {comparison['ks_test']['p_value']:.4f}")
```

---

## 📋 Validation Checks

### Sample Size & Power

| Check | Code | Description |
|-------|------|-------------|
| Minimum Sample Size | SG101 | Validates minimum observations |
| Insufficient Power | SG102 | Estimates statistical power |
| Unbalanced Groups | SG103 | Checks group size ratios |
| Covariate Imbalance | SG104 | Standardized Mean Difference |

### Distribution

| Check | Code | Description |
|-------|------|-------------|
| Zero Variance | SG201 | Detects constant values |
| Near-Zero Variance | SG202 | Flags dominant values |
| High Skewness | SG203 | Asymmetry detection |
| High Kurtosis | SG204 | Tail behavior analysis |
| Non-Normal | SG205 | Shapiro-Wilk test |
| Heteroscedasticity | SG206 | Levene's test |

### Data Integrity

| Check | Code | Description |
|-------|------|-------------|
| Duplicate Observations | SG301 | Unit-level duplicates |
| Duplicate Rows | SG302 | Complete row duplicates |
| Unit Leakage | SG303 | Cross-group contamination |
| Missing Unit ID | SG304 | Null identifiers |

### Outliers

| Check | Code | Description |
|-------|------|-------------|
| Extreme Outliers | SG501 | IQR/Z-score/MAD detection |
| Moderate Outliers | SG502 | Borderline cases |
| Outlier Clusters | SG503 | Pattern detection |

### Correlation

| Check | Code | Description |
|-------|------|-------------|
| High Correlation | SG401 | Inter-feature correlation |
| Multicollinearity | SG402 | VIF analysis |
| Perfect Correlation | SG403 | Redundant features |

---

## ⚙️ Policies

### Predefined Policies

```python
# Default - Balanced for general use
report = sg.validate(data, target_col="x", policy="default")

# Strict - Conservative thresholds
report = sg.validate(data, target_col="x", policy="strict")

# Lenient - Permissive for exploration
report = sg.validate(data, target_col="x", policy="lenient")

# Experiment - For A/B tests
report = sg.validate(data, target_col="x", policy="experiment")

# Time Series - For temporal data
report = sg.validate(data, target_col="x", policy="time_series")
```

### Custom Policies

```python
# Create custom policy
custom_policy = sg.create_custom_policy(
    base="default",
    min_sample_size=100,
    max_missing_pct=0.10,
    max_skewness=1.5
)

# Use it
report = sg.validate(data, target_col="x", policy=custom_policy)
```

---

## 🖥️ CLI Usage

```bash
# Validate a CSV file
statguard validate data.csv --target metric --group treatment --output report.html

# Profile a dataset
statguard profile data.csv --output profile.json

# Compare two datasets
statguard compare train.csv test.csv --target metric --output comparison.json

# Use strict policy
statguard validate data.csv --target metric --policy strict

# Fail fast on first error
statguard validate data.csv --target metric --fail-fast
```

---

## 📊 Report Examples

### HTML Report

```python
report.save_html("report.html", theme="light")  # or "dark"
```

Features:
- ✅ Interactive summary cards
- ✅ Color-coded violations by severity
- ✅ Expandable context for each issue
- ✅ Responsive design

### JSON Report

```python
report.save_json("report.json", indent=2)
```

### Markdown Report

```python
report.save_markdown("report.md")
```

---

## 🔧 Advanced Usage

### Custom Validation Checks

```python
from stat_guard import register_validator, Violation, Severity

class MyCustomCheck:
    name = "My Custom Check"
    
    def run(self, data, target_col, **kwargs):
        values = data[target_col]
        if values.min() < 0:
            return Violation(
                code="SG999",
                severity=Severity.ERROR,
                message="Negative values detected",
                suggestion="Check data collection process"
            )
        return None

# Register and use
register_validator(MyCustomCheck())
report = sg.validate(data, target_col="x")
```

### Multiple Column Validation

```python
# Validate multiple columns at once
reports = sg.validate_multiple(
    data,
    target_cols=["col1", "col2", "col3"],
    group_col="group"
)

for col, report in reports.items():
    print(f"{col}: {'✅' if report.is_valid else '❌'}")
```

### Batch Processing

```python
from stat_guard import DataValidator

validator = DataValidator(policy="strict")

for file in data_files:
    data = pd.read_csv(file)
    if validator.is_valid(data, target_col="metric"):
        process(data)
    else:
        log_error(file)
```

---

## 📈 Comparison with ydata-profiling

| Feature | StatGuard | ydata-profiling |
|---------|-----------|-----------------|
| Data Profiling | ✅ | ✅ |
| Statistical Validation | ✅ Extensive | ⚠️ Basic |
| A/B Test Checks | ✅ | ❌ |
| Custom Policies | ✅ | ❌ |
| CLI Interface | ✅ | ❌ |
| Data Drift | ✅ | ❌ |
| HTML Reports | ✅ | ✅ |
| Alert System | ✅ Severity-based | ✅ Threshold-based |

**Use StatGuard when:**
- You need statistical validation before analysis
- You're running A/B tests or experiments
- You want policy-based validation
- You need data drift detection

**Use ydata-profiling when:**
- You want the most comprehensive EDA
- You need visualizations out of the box
- You're doing initial data exploration

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=stat_guard tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/stat-guard/issues)
- **Documentation**: [Full Docs](https://stat-guard.readthedocs.io)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/stat-guard/discussions)

---

## 🙏 Acknowledgments

Inspired by:
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) for data profiling concepts
- [Great Expectations](https://greatexpectations.io/) for validation patterns
- [pandera](https://pandera.readthedocs.io/) for schema validation ideas
