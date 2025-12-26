import numpy as np
from scipy.stats import skew, shapiro
from .base import StatisticalCheck
from ..violations import Violation, Severity, ViolationCodes


class ZeroVarianceCheck(StatisticalCheck):

    @property
    def name(self):
        return "Zero Variance"

    def run(self, data, target_col, group_col, **_):
        for g, vals in self._groups(data, target_col, group_col).items():
            if vals.nunique() <= 1:
                return Violation(
                    code=ViolationCodes.ZERO_VARIANCE,
                    severity=Severity.ERROR,
                    message=f"Zero variance in group '{g}'",
                    suggestion="Metric has no variability",
                )
        return None


class SkewnessCheck(StatisticalCheck):

    @property
    def name(self):
        return "Skewness"

    def run(self, data, target_col, group_col, max_skewness, **_):
        values = data[target_col].dropna()
        if len(values) < 10:
            return None

        s = skew(values)
        if abs(s) > max_skewness:
            return Violation(
                code=ViolationCodes.HIGH_SKEWNESS,
                severity=Severity.WARNING,
                message=f"High skewness detected ({s:.2f})",
                suggestion="Mean may be misleading; consider median or transform",
            )
        return None


class NormalityCheck(StatisticalCheck):

    @property
    def name(self):
        return "Normality"

    def run(self, data, target_col, group_col, **_):
        values = data[target_col].dropna()
        if len(values) < 20:
            return None

        _, p = shapiro(values.sample(min(len(values), 500)))
        if p < 0.05:
            return Violation(
                code=ViolationCodes.NON_NORMAL,
                severity=Severity.WARNING,
                message="Normality assumption violated",
                suggestion="Use non-parametric tests",
                context={"p_value": p},
            )
        return None
