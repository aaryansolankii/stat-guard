import pandas as pd
from typing import Optional
from .engine import ValidationEngine
from .report import ValidationReport

_engine = ValidationEngine()


def validate(
    data: pd.DataFrame,
    *,
    target_col: str,
    group_col: Optional[str] = None,
    policy: str = "default",
    fail_fast: bool = False,
) -> ValidationReport:
    return _engine.validate(
        data=data,
        target_col=target_col,
        group_col=group_col,
        policy=policy,
        fail_fast=fail_fast,
    )
