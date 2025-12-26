from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from ..violations import Violation


class StatisticalCheck(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_col: Optional[str],
        **policy
    ) -> Optional[Violation]:
        pass

    def _groups(self, data, target_col, group_col):
        if group_col is None:
            return {"all": data[target_col].dropna()}
        return {
            str(k): v[target_col].dropna()
            for k, v in data.groupby(group_col)
        }
