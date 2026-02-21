"""
Comprehensive data profiling for StatGuard.

Provides detailed statistical summaries similar to ydata-profiling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import CategoricalDtype


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    
    # For numeric columns
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # For categorical columns
    top_categories: Optional[List[Dict]] = None
    
    # Computed properties
    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_constant: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }


@dataclass
class DatasetProfile:
    """Profile for an entire dataset."""
    
    # Basic info
    n_rows: int
    n_columns: int
    memory_usage_mb: float
    
    # Column type counts
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    
    # Missing data
    total_missing_cells: int
    missing_cell_pct: float
    complete_rows: int
    complete_row_pct: float
    
    # Column profiles
    columns: Dict[str, ColumnProfile]
    
    # Correlations
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    # Warnings
    warnings: List[Dict] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "memory_usage_mb": self.memory_usage_mb,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "total_missing_cells": self.total_missing_cells,
            "missing_cell_pct": self.missing_cell_pct,
            "complete_rows": self.complete_rows,
            "complete_row_pct": self.complete_row_pct,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "correlations": self.correlations,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat(),
        }


class DataProfiler:
    """
    Comprehensive data profiler similar to ydata-profiling.
    
    Generates detailed statistical summaries of datasets.
    """
    
    def __init__(
        self,
        compute_correlations: bool = True,
        correlation_method: str = "pearson",
        max_categories: int = 10,
        histogram_bins: int = 20
    ):
        self.compute_correlations = compute_correlations
        self.correlation_method = correlation_method
        self.max_categories = max_categories
        self.histogram_bins = histogram_bins
    
    def profile(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> DatasetProfile:
        """
        Generate comprehensive profile of a dataset.
        
        Args:
            data: Input DataFrame
            target_col: Optional target column for focused analysis
            
        Returns:
            DatasetProfile with complete statistics
        """
        # Basic info
        n_rows, n_columns = data.shape
        memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Column type detection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Missing data summary
        total_missing = data.isna().sum().sum()
        missing_cell_pct = total_missing / (n_rows * n_columns)
        complete_rows = data.dropna().shape[0]
        complete_row_pct = complete_rows / n_rows
        
        # Profile each column
        columns = {}
        warnings_list = []
        
        for col in data.columns:
            profile = self._profile_column(data[col], col)
            columns[col] = profile
            
            # Generate warnings
            col_warnings = self._generate_warnings(profile)
            warnings_list.extend(col_warnings)
        
        # Compute correlations
        correlations = None
        if self.compute_correlations and len(numeric_columns) >= 2:
            correlations = self._compute_correlations(data[numeric_columns])
        
        return DatasetProfile(
            n_rows=n_rows,
            n_columns=n_columns,
            memory_usage_mb=memory_usage,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            total_missing_cells=total_missing,
            missing_cell_pct=missing_cell_pct,
            complete_rows=complete_rows,
            complete_row_pct=complete_row_pct,
            columns=columns,
            correlations=correlations,
            warnings=warnings_list
        )
    
    def _profile_column(
        self,
        series: pd.Series,
        name: str
    ) -> ColumnProfile:
        """Profile a single column."""
        
        dtype = str(series.dtype)
        count = len(series)
        missing_count = series.isna().sum()
        missing_pct = missing_count / count if count > 0 else 0
        unique_count = series.nunique(dropna=False)
        unique_pct = unique_count / count if count > 0 else 0
        
        # Determine column type
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_categorical = isinstance(series.dtype, CategoricalDtype) or series.dtype == "object"
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_constant = unique_count == 1
        
        profile_kwargs = {
            "name": name,
            "dtype": dtype,
            "count": count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "is_datetime": is_datetime,
            "is_constant": is_constant,
        }
        
        # Add numeric statistics
        if is_numeric:
            numeric_stats = self._compute_numeric_stats(series)
            profile_kwargs.update(numeric_stats)
        
        # Add categorical statistics
        if is_categorical:
            top_categories = self._compute_categorical_stats(series)
            profile_kwargs["top_categories"] = top_categories
        
        return ColumnProfile(**profile_kwargs)
    
    def _compute_numeric_stats(
        self,
        series: pd.Series
    ) -> Dict[str, float]:
        """Compute statistics for numeric column."""
        values = series.dropna()
        
        if len(values) == 0:
            return {}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            stats_dict = {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "q25": values.quantile(0.25),
                "q50": values.quantile(0.50),
                "q75": values.quantile(0.75),
            }
            
            # Only compute skewness/kurtosis for sufficient data
            if len(values) >= 8:
                stats_dict["skewness"] = stats.skew(values)
                stats_dict["kurtosis"] = stats.kurtosis(values)
        
        return stats_dict
    
    def _compute_categorical_stats(
        self,
        series: pd.Series
    ) -> List[Dict]:
        """Compute statistics for categorical column."""
        value_counts = series.value_counts(dropna=False)
        total = len(series)
        
        top_categories = []
        for value, count in value_counts.head(self.max_categories).items():
            top_categories.append({
                "value": str(value) if pd.notna(value) else "(missing)",
                "count": int(count),
                "percentage": count / total * 100
            })
        
        return top_categories
    
    def _compute_correlations(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlation matrix."""
        try:
            corr_matrix = data.corr(method=self.correlation_method)
            return corr_matrix.to_dict()
        except Exception:
            return {}
    
    def _generate_warnings(
        self,
        profile: ColumnProfile
    ) -> List[Dict]:
        """Generate warnings based on column profile."""
        warnings = []
        
        # High missing percentage
        if profile.missing_pct > 0.5:
            warnings.append({
                "column": profile.name,
                "type": "high_missing",
                "message": f"Column has {profile.missing_pct:.1%} missing values",
                "severity": "warning"
            })
        
        # High cardinality (potential ID)
        if profile.unique_pct > 0.9 and profile.is_categorical:
            warnings.append({
                "column": profile.name,
                "type": "high_cardinality",
                "message": "Column appears to be an identifier (all unique values)",
                "severity": "info"
            })
        
        # Constant column
        if profile.is_constant:
            warnings.append({
                "column": profile.name,
                "type": "constant",
                "message": "Column has constant value",
                "severity": "warning"
            })
        
        # High skewness
        if profile.skewness and abs(profile.skewness) > 3:
            warnings.append({
                "column": profile.name,
                "type": "high_skewness",
                "message": f"Column has high skewness ({profile.skewness:.2f})",
                "severity": "info"
            })
        
        # Zero variance
        if profile.std == 0:
            warnings.append({
                "column": profile.name,
                "type": "zero_variance",
                "message": "Column has zero variance",
                "severity": "error"
            })
        
        return warnings
    
    def quick_profile(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a quick profile with essential statistics only.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with key statistics
        """
        return {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing": data.isna().sum().to_dict(),
            "memory_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }
