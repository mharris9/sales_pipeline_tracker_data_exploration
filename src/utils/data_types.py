"""
Data type utilities and enums for the Sales Pipeline Explorer.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class DataType(Enum):
    """Enumeration of supported data types."""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATE = "date"
    TEXT = "text"
    BOOLEAN = "boolean"

class ColumnCompatibility(Enum):
    """Column compatibility for different chart types."""
    X_AXIS = "x_axis"
    Y_AXIS = "y_axis"
    COLOR = "color"
    SIZE = "size"
    GROUP_BY = "group_by"

# Chart type compatibility matrix
CHART_COMPATIBILITY: Dict[str, Dict[str, List[DataType]]] = {
    "histogram": {
        "x_axis": [DataType.NUMERICAL, DataType.DATE],
        "color": [DataType.CATEGORICAL, DataType.BOOLEAN]
    },
    "bar_chart": {
        "x_axis": [DataType.CATEGORICAL, DataType.BOOLEAN],
        "y_axis": [DataType.NUMERICAL],
        "color": [DataType.CATEGORICAL, DataType.BOOLEAN]
    },
    "scatter_plot": {
        "x_axis": [DataType.NUMERICAL, DataType.DATE],
        "y_axis": [DataType.NUMERICAL],
        "color": [DataType.CATEGORICAL, DataType.BOOLEAN],
        "size": [DataType.NUMERICAL]
    },
    "line_chart": {
        "x_axis": [DataType.DATE, DataType.NUMERICAL],
        "y_axis": [DataType.NUMERICAL],
        "color": [DataType.CATEGORICAL, DataType.BOOLEAN]
    },
    "correlation_heatmap": {
        "variables": [DataType.NUMERICAL]
    }
}

def detect_data_type(series: pd.Series, column_name: str = None) -> DataType:
    """
    Automatically detect the data type of a pandas Series.
    
    Args:
        series: The pandas Series to analyze
        column_name: Optional column name for context-based detection
        
    Returns:
        DataType enum value
    """
    # Remove null values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return DataType.TEXT
    
    # Context-based detection for known column patterns
    if column_name:
        column_lower = column_name.lower()
        if 'date' in column_lower or 'time' in column_lower:
            return DataType.DATE
        if 'id' in column_lower:
            # Try to convert to numeric first
            try:
                pd.to_numeric(clean_series)
                return DataType.NUMERICAL
            except:
                return DataType.CATEGORICAL
    
    # Check if it's already a datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return DataType.DATE
    
    # Check if it's boolean
    if pd.api.types.is_bool_dtype(series):
        return DataType.BOOLEAN
    
    # Check if it's numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it looks like categorical numeric (few unique values)
        unique_ratio = len(clean_series.unique()) / len(clean_series)
        if unique_ratio < 0.05 and len(clean_series.unique()) < 20:
            return DataType.CATEGORICAL
        return DataType.NUMERICAL
    
    # Try to convert to datetime
    if _is_date_column(clean_series):
        return DataType.DATE
    
    # Check if categorical (limited unique values)
    unique_ratio = len(clean_series.unique()) / len(clean_series)
    if unique_ratio < 0.5 and len(clean_series.unique()) < 50:
        return DataType.CATEGORICAL
    
    return DataType.TEXT

def _is_date_column(series: pd.Series) -> bool:
    """
    Check if a series contains date-like strings.
    
    Args:
        series: The pandas Series to check
        
    Returns:
        True if the series appears to contain dates
    """
    # Sample a few values to test
    sample_size = min(100, len(series))
    sample = series.sample(n=sample_size, random_state=42)
    
    date_count = 0
    for value in sample:
        if pd.isna(value):
            continue
        try:
            # Try common date formats
            pd.to_datetime(str(value), format="%m/%d/%Y")
            date_count += 1
        except:
            try:
                pd.to_datetime(str(value))
                date_count += 1
            except:
                pass
    
    # If more than 80% of samples look like dates, consider it a date column
    return (date_count / len(sample)) > 0.8

def convert_to_proper_type(series: pd.Series, data_type: DataType, 
                          date_format: str = "%m/%d/%Y") -> pd.Series:
    """
    Convert a pandas Series to the proper data type.
    
    Args:
        series: The pandas Series to convert
        data_type: The target data type
        date_format: Date format for parsing dates
        
    Returns:
        Converted pandas Series
    """
    if data_type == DataType.DATE:
        return pd.to_datetime(series, format=date_format, errors='coerce')
    elif data_type == DataType.NUMERICAL:
        return pd.to_numeric(series, errors='coerce')
    elif data_type == DataType.BOOLEAN:
        return series.astype(bool, errors='ignore')
    elif data_type == DataType.CATEGORICAL:
        return series.astype('category')
    else:  # TEXT
        return series.astype(str)

def get_compatible_columns(columns_info: Dict[str, DataType], 
                          chart_type: str, 
                          axis_type: str) -> List[str]:
    """
    Get columns compatible with a specific chart type and axis.
    
    Args:
        columns_info: Dictionary mapping column names to data types
        chart_type: Type of chart
        axis_type: Type of axis (x_axis, y_axis, etc.)
        
    Returns:
        List of compatible column names
    """
    if chart_type not in CHART_COMPATIBILITY:
        return list(columns_info.keys())
    
    if axis_type not in CHART_COMPATIBILITY[chart_type]:
        return list(columns_info.keys())
    
    compatible_types = CHART_COMPATIBILITY[chart_type][axis_type]
    
    return [
        col for col, dtype in columns_info.items() 
        if dtype in compatible_types
    ]

def calculate_statistics(series: pd.Series, data_type: DataType) -> Dict[str, Any]:
    """
    Calculate appropriate statistics for a series based on its data type.
    
    Args:
        series: The pandas Series to analyze
        data_type: The data type of the series
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    clean_series = series.dropna()
    
    # Common statistics
    stats['count'] = len(series)
    stats['non_null_count'] = len(clean_series)
    stats['null_count'] = stats['count'] - stats['non_null_count']
    stats['null_percentage'] = (stats['null_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
    
    if len(clean_series) == 0:
        return stats
    
    if data_type == DataType.NUMERICAL:
        stats.update({
            'mean': clean_series.mean(),
            'median': clean_series.median(),
            'std': clean_series.std(),
            'min': clean_series.min(),
            'max': clean_series.max(),
            'q25': clean_series.quantile(0.25),
            'q75': clean_series.quantile(0.75),
            'skewness': clean_series.skew(),
            'kurtosis': clean_series.kurtosis()
        })
    elif data_type == DataType.CATEGORICAL:
        value_counts = clean_series.value_counts()
        stats.update({
            'unique_count': len(clean_series.unique()),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
        })
    elif data_type == DataType.DATE:
        stats.update({
            'earliest': clean_series.min(),
            'latest': clean_series.max(),
            'range_days': (clean_series.max() - clean_series.min()).days if len(clean_series) > 1 else 0
        })
    elif data_type == DataType.TEXT:
        stats.update({
            'unique_count': len(clean_series.unique()),
            'avg_length': clean_series.astype(str).str.len().mean(),
            'max_length': clean_series.astype(str).str.len().max(),
            'min_length': clean_series.astype(str).str.len().min()
        })
    
    return stats

