"""
FilterManager class for creating and managing data filters.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import streamlit as st
from datetime import datetime, date

from utils.data_types import DataType

class FilterManager:
    """
    Manages filtering of data based on column types and user selections.
    """
    
    def __init__(self):
        """Initialize the FilterManager."""
        self.filters: Dict[str, Dict[str, Any]] = {}
        self.active_filters: Dict[str, bool] = {}
        
    def create_filters(self, df: pd.DataFrame, column_types: Dict[str, DataType]) -> None:
        """
        Create filter configurations for all columns.
        
        Args:
            df: The DataFrame to create filters for
            column_types: Dictionary mapping column names to data types
        """
        self.filters = {}
        self.active_filters = {}
        
        for column, data_type in column_types.items():
            if column not in df.columns:
                continue
                
            self.active_filters[column] = False
            
            if data_type == DataType.CATEGORICAL:
                self.filters[column] = self._create_categorical_filter(df[column])
            elif data_type == DataType.NUMERICAL:
                self.filters[column] = self._create_numerical_filter(df[column])
            elif data_type == DataType.DATE:
                self.filters[column] = self._create_date_filter(df[column])
            elif data_type == DataType.TEXT:
                self.filters[column] = self._create_text_filter(df[column])
            elif data_type == DataType.BOOLEAN:
                self.filters[column] = self._create_boolean_filter(df[column])
    
    def _create_categorical_filter(self, series: pd.Series) -> Dict[str, Any]:
        """Create filter configuration for categorical data."""
        unique_values = sorted([str(val) for val in series.dropna().unique() if str(val) != 'nan'])
        
        return {
            'type': 'categorical',
            'unique_values': unique_values,
            'selected_values': unique_values.copy(),  # Default: all selected
            'filter_type': 'include'  # 'include' or 'exclude'
        }
    
    def _create_numerical_filter(self, series: pd.Series) -> Dict[str, Any]:
        """Create filter configuration for numerical data."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'type': 'numerical',
                'min_value': 0,
                'max_value': 1,
                'selected_min': 0,
                'selected_max': 1,
                'filter_type': 'range'
            }
        
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        
        return {
            'type': 'numerical',
            'min_value': min_val,
            'max_value': max_val,
            'selected_min': min_val,
            'selected_max': max_val,
            'filter_type': 'range',  # 'range', 'greater_than', 'less_than'
            'percentiles': {
                '25': float(clean_series.quantile(0.25)),
                '50': float(clean_series.quantile(0.50)),
                '75': float(clean_series.quantile(0.75))
            }
        }
    
    def _create_date_filter(self, series: pd.Series) -> Dict[str, Any]:
        """Create filter configuration for date data."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            default_date = datetime.now().date()
            return {
                'type': 'date',
                'min_date': default_date,
                'max_date': default_date,
                'selected_min_date': default_date,
                'selected_max_date': default_date,
                'filter_type': 'range'
            }
        
        min_date = clean_series.min().date()
        max_date = clean_series.max().date()
        
        return {
            'type': 'date',
            'min_date': min_date,
            'max_date': max_date,
            'selected_min_date': min_date,
            'selected_max_date': max_date,
            'filter_type': 'range'  # 'range', 'after', 'before'
        }
    
    def _create_text_filter(self, series: pd.Series) -> Dict[str, Any]:
        """Create filter configuration for text data."""
        return {
            'type': 'text',
            'search_text': '',
            'filter_type': 'contains',  # 'contains', 'starts_with', 'ends_with', 'exact'
            'case_sensitive': False
        }
    
    def _create_boolean_filter(self, series: pd.Series) -> Dict[str, Any]:
        """Create filter configuration for boolean data."""
        return {
            'type': 'boolean',
            'selected_values': [True, False],  # Default: both selected
            'filter_type': 'include'
        }
    
    def render_filter_ui(self, column: str, data_type: DataType) -> None:
        """
        Render the filter UI for a specific column.
        
        Args:
            column: Column name
            data_type: Data type of the column
        """
        if column not in self.filters:
            return
        
        filter_config = self.filters[column]
        
        # Filter activation checkbox
        self.active_filters[column] = st.checkbox(
            f"Filter {column}", 
            value=self.active_filters.get(column, False),
            key=f"filter_active_{column}"
        )
        
        if not self.active_filters[column]:
            return
        
        # Render specific filter UI based on data type
        if data_type == DataType.CATEGORICAL:
            self._render_categorical_filter_ui(column, filter_config)
        elif data_type == DataType.NUMERICAL:
            self._render_numerical_filter_ui(column, filter_config)
        elif data_type == DataType.DATE:
            self._render_date_filter_ui(column, filter_config)
        elif data_type == DataType.TEXT:
            self._render_text_filter_ui(column, filter_config)
        elif data_type == DataType.BOOLEAN:
            self._render_boolean_filter_ui(column, filter_config)
    
    def _render_categorical_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for categorical filters."""
        unique_values = filter_config['unique_values']
        
        # Filter type selection
        filter_config['filter_type'] = st.radio(
            f"Filter type for {column}",
            ['include', 'exclude'],
            index=0 if filter_config['filter_type'] == 'include' else 1,
            key=f"filter_type_{column}",
            horizontal=True
        )
        
        # Value selection
        if len(unique_values) <= 10:
            # Use checkboxes for small number of values
            selected_values = []
            for value in unique_values:
                if st.checkbox(
                    value, 
                    value=value in filter_config['selected_values'],
                    key=f"filter_cat_{column}_{value}"
                ):
                    selected_values.append(value)
            filter_config['selected_values'] = selected_values
        else:
            # Use multiselect for large number of values
            filter_config['selected_values'] = st.multiselect(
                f"Select values for {column}",
                options=unique_values,
                default=filter_config['selected_values'],
                key=f"filter_multiselect_{column}"
            )
    
    def _render_numerical_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for numerical filters."""
        min_val = filter_config['min_value']
        max_val = filter_config['max_value']
        
        # Filter type selection
        filter_type = st.selectbox(
            f"Filter type for {column}",
            ['range', 'greater_than', 'less_than'],
            index=['range', 'greater_than', 'less_than'].index(filter_config['filter_type']),
            key=f"filter_type_{column}"
        )
        filter_config['filter_type'] = filter_type
        
        if filter_type == 'range':
            # Range slider
            selected_range = st.slider(
                f"Range for {column}",
                min_value=min_val,
                max_value=max_val,
                value=(filter_config['selected_min'], filter_config['selected_max']),
                key=f"filter_range_{column}"
            )
            filter_config['selected_min'], filter_config['selected_max'] = selected_range
        elif filter_type == 'greater_than':
            filter_config['selected_min'] = st.number_input(
                f"Greater than",
                min_value=min_val,
                max_value=max_val,
                value=filter_config.get('selected_min', min_val),
                key=f"filter_gt_{column}"
            )
        elif filter_type == 'less_than':
            filter_config['selected_max'] = st.number_input(
                f"Less than",
                min_value=min_val,
                max_value=max_val,
                value=filter_config.get('selected_max', max_val),
                key=f"filter_lt_{column}"
            )
        
        # Show percentile information
        if 'percentiles' in filter_config:
            percentiles = filter_config['percentiles']
            st.caption(f"25th: {percentiles['25']:.2f} | 50th: {percentiles['50']:.2f} | 75th: {percentiles['75']:.2f}")
    
    def _render_date_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for date filters."""
        min_date = filter_config['min_date']
        max_date = filter_config['max_date']
        
        # Filter type selection
        filter_type = st.selectbox(
            f"Filter type for {column}",
            ['range', 'after', 'before'],
            index=['range', 'after', 'before'].index(filter_config['filter_type']),
            key=f"filter_type_{column}"
        )
        filter_config['filter_type'] = filter_type
        
        if filter_type == 'range':
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                filter_config['selected_min_date'] = st.date_input(
                    "From date",
                    value=filter_config['selected_min_date'],
                    min_value=min_date,
                    max_value=max_date,
                    key=f"filter_date_min_{column}"
                )
            with col2:
                filter_config['selected_max_date'] = st.date_input(
                    "To date",
                    value=filter_config['selected_max_date'],
                    min_value=min_date,
                    max_value=max_date,
                    key=f"filter_date_max_{column}"
                )
        elif filter_type == 'after':
            filter_config['selected_min_date'] = st.date_input(
                "After date",
                value=filter_config.get('selected_min_date', min_date),
                min_value=min_date,
                max_value=max_date,
                key=f"filter_date_after_{column}"
            )
        elif filter_type == 'before':
            filter_config['selected_max_date'] = st.date_input(
                "Before date",
                value=filter_config.get('selected_max_date', max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"filter_date_before_{column}"
            )
    
    def _render_text_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for text filters."""
        # Filter type selection
        filter_type = st.selectbox(
            f"Text filter type for {column}",
            ['contains', 'starts_with', 'ends_with', 'exact'],
            index=['contains', 'starts_with', 'ends_with', 'exact'].index(filter_config['filter_type']),
            key=f"filter_type_{column}"
        )
        filter_config['filter_type'] = filter_type
        
        # Search text input
        filter_config['search_text'] = st.text_input(
            f"Search text for {column}",
            value=filter_config['search_text'],
            key=f"filter_text_{column}"
        )
        
        # Case sensitivity
        filter_config['case_sensitive'] = st.checkbox(
            "Case sensitive",
            value=filter_config['case_sensitive'],
            key=f"filter_case_{column}"
        )
    
    def _render_boolean_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for boolean filters."""
        selected_values = []
        
        if st.checkbox(
            "True", 
            value=True in filter_config['selected_values'],
            key=f"filter_bool_true_{column}"
        ):
            selected_values.append(True)
        
        if st.checkbox(
            "False", 
            value=False in filter_config['selected_values'],
            key=f"filter_bool_false_{column}"
        ):
            selected_values.append(False)
        
        filter_config['selected_values'] = selected_values
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all active filters to the DataFrame.
        
        Args:
            df: The DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for column, is_active in self.active_filters.items():
            if not is_active or column not in self.filters or column not in df.columns:
                continue
            
            filter_config = self.filters[column]
            filtered_df = self._apply_single_filter(filtered_df, column, filter_config)
        
        return filtered_df
    
    def _apply_single_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single filter to the DataFrame."""
        filter_type = filter_config['type']
        
        if filter_type == 'categorical':
            return self._apply_categorical_filter(df, column, filter_config)
        elif filter_type == 'numerical':
            return self._apply_numerical_filter(df, column, filter_config)
        elif filter_type == 'date':
            return self._apply_date_filter(df, column, filter_config)
        elif filter_type == 'text':
            return self._apply_text_filter(df, column, filter_config)
        elif filter_type == 'boolean':
            return self._apply_boolean_filter(df, column, filter_config)
        
        return df
    
    def _apply_categorical_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply categorical filter."""
        selected_values = filter_config['selected_values']
        filter_type = filter_config['filter_type']
        
        if not selected_values:
            return df.iloc[0:0]  # Return empty DataFrame
        
        # Convert to string for comparison
        column_str = df[column].astype(str)
        
        if filter_type == 'include':
            return df[column_str.isin(selected_values)]
        else:  # exclude
            return df[~column_str.isin(selected_values)]
    
    def _apply_numerical_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply numerical filter."""
        filter_type = filter_config['filter_type']
        
        if filter_type == 'range':
            min_val = filter_config['selected_min']
            max_val = filter_config['selected_max']
            return df[(df[column] >= min_val) & (df[column] <= max_val)]
        elif filter_type == 'greater_than':
            min_val = filter_config['selected_min']
            return df[df[column] > min_val]
        elif filter_type == 'less_than':
            max_val = filter_config['selected_max']
            return df[df[column] < max_val]
        
        return df
    
    def _apply_date_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply date filter."""
        filter_type = filter_config['filter_type']
        
        if filter_type == 'range':
            min_date = pd.to_datetime(filter_config['selected_min_date'])
            max_date = pd.to_datetime(filter_config['selected_max_date'])
            return df[(df[column] >= min_date) & (df[column] <= max_date)]
        elif filter_type == 'after':
            min_date = pd.to_datetime(filter_config['selected_min_date'])
            return df[df[column] > min_date]
        elif filter_type == 'before':
            max_date = pd.to_datetime(filter_config['selected_max_date'])
            return df[df[column] < max_date]
        
        return df
    
    def _apply_text_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply text filter."""
        search_text = filter_config['search_text']
        filter_type = filter_config['filter_type']
        case_sensitive = filter_config['case_sensitive']
        
        if not search_text:
            return df
        
        column_data = df[column].astype(str)
        
        if not case_sensitive:
            column_data = column_data.str.lower()
            search_text = search_text.lower()
        
        if filter_type == 'contains':
            mask = column_data.str.contains(search_text, na=False, regex=False)
        elif filter_type == 'starts_with':
            mask = column_data.str.startswith(search_text, na=False)
        elif filter_type == 'ends_with':
            mask = column_data.str.endswith(search_text, na=False)
        elif filter_type == 'exact':
            mask = column_data == search_text
        else:
            return df
        
        return df[mask]
    
    def _apply_boolean_filter(self, df: pd.DataFrame, column: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply boolean filter."""
        selected_values = filter_config['selected_values']
        
        if not selected_values:
            return df.iloc[0:0]  # Return empty DataFrame
        
        return df[df[column].isin(selected_values)]
    
    def get_active_filters_summary(self) -> Dict[str, str]:
        """
        Get a summary of active filters.
        
        Returns:
            Dictionary with filter summaries
        """
        summary = {}
        
        for column, is_active in self.active_filters.items():
            if not is_active or column not in self.filters:
                continue
            
            filter_config = self.filters[column]
            filter_type = filter_config['type']
            
            if filter_type == 'categorical':
                selected = filter_config['selected_values']
                action = filter_config['filter_type']
                summary[column] = f"{action.title()} {len(selected)} values"
            elif filter_type == 'numerical':
                ftype = filter_config['filter_type']
                if ftype == 'range':
                    summary[column] = f"Range: {filter_config['selected_min']:.2f} - {filter_config['selected_max']:.2f}"
                elif ftype == 'greater_than':
                    summary[column] = f"> {filter_config['selected_min']:.2f}"
                elif ftype == 'less_than':
                    summary[column] = f"< {filter_config['selected_max']:.2f}"
            elif filter_type == 'date':
                ftype = filter_config['filter_type']
                if ftype == 'range':
                    summary[column] = f"Date range: {filter_config['selected_min_date']} - {filter_config['selected_max_date']}"
                elif ftype == 'after':
                    summary[column] = f"After: {filter_config['selected_min_date']}"
                elif ftype == 'before':
                    summary[column] = f"Before: {filter_config['selected_max_date']}"
            elif filter_type == 'text':
                search_text = filter_config['search_text']
                ftype = filter_config['filter_type']
                summary[column] = f"{ftype.replace('_', ' ').title()}: '{search_text}'"
            elif filter_type == 'boolean':
                selected = filter_config['selected_values']
                summary[column] = f"Values: {', '.join(map(str, selected))}"
        
        return summary
    
    def clear_all_filters(self) -> None:
        """Clear all active filters."""
        for column in self.active_filters:
            self.active_filters[column] = False
    
    def get_filter_state(self) -> Dict[str, Any]:
        """Get the current filter state for serialization."""
        return {
            'filters': self.filters.copy(),
            'active_filters': self.active_filters.copy()
        }
    
    def set_filter_state(self, state: Dict[str, Any]) -> None:
        """Set the filter state from serialized data."""
        self.filters = state.get('filters', {})
        self.active_filters = state.get('active_filters', {})

