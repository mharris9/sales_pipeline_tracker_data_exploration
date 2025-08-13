"""
FilterManager class for creating and managing data filters.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import streamlit as st
from datetime import datetime, date

from src.utils.data_types import DataType

class FilterManager:
    """
    Manages filtering of data based on column types and user selections.
    """
    
    def __init__(self, state_manager=None):
        """Initialize the FilterManager."""
        # Get state manager instance
        if state_manager is not None:
            self.state_manager = state_manager
        elif hasattr(st.session_state, 'state_manager'):
            self.state_manager = st.session_state.state_manager
        else:
            # Create a temporary state manager for testing
            from src.services.state_manager import StateManager
            self.state_manager = StateManager()
        
        # Initialize state if needed
        if not self.state_manager.get_state('filters.filter_configs'):
            self.state_manager.update_state('filters.filter_configs', {})
        if not self.state_manager.get_state('filters.active_filters'):
            self.state_manager.update_state('filters.active_filters', {})
        if not self.state_manager.get_state('filters.filter_results'):
            self.state_manager.update_state('filters.filter_results', {})
        if not self.state_manager.get_state('filters.filter_summary'):
            self.state_manager.update_state('filters.filter_summary', {})
        
    def create_filters(self, df: pd.DataFrame, column_types: Dict[str, DataType]) -> None:
        """
        Create filter configurations for all columns.
        
        Args:
            df: The DataFrame to create filters for
            column_types: Dictionary mapping column names to data types
        """
        filter_configs = {}
        active_filters = {}
        
        for column, data_type in column_types.items():
            if column not in df.columns:
                continue
                
            active_filters[column] = False
            
            if data_type == DataType.CATEGORICAL:
                filter_configs[column] = self._create_categorical_filter(df[column])
            elif data_type == DataType.NUMERICAL:
                filter_configs[column] = self._create_numerical_filter(df[column])
            elif data_type == DataType.DATE:
                filter_configs[column] = self._create_date_filter(df[column])
            elif data_type == DataType.TEXT:
                filter_configs[column] = self._create_text_filter(df[column])
            elif data_type == DataType.BOOLEAN:
                filter_configs[column] = self._create_boolean_filter(df[column])
            
            # Convert any non-serializable values to serializable types
            if data_type == DataType.DATE:
                filter_configs[column]['min_date'] = filter_configs[column]['min_date'].isoformat()
                filter_configs[column]['max_date'] = filter_configs[column]['max_date'].isoformat()
                filter_configs[column]['selected_min_date'] = filter_configs[column]['selected_min_date'].isoformat()
                filter_configs[column]['selected_max_date'] = filter_configs[column]['selected_max_date'].isoformat()
            elif data_type == DataType.NUMERICAL:
                filter_configs[column]['min_value'] = float(filter_configs[column]['min_value'])
                filter_configs[column]['max_value'] = float(filter_configs[column]['max_value'])
                filter_configs[column]['selected_min'] = float(filter_configs[column]['selected_min'])
                filter_configs[column]['selected_max'] = float(filter_configs[column]['selected_max'])
                if 'percentiles' in filter_configs[column]:
                    filter_configs[column]['percentiles'] = {
                        k: float(v) for k, v in filter_configs[column]['percentiles'].items()
                    }
        
        # Update state
        self.state_manager.update_state('filters.filter_configs', filter_configs)
        self.state_manager.update_state('filters.active_filters', active_filters)
        self.state_manager.update_state('filters.filter_results', {})
    
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
        filter_configs = self.state_manager.get_state('filters.filter_configs', {})
        if column not in filter_configs:
            return
        
        filter_config = filter_configs[column]
        active_filters = self.state_manager.get_state('filters.active_filters', {})
        
        # Filter activation checkbox
        is_active = st.checkbox(
            f"Filter {column}", 
            value=active_filters.get(column, False),
            key=f"filter_active_{column}"
        )
        
        # Update active state
        active_filters[column] = is_active
        self.state_manager.update_state('filters.active_filters', active_filters)
        
        if not is_active:
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
        
        # Update filter config
        filter_configs[column] = filter_config
        self.state_manager.update_state('filters.filter_configs', filter_configs)

    def render_all_filters_ui(self) -> None:
        """
        Render filter UI for all available columns.
        This method is called from the filters page.
        """
        df = self.state_manager.get_state('data.current_df')
        column_types = self.state_manager.get_state('data.column_types', {})
        
        if df is None or not column_types:
            st.warning("No data available for filtering.")
            return
        
        # Create filters for all columns if they don't exist
        self.create_filters(df, column_types)
        
        # Render filter UI for each column
        for column in df.columns:
            if column in column_types:
                data_type = column_types[column]
                self.render_filter_ui(column, data_type)
    
    def _render_categorical_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for categorical filters with proper callbacks and validation."""
        unique_values = filter_config['unique_values']
        
        # Initialize session state for this filter if not exists
        state_key = f"filter_state_{column}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                'filter_type': filter_config['filter_type'],
                'selected_values': filter_config['selected_values'].copy()
            }
        
        # Filter type selection with callback
        def update_filter_type():
            st.session_state[state_key]['filter_type'] = st.session_state[f"filter_type_{column}"]
            filter_config['filter_type'] = st.session_state[f"filter_type_{column}"]
            self.state_manager.trigger_rerun()
        
        filter_type = st.radio(
            f"Filter type for {column}",
            ['include', 'exclude'],
            index=0 if st.session_state[state_key]['filter_type'] == 'include' else 1,
            key=f"filter_type_{column}",
            horizontal=True,
            on_change=update_filter_type
        )
        
        # Value selection with validation
        if len(unique_values) <= 10:
            # Use checkboxes for small number of values
            st.write("Select values to include/exclude:")
            selected_values = []
            
            # Create columns for better layout
            cols = st.columns(2)
            for i, value in enumerate(unique_values):
                with cols[i % 2]:
                    checkbox_key = f"filter_cat_{column}_{value}"
                    
                    def update_selected_values(val=value):
                        current_values = st.session_state[state_key]['selected_values']
                        if st.session_state[checkbox_key]:
                            if val not in current_values:
                                current_values.append(val)
                        else:
                            if val in current_values:
                                current_values.remove(val)
                        st.session_state[state_key]['selected_values'] = current_values
                        filter_config['selected_values'] = current_values
                        self.state_manager.trigger_rerun()
                    
                    if st.checkbox(
                        value,
                        value=value in st.session_state[state_key]['selected_values'],
                        key=checkbox_key,
                        help=f"Toggle {value} in filter",
                        on_change=update_selected_values
                    ):
                        selected_values.append(value)
            
            # Show summary of selected values
            if selected_values:
                st.caption(f"Selected: {', '.join(selected_values)}")
            else:
                st.caption("No values selected")
        else:
            # Use multiselect for large number of values
            def update_multiselect():
                st.session_state[state_key]['selected_values'] = st.session_state[f"filter_multiselect_{column}"]
                filter_config['selected_values'] = st.session_state[f"filter_multiselect_{column}"]
                self.state_manager.trigger_rerun()
            
            selected_values = st.multiselect(
                f"Select values for {column}",
                options=unique_values,
                default=st.session_state[state_key]['selected_values'],
                key=f"filter_multiselect_{column}",
                on_change=update_multiselect
            )
    
    def _render_numerical_filter_ui(self, column: str, filter_config: Dict[str, Any]) -> None:
        """Render UI for numerical filters with proper callbacks and validation."""
        min_val = filter_config['min_value']
        max_val = filter_config['max_value']
        
        # Initialize session state for this filter if not exists
        state_key = f"filter_state_{column}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                'filter_type': filter_config['filter_type'],
                'selected_min': filter_config['selected_min'],
                'selected_max': filter_config['selected_max']
            }
        
        # Filter type selection with callback
        def update_filter_type():
            st.session_state[state_key]['filter_type'] = st.session_state[f"filter_type_{column}"]
            filter_config['filter_type'] = st.session_state[f"filter_type_{column}"]
            self.state_manager.trigger_rerun()
        
        filter_type = st.selectbox(
            f"Filter type for {column}",
            ['range', 'greater_than', 'less_than'],
            index=['range', 'greater_than', 'less_than'].index(st.session_state[state_key]['filter_type']),
            key=f"filter_type_{column}",
            on_change=update_filter_type
        )
        
        if filter_type == 'range':
            st.write("Enter exact values or use the slider:")
            
            # Manual number inputs in columns
            col1, col2 = st.columns(2)
            with col1:
                manual_min = st.number_input(
                    f"Min {column}",
                    value=float(st.session_state[state_key]['selected_min']),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    format="%.2f",
                    key=f"filter_range_min_{column}"
                )
            with col2:
                manual_max = st.number_input(
                    f"Max {column}",
                    value=float(st.session_state[state_key]['selected_max']),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    format="%.2f",
                    key=f"filter_range_max_{column}"
                )
            
            # Update session state and config
            st.session_state[state_key]['selected_min'] = manual_min
            st.session_state[state_key]['selected_max'] = manual_max
            filter_config['selected_min'] = manual_min
            filter_config['selected_max'] = manual_max
            
            # Optional slider for visual reference
            st.slider(
                f"Visual range for {column}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(manual_min), float(manual_max)),
                key=f"filter_range_slider_{column}",
                disabled=True  # Make it read-only since we're using manual inputs
            )
            
        elif filter_type == 'greater_than':
            manual_min = st.number_input(
                f"Greater than",
                value=float(st.session_state[state_key]['selected_min']),
                min_value=float(min_val),
                max_value=float(max_val),
                format="%.2f",
                key=f"filter_gt_{column}"
            )
            st.session_state[state_key]['selected_min'] = manual_min
            filter_config['selected_min'] = manual_min
            
        elif filter_type == 'less_than':
            manual_max = st.number_input(
                f"Less than",
                value=float(st.session_state[state_key]['selected_max']),
                min_value=float(min_val),
                max_value=float(max_val),
                format="%.2f",
                key=f"filter_lt_{column}"
            )
            st.session_state[state_key]['selected_max'] = manual_max
            filter_config['selected_max'] = manual_max
        
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
        original_count = len(filtered_df)
        
        # Store the original DataFrame
        self.state_manager.update_state('filters.original_df', df.copy())
        
        # Get filter state
        active_filters = self.state_manager.get_state('filters.active_filters', {})
        filter_configs = self.state_manager.get_state('filters.filter_configs', {})
        
        # Apply filters sequentially and track changes
        filter_results = {}
        
        for column, is_active in active_filters.items():
            if not is_active or column not in filter_configs or column not in df.columns:
                continue
            
            filter_config = filter_configs[column]
            pre_filter_count = len(filtered_df)
            filtered_df = self._apply_single_filter(filtered_df, column, filter_config)
            post_filter_count = len(filtered_df)
            
            # Collect debug info
            filter_results[column] = {
                'type': filter_config['type'],
                'pre_count': pre_filter_count,
                'post_count': post_filter_count,
                'filtered_count': post_filter_count,  # This is the number of records that match the filter
                'total_count': original_count
            }
            
            if filter_config['type'] == 'categorical':
                filter_results[column].update({
                    'mode': filter_config['filter_type'],
                    'selected_values': filter_config['selected_values'],
                    'value_counts': filtered_df[column].value_counts().to_dict()
                })
        
        # Store debug info in state
        self.state_manager.update_state('filters.filter_results', filter_results)
        self.state_manager.update_state('filters.filter_summary', {
            'original_count': original_count,
            'filtered_count': len(filtered_df),
            'total_filtered': original_count - len(filtered_df)
        })
        
        # Display debug information
        for column, info in filter_results.items():
            st.write(f"Filter applied to {column}:")
            st.write(f"- Filter type: {info['type']}")
            if info['type'] == 'categorical':
                st.write(f"- Mode: {info['mode']}")
                st.write(f"- Selected values: {info['selected_values']}")
            st.write(f"- Records before: {info['pre_count']}")
            st.write(f"- Records after: {info['post_count']}")
            st.write(f"- Records filtered: {info['filtered_count']}")
            
            if info['type'] == 'categorical':
                st.write("Value counts in filtered data:")
                st.write(pd.Series(info['value_counts']))
        
        st.write(f"Total records: {original_count} → {len(filtered_df)}")
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
        unique_values = filter_config['unique_values']
        
        # If all values are selected, return the original DataFrame
        if set(selected_values) == set(unique_values):
            return df
        
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
        """Clear all active filters and reset filter configurations."""
        # Get filter state
        active_filters = self.state_manager.get_state('filters.active_filters', {})
        filter_configs = self.state_manager.get_state('filters.filter_configs', {})
        
        # Clear active states and session state
        for column in active_filters:
            # Clear active state
            active_filters[column] = False
            
            # Clear checkbox state in session
            checkbox_key = f"filter_active_{column}"
            if checkbox_key in st.session_state:
                st.session_state[checkbox_key] = False
            
            # Reset filter configurations to default
            if column in filter_configs:
                filter_config = filter_configs[column]
                state_key = f"filter_state_{column}"
                
                if filter_config['type'] == 'categorical':
                    unique_values = filter_config['unique_values']
                    filter_config['selected_values'] = unique_values.copy()
                    filter_config['filter_type'] = 'include'
                    if state_key in st.session_state:
                        st.session_state[state_key] = {
                            'filter_type': 'include',
                            'selected_values': unique_values.copy()
                        }
                elif filter_config['type'] == 'numerical':
                    min_val = filter_config['min_value']
                    max_val = filter_config['max_value']
                    filter_config['selected_min'] = min_val
                    filter_config['selected_max'] = max_val
                    filter_config['filter_type'] = 'range'
                    if state_key in st.session_state:
                        st.session_state[state_key] = {
                            'filter_type': 'range',
                            'selected_min': min_val,
                            'selected_max': max_val
                        }
                elif filter_config['type'] == 'date':
                    min_date = filter_config['min_date']
                    max_date = filter_config['max_date']
                    filter_config['selected_min_date'] = min_date
                    filter_config['selected_max_date'] = max_date
                    filter_config['filter_type'] = 'range'
                    if state_key in st.session_state:
                        st.session_state[state_key] = {
                            'filter_type': 'range',
                            'selected_min_date': min_date,
                            'selected_max_date': max_date
                        }
                elif filter_config['type'] == 'text':
                    filter_config['search_text'] = ''
                    filter_config['filter_type'] = 'contains'
                    filter_config['case_sensitive'] = False
                    if state_key in st.session_state:
                        st.session_state[state_key] = {
                            'filter_type': 'contains',
                            'search_text': '',
                            'case_sensitive': False
                        }
                elif filter_config['type'] == 'boolean':
                    filter_config['selected_values'] = [True, False]
                    filter_config['filter_type'] = 'include'
                    if state_key in st.session_state:
                        st.session_state[state_key] = {
                            'filter_type': 'include',
                            'selected_values': [True, False]
                        }
                
                # Update filter config
                filter_configs[column] = filter_config
        
        # Update state
        self.state_manager.update_state('filters.active_filters', active_filters)
        self.state_manager.update_state('filters.filter_configs', filter_configs)
        self.state_manager.update_state('filters.filter_results', {})
        self.state_manager.update_state('filters.filter_summary', {})
        
        # Clear any remaining filter-related session state
        for key in list(st.session_state.keys()):
            if key.startswith('filter_'):
                del st.session_state[key]
    
    def update_filter(self, column: str, filter_config: Dict[str, Any]) -> None:
        """
        Update a filter configuration.
        
        Args:
            column: Column name
            filter_config: New filter configuration
        """
        # Get current filter configs
        filter_configs = self.state_manager.get_state('filters.filter_configs', {})
        active_filters = self.state_manager.get_state('filters.active_filters', {})
        
        # Get current filter config
        current_config = filter_configs.get(column, {})
        
        # Merge new config with current config
        merged_config = current_config.copy()
        merged_config.update(filter_config)
        
        # Convert any non-serializable values to serializable types
        if merged_config['type'] == 'date':
            if isinstance(merged_config.get('selected_min_date'), pd.Timestamp):
                merged_config['selected_min_date'] = merged_config['selected_min_date'].isoformat()
            if isinstance(merged_config.get('selected_max_date'), pd.Timestamp):
                merged_config['selected_max_date'] = merged_config['selected_max_date'].isoformat()
        elif merged_config['type'] == 'numerical':
            if 'selected_min' in merged_config:
                merged_config['selected_min'] = float(merged_config['selected_min'])
            if 'selected_max' in merged_config:
                merged_config['selected_max'] = float(merged_config['selected_max'])
        
        # Update filter config and active state
        filter_configs[column] = merged_config
        active_filters[column] = True
        
        # Update state
        self.state_manager.update_state('filters.filter_configs', filter_configs)
        self.state_manager.update_state('filters.active_filters', active_filters)
        
        # Apply filters to update results
        df = self.state_manager.get_state('data.original_df')
        if df is not None:
            self.apply_filters(df)
    
    def get_filter_state(self) -> Dict[str, Any]:
        """Get the current filter state for serialization."""
        return {
            'filter_configs': self.state_manager.get_state('filters.filter_configs', {}),
            'active_filters': self.state_manager.get_state('filters.active_filters', {}),
            'filter_results': self.state_manager.get_state('filters.filter_results', {}),
            'filter_summary': self.state_manager.get_state('filters.filter_summary', {})
        }
    
    def set_filter_state(self, state: Dict[str, Any]) -> None:
        """Set the filter state from serialized data."""
        # Get current state
        current_configs = self.state_manager.get_state('filters.filter_configs', {})
        current_active = self.state_manager.get_state('filters.active_filters', {})
        
        # Merge new state with current state
        filter_configs = current_configs.copy()
        filter_configs.update(state.get('filter_configs', {}))
        
        active_filters = current_active.copy()
        active_filters.update(state.get('active_filters', {}))
        
        # Update state
        self.state_manager.update_state('filters.filter_configs', filter_configs)
        self.state_manager.update_state('filters.active_filters', active_filters)
        self.state_manager.update_state('filters.filter_results', state.get('filter_results', {}))
        self.state_manager.update_state('filters.filter_summary', state.get('filter_summary', {}))

