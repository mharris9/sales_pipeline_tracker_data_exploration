"""
ReportEngine class for generating various types of reports and visualizations.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt

from config.settings import CHART_HEIGHT, CHART_THEME, COLOR_PALETTE
from utils.data_types import DataType, get_compatible_columns, calculate_statistics
from utils.column_mapping import column_mapper

class ReportEngine:
    """
    Generates reports and visualizations for sales pipeline data analysis.
    """
    
    def __init__(self):
        """Initialize the ReportEngine."""
        self.available_reports: Dict[str, Dict[str, Any]] = {}
        self._register_default_reports()
    
    def _register_default_reports(self) -> None:
        """Register default report types."""
        
        self.available_reports = {
            'descriptive_statistics': {
                'name': 'Descriptive Statistics',
                'description': 'Summary statistics for selected columns',
                'function': self.generate_descriptive_statistics,
                'requires_axes': False,
                'supports_groupby': True
            },
            'histogram': {
                'name': 'Histogram',
                'description': 'Distribution of numerical or date data',
                'function': self.generate_histogram,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.NUMERICAL, DataType.DATE]
                },
                'supports_groupby': True
            },
            'bar_chart': {
                'name': 'Bar Chart',
                'description': 'Categorical data visualization',
                'function': self.generate_bar_chart,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.CATEGORICAL, DataType.BOOLEAN],
                    'y_axis': [DataType.NUMERICAL]
                },
                'supports_groupby': True
            },
            'scatter_plot': {
                'name': 'Scatter Plot',
                'description': 'Relationship between two numerical variables',
                'function': self.generate_scatter_plot,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.NUMERICAL, DataType.DATE],
                    'y_axis': [DataType.NUMERICAL]
                },
                'supports_groupby': True
            },
            'line_chart': {
                'name': 'Line Chart',
                'description': 'Trends over time or numerical progression',
                'function': self.generate_line_chart,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.DATE, DataType.NUMERICAL],
                    'y_axis': [DataType.NUMERICAL]
                },
                'supports_groupby': True
            },
            'correlation_heatmap': {
                'name': 'Correlation Heatmap',
                'description': 'Correlation matrix of numerical variables',
                'function': self.generate_correlation_heatmap,
                'requires_axes': False,
                'supports_groupby': False
            },
            'box_plot': {
                'name': 'Box Plot',
                'description': 'Distribution and outliers in numerical data',
                'function': self.generate_box_plot,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.CATEGORICAL, DataType.BOOLEAN],
                    'y_axis': [DataType.NUMERICAL]
                },
                'supports_groupby': True
            },
            'time_series': {
                'name': 'Time Series',
                'description': 'Data trends over time',
                'function': self.generate_time_series,
                'requires_axes': True,
                'axis_requirements': {
                    'x_axis': [DataType.DATE],
                    'y_axis': [DataType.NUMERICAL]
                },
                'supports_groupby': True
            }
        }
    
    def get_available_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get all available report types."""
        return self.available_reports.copy()
    
    def get_compatible_columns_for_report(self, report_type: str, axis: str, 
                                        column_types: Dict[str, DataType]) -> List[str]:
        """
        Get columns compatible with a specific report type and axis.
        
        Args:
            report_type: Type of report
            axis: Axis type (x_axis, y_axis, etc.)
            column_types: Dictionary mapping column names to data types
            
        Returns:
            List of compatible column names
        """
        if report_type not in self.available_reports:
            return list(column_types.keys())
        
        report_config = self.available_reports[report_type]
        
        if 'axis_requirements' not in report_config:
            return list(column_types.keys())
        
        if axis not in report_config['axis_requirements']:
            return list(column_types.keys())
        
        compatible_types = report_config['axis_requirements'][axis]
        
        return [
            col for col, dtype in column_types.items() 
            if dtype in compatible_types
        ]
    
    def generate_report(self, report_type: str, df: pd.DataFrame, 
                       config: Dict[str, Any], 
                       exclusion_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
        """
        Generate a report of the specified type.
        
        Args:
            report_type: Type of report to generate
            df: DataFrame to analyze
            config: Report configuration
            
        Returns:
            Tuple of (figure, data_table) - either can be None
        """
        if report_type not in self.available_reports:
            st.error(f"Unknown report type: {report_type}")
            return None, None
        
        if df.empty:
            st.warning("No data available for report generation")
            return None, None
        
        try:
            report_function = self.available_reports[report_type]['function']
            figure, data_table = report_function(df, config)
            
            # Add outlier exclusion notes to charts
            if figure is not None and exclusion_info is not None and exclusion_info.get('outliers_excluded', False):
                exclusion_note = self._get_exclusion_note(exclusion_info)
                figure.add_annotation(
                    text=exclusion_note,
                    xref="paper", yref="paper",
                    x=0.02, y=-0.1,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
                
                # Adjust layout to make room for annotation
                figure.update_layout(margin=dict(b=100))
            
            return figure, data_table
            
        except Exception as e:
            st.error(f"Error generating {report_type}: {str(e)}")
            return None, None
    
    def generate_descriptive_statistics(self, df: pd.DataFrame, 
                                      config: Dict[str, Any]) -> Tuple[None, pd.DataFrame]:
        """Generate descriptive statistics report."""
        selected_columns = config.get('selected_columns', df.columns.tolist())
        group_by_column = config.get('group_by_column')
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        if group_by_column and group_by_column in df_deduplicated.columns:
            # Generate grouped statistics
            stats_data = []
            
            for group_value in df_deduplicated[group_by_column].unique():
                if pd.isna(group_value):
                    continue
                
                group_df = df_deduplicated[df_deduplicated[group_by_column] == group_value]
                
                for column in selected_columns:
                    if column not in group_df.columns or column == group_by_column:
                        continue
                    
                    # Detect column type for this subset
                    from utils.data_types import detect_data_type
                    col_type = detect_data_type(group_df[column], column)
                    
                    stats = calculate_statistics(group_df[column], col_type)
                    
                    row = {
                        'Group': str(group_value),
                        'Column': column_mapper.map_column_name(column),
                        'Data Type': col_type.value,
                        'Count': f"{stats.get('count', 0):,}",
                        'Non-Null': f"{stats.get('non_null_count', 0):,}",
                        'Missing': f"{stats.get('null_count', 0):,}",
                        'Missing %': f"{stats.get('null_percentage', 0):.1f}%"
                    }
                    
                    # Add type-specific statistics
                    if col_type == DataType.NUMERICAL:
                        row.update({
                            'Mean': f"{stats.get('mean', 0):,.0f}",
                            'Median': f"{stats.get('median', 0):,.0f}",
                            'Std Dev': f"{stats.get('std', 0):,.0f}",
                            'Min': f"{stats.get('min', 0):,.0f}",
                            'Max': f"{stats.get('max', 0):,.0f}",
                            'Skewness': f"{stats.get('skewness', 0):,.0f}",
                            'Kurtosis': f"{stats.get('kurtosis', 0):,.0f}"
                        })
                    elif col_type == DataType.CATEGORICAL:
                        row.update({
                            'Unique Values': f"{stats.get('unique_count', 0):,}",
                            'Most Frequent': stats.get('most_frequent', 'N/A'),
                            'Most Freq Count': f"{stats.get('most_frequent_count', 0):,}"
                        })
                    elif col_type == DataType.DATE:
                        row.update({
                            'Earliest': str(stats.get('earliest', 'N/A')),
                            'Latest': str(stats.get('latest', 'N/A')),
                            'Range (Days)': f"{stats.get('range_days', 0):,}"
                        })
                    
                    stats_data.append(row)
            
            stats_df = pd.DataFrame(stats_data)
        else:
            # Generate overall statistics
            stats_data = []
            
            for column in selected_columns:
                if column not in df_deduplicated.columns:
                    continue
                
                # Detect column type
                from utils.data_types import detect_data_type
                col_type = detect_data_type(df_deduplicated[column], column)
                
                stats = calculate_statistics(df_deduplicated[column], col_type)
                
                row = {
                    'Column': column_mapper.map_column_name(column),
                    'Data Type': col_type.value,
                    'Count': f"{stats.get('count', 0):,}",
                    'Non-Null': f"{stats.get('non_null_count', 0):,}",
                    'Missing': f"{stats.get('null_count', 0):,}",
                    'Missing %': f"{stats.get('null_percentage', 0):.1f}%"
                }
                
                # Add type-specific statistics
                if col_type == DataType.NUMERICAL:
                    row.update({
                        'Mean': f"{stats.get('mean', 0):,.0f}",
                        'Median': f"{stats.get('median', 0):,.0f}",
                        'Std Dev': f"{stats.get('std', 0):,.0f}",
                        'Min': f"{stats.get('min', 0):,.0f}",
                        'Max': f"{stats.get('max', 0):,.0f}",
                        'Skewness': f"{stats.get('skewness', 0):,.0f}",
                        'Kurtosis': f"{stats.get('kurtosis', 0):,.0f}"
                    })
                elif col_type == DataType.CATEGORICAL:
                    row.update({
                        'Unique Values': f"{stats.get('unique_count', 0):,}",
                        'Most Frequent': stats.get('most_frequent', 'N/A'),
                        'Most Freq Count': f"{stats.get('most_frequent_count', 0):,}"
                    })
                elif col_type == DataType.DATE:
                    row.update({
                        'Earliest': str(stats.get('earliest', 'N/A')),
                        'Latest': str(stats.get('latest', 'N/A')),
                        'Range (Days)': f"{stats.get('range_days', 0):,}"
                    })
                
                stats_data.append(row)
            
            stats_df = pd.DataFrame(stats_data)
        
        # Apply sorting to the statistics DataFrame
        sort_by = config.get('sort_by', 'none')
        if sort_by != 'none' and not stats_df.empty:
            if sort_by == 'column_name':
                stats_df = stats_df.sort_values('Column')
            elif sort_by == 'data_type':
                stats_df = stats_df.sort_values('Data Type')
            elif sort_by == 'count_descending':
                # Convert Count column to numeric for sorting (remove commas)
                stats_df['_sort_count'] = stats_df['Count'].str.replace(',', '').astype(int)
                stats_df = stats_df.sort_values('_sort_count', ascending=False)
                stats_df = stats_df.drop('_sort_count', axis=1)
            elif sort_by == 'missing_ascending':
                # Convert Missing column to numeric for sorting (remove commas)
                stats_df['_sort_missing'] = stats_df['Missing'].str.replace(',', '').astype(int)
                stats_df = stats_df.sort_values('_sort_missing', ascending=True)
                stats_df = stats_df.drop('_sort_missing', axis=1)
            
            # Reset index after sorting
            stats_df = stats_df.reset_index(drop=True)
        
        return None, stats_df
    
    def generate_histogram(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate histogram visualization."""
        x_column = config.get('x_axis')
        group_by_column = config.get('group_by_column')
        bins = config.get('bins', 30)
        upper_limit = config.get('upper_limit')
        upper_limit_label = config.get('upper_limit_label', f">= {upper_limit}")
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        def prepare_data_with_limit(data_series):
            """Prepare data by applying upper limit if specified."""
            clean_data = data_series.dropna()
            
            if upper_limit is not None and pd.api.types.is_numeric_dtype(clean_data):
                # Create a copy to avoid modifying original data
                limited_data = clean_data.copy()
                
                # Count values above the limit
                above_limit_count = (limited_data > upper_limit).sum()
                
                if above_limit_count > 0:
                    # Replace values above limit with the limit value
                    limited_data[limited_data > upper_limit] = upper_limit
                    
                return limited_data, above_limit_count
            
            return clean_data, 0
        
        fig = go.Figure()
        title_suffix = ""
        
        if group_by_column and group_by_column in df_deduplicated.columns:
            # Grouped histogram
            total_above_limit = 0
            
            for i, group_value in enumerate(df_deduplicated[group_by_column].unique()):
                if pd.isna(group_value):
                    continue
                
                group_data = df_deduplicated[df_deduplicated[group_by_column] == group_value][x_column]
                limited_data, above_count = prepare_data_with_limit(group_data)
                total_above_limit += above_count
                
                fig.add_trace(go.Histogram(
                    x=limited_data,
                    name=str(group_value),
                    nbinsx=bins,
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    opacity=0.7
                ))
            
            if total_above_limit > 0:
                title_suffix = f" (Upper Limit: {upper_limit_label}, {total_above_limit:,} values grouped)"
                
        else:
            # Simple histogram
            limited_data, above_count = prepare_data_with_limit(df_deduplicated[x_column])
            
            fig.add_trace(go.Histogram(
                x=limited_data,
                nbinsx=bins,
                marker_color=COLOR_PALETTE[0]
            ))
            
            if above_count > 0:
                title_suffix = f" (Upper Limit: {upper_limit_label}, {above_count:,} values grouped)"
        
        # Create user-friendly labels
        x_display = column_mapper.map_column_name(x_column)
        
        # Update x-axis to show the upper limit clearly
        layout_updates = {
            'title': f"Distribution of {x_display}{title_suffix}",
            'xaxis_title': x_display,
            'yaxis_title': "Frequency",
            'template': CHART_THEME,
            'height': CHART_HEIGHT,
            'bargap': 0.1
        }
        
        # Add annotation if upper limit is used
        if upper_limit is not None and (title_suffix != ""):
            fig.add_annotation(
                text=f"Note: Values above {upper_limit:,.0f} are grouped into the rightmost bin",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        fig.update_layout(**layout_updates)
        
        return fig, None
    
    def generate_bar_chart(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate bar chart visualization."""
        x_column = config.get('x_axis')
        y_column = config.get('y_axis')
        group_by_column = config.get('group_by_column')
        aggregation = config.get('aggregation', 'sum')
        sort_by = config.get('sort_by', 'none')
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        if not y_column or y_column not in df.columns:
            raise ValueError("Y-axis column not specified or not found")
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        def apply_sorting(data, x_col, y_col, sort_option):
            """Apply sorting to aggregated data."""
            if sort_option == 'category_name':
                return data.sort_values(x_col)
            elif sort_option == 'value_ascending':
                return data.sort_values(y_col, ascending=True)
            elif sort_option == 'value_descending':
                return data.sort_values(y_col, ascending=False)
            else:  # 'none' or any other value
                return data
        
        # Aggregate data
        if group_by_column and group_by_column in df_deduplicated.columns:
            agg_data = df_deduplicated.groupby([x_column, group_by_column])[y_column].agg(aggregation).reset_index()
            
            # Check if all aggregated values are NaN
            if agg_data[y_column].isna().all():
                # Create empty figure with informative message
                fig = go.Figure()
                fig.add_annotation(
                    text="📊 The report returned no results with current filters<br><br>" +
                         "This usually means:<br>" +
                         "• No data matches the selected criteria<br>" +
                         "• All values for this metric are unavailable<br>" +
                         "• Try adjusting your filters or selecting different columns",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                    align="center"
                )
            else:
                fig = go.Figure()
                
                # Get unique group values and sort them for consistent ordering
                group_values = sorted([gv for gv in agg_data[group_by_column].unique() if not pd.isna(gv)])
                has_nan_values = agg_data[y_column].isna().any()
                
                for i, group_value in enumerate(group_values):
                    group_data = agg_data[agg_data[group_by_column] == group_value]
                    
                    # Apply sorting to this group's data
                    group_data_sorted = apply_sorting(group_data, x_column, y_column, sort_by)
                    
                    # Replace NaN values with 0 for display
                    group_data_sorted[y_column] = group_data_sorted[y_column].fillna(0)
                    
                    fig.add_trace(go.Bar(
                        x=group_data_sorted[x_column],
                        y=group_data_sorted[y_column],
                        name=str(group_value),
                        marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)]
                    ))
                
                # Add note about NaN values if any exist
                if has_nan_values:
                    fig.add_annotation(
                        text="Note: Categories with no data are shown as zero",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
        else:
            agg_data = df_deduplicated.groupby(x_column)[y_column].agg(aggregation).reset_index()
            
            # Check if all aggregated values are NaN
            if agg_data[y_column].isna().all():
                # Create empty figure with informative message
                fig = go.Figure()
                fig.add_annotation(
                    text="📊 The report returned no results with current filters<br><br>" +
                         "This usually means:<br>" +
                         "• No data matches the selected criteria<br>" +
                         "• All values for this metric are unavailable<br>" +
                         "• Try adjusting your filters or selecting different columns",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                    align="center"
                )
            else:
                # Filter out NaN values for display but keep category names
                display_data = agg_data.copy()
                
                # Apply sorting to the aggregated data
                agg_data_sorted = apply_sorting(display_data, x_column, y_column, sort_by)
                
                # Replace NaN values with 0 for display purposes, but add annotation
                has_nan_values = agg_data_sorted[y_column].isna().any()
                agg_data_sorted[y_column] = agg_data_sorted[y_column].fillna(0)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=agg_data_sorted[x_column],
                        y=agg_data_sorted[y_column],
                        marker_color=COLOR_PALETTE[0]
                    )
                ])
                
                # Add note about NaN values if any exist
                if has_nan_values:
                    fig.add_annotation(
                        text="Note: Categories with no data are shown as zero",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
        
        # Add sorting info to title
        sort_suffix = ""
        if sort_by != 'none':
            sort_labels = {
                'category_name': ' (Sorted by Category)',
                'value_ascending': ' (Sorted by Value ↑)',
                'value_descending': ' (Sorted by Value ↓)'
            }
            sort_suffix = sort_labels.get(sort_by, '')
        
        # Create user-friendly labels
        x_display = column_mapper.map_column_name(x_column)
        y_display = column_mapper.map_column_name(y_column)
        agg_display = aggregation.title()
        
        # Map aggregation to friendly terms
        agg_mapping = {
            'Sum': 'Total',
            'Mean': 'Average',
            'Count': 'Count of',
            'Min': 'Minimum',
            'Max': 'Maximum'
        }
        agg_friendly = agg_mapping.get(agg_display, agg_display)
        
        fig.update_layout(
            title=f"{agg_friendly} {y_display} by {x_display}{sort_suffix} (Most Recent Snapshots)",
            xaxis_title=x_display,
            yaxis_title=f"{agg_friendly} {y_display}",
            template=CHART_THEME,
            height=CHART_HEIGHT
        )
        
        return fig, None
    
    def generate_scatter_plot(self, df: pd.DataFrame, 
                             config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate scatter plot visualization."""
        x_column = config.get('x_axis')
        y_column = config.get('y_axis')
        group_by_column = config.get('group_by_column')
        size_column = config.get('size_column')
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        if not y_column or y_column not in df.columns:
            raise ValueError("Y-axis column not specified or not found")
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        fig = go.Figure()
        
        if group_by_column and group_by_column in df_deduplicated.columns:
            for i, group_value in enumerate(df_deduplicated[group_by_column].unique()):
                if pd.isna(group_value):
                    continue
                
                group_data = df_deduplicated[df_deduplicated[group_by_column] == group_value]
                
                scatter_kwargs = {
                    'x': group_data[x_column],
                    'y': group_data[y_column],
                    'mode': 'markers',
                    'name': str(group_value),
                    'marker': dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
                }
                
                if size_column and size_column in df_deduplicated.columns:
                    scatter_kwargs['marker']['size'] = group_data[size_column]
                    scatter_kwargs['marker']['sizemode'] = 'diameter'
                    scatter_kwargs['marker']['sizeref'] = 2. * max(group_data[size_column]) / (40.**2)
                    scatter_kwargs['marker']['sizemin'] = 4
                
                fig.add_trace(go.Scatter(**scatter_kwargs))
        else:
            scatter_kwargs = {
                'x': df_deduplicated[x_column],
                'y': df_deduplicated[y_column],
                'mode': 'markers',
                'marker': dict(color=COLOR_PALETTE[0])
            }
            
            if size_column and size_column in df_deduplicated.columns:
                scatter_kwargs['marker']['size'] = df_deduplicated[size_column]
                scatter_kwargs['marker']['sizemode'] = 'diameter'
                scatter_kwargs['marker']['sizeref'] = 2. * max(df_deduplicated[size_column]) / (40.**2)
                scatter_kwargs['marker']['sizemin'] = 4
            
            fig.add_trace(go.Scatter(**scatter_kwargs))
        
        # Create user-friendly labels
        x_display = column_mapper.map_column_name(x_column)
        y_display = column_mapper.map_column_name(y_column)
        
        fig.update_layout(
            title=f"{y_display} vs {x_display} (Most Recent Snapshots)",
            xaxis_title=x_display,
            yaxis_title=y_display,
            template=CHART_THEME,
            height=CHART_HEIGHT
        )
        
        return fig, None
    
    def generate_line_chart(self, df: pd.DataFrame, 
                           config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate line chart visualization."""
        x_column = config.get('x_axis')
        y_column = config.get('y_axis')
        group_by_column = config.get('group_by_column')
        aggregation = config.get('aggregation', 'mean')
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        if not y_column or y_column not in df.columns:
            raise ValueError("Y-axis column not specified or not found")
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        fig = go.Figure()
        
        if group_by_column and group_by_column in df_deduplicated.columns:
            for i, group_value in enumerate(df_deduplicated[group_by_column].unique()):
                if pd.isna(group_value):
                    continue
                
                group_data = df_deduplicated[df_deduplicated[group_by_column] == group_value]
                agg_data = group_data.groupby(x_column)[y_column].agg(aggregation).reset_index()
                agg_data = agg_data.sort_values(x_column)
                
                fig.add_trace(go.Scatter(
                    x=agg_data[x_column],
                    y=agg_data[y_column],
                    mode='lines+markers',
                    name=str(group_value),
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
                ))
        else:
            agg_data = df_deduplicated.groupby(x_column)[y_column].agg(aggregation).reset_index()
            agg_data = agg_data.sort_values(x_column)
            
            fig.add_trace(go.Scatter(
                x=agg_data[x_column],
                y=agg_data[y_column],
                mode='lines+markers',
                line=dict(color=COLOR_PALETTE[0])
            ))
        
        x_display = column_mapper.map_column_name(x_column)
        y_display = column_mapper.map_column_name(y_column)
        
        fig.update_layout(
            title=f"{aggregation.title()} of {y_display} over {x_display}",
            xaxis_title=x_display,
            yaxis_title=f"{aggregation.title()} of {y_display}",
            template=CHART_THEME,
            height=CHART_HEIGHT
        )
        
        return fig, None
    
    def generate_correlation_heatmap(self, df: pd.DataFrame, 
                                   config: Dict[str, Any]) -> Tuple[go.Figure, pd.DataFrame]:
        """Generate correlation heatmap."""
        selected_columns = config.get('selected_columns', [])
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        # Filter to numerical columns only
        numerical_columns = []
        for col in df_deduplicated.columns:
            if pd.api.types.is_numeric_dtype(df_deduplicated[col]):
                if not selected_columns or col in selected_columns:
                    numerical_columns.append(col)
        
        if len(numerical_columns) < 2:
            raise ValueError("At least 2 numerical columns are required for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df_deduplicated[numerical_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Create user-friendly column labels for the heatmap
        display_columns = [column_mapper.map_column_name(col) for col in corr_matrix.columns]
        
        # Update the heatmap with friendly labels
        fig.data[0].x = display_columns
        fig.data[0].y = display_columns
        
        fig.update_layout(
            title="Correlation Matrix (Most Recent Snapshots)",
            template=CHART_THEME,
            height=CHART_HEIGHT,
            width=CHART_HEIGHT  # Make it square
        )
        
        return fig, corr_matrix
    
    def generate_box_plot(self, df: pd.DataFrame, 
                         config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate box plot visualization."""
        x_column = config.get('x_axis')
        y_column = config.get('y_axis')
        sort_by = config.get('sort_by', 'none')
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        if not y_column or y_column not in df.columns:
            raise ValueError("Y-axis column not specified or not found")
        
        # Use most recent snapshots to avoid double counting opportunities
        df_deduplicated = self._get_most_recent_snapshots(df)
        
        # Calculate statistics for each category to enable value-based sorting
        category_stats = []
        categories = [cat for cat in df_deduplicated[x_column].unique() if not pd.isna(cat)]
        
        for category in categories:
            category_data = df_deduplicated[df_deduplicated[x_column] == category][y_column].dropna()
            if not category_data.empty:
                category_stats.append({
                    'category': category,
                    'median': category_data.median(),
                    'mean': category_data.mean(),
                    'count': len(category_data)
                })
        
        # Sort categories based on user selection
        if sort_by == 'category_name':
            category_stats.sort(key=lambda x: str(x['category']))
        elif sort_by == 'value_ascending':
            category_stats.sort(key=lambda x: x['median'])  # Sort by median value
        elif sort_by == 'value_descending':
            category_stats.sort(key=lambda x: x['median'], reverse=True)
        # else: keep original order
        
        fig = go.Figure()
        
        # Create box plots in the sorted order
        for i, cat_stat in enumerate(category_stats):
            category = cat_stat['category']
            category_data = df_deduplicated[df_deduplicated[x_column] == category][y_column].dropna()
            
            fig.add_trace(go.Box(
                y=category_data,
                name=str(category),
                marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)]
            ))
        
        # Add sorting info to title
        sort_suffix = ""
        if sort_by != 'none':
            sort_labels = {
                'category_name': ' (Sorted by Category)',
                'value_ascending': ' (Sorted by Median ↑)',
                'value_descending': ' (Sorted by Median ↓)'
            }
            sort_suffix = sort_labels.get(sort_by, '')
        
        # Create user-friendly labels
        x_display = column_mapper.map_column_name(x_column)
        y_display = column_mapper.map_column_name(y_column)
        
        fig.update_layout(
            title=f"Distribution of {y_display} by {x_display}{sort_suffix} (Most Recent Snapshots)",
            xaxis_title=x_display,
            yaxis_title=y_display,
            template=CHART_THEME,
            height=CHART_HEIGHT
        )
        
        return fig, None
    
    def generate_time_series(self, df: pd.DataFrame, 
                            config: Dict[str, Any]) -> Tuple[go.Figure, None]:
        """Generate time series visualization with proper deduplication for pipeline snapshots."""
        x_column = config.get('x_axis')
        y_column = config.get('y_axis')
        group_by_column = config.get('group_by_column')
        aggregation = config.get('aggregation', 'sum')
        time_period = config.get('time_period', 'D')  # D=daily, W=weekly, M=monthly
        
        if not x_column or x_column not in df.columns:
            raise ValueError("X-axis column not specified or not found")
        
        if not y_column or y_column not in df.columns:
            raise ValueError("Y-axis column not specified or not found")
        
        # Import here to avoid circular imports
        from config.settings import ID_COLUMN, SNAPSHOT_DATE_COLUMN
        
        # Prepare dataframe
        df_temp = df.copy()
        df_temp[x_column] = pd.to_datetime(df_temp[x_column])
        
        # Check if we have opportunity ID and snapshot date columns for deduplication
        id_col = self._find_column(df_temp, ID_COLUMN)
        snapshot_col = self._find_column(df_temp, SNAPSHOT_DATE_COLUMN)
        
        if id_col and snapshot_col and id_col in df_temp.columns and snapshot_col in df_temp.columns:
            # Deduplicate by keeping only the most recent snapshot for each opportunity within each time period
            df_temp = self._deduplicate_for_time_series(df_temp, x_column, y_column, id_col, snapshot_col, time_period)
        
        # Set index for resampling
        df_temp.set_index(x_column, inplace=True)
        
        fig = go.Figure()
        
        if group_by_column and group_by_column in df.columns:
            for i, group_value in enumerate(df[group_by_column].unique()):
                if pd.isna(group_value):
                    continue
                
                group_data = df_temp[df_temp[group_by_column] == group_value]
                resampled = group_data[y_column].resample(time_period).agg(aggregation)
                
                fig.add_trace(go.Scatter(
                    x=resampled.index,
                    y=resampled.values,
                    mode='lines+markers',
                    name=str(group_value),
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
                ))
        else:
            resampled = df_temp[y_column].resample(time_period).agg(aggregation)
            
            fig.add_trace(go.Scatter(
                x=resampled.index,
                y=resampled.values,
                mode='lines+markers',
                line=dict(color=COLOR_PALETTE[0])
            ))
        
        # Update title to indicate deduplication
        title_suffix = ""
        if id_col and snapshot_col:
            title_suffix = " (deduplicated by most recent snapshot)"
        
        x_display = column_mapper.map_column_name(x_column)
        y_display = column_mapper.map_column_name(y_column)
        
        fig.update_layout(
            title=f"{aggregation.title()} of {y_display} over time{title_suffix}",
            xaxis_title=x_display,
            yaxis_title=f"{aggregation.title()} of {y_display}",
            template=CHART_THEME,
            height=CHART_HEIGHT
        )
        
        return fig, None
    
    def _find_column(self, df: pd.DataFrame, target_column: str) -> Optional[str]:
        """Find column by case-insensitive search."""
        for col in df.columns:
            if col.lower() == target_column.lower():
                return col
        return None
    
    def _deduplicate_for_time_series(self, df: pd.DataFrame, x_column: str, y_column: str, 
                                   id_col: str, snapshot_col: str, time_period: str) -> pd.DataFrame:
        """
        Deduplicate opportunities for time series analysis by keeping only the most recent 
        snapshot for each opportunity within each time period.
        
        Args:
            df: Input dataframe
            x_column: Time column for grouping (e.g., Created date)
            y_column: Value column (e.g., SellPrice)
            id_col: Opportunity ID column
            snapshot_col: Snapshot date column
            time_period: Time period for resampling ('D', 'W', 'M', etc.)
            
        Returns:
            Deduplicated dataframe
        """
        df_clean = df.copy()
        
        # Ensure snapshot column is datetime
        df_clean[snapshot_col] = pd.to_datetime(df_clean[snapshot_col])
        
        # Create time period grouping based on x_column
        df_clean['time_group'] = pd.to_datetime(df_clean[x_column])
        
        # Map time_period to pandas frequency
        freq_map = {
            'D': 'D',     # Daily
            'W': 'W',     # Weekly  
            'M': 'M',     # Monthly - use M for periods
            'Q': 'Q',     # Quarterly - use Q for periods
            'Y': 'Y'      # Yearly - use Y for periods
        }
        
        freq = freq_map.get(time_period, 'D')
        
        # Group by time period based on x_column (e.g., creation date)
        df_clean['time_group'] = df_clean['time_group'].dt.to_period(freq).dt.start_time
        
        # For each time group and opportunity ID, keep only the most recent snapshot
        deduplicated_rows = []
        
        for time_group, time_data in df_clean.groupby('time_group'):
            for opp_id, opp_data in time_data.groupby(id_col):
                if pd.isna(opp_id):
                    continue
                
                # Sort by snapshot date and take the most recent
                opp_data_sorted = opp_data.sort_values(snapshot_col)
                most_recent = opp_data_sorted.iloc[-1:].copy()
                
                # Use the time_group for the x_column value to ensure proper time series grouping
                most_recent[x_column] = time_group
                
                deduplicated_rows.append(most_recent)
        
        if deduplicated_rows:
            result_df = pd.concat(deduplicated_rows, ignore_index=True)
            # Remove the temporary time_group column
            result_df = result_df.drop('time_group', axis=1)
            return result_df
        else:
            # Return empty dataframe with same structure if no data
            return df_clean.iloc[0:0].drop('time_group', axis=1)
    
    def _get_exclusion_note(self, exclusion_info: Dict[str, Any]) -> str:
        """
        Generate a note about outlier exclusion for charts.
        
        Args:
            exclusion_info: Information about outlier exclusion
            
        Returns:
            String note about exclusion
        """
        if not exclusion_info.get('outliers_excluded', False):
            return ""
        
        excluded_rows = exclusion_info['excluded_rows']
        percentage = exclusion_info['exclusion_percentage']
        columns = exclusion_info['excluded_columns']
        method = exclusion_info['detection_info']['method'].replace('_', ' ').title()
        sensitivity = exclusion_info['detection_info']['sensitivity'].replace('_', ' ').title()
        
        note = f"Note: {excluded_rows:,} outlier rows ({percentage:.1f}%) excluded using {method} method "
        note += f"({sensitivity} sensitivity) on: {', '.join(columns)}"
        
        return note
    
    def _get_most_recent_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the most recent snapshot for each opportunity ID to avoid double counting.
        
        Args:
            df: DataFrame with potentially multiple snapshots per opportunity
            
        Returns:
            DataFrame with only the most recent snapshot for each opportunity
        """
        from config.settings import ID_COLUMN, SNAPSHOT_DATE_COLUMN
        
        # Find ID and date columns (case-insensitive)
        id_col = None
        date_col = None
        
        for col in df.columns:
            if col.lower() == ID_COLUMN.lower():
                id_col = col
            elif col.lower() == SNAPSHOT_DATE_COLUMN.lower():
                date_col = col
        
        # If we don't have both ID and date columns, return original DataFrame
        if not id_col or not date_col:
            st.info("Note: Could not identify ID and Snapshot Date columns for deduplication. Using all records.")
            return df
        
        try:
            # Ensure date column is datetime
            df_temp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            
            # Remove rows where we couldn't parse the date or ID is missing
            clean_df = df_temp.dropna(subset=[id_col, date_col])
            
            if clean_df.empty:
                return df
            
            # Get the index of the most recent snapshot for each opportunity
            most_recent_idx = clean_df.groupby(id_col)[date_col].idxmax()
            
            # Handle any NaN indices (shouldn't happen with clean data, but safety check)
            valid_indices = most_recent_idx.dropna()
            
            if valid_indices.empty:
                return df
            
            # Return only the most recent snapshots
            most_recent_df = df.loc[valid_indices]
            
            # Add informational message about deduplication
            original_count = len(df)
            deduplicated_count = len(most_recent_df)
            unique_opportunities = len(valid_indices)
            
            if original_count != deduplicated_count:
                st.info(f"📊 Deduplication applied: Using {deduplicated_count:,} most recent snapshots "
                       f"from {original_count:,} total records ({unique_opportunities:,} unique opportunities)")
            
            return most_recent_df
            
        except Exception as e:
            st.warning(f"Error during deduplication: {str(e)}. Using all records.")
            return df

