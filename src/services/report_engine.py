"""
ReportEngine class for generating data reports.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import streamlit as st
import logging

from src.utils.data_types import DataType

logger = logging.getLogger(__name__)

class ReportEngine:
    """
    Generates and manages data reports using centralized state management.
    """
    
    def __init__(self, state_manager=None):
        """Initialize the ReportEngine."""
        # Get state manager instance
        if state_manager is not None:
            self.state_manager = state_manager
        elif hasattr(st.session_state, 'state_manager'):
            self.state_manager = st.session_state.state_manager
        else:
            # Create a temporary state manager for testing
            from src.services.state_manager import StateManager
            self.state_manager = StateManager()
        
        # Store report functions separately (not in state)
        self.report_functions: Dict[str, Callable] = {}
        
        # Initialize state if needed
        if not self.state_manager.get_state('report_configs'):
            self.state_manager.set_state('report_configs', {})
        if not self.state_manager.get_state('report_active'):
            self.state_manager.set_state('report_active', {})
        if not self.state_manager.get_state('report_results'):
            self.state_manager.set_state('report_results', {})
        
        # Register test reports
        self._register_test_reports()
        
        logger.info("ReportEngine initialized")
    
    def _register_test_reports(self) -> None:
        """Register test reports."""
        # Categorical distribution report
        self.register_report(
            name="category_distribution",
            description="Distribution of categorical values",
            requirements=["category"],
            data_type=DataType.CATEGORICAL,
            function=self._generate_categorical_report,
            source_column="category"
        )
        
        # Numerical statistics report
        self.register_report(
            name="value_statistics",
            description="Statistical analysis of numerical values",
            requirements=["value"],
            data_type=DataType.NUMERICAL,
            function=self._generate_numerical_report,
            source_column="value"
        )
        
        # Date trends report
        self.register_report(
            name="date_trends",
            description="Trends over time",
            requirements=["date"],
            data_type=DataType.DATE,
            function=self._generate_date_report,
            source_column="date"
        )
        
        logger.debug("Test reports registered")
    
    def register_report(self, name: str, description: str, requirements: List[str],
                       data_type: DataType, function: Callable,
                       source_column: str) -> None:
        """
        Register a new report generator.
        
        Args:
            name: Report name
            description: Report description
            requirements: Required columns for the report
            data_type: Expected data type to analyze
            function: Function to generate the report
            source_column: Column to analyze
        """
        # Create report config
        report_config = {
            'name': name,
            'description': description,
            'requirements': requirements,
            'data_type': data_type.value,  # Store as string for serialization
            'source_column': source_column,
            'type': data_type.value  # For compatibility with test expectations
        }
        
        # Store function separately
        self.report_functions[name] = function
        
        # Update state
        self.state_manager.set_state(f'report_configs/{name}', report_config)
        self.state_manager.set_state(f'report_active/{name}', False)
        
        logger.debug("Report registered: %s", name)
    
    def set_active_reports(self, reports: List[str]) -> None:
        """Set which reports are active."""
        # Get all report names
        report_configs = self.state_manager.get_state('report_configs', {})
        
        # Reset all reports
        for name in report_configs:
            self.state_manager.set_state(f'report_active/{name}', False)
        
        # Activate selected reports
        for report in reports:
            if report in report_configs:
                self.state_manager.set_state(f'report_active/{report}', True)
        
        logger.debug("Active reports updated: %s", reports)
    
    def get_active_reports(self) -> List[str]:
        """Get list of active reports."""
        report_active = self.state_manager.get_state('report_active', {})
        return [name for name, active in report_active.items() if active]

    def get_available_reports(self) -> List[Dict[str, Any]]:
        """Get list of all available reports with their configurations."""
        report_configs = self.state_manager.get_state('report_configs', {})
        report_active = self.state_manager.get_state('report_active', {})
        
        reports = []
        for name, config in report_configs.items():
            reports.append({
                'name': name,
                'description': config.get('description', name),
                'active': report_active.get(name, False)
            })
        
        return reports

    def generate_reports(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all active reports.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with report generation results
        """
        if df.empty:
            return df
        
        df_with_reports = df.copy()
        active_reports = self.get_active_reports()
        report_configs = self.state_manager.get_state('report_configs', {})
        
        for report_name in active_reports:
            if report_name not in report_configs:
                logger.warning("Report '%s' not found", report_name)
                continue
            
            try:
                # Get report configuration
                config = report_configs[report_name]
                source_column = config['source_column']
                
                # Check if source column exists
                if source_column not in df.columns:
                    logger.warning("Source column '%s' not found for report '%s'", source_column, report_name)
                    continue
                
                # Get report function
                report_function = self.report_functions[report_name]
                
                # Generate report
                df_with_reports = report_function(df_with_reports)
                logger.debug("Report generated: %s", report_name)
                
            except Exception as e:
                logger.error("Error generating report '%s': %s", report_name, str(e))
        
        return df_with_reports
    
    def _generate_categorical_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical distribution report."""
        df = df.copy()
        
        # Calculate value counts and percentages
        value_counts = df['category'].value_counts()
        total = len(df)
        percentages = (value_counts / total) * 100
        
        # Store results
        results = {
            'value_counts': {str(k): int(v) for k, v in value_counts.items()},
            'percentages': {str(k): float(v) for k, v in percentages.items()},
            'total_count': int(total)
        }
        self.state_manager.set_state(f'report_results/category_distribution', results)
        
        return df
    
    def _generate_numerical_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate numerical statistics report."""
        df = df.copy()
        
        # Calculate statistics
        series = df['value']
        statistics = {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'q1': float(series.quantile(0.25)),
            'q3': float(series.quantile(0.75))
        }
        
        # Store results
        results = {
            'statistics': statistics,
            'total_count': int(len(series)),
            'non_null_count': int(series.count())
        }
        self.state_manager.set_state(f'report_results/value_statistics', results)
        
        return df
    
    def _generate_date_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate date trends report."""
        df = df.copy()
        
        # Calculate date distributions
        series = df['date']
        daily_counts = series.dt.day.value_counts()
        weekly_counts = series.dt.isocalendar().week.value_counts()
        monthly_counts = series.dt.month.value_counts()
        
        # Store results
        results = {
            'daily_counts': {int(k): int(v) for k, v in daily_counts.items()},
            'weekly_counts': {int(k): int(v) for k, v in weekly_counts.items()},
            'monthly_counts': {int(k): int(v) for k, v in monthly_counts.items()},
            'date_range': {
                'start': series.min().isoformat(),
                'end': series.max().isoformat(),
                'total_days': int((series.max() - series.min()).days)
            }
        }
        self.state_manager.set_state(f'report_results/date_trends', results)
        
        return df

    def _get_most_recent_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the most recent snapshot for each unique ID.
        
        Args:
            df: DataFrame with ID and date columns
            
        Returns:
            DataFrame with deduplicated records (most recent per ID)
        """
        if df.empty:
            return df
        
        # Try to find potential ID and date columns
        id_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['id', 'identifier', 'key', 'primary'])]
        date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp'])]
        
        # Use the first found columns or return original dataframe if not found
        if not id_columns or not date_columns:
            logger.warning("Could not find suitable ID and date columns for deduplication")
            return df
        
        id_column = id_columns[0]
        date_column = date_columns[0]
        
        logger.info(f"Using '{id_column}' as ID column and '{date_column}' as date column for deduplication")
        
        # Ensure date column is datetime
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=[date_column])
        
        if df_copy.empty:
            logger.warning("No valid dates found after conversion")
            return df_copy
        
        # Get the most recent snapshot for each ID
        deduplicated = df_copy.loc[df_copy.groupby(id_column)[date_column].idxmax()]
        
        logger.info(f"Deduplicated {len(df)} records to {len(deduplicated)} records")
        return deduplicated

    def generate_time_series(self, df: pd.DataFrame, config: Dict[str, Any]) -> tuple:
        """
        Generate time series visualization.
        
        Args:
            df: DataFrame to analyze
            config: Configuration dictionary with keys:
                - 'date_column': Column name for dates
                - 'value_column': Column name for values
                - 'group_by': Optional column to group by
                - 'aggregation': Aggregation method ('sum', 'mean', 'count')
                - 'freq': Frequency ('D', 'W', 'M', 'Q', 'Y')
                
        Returns:
            Tuple of (plotly figure, data table)
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        try:
            date_column = config.get('date_column', 'Snapshot Date')
            value_column = config.get('value_column')
            group_by = config.get('group_by')
            aggregation = config.get('aggregation', 'count')
            freq = config.get('freq', 'M')
            
            # Validate columns exist
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' not found")
            
            if value_column and value_column not in df.columns:
                raise ValueError(f"Value column '{value_column}' not found")
            
            # Prepare data
            df_copy = df.copy()
            
            # Convert date column
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            
            df_copy = df_copy.dropna(subset=[date_column])
            
            if df_copy.empty:
                raise ValueError("No valid dates found after conversion")
            
            # Set date as index for resampling
            df_copy = df_copy.set_index(date_column)
            
            # Prepare aggregation
            if value_column:
                if aggregation == 'sum':
                    agg_func = 'sum'
                elif aggregation == 'mean':
                    agg_func = 'mean'
                elif aggregation == 'count':
                    agg_func = 'count'
                else:
                    agg_func = 'sum'
                
                if group_by and group_by in df_copy.columns:
                    # Group by both date and category
                    grouped = df_copy.groupby([pd.Grouper(freq=freq), group_by])[value_column].agg(agg_func).reset_index()
                    
                    # Create line plot
                    fig = px.line(grouped, x=date_column, y=value_column, color=group_by,
                                title=f"Time Series: {value_column} by {group_by} ({freq})")
                else:
                    # Simple time series
                    resampled = df_copy[value_column].resample(freq).agg(agg_func)
                    fig = px.line(x=resampled.index, y=resampled.values,
                                title=f"Time Series: {value_column} ({freq})")
            else:
                # Count records over time
                if group_by and group_by in df_copy.columns:
                    grouped = df_copy.groupby([pd.Grouper(freq=freq), group_by]).size().reset_index(name='count')
                    fig = px.line(grouped, x=date_column, y='count', color=group_by,
                                title=f"Record Count by {group_by} ({freq})")
                else:
                    resampled = df_copy.resample(freq).size()
                    fig = px.line(x=resampled.index, y=resampled.values,
                                title=f"Record Count ({freq})")
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=True,
                xaxis_title="Date",
                yaxis_title=value_column if value_column else "Count"
            )
            
            # Create data table
            if value_column and group_by and group_by in df_copy.columns:
                data_table = grouped
            elif value_column:
                data_table = resampled.reset_index()
                data_table.columns = [date_column, value_column]
            elif group_by and group_by in df_copy.columns:
                data_table = grouped
            else:
                data_table = resampled.reset_index()
                data_table.columns = [date_column, 'count']
            
            return fig, data_table
            
        except Exception as e:
            logger.error(f"Error generating time series: {str(e)}")
            # Return empty figure and empty DataFrame
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig, pd.DataFrame()

    def get_current_chart(self) -> Optional[Any]:
        """
        Get the current chart figure.
        
        Returns:
            Current chart figure or None if no chart is available
        """
        return self.state_manager.get_state('reports.current_chart')
