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
    
    def __init__(self):
        """Initialize the ReportEngine."""
        # Get state manager instance
        if not hasattr(st.session_state, 'state_manager'):
            raise RuntimeError("StateManager not initialized")
        self.state_manager = st.session_state.state_manager
        
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
