"""
Tests for the ReportEngine class.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.services.report_engine import ReportEngine
from src.utils.data_types import DataType


class TestReportEngine:
    """Test suite for ReportEngine class."""
    
    def test_initialization(self, report_engine):
        """Test ReportEngine initialization."""
        assert hasattr(report_engine, 'available_reports')
        assert isinstance(report_engine.available_reports, dict)
        
        # Should have expected report types
        expected_reports = ['descriptive_statistics', 'histogram', 'bar_chart', 
                          'scatter_plot', 'line_chart', 'correlation_heatmap', 
                          'box_plot', 'time_series']
        
        for report_type in expected_reports:
            assert report_type in report_engine.available_reports
    
    def test_get_most_recent_snapshots(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test the _get_most_recent_snapshots method."""
        # This is the critical deduplication logic
        deduplicated_df = report_engine._get_most_recent_snapshots(sample_pipeline_data)
        
        # Should have fewer or equal rows than original
        assert len(deduplicated_df) <= len(sample_pipeline_data)
        
        # Should have unique opportunity IDs
        unique_opps_original = sample_pipeline_data['Id'].nunique()
        unique_opps_deduplicated = deduplicated_df['Id'].nunique()
        
        # Should have the same number of unique opportunities
        assert unique_opps_deduplicated == unique_opps_original
        
        # Each opportunity should appear only once
        id_counts = deduplicated_df['Id'].value_counts()
        assert all(count == 1 for count in id_counts)
    
    def test_get_most_recent_snapshots_with_missing_columns(self, report_engine, mock_streamlit):
        """Test deduplication when required columns are missing."""
        # Create dataframe without Id or SnapshotDate columns
        df_no_id = pd.DataFrame({
            'SomeColumn': [1, 2, 3],
            'AnotherColumn': ['a', 'b', 'c']
        })
        
        # Should return original dataframe unchanged
        result_df = report_engine._get_most_recent_snapshots(df_no_id)
        pd.testing.assert_frame_equal(result_df, df_no_id)
    
    def test_get_most_recent_snapshots_preserves_most_recent(self, report_engine, mock_streamlit):
        """Test that deduplication keeps the most recent snapshot."""
        # Create test data with known dates
        test_data = pd.DataFrame({
            'Id': ['OPP-001', 'OPP-001', 'OPP-001'],
            'Snapshot Date': ['2024-01-01', '2024-01-15', '2024-01-10'],  # Middle one is most recent
            'Stage': ['Lead', 'Negotiation', 'Budget'],
            'SellPrice': [100000, 150000, 125000]
        })
        
        deduplicated_df = report_engine._get_most_recent_snapshots(test_data)
        
        # Should have only one row
        assert len(deduplicated_df) == 1
        
        # Should be the most recent record (2024-01-15)
        assert deduplicated_df.iloc[0]['Stage'] == 'Negotiation'
        assert deduplicated_df.iloc[0]['SellPrice'] == 150000
    
    def test_time_series_deduplication(self, report_engine, mock_streamlit):
        """Test the time series specific deduplication logic."""
        # Create test data with multiple opportunities across time periods
        test_data = pd.DataFrame({
            'Id': ['OPP-001', 'OPP-001', 'OPP-002', 'OPP-002'],
            'Created': ['2024-01-01', '2024-01-01', '2024-02-01', '2024-02-01'],  # Same creation dates
            'Snapshot Date': ['2024-01-05', '2024-01-10', '2024-02-05', '2024-02-10'],  # Different snapshot dates
            'SellPrice': [100000, 120000, 200000, 250000]  # Different values
        })
        
        # Test monthly deduplication
        deduplicated_df = report_engine._deduplicate_for_time_series(
            test_data, 'Created', 'SellPrice', 'Id', 'Snapshot Date', 'M'
        )
        
        # Should have 2 rows (one per opportunity per month)
        assert len(deduplicated_df) == 2
        
        # Should use most recent snapshot values
        opp1_row = deduplicated_df[deduplicated_df['Id'] == 'OPP-001'].iloc[0]
        opp2_row = deduplicated_df[deduplicated_df['Id'] == 'OPP-002'].iloc[0]
        
        assert opp1_row['SellPrice'] == 120000  # Most recent for OPP-001
        assert opp2_row['SellPrice'] == 250000  # Most recent for OPP-002
    
    def test_generate_descriptive_statistics(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test descriptive statistics generation."""
        config = {
            'selected_columns': ['SellPrice', 'GM%'],
            'group_by_column': None
        }
        
        fig, data_table = report_engine.generate_descriptive_statistics(sample_pipeline_data, config)
        
        # Should return None for figure, DataFrame for data
        assert fig is None
        assert isinstance(data_table, pd.DataFrame)
        assert len(data_table) > 0
    
    def test_generate_bar_chart(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test bar chart generation."""
        config = {
            'x_axis': 'Stage',
            'y_axis': 'SellPrice',
            'aggregation': 'sum',
            'group_by_column': None
        }
        
        fig, data_table = report_engine.generate_bar_chart(sample_pipeline_data, config)
        
        # Should return plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure should have data attribute
        assert data_table is None
    
    def test_generate_scatter_plot(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test scatter plot generation."""
        config = {
            'x_axis': 'SellPrice',
            'y_axis': 'GM%',
            'size_column': None,
            'group_by_column': None
        }
        
        fig, data_table = report_engine.generate_scatter_plot(sample_pipeline_data, config)
        
        # Should return plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert data_table is None
    
    def test_generate_line_chart(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test line chart generation."""
        config = {
            'x_axis': 'Snapshot Date',
            'y_axis': 'SellPrice',
            'aggregation': 'mean',
            'group_by_column': None
        }
        
        fig, data_table = report_engine.generate_line_chart(sample_pipeline_data, config)
        
        # Should return plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert data_table is None
    
    def test_generate_time_series(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test time series generation with deduplication."""
        config = {
            'x_axis': 'Created',
            'y_axis': 'SellPrice',
            'aggregation': 'sum',
            'time_period': 'M',
            'group_by_column': None
        }
        
        fig, data_table = report_engine.generate_time_series(sample_pipeline_data, config)
        
        # Should return plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert data_table is None
        
        # Title should indicate deduplication was applied
        assert 'deduplicated' in fig.layout.title.text.lower()
    
    def test_correlation_heatmap(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test correlation heatmap generation."""
        config = {
            'selected_columns': ['SellPrice', 'GM%']
        }
        
        fig, data_table = report_engine.generate_correlation_heatmap(sample_pipeline_data, config)
        
        # Should return both figure and data table
        assert fig is not None
        assert isinstance(data_table, pd.DataFrame)
    
    def test_box_plot(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test box plot generation."""
        config = {
            'x_axis': 'BusinessUnit',
            'y_axis': 'SellPrice',
            'sort_by': 'none'
        }
        
        fig, data_table = report_engine.generate_box_plot(sample_pipeline_data, config)
        
        # Should return plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert data_table is None
    
    def test_report_generation_with_missing_columns(self, report_engine, sample_pipeline_data, mock_streamlit):
        """Test report generation when required columns are missing."""
        config = {
            'x_axis': 'NonExistentColumn',
            'y_axis': 'SellPrice'
        }
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="X-axis column not specified or not found"):
            report_engine.generate_bar_chart(sample_pipeline_data, config)
    
    def test_get_compatible_columns_for_report(self, report_engine):
        """Test getting compatible columns for different report types."""
        column_types = {
            'Id': DataType.CATEGORICAL,
            'SellPrice': DataType.NUMERICAL,
            'SnapshotDate': DataType.DATE,
            'Stage': DataType.CATEGORICAL
        }
        
        # Test scatter plot compatibility
        x_compatible = report_engine.get_compatible_columns_for_report('scatter_plot', 'x_axis', column_types)
        y_compatible = report_engine.get_compatible_columns_for_report('scatter_plot', 'y_axis', column_types)
        
        # X-axis should include numerical and date columns
        assert 'SellPrice' in x_compatible
        assert 'SnapshotDate' in x_compatible
        assert 'Stage' not in x_compatible  # Categorical not compatible with scatter plot x-axis
        
        # Y-axis should include only numerical columns
        assert 'SellPrice' in y_compatible
        assert 'SnapshotDate' not in y_compatible
        assert 'Stage' not in y_compatible
    
    def test_empty_dataframe_handling(self, report_engine, empty_dataframe, mock_streamlit):
        """Test report generation with empty dataframes."""
        config = {
            'x_axis': 'Stage',
            'y_axis': 'SellPrice',
            'aggregation': 'sum'
        }
        
        # Should handle empty dataframes gracefully
        try:
            fig, data_table = report_engine.generate_bar_chart(empty_dataframe, config)
            # If it succeeds, that's good
        except Exception as e:
            # If it fails, should be a meaningful error, not a crash
            assert "not found" in str(e).lower() or "empty" in str(e).lower()
    
    def test_find_column_method(self, report_engine, sample_pipeline_data):
        """Test the _find_column helper method."""
        # Should find exact matches
        assert report_engine._find_column(sample_pipeline_data, 'Id') == 'Id'
        
        # Should find case-insensitive matches  
        assert report_engine._find_column(sample_pipeline_data, 'id') == 'Id'
        assert report_engine._find_column(sample_pipeline_data, 'ID') == 'Id'
        
        # Should return None for non-existent columns
        assert report_engine._find_column(sample_pipeline_data, 'NonExistentColumn') is None
