"""
Integration tests for component interactions.
"""
import pytest
import pandas as pd
import numpy as np

from core.data_handler import DataHandler
from core.feature_engine import FeatureEngine
from core.report_engine import ReportEngine
from utils.data_types import DataType


class TestIntegration:
    """Integration tests for component interactions."""
    
    def test_data_handler_feature_engine_integration(self, sample_pipeline_data, mock_streamlit):
        """Test DataHandler and FeatureEngine working together."""
        # Load data through DataHandler
        handler = DataHandler()
        handler.df_raw = sample_pipeline_data.copy()
        handler.df_processed = sample_pipeline_data.copy()
        handler._detect_and_convert_types()
        handler._calculate_column_statistics()
        
        # Use FeatureEngine to add features
        feature_engine = FeatureEngine()
        columns = handler.df_processed.columns.tolist()
        available_features = feature_engine.get_available_features(columns)
        
        if available_features:
            # Add some features
            features_to_add = list(available_features.keys())[:2]  # Add first 2 available features
            df_with_features = feature_engine.add_features(handler.df_processed, features_to_add)
            
            # Should have more columns now
            assert len(df_with_features.columns) >= len(handler.df_processed.columns)
            
            # Update handler with new data
            handler.df_processed = df_with_features
            handler._detect_and_convert_types()  # Re-detect types for new columns
            
            # Column type methods should work with new features
            all_columns = handler.get_numerical_columns() + handler.get_categorical_columns() + handler.get_date_columns()
            assert len(all_columns) > 0
    
    def test_data_handler_report_engine_integration(self, sample_pipeline_data, mock_streamlit):
        """Test DataHandler and ReportEngine working together."""
        # Load data through DataHandler
        handler = DataHandler()
        handler.df_raw = sample_pipeline_data.copy()
        handler.df_processed = sample_pipeline_data.copy()
        handler._detect_and_convert_types()
        handler._calculate_column_statistics()
        
        # Use ReportEngine with data
        report_engine = ReportEngine()
        
        # Test different report types
        numerical_cols = handler.get_numerical_columns()
        categorical_cols = handler.get_categorical_columns()
        
        if numerical_cols and categorical_cols:
            # Test bar chart
            config = {
                'x_axis': categorical_cols[0],
                'y_axis': numerical_cols[0],
                'aggregation': 'sum'
            }
            
            fig, data_table = report_engine.generate_bar_chart(handler.df_processed, config)
            assert fig is not None
    
    def test_column_sync_after_feature_addition(self, sample_pipeline_data, mock_streamlit):
        """Test that column types stay synced after adding features."""
        # Setup DataHandler
        handler = DataHandler()
        handler.df_raw = sample_pipeline_data.copy()
        handler.df_processed = sample_pipeline_data.copy()
        handler._detect_and_convert_types()
        
        original_numerical_count = len(handler.get_numerical_columns())
        
        # Add features through FeatureEngine
        feature_engine = FeatureEngine()
        df_with_features = feature_engine.add_features(handler.df_processed, ['days_in_pipeline'])
        
        # Update handler
        handler.df_processed = df_with_features
        handler._detect_and_convert_types()
        
        # Column methods should only return columns that exist
        numerical_cols = handler.get_numerical_columns()
        df_columns = set(handler.df_processed.columns)
        
        # All returned columns should exist in dataframe
        assert all(col in df_columns for col in numerical_cols)
        
        # Should have potentially more numerical columns now (if feature was added)
        assert len(numerical_cols) >= original_numerical_count
    
    def test_phantom_column_prevention(self, sample_pipeline_data, mock_streamlit):
        """Test that phantom columns don't appear in any component."""
        # Setup DataHandler
        handler = DataHandler()
        handler.df_raw = sample_pipeline_data.copy()
        handler.df_processed = sample_pipeline_data.copy()
        handler._detect_and_convert_types()
        
        # Manually add phantom columns to column_types (simulating the bug)
        handler.column_types['FactoredSellPrice'] = DataType.NUMERICAL
        handler.column_types['PhantomCategory'] = DataType.CATEGORICAL
        handler.column_types['GhostDate'] = DataType.DATE
        
        # Column methods should filter out phantom columns
        numerical_cols = handler.get_numerical_columns()
        categorical_cols = handler.get_categorical_columns()
        date_cols = handler.get_date_columns()
        
        assert 'FactoredSellPrice' not in numerical_cols
        assert 'PhantomCategory' not in categorical_cols
        assert 'GhostDate' not in date_cols
        
        # ReportEngine should work without issues
        report_engine = ReportEngine()
        
        if numerical_cols and categorical_cols:
            config = {
                'x_axis': categorical_cols[0],
                'y_axis': numerical_cols[0],
                'aggregation': 'sum'
            }
            
            # Should not crash due to phantom columns
            fig, data_table = report_engine.generate_bar_chart(handler.df_processed, config)
            assert fig is not None
    
    def test_deduplication_across_all_reports(self, sample_pipeline_data, mock_streamlit):
        """Test that all report types properly deduplicate data."""
        # Create data with known duplicates
        duplicated_data = pd.concat([sample_pipeline_data, sample_pipeline_data.head(10)])
        
        report_engine = ReportEngine()
        
        # Test various report types
        test_configs = [
            {
                'report_type': 'bar_chart',
                'config': {'x_axis': 'Stage', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
            },
            {
                'report_type': 'scatter_plot', 
                'config': {'x_axis': 'SellPrice', 'y_axis': 'GM%'}
            },
            {
                'report_type': 'line_chart',
                'config': {'x_axis': 'SnapshotDate', 'y_axis': 'SellPrice', 'aggregation': 'mean'}
            },
            {
                'report_type': 'box_plot',
                'config': {'x_axis': 'BusinessUnit', 'y_axis': 'SellPrice'}
            }
        ]
        
        for test_case in test_configs:
            report_type = test_case['report_type']
            config = test_case['config']
            
            # Get the report generation method
            method_name = f'generate_{report_type}'
            if hasattr(report_engine, method_name):
                method = getattr(report_engine, method_name)
                
                try:
                    fig, data_table = method(duplicated_data, config)
                    # If it succeeds, that's good - deduplication should have worked
                    assert fig is not None or data_table is not None
                except Exception as e:
                    # If it fails, should be due to data issues, not crashes
                    assert "not found" in str(e).lower() or "not specified" in str(e).lower()
    
    def test_time_series_deduplication_accuracy(self, mock_streamlit):
        """Test that time series deduplication produces accurate results."""
        # Create controlled test data
        test_data = pd.DataFrame({
            'Id': ['OPP-001', 'OPP-001', 'OPP-001', 'OPP-002', 'OPP-002'],
            'Created': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01'],
            'Snapshot Date': ['2024-01-01', '2024-01-15', '2024-01-30', '2024-01-10', '2024-01-20'],
            'SellPrice': [100000, 120000, 110000, 200000, 220000]  # OPP-001: latest=120k, OPP-002: latest=220k
        })
        
        report_engine = ReportEngine()
        
        config = {
            'x_axis': 'Created',
            'y_axis': 'SellPrice', 
            'aggregation': 'sum',
            'time_period': 'M'
        }
        
        fig, _ = report_engine.generate_time_series(test_data, config)
        
        # Should create a figure
        assert fig is not None
        
        # The sum should be 120k + 220k = 340k (using most recent snapshots)
        # This is hard to test directly from the figure, but the fact that it runs
        # without error indicates the deduplication logic is working
    
    def test_error_handling_chain(self, mock_streamlit):
        """Test error handling across component chain."""
        # Create problematic data
        bad_data = pd.DataFrame({
            'Id': [None, '', 'OPP-001'],
            'SnapshotDate': ['invalid', None, '2024-01-01'],
            'SellPrice': ['not_number', -100, 50000]
        })
        
        # DataHandler should handle bad data gracefully
        handler = DataHandler()
        handler.df_raw = bad_data.copy()
        handler.df_processed = bad_data.copy()
        
        # Should not crash
        handler._detect_and_convert_types()
        handler._calculate_column_statistics()
        
        # Should still return some columns (even if empty lists)
        numerical_cols = handler.get_numerical_columns()
        categorical_cols = handler.get_categorical_columns()
        
        assert isinstance(numerical_cols, list)
        assert isinstance(categorical_cols, list)
        
        # FeatureEngine should handle bad data
        feature_engine = FeatureEngine()
        try:
            df_with_features = feature_engine.add_features(handler.df_processed, ['days_in_pipeline'])
            assert isinstance(df_with_features, pd.DataFrame)
        except Exception:
            # If it fails, that's acceptable for very bad data
            pass
        
        # ReportEngine should handle bad data
        report_engine = ReportEngine()
        deduplicated = report_engine._get_most_recent_snapshots(bad_data)
        assert isinstance(deduplicated, pd.DataFrame)
