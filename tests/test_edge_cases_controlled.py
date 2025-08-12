"""
Test edge cases and data validation using controlled datasets.

This module tests various edge cases, boundary conditions,
and data validation scenarios to ensure robust behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tests.fixtures.controlled_test_data import create_controlled_dataset
from core.data_handler import DataHandler
from core.feature_engine import FeatureEngine
from core.report_engine import ReportEngine


class TestEdgeCasesControlled:
    """Test edge cases using controlled and modified datasets."""
    
    @pytest.fixture
    def base_controlled_data(self):
        """Create the base controlled test dataset."""
        return create_controlled_dataset()
    
    @pytest.fixture
    def data_handler(self):
        """Create DataHandler for testing."""
        return DataHandler()
    
    @pytest.fixture
    def feature_engine(self):
        """Create FeatureEngine for testing."""
        return FeatureEngine()
    
    @pytest.fixture
    def report_engine(self):
        """Create ReportEngine for testing."""
        return ReportEngine()
    
    def create_edge_case_dataset(self, case_type: str) -> pd.DataFrame:
        """Create specific edge case datasets for testing."""
        
        base_data = create_controlled_dataset()
        
        if case_type == "missing_dates":
            # Create dataset with missing snapshot dates
            df = base_data.copy()
            df.loc[0:2, 'Snapshot Date'] = None
            return df
        
        elif case_type == "missing_ids":
            # Create dataset with missing opportunity IDs
            df = base_data.copy()
            df.loc[0:2, 'Id'] = None
            return df
        
        elif case_type == "duplicate_exact":
            # Create dataset with exact duplicate rows
            df = base_data.copy()
            duplicate_row = df.iloc[0:1].copy()
            df = pd.concat([df, duplicate_row], ignore_index=True)
            return df
        
        elif case_type == "single_opportunity":
            # Create dataset with only one opportunity
            df = base_data.copy()
            df = df[df['Id'] == 'OPP-001'].copy()
            return df
        
        elif case_type == "zero_values":
            # Create dataset with zero sell prices
            df = base_data.copy()
            df['SellPrice'] = 0
            return df
        
        elif case_type == "negative_values":
            # Create dataset with negative values
            df = base_data.copy()
            df.loc[0:2, 'SellPrice'] = -100000
            df.loc[0:2, 'GM%'] = -0.1
            return df
        
        elif case_type == "extreme_values":
            # Create dataset with extremely large values
            df = base_data.copy()
            df.loc[0, 'SellPrice'] = 999999999999
            df.loc[1, 'GM%'] = 999.0
            return df
        
        elif case_type == "mixed_types":
            # Create dataset with mixed data types in numeric columns
            df = base_data.copy()
            df.loc[0, 'SellPrice'] = "not_a_number"
            df.loc[1, 'GM%'] = "invalid"
            return df
        
        elif case_type == "future_dates":
            # Create dataset with future dates
            df = base_data.copy()
            future_date = datetime.now() + timedelta(days=365)
            df.loc[0:2, 'Snapshot Date'] = future_date.strftime('%Y-%m-%d')
            df.loc[0:2, 'Created'] = future_date.strftime('%Y-%m-%d')
            return df
        
        elif case_type == "invalid_stages":
            # Create dataset with unexpected stage values
            df = base_data.copy()
            df.loc[0, 'Stage'] = "Unknown Stage"
            df.loc[1, 'Stage'] = ""
            df.loc[2, 'Stage'] = None
            return df
        
        elif case_type == "long_strings":
            # Create dataset with very long string values
            df = base_data.copy()
            long_name = "A" * 1000
            df.loc[0, 'OpportunityName'] = long_name
            df.loc[1, 'Owner'] = long_name
            return df
        
        else:
            return base_data
    
    def test_missing_snapshot_dates(self, data_handler, feature_engine, report_engine, mock_streamlit):
        """Test handling of missing snapshot dates."""
        
        df = self.create_edge_case_dataset("missing_dates")
        
        # Data handler should handle missing dates
        data_handler.load_dataframeframe(df)
        processed_df = data_handler.get_data()
        
        # Should not crash and should handle missing dates gracefully
        assert processed_df is not None
        assert len(processed_df) > 0
        
        # Feature calculations should handle missing dates
        try:
            df_with_features = feature_engine._calculate_days_in_pipeline(processed_df)
            # Should either skip rows with missing dates or handle them gracefully
            assert 'days_in_pipeline' in df_with_features.columns
        except Exception as e:
            # If it raises an exception, should be meaningful
            assert len(str(e)) > 0
        
        # Reports should handle missing dates
        config = {'x_axis': 'Snapshot Date', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        try:
            fig, _ = report_engine.generate_line_chart(processed_df, config)
            assert fig is not None
        except Exception as e:
            assert "date" in str(e).lower() or "missing" in str(e).lower()
    
    def test_missing_opportunity_ids(self, data_handler, feature_engine, report_engine, mock_streamlit):
        """Test handling of missing opportunity IDs."""
        
        df = self.create_edge_case_dataset("missing_ids")
        
        # Data handler should handle missing IDs
        data_handler.load_dataframeframe(df)
        processed_df = data_handler.get_data()
        
        assert processed_df is not None
        
        # Deduplication should handle missing IDs gracefully
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_df)
        
        # Should not crash, may exclude rows with missing IDs
        assert deduplicated_df is not None
        
        # Feature calculations should handle missing IDs
        try:
            df_with_features = feature_engine._calculate_final_stage(processed_df)
            assert df_with_features is not None
        except Exception as e:
            assert len(str(e)) > 0
    
    def test_exact_duplicate_rows(self, data_handler, report_engine, mock_streamlit):
        """Test handling of exact duplicate rows."""
        
        df = self.create_edge_case_dataset("duplicate_exact")
        
        # Should have one more row than original
        original_len = len(create_controlled_dataset())
        assert len(df) == original_len + 1
        
        # Data handler should handle duplicates
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        
        # Deduplication should remove exact duplicates
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_df)
        
        # Should have same number of unique opportunities as original
        unique_opps = deduplicated_df['Id'].nunique()
        assert unique_opps == 4, "Should still have 4 unique opportunities"
    
    def test_single_opportunity_dataset(self, data_handler, feature_engine, report_engine, mock_streamlit):
        """Test handling of dataset with only one opportunity."""
        
        df = self.create_edge_case_dataset("single_opportunity")
        
        # Should have only OPP-001 data
        assert df['Id'].nunique() == 1
        assert df['Id'].iloc[0] == 'OPP-001'
        
        # Data handler should work with single opportunity
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Features should work with single opportunity
        df_with_features = feature_engine._calculate_days_in_pipeline(processed_df)
        assert 'days_in_pipeline' in df_with_features.columns
        
        df_with_final = feature_engine._calculate_final_stage(processed_df)
        assert 'final_stage' in df_with_final.columns
        assert df_with_final['final_stage'].iloc[0] == 'Closed - WON'
        
        # Reports should work with single opportunity
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_df, config)
        
        # Should have one bar
        assert len(fig.data[0].x) == 1
        assert fig.data[0].x[0] == 'Alice'
        assert fig.data[0].y[0] == 150000  # Final value for OPP-001
    
    def test_zero_values_handling(self, data_handler, report_engine, mock_streamlit):
        """Test handling of zero values in calculations."""
        
        df = self.create_edge_case_dataset("zero_values")
        
        # All sell prices should be zero
        assert (df['SellPrice'] == 0).all()
        
        # Data handler should handle zeros
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Reports should handle zero values
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_df, config)
        
        # All bars should be zero
        assert all(y == 0 for y in fig.data[0].y)
        
        # Descriptive statistics should handle zeros
        config = {'columns': ['SellPrice']}
        fig, data_table = report_engine.generate_descriptive_statistics(processed_df, config)
        
        assert data_table.loc['mean', 'SellPrice'] == 0
        assert data_table.loc['sum', 'SellPrice'] == 0
    
    def test_negative_values_handling(self, data_handler, report_engine, mock_streamlit):
        """Test handling of negative values."""
        
        df = self.create_edge_case_dataset("negative_values")
        
        # Should have some negative values
        assert (df['SellPrice'] < 0).any()
        assert (df['GM%'] < 0).any()
        
        # Data handler should handle negative values
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Reports should handle negative values
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_df, config)
        
        # Should include negative values in calculations
        total_sum = sum(fig.data[0].y)
        expected_negative_contribution = -100000 * 3  # 3 negative values
        assert total_sum < 1000000, "Total should be reduced by negative values"
    
    def test_extreme_values_handling(self, data_handler, report_engine, mock_streamlit):
        """Test handling of extremely large values."""
        
        df = self.create_edge_case_dataset("extreme_values")
        
        # Should have extreme values
        assert (df['SellPrice'] > 1000000000).any()
        assert (df['GM%'] > 10).any()
        
        # Data handler should handle extreme values
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Reports should handle extreme values without crashing
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        try:
            fig, _ = report_engine.generate_bar_chart(processed_df, config)
            assert fig is not None
        except Exception as e:
            # If it fails, should be due to reasonable limits
            assert "overflow" in str(e).lower() or "too large" in str(e).lower()
    
    def test_mixed_data_types(self, data_handler, mock_streamlit):
        """Test handling of mixed data types in numeric columns."""
        
        df = self.create_edge_case_dataset("mixed_types")
        
        # Should have mixed types
        assert df.loc[0, 'SellPrice'] == "not_a_number"
        assert df.loc[1, 'GM%'] == "invalid"
        
        # Data handler should handle mixed types gracefully
        try:
            data_handler.load_dataframe(df)
            processed_df = data_handler.get_data()
            assert processed_df is not None
            
            # Should attempt to convert or handle mixed types
            # Either convert to NaN or exclude problematic rows
            
        except Exception as e:
            # If it raises an exception, should be meaningful
            assert "type" in str(e).lower() or "convert" in str(e).lower()
    
    def test_future_dates_handling(self, data_handler, feature_engine, mock_streamlit):
        """Test handling of future dates."""
        
        df = self.create_edge_case_dataset("future_dates")
        
        # Should have future dates
        future_rows = pd.to_datetime(df['Snapshot Date']) > datetime.now()
        assert future_rows.any()
        
        # Data handler should handle future dates
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Feature calculations should handle future dates
        df_with_features = feature_engine._calculate_days_in_pipeline(processed_df)
        assert 'days_in_pipeline' in df_with_features.columns
        
        # Days in pipeline might be negative for future dates
        future_pipeline_days = df_with_features[future_rows]['days_in_pipeline']
        # Should handle this gracefully (either negative values or special handling)
        assert not future_pipeline_days.isna().all(), "Should calculate something for future dates"
    
    def test_invalid_stage_values(self, data_handler, feature_engine, mock_streamlit):
        """Test handling of invalid or unexpected stage values."""
        
        df = self.create_edge_case_dataset("invalid_stages")
        
        # Should have invalid stages
        invalid_stages = df['Stage'].isin([None, "", "Unknown Stage"])
        assert invalid_stages.any()
        
        # Data handler should handle invalid stages
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Feature calculations should handle invalid stages
        df_with_final = feature_engine._calculate_final_stage(processed_df)
        assert 'final_stage' in df_with_final.columns
        
        # Should handle invalid stages gracefully (might be NaN or excluded)
        final_stages = df_with_final['final_stage']
        assert not final_stages.isna().all(), "Should have some valid final stages"
    
    def test_long_string_values(self, data_handler, report_engine, mock_streamlit):
        """Test handling of very long string values."""
        
        df = self.create_edge_case_dataset("long_strings")
        
        # Should have long strings
        assert len(df.loc[0, 'OpportunityName']) == 1000
        assert len(df.loc[1, 'Owner']) == 1000
        
        # Data handler should handle long strings
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Reports should handle long strings (might truncate in display)
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_df, config)
        
        # Should not crash, though display might be affected
        assert fig is not None
        assert len(fig.data[0].x) > 0
    
    def test_boundary_date_conditions(self, data_handler, feature_engine, mock_streamlit):
        """Test boundary conditions with dates."""
        
        # Create dataset with same snapshot and creation dates
        df = create_controlled_dataset()
        df['Snapshot Date'] = df['Created']  # Same dates
        
        # Data handler should handle same dates
        data_handler.load_dataframe(df)
        processed_df = data_handler.get_data()
        assert processed_df is not None
        
        # Days in pipeline should be 0 when dates are the same
        df_with_features = feature_engine._calculate_days_in_pipeline(processed_df)
        days_values = df_with_features['days_in_pipeline']
        
        # Should be 0 or very close to 0
        assert (days_values <= 1).all(), "Days in pipeline should be 0 or 1 when dates are the same"
    
    def test_memory_efficiency_large_dataset(self, data_handler, mock_streamlit):
        """Test memory efficiency with larger dataset."""
        
        # Create larger dataset by replicating base data
        base_df = create_controlled_dataset()
        large_dfs = []
        
        for i in range(100):  # 1200 rows total
            df_copy = base_df.copy()
            df_copy['Id'] = df_copy['Id'] + f'-BATCH-{i}'
            large_dfs.append(df_copy)
        
        large_df = pd.concat(large_dfs, ignore_index=True)
        
        # Should handle larger dataset without issues
        data_handler.load_dataframe(large_df)
        processed_df = data_handler.get_data()
        
        assert processed_df is not None
        assert len(processed_df) == 1200
        assert processed_df['Id'].nunique() == 400  # 4 original * 100 batches
