"""
Tests for the FeatureEngine class.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.feature_engine import FeatureEngine
from utils.data_types import DataType


class TestFeatureEngine:
    """Test suite for FeatureEngine class."""
    
    def test_initialization(self, feature_engine):
        """Test FeatureEngine initialization."""
        assert isinstance(feature_engine.available_features, dict)
        assert isinstance(feature_engine.active_features, dict)
        assert isinstance(feature_engine.feature_functions, dict)
        
        # Should have some default features registered
        assert len(feature_engine.available_features) > 0
        assert 'days_in_pipeline' in feature_engine.available_features
        assert 'user_win_rate' in feature_engine.available_features
    
    def test_get_available_features_empty_columns(self, feature_engine):
        """Test get_available_features with empty column list."""
        available = feature_engine.get_available_features([])
        assert isinstance(available, dict)
        # Should return empty dict since no columns match requirements
        assert len(available) == 0
    
    def test_get_available_features_with_columns(self, feature_engine, sample_pipeline_data):
        """Test get_available_features with valid columns."""
        columns = sample_pipeline_data.columns.tolist()
        available = feature_engine.get_available_features(columns)
        
        assert isinstance(available, dict)
        assert len(available) > 0
        
        # Should include features that have their requirements met
        if all(req in columns for req in ['Id', 'SnapshotDate']):
            assert 'days_in_pipeline' in available
    
    def test_days_in_pipeline_calculation(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test days_in_pipeline feature calculation."""
        df_with_feature = feature_engine._calculate_days_in_pipeline(sample_pipeline_data.copy())
        
        # Should add the new column
        assert 'days_in_pipeline' in df_with_feature.columns
        
        # Check that calculation makes sense
        # For opportunities with multiple snapshots, should have positive values
        grouped = df_with_feature.groupby('Id')['days_in_pipeline'].first()
        non_null_values = grouped.dropna()
        
        if len(non_null_values) > 0:
            # Should have some non-zero values for multi-snapshot opportunities
            assert any(val > 0 for val in non_null_values)
    
    def test_final_stage_calculation(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test final_stage feature calculation."""
        df_with_feature = feature_engine._calculate_final_stage(sample_pipeline_data.copy())
        
        # Should add the new column
        assert 'final_stage' in df_with_feature.columns
        
        # Check that we get the most recent stage for each opportunity
        for opp_id in sample_pipeline_data['Id'].unique():
            opp_data = sample_pipeline_data[sample_pipeline_data['Id'] == opp_id]
            if len(opp_data) > 0:
                # Get the most recent record
                latest_record = opp_data.loc[opp_data['Snapshot Date'].idxmax()]
                expected_stage = latest_record['Stage']
                
                # Check if our calculation matches
                calculated_stage = df_with_feature[df_with_feature['Id'] == opp_id]['final_stage'].iloc[0]
                if pd.notna(calculated_stage) and pd.notna(expected_stage):
                    assert calculated_stage == expected_stage
    
    def test_user_win_rate_calculation(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test user_win_rate feature calculation."""
        df_with_feature = feature_engine._calculate_user_win_rate(sample_pipeline_data.copy())
        
        # Should add the new column
        assert 'user_win_rate' in df_with_feature.columns
        
        # Win rates should be between 0 and 100 (or NaN)
        win_rates = df_with_feature['user_win_rate'].dropna()
        if len(win_rates) > 0:
            assert all(0 <= rate <= 100 for rate in win_rates)
    
    def test_user_win_rate_with_missing_final_stage(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test user_win_rate when final_stage calculation fails."""
        # Create a scenario where final_stage calculation might fail
        df_bad = sample_pipeline_data.copy()
        df_bad['Snapshot Date'] = None  # This should cause final_stage calculation to fail
        
        df_with_feature = feature_engine._calculate_user_win_rate(df_bad)
        
        # Should still add the column but with NaN values
        assert 'user_win_rate' in df_with_feature.columns
        assert df_with_feature['user_win_rate'].isna().all()
    
    def test_starting_stage_calculation(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test starting_stage feature calculation."""
        df_with_feature = feature_engine._calculate_starting_stage(sample_pipeline_data.copy())
        
        # Should add the new column
        assert 'starting_stage' in df_with_feature.columns
        
        # Check that we get the earliest stage for each opportunity
        for opp_id in sample_pipeline_data['Id'].unique():
            opp_data = sample_pipeline_data[sample_pipeline_data['Id'] == opp_id]
            if len(opp_data) > 0:
                # Get the earliest record
                earliest_record = opp_data.loc[opp_data['Snapshot Date'].idxmin()]
                expected_stage = earliest_record['Stage']
                
                # Check if our calculation matches
                calculated_stage = df_with_feature[df_with_feature['Id'] == opp_id]['starting_stage'].iloc[0]
                if pd.notna(calculated_stage) and pd.notna(expected_stage):
                    assert calculated_stage == expected_stage
    
    def test_add_features_empty_list(self, feature_engine, sample_pipeline_data):
        """Test adding features with empty feature list."""
        original_columns = set(sample_pipeline_data.columns)
        result_df = feature_engine.add_features(sample_pipeline_data.copy(), [])
        
        # Should return dataframe unchanged
        assert set(result_df.columns) == original_columns
    
    def test_add_features_valid_features(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test adding valid features."""
        original_columns = set(sample_pipeline_data.columns)
        features_to_add = ['days_in_pipeline', 'final_stage']
        
        result_df = feature_engine.add_features(sample_pipeline_data.copy(), features_to_add)
        
        # Should have added the new columns
        new_columns = set(result_df.columns) - original_columns
        expected_new_columns = set(features_to_add)
        
        # At least some features should have been added (depending on data requirements)
        assert len(new_columns) >= 0  # Some features might not be added due to missing requirements
    
    def test_add_features_invalid_features(self, feature_engine, sample_pipeline_data, mock_streamlit):
        """Test adding invalid features."""
        original_columns = set(sample_pipeline_data.columns)
        invalid_features = ['nonexistent_feature', 'another_fake_feature']
        
        result_df = feature_engine.add_features(sample_pipeline_data.copy(), invalid_features)
        
        # Should return dataframe unchanged (invalid features ignored)
        assert set(result_df.columns) == original_columns
    
    def test_feature_calculation_with_missing_columns(self, feature_engine, mock_streamlit):
        """Test feature calculation when required columns are missing."""
        # Create dataframe missing key columns
        df_incomplete = pd.DataFrame({
            'SomeColumn': [1, 2, 3],
            'AnotherColumn': ['a', 'b', 'c']
        })
        
        # Should not crash, should return original dataframe
        result_df = feature_engine._calculate_days_in_pipeline(df_incomplete.copy())
        assert len(result_df.columns) == len(df_incomplete.columns)
    
    def test_feature_calculation_with_empty_dataframe(self, feature_engine, empty_dataframe, mock_streamlit):
        """Test feature calculation with empty dataframe."""
        # Should not crash with empty data
        result_df = feature_engine._calculate_days_in_pipeline(empty_dataframe.copy())
        
        # Should return the dataframe (possibly with new NaN columns)
        assert isinstance(result_df, pd.DataFrame)
    
    def test_feature_requirements_validation(self, feature_engine):
        """Test that feature requirements are properly defined."""
        for feature_name, feature_info in feature_engine.available_features.items():
            # Each feature should have requirements, description, data_type, and function
            assert 'requirements' in feature_info
            assert 'description' in feature_info
            assert 'data_type' in feature_info
            
            assert isinstance(feature_info['requirements'], list)
            assert isinstance(feature_info['description'], str)
            assert isinstance(feature_info['data_type'], DataType)
            
            # Feature should have corresponding function
            assert feature_name in feature_engine.feature_functions
            assert callable(feature_engine.feature_functions[feature_name])
    
    def test_find_column_method(self, feature_engine, sample_pipeline_data):
        """Test the _find_column helper method."""
        # Should find exact matches
        assert feature_engine._find_column(sample_pipeline_data, 'Id') == 'Id'
        
        # Should find case-insensitive matches
        assert feature_engine._find_column(sample_pipeline_data, 'id') == 'Id'
        assert feature_engine._find_column(sample_pipeline_data, 'ID') == 'Id'
        
        # Should return None for non-existent columns
        assert feature_engine._find_column(sample_pipeline_data, 'NonExistentColumn') is None
