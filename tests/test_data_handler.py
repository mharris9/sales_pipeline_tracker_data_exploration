"""
Tests for the DataHandler class.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from core.data_handler import DataHandler
from utils.data_types import DataType


class TestDataHandler:
    """Test suite for DataHandler class."""
    
    def test_initialization(self, data_handler):
        """Test DataHandler initialization."""
        assert data_handler.df_raw is None
        assert data_handler.df_processed is None
        assert data_handler.column_types == {}
        assert data_handler.column_stats == {}
        assert data_handler.file_info == {}
    
    def test_get_data_empty(self, data_handler):
        """Test get_data with no data loaded."""
        assert data_handler.get_data(processed=True) is None
        assert data_handler.get_data(processed=False) is None
    
    def test_get_data_with_data(self, loaded_data_handler):
        """Test get_data with loaded data."""
        processed_data = loaded_data_handler.get_data(processed=True)
        raw_data = loaded_data_handler.get_data(processed=False)
        
        assert processed_data is not None
        assert raw_data is not None
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(raw_data, pd.DataFrame)
        
        # Should return copies, not the original
        assert processed_data is not loaded_data_handler.df_processed
        assert raw_data is not loaded_data_handler.df_raw
    
    def test_column_type_methods_empty_handler(self, data_handler):
        """Test column type methods with empty handler."""
        assert data_handler.get_numerical_columns() == []
        assert data_handler.get_categorical_columns() == []
        assert data_handler.get_date_columns() == []
        assert data_handler.get_text_columns() == []
    
    def test_column_type_methods_with_data(self, loaded_data_handler):
        """Test column type methods with loaded data."""
        numerical = loaded_data_handler.get_numerical_columns()
        categorical = loaded_data_handler.get_categorical_columns()
        date_cols = loaded_data_handler.get_date_columns()
        
        # Should have some of each type
        assert len(numerical) > 0
        assert len(categorical) > 0
        assert len(date_cols) > 0
        
        # All returned columns should exist in the dataframe
        df_columns = set(loaded_data_handler.df_processed.columns)
        assert all(col in df_columns for col in numerical)
        assert all(col in df_columns for col in categorical)
        assert all(col in df_columns for col in date_cols)
    
    def test_phantom_column_filtering(self, loaded_data_handler):
        """Test that phantom columns are filtered out."""
        # Add a phantom column to column_types that doesn't exist in dataframe
        loaded_data_handler.column_types['FactoredSellPrice'] = DataType.NUMERICAL
        loaded_data_handler.column_types['PhantomCategory'] = DataType.CATEGORICAL
        loaded_data_handler.column_types['GhostDate'] = DataType.DATE
        
        # These phantom columns should NOT appear in the results
        numerical = loaded_data_handler.get_numerical_columns()
        categorical = loaded_data_handler.get_categorical_columns()
        date_cols = loaded_data_handler.get_date_columns()
        
        assert 'FactoredSellPrice' not in numerical
        assert 'PhantomCategory' not in categorical
        assert 'GhostDate' not in date_cols
        
        # But real columns should still be there
        df_columns = set(loaded_data_handler.df_processed.columns)
        assert all(col in df_columns for col in numerical)
        assert all(col in df_columns for col in categorical)
        assert all(col in df_columns for col in date_cols)
    
    def test_column_type_detection(self, loaded_data_handler):
        """Test that column types are detected correctly."""
        column_types = loaded_data_handler.column_types
        
        # Check specific expected types
        assert column_types.get('Id') == DataType.CATEGORICAL
        assert column_types.get('SellPrice') in [DataType.NUMERICAL]
        assert column_types.get('Snapshot Date') == DataType.DATE
        assert column_types.get('Stage') == DataType.CATEGORICAL
        assert column_types.get('BusinessUnit') == DataType.CATEGORICAL
    
    def test_get_column_info(self, loaded_data_handler):
        """Test get_column_info method."""
        column_info = loaded_data_handler.get_column_info()
        
        assert isinstance(column_info, dict)
        assert len(column_info) > 0
        
        # Each column should have type and stats
        for col_name, info in column_info.items():
            assert 'type' in info
            assert 'stats' in info
            assert isinstance(info['type'], DataType)
            assert isinstance(info['stats'], dict)
    
    @pytest.mark.parametrize("data_type,expected_method", [
        (DataType.NUMERICAL, "get_numerical_columns"),
        (DataType.CATEGORICAL, "get_categorical_columns"),
        (DataType.DATE, "get_date_columns"),
        (DataType.TEXT, "get_text_columns")
    ])
    def test_column_methods_consistency(self, loaded_data_handler, data_type, expected_method):
        """Test that column type methods are consistent with column_types."""
        method = getattr(loaded_data_handler, expected_method)
        returned_columns = method()
        
        # Find columns of this type in column_types
        expected_columns = [
            col for col, dtype in loaded_data_handler.column_types.items()
            if dtype == data_type and col in loaded_data_handler.df_processed.columns
        ]
        
        # Should match (order doesn't matter)
        assert set(returned_columns) == set(expected_columns)
    
    def test_malformed_data_handling(self, data_handler, malformed_data, mock_streamlit):
        """Test handling of malformed data."""
        # Simulate loading malformed data
        data_handler.df_raw = malformed_data.copy()
        data_handler.df_processed = malformed_data.copy()
        
        # Should not crash when detecting types
        data_handler._detect_and_convert_types()
        data_handler._calculate_column_statistics()
        
        # Should still have some column types detected
        assert len(data_handler.column_types) > 0
        
        # Methods should still work without crashing
        assert isinstance(data_handler.get_numerical_columns(), list)
        assert isinstance(data_handler.get_categorical_columns(), list)
        assert isinstance(data_handler.get_date_columns(), list)
    
    def test_empty_dataframe_handling(self, data_handler, empty_dataframe):
        """Test handling of empty dataframes."""
        data_handler.df_raw = empty_dataframe.copy()
        data_handler.df_processed = empty_dataframe.copy()
        
        # Should not crash with empty data
        data_handler._detect_and_convert_types()
        data_handler._calculate_column_statistics()
        
        # Should return empty lists
        assert data_handler.get_numerical_columns() == []
        assert data_handler.get_categorical_columns() == []
        assert data_handler.get_date_columns() == []
        assert data_handler.get_text_columns() == []
    
    def test_data_validation(self, loaded_data_handler):
        """Test sales pipeline data validation."""
        validation_result = loaded_data_handler.validate_sales_pipeline_data()
        
        assert isinstance(validation_result, dict)
        assert 'warnings' in validation_result
        assert 'suggestions' in validation_result
        assert isinstance(validation_result['warnings'], list)
        assert isinstance(validation_result['suggestions'], list)
