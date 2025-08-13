"""
Tests for Phase 2 - DataHandler with StateManager integration.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import logging
from io import StringIO

from src.services.state_manager import StateManager
from src.services.data_handler import DataHandler
from src.utils.data_types import DataType

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_session_state(monkeypatch):
    """Mock Streamlit session state."""
    class MockSessionState(dict):
        def __init__(self):
            super().__init__()
            self._state = {}
        
        def __getattr__(self, key):
            if key not in self._state:
                return None
            return self._state[key]
        
        def __setattr__(self, key, value):
            if key == '_state':
                super().__setattr__(key, value)
            else:
                self._state[key] = value
        
        def __getitem__(self, key):
            return self._state[key]
        
        def __setitem__(self, key, value):
            self._state[key] = value
        
        def __contains__(self, key):
            return key in self._state
    
    session_state = MockSessionState()
    monkeypatch.setattr(st, 'session_state', session_state)
    return session_state

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Id': range(1, 11),  # Required ID column
        'Snapshot Date': pd.date_range(start='2024-01-01', periods=10).strftime('%m/%d/%Y'),  # Required date column
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'value': [10.5, 20.0, 15.5, 30.0, 25.5, 12.0, 35.0, 22.5, 18.0, 28.0],
        'date': pd.date_range(start='2024-01-01', periods=10),
        'active': [True, False, True, True, False, True, False, True, False, True]
    })

@pytest.fixture
def mock_uploaded_file(sample_df):
    """Create a mock uploaded file."""
    class MockUploadedFile:
        def __init__(self, df):
            self.name = "test.csv"
            self.size = 1024  # 1KB
            self.type = "text/csv"
            self._df = df
            self._buffer = StringIO()
            self._df.to_csv(self._buffer, index=False)
            self._buffer.seek(0)
            self._content = self._buffer.getvalue()
            self._pos = 0
            
        def read(self):
            return self._content
            
        def seek(self, pos):
            self._pos = pos
            
        def __iter__(self):
            return iter(self._content.splitlines())
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def __str__(self):
            return self._content
            
        def __getattr__(self, name):
            # Handle any other attributes by returning None
            return None
            
        def __getitem__(self, key):
            # Handle dictionary-like access
            return getattr(self, key)
            
        def __bool__(self):
            # Handle truthiness
            return True
            
        def tell(self):
            return self._pos
            
        def readable(self):
            return True
            
        def seekable(self):
            return True
    
    return MockUploadedFile(sample_df)

def test_data_handler_initialization(mock_session_state):
    """Test DataHandler initialization with StateManager."""
    # Initialize state manager
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    
    # Get DataHandler instance
    data_handler = state_manager.get_state('data.data_handler')
    assert isinstance(data_handler, DataHandler)
    assert data_handler.df_raw is None
    assert data_handler.df_processed is None
    assert data_handler.column_types == {}

def test_load_dataframe(mock_session_state, sample_df):
    """Test loading DataFrame directly."""
    # Initialize state manager and data handler
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    data_handler = state_manager.get_state('data.data_handler')
    
    # Load DataFrame
    success = data_handler.load_dataframe(sample_df)
    assert success
    
    # Check state updates
    assert state_manager.get_state('data.data_loaded') is True
    loaded_df = state_manager.get_state('data.current_df')
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(sample_df)
    
    # Check column types
    column_types = data_handler.column_types
    assert column_types['Id'] == DataType.NUMERICAL
    assert column_types['category'] == DataType.CATEGORICAL
    assert column_types['value'] == DataType.NUMERICAL
    assert column_types['date'] == DataType.DATE
    assert column_types['active'] == DataType.BOOLEAN

def test_load_file(mock_session_state, mock_uploaded_file):
    """Test loading file through Streamlit's file uploader."""
    # Initialize state manager and data handler
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    data_handler = state_manager.get_state('data.data_handler')
    
    # Load file
    success = data_handler.load_file(mock_uploaded_file)
    assert success
    
    # Check state updates
    assert state_manager.get_state('data.data_loaded') is True
    loaded_df = state_manager.get_state('data.current_df')
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 10  # From sample_df
    
    # Check file info
    file_info = data_handler.get_file_info()
    assert file_info['name'] == "test.csv"
    assert file_info['size'] == 1024
    assert file_info['type'] == "text/csv"

def test_data_processing(mock_session_state, sample_df):
    """Test data processing and type detection."""
    # Initialize state manager and data handler
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    data_handler = state_manager.get_state('data.data_handler')
    
    # Load and process data
    data_handler.load_dataframe(sample_df)
    
    # Check processed data
    processed_df = data_handler.get_data(processed=True)
    assert isinstance(processed_df, pd.DataFrame)
    assert len(processed_df) == len(sample_df)
    
    # Check column statistics
    stats = data_handler.get_column_info()
    assert 'Id' in stats
    assert stats['Id']['type'] == 'numerical'
    assert stats['Id']['min'] == 1
    assert stats['Id']['max'] == 10
    assert stats['Id']['mean'] == 5.5
    
    assert 'category' in stats
    assert stats['category']['type'] == 'categorical'
    assert stats['category']['unique_count'] == 3
    assert stats['category']['most_frequent'] == 'A'

def test_error_handling(mock_session_state):
    """Test error handling in DataHandler."""
    # Initialize state manager and data handler
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    data_handler = state_manager.get_state('data.data_handler')
    
    # Test loading invalid DataFrame
    success = data_handler.load_dataframe(None)
    assert not success
    assert state_manager.get_state('data.data_loaded') is False
    
    # Test loading empty DataFrame
    empty_df = pd.DataFrame()
    success = data_handler.load_dataframe(empty_df)
    assert not success
    assert state_manager.get_state('data.data_loaded') is False
    
    # Test loading DataFrame with missing required columns
    invalid_df = pd.DataFrame({'A': [1, 2, 3]})
    success = data_handler.load_dataframe(invalid_df)
    assert not success
    assert state_manager.get_state('data.data_loaded') is False

def test_state_persistence(mock_session_state, sample_df):
    """Test state persistence across operations."""
    # Initialize state manager and data handler
    state_manager = StateManager()
    state_manager.register_extension('data', {
        'data_handler': DataHandler(),
        'current_df': None,
        'data_loaded': False,
        'data_info': {}
    })
    data_handler = state_manager.get_state('data.data_handler')
    
    # Load data and check state
    data_handler.load_dataframe(sample_df)
    assert state_manager.get_state('data.data_loaded') is True
    
    # Save state
    saved_state = state_manager.save_state(['data'])
    
    # Clear state
    state_manager.clear_state('data')
    assert state_manager.get_state('data.data_loaded') is False
    
    # Restore state
    state_manager.load_state(saved_state)
    assert state_manager.get_state('data.data_loaded') is True
    restored_df = state_manager.get_state('data.current_df')
    assert isinstance(restored_df, pd.DataFrame)
    assert len(restored_df) == len(sample_df)