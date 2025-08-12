"""
Tests for Phase 2 - OutlierManager with StateManager integration.
"""
import pytest
import pandas as pd
import numpy as np
import streamlit as st
import logging

from core.state_manager import StateManager
from core.outlier_manager import OutlierManager
from utils.data_types import DataType

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
    
    # Initialize state manager
    state_manager = StateManager()
    session_state.state_manager = state_manager
    
    return session_state

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame({
        'Id': range(1, 11),  # Required ID column
        'Snapshot Date': pd.date_range(start='2024-01-01', periods=10).strftime('%m/%d/%Y'),  # Required date column
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'value': [10.5, 20.0, 15.5, 30.0, 25.5, 12.0, 35.0, 22.5, 18.0, 28.0],
        'date': pd.date_range(start='2024-01-01', periods=10),
        'active': [True, False, True, True, False, True, False, True, False, True]
    })
    
    # Convert date columns to datetime
    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'])
    df['date'] = pd.to_datetime(df['date'])
    
    return df

@pytest.fixture
def column_types():
    """Create column type mappings for testing."""
    return {
        'Id': DataType.NUMERICAL,
        'Snapshot Date': DataType.DATE,
        'category': DataType.CATEGORICAL,
        'value': DataType.NUMERICAL,
        'date': DataType.DATE,
        'active': DataType.BOOLEAN
    }

def test_outlier_manager_initialization(mock_session_state):
    """Test OutlierManager initialization with StateManager."""
    # Create outlier manager
    outlier_manager = OutlierManager()
    
    # Check initial state
    outlier_configs = mock_session_state.state_manager.get_state('outlier_configs')
    assert isinstance(outlier_configs, dict)
    assert len(outlier_configs) > 0

def test_create_outlier_detectors(mock_session_state, sample_df, column_types):
    """Test creating outlier detectors."""
    # Create outlier manager
    outlier_manager = OutlierManager()
    
    # Check outlier configurations
    outlier_configs = mock_session_state.state_manager.get_state('outlier_configs')
    
    # Check numerical outlier detector
    assert mock_session_state.state_manager.get_state('outlier_configs/value_zscore') is not None
    config = mock_session_state.state_manager.get_state('outlier_configs/value_zscore')
    assert config['type'] == 'numerical'
    assert config['method'] == 'zscore'
    assert config['source_column'] == 'value'
    
    # Check date outlier detector
    assert mock_session_state.state_manager.get_state('outlier_configs/date_range') is not None
    config = mock_session_state.state_manager.get_state('outlier_configs/date_range')
    assert config['type'] == 'date'
    assert config['method'] == 'range'
    assert config['source_column'] == 'date'

def test_detect_numerical_outliers(mock_session_state, sample_df, column_types):
    """Test detecting numerical outliers."""
    # Create outlier manager
    outlier_manager = OutlierManager()
    
    # Set active detectors
    outlier_manager.set_active_detectors(['value_zscore'])
    
    # Detect outliers
    outlier_manager.detect_outliers(sample_df)
    
    # Check outlier results
    results = mock_session_state.state_manager.get_state('outlier_results/value_zscore')
    assert results is not None
    assert isinstance(results['outlier_indices'], list)
    assert isinstance(results['zscore_values'], dict)
    assert isinstance(results['threshold'], float)

def test_detect_date_outliers(mock_session_state, sample_df, column_types):
    """Test detecting date outliers."""
    # Create outlier manager
    outlier_manager = OutlierManager()
    
    # Set active detectors
    outlier_manager.set_active_detectors(['date_range'])
    
    # Detect outliers
    outlier_manager.detect_outliers(sample_df)
    
    # Check outlier results
    results = mock_session_state.state_manager.get_state('outlier_results/date_range')
    assert results is not None
    assert isinstance(results['outlier_indices'], list)
    assert isinstance(results['min_date'], str)
    assert isinstance(results['max_date'], str)

def test_outlier_state_persistence(mock_session_state, sample_df, column_types):
    """Test outlier state persistence."""
    # Create outlier manager
    outlier_manager = OutlierManager()
    
    # Set active detectors
    outlier_manager.set_active_detectors(['value_zscore', 'date_range'])
    
    # Detect outliers
    outlier_manager.detect_outliers(sample_df)
    
    # Save state
    saved_state = mock_session_state.state_manager.save_state()
    
    # Clear state
    mock_session_state.state_manager.clear_state()
    assert mock_session_state.state_manager.get_state('outlier_configs/value_zscore') is None
    
    # Restore state
    mock_session_state.state_manager.load_state(saved_state)
    
    # Check restored state
    assert mock_session_state.state_manager.get_state('outlier_configs/value_zscore') is not None
    assert mock_session_state.state_manager.get_state('outlier_configs/date_range') is not None
    
    assert mock_session_state.state_manager.get_state('outlier_results/value_zscore') is not None
    assert mock_session_state.state_manager.get_state('outlier_results/date_range') is not None
