"""
Tests for Phase 2 - FeatureEngine with StateManager integration.
"""
import pytest
import pandas as pd
import numpy as np
import streamlit as st
import logging

from core.state_manager import StateManager
from core.feature_engine import FeatureEngine
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

def test_feature_engine_initialization(mock_session_state):
    """Test FeatureEngine initialization with StateManager."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Check initial state
    feature_configs = mock_session_state.state_manager.get_state('feature_configs')
    assert isinstance(feature_configs, dict)
    assert len(feature_configs) > 0

def test_create_features(mock_session_state, sample_df, column_types):
    """Test creating features from DataFrame."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Check feature configurations
    feature_configs = mock_session_state.state_manager.get_state('feature_configs')
    
    # Check categorical features
    assert mock_session_state.state_manager.get_state('feature_configs/category_counts') is not None
    config = mock_session_state.state_manager.get_state('feature_configs/category_counts')
    assert config['type'] == 'categorical'
    assert config['source_column'] == 'category'
    
    # Check numerical features
    assert mock_session_state.state_manager.get_state('feature_configs/value_stats') is not None
    config = mock_session_state.state_manager.get_state('feature_configs/value_stats')
    assert config['type'] == 'numerical'
    assert config['source_column'] == 'value'
    
    # Check date features
    assert mock_session_state.state_manager.get_state('feature_configs/date_trends') is not None
    config = mock_session_state.state_manager.get_state('feature_configs/date_trends')
    assert config['type'] == 'date'
    assert config['source_column'] == 'date'

def test_calculate_categorical_features(mock_session_state, sample_df, column_types):
    """Test calculating categorical features."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Set active features
    feature_engine.set_active_features(['category_counts'])
    
    # Calculate features
    feature_engine.calculate_features(sample_df)
    
    # Check feature results
    results = mock_session_state.state_manager.get_state('feature_results/category_counts')
    assert results is not None
    assert results['A'] == 4
    assert results['B'] == 3
    assert results['C'] == 3

def test_calculate_numerical_features(mock_session_state, sample_df, column_types):
    """Test calculating numerical features."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Set active features
    feature_engine.set_active_features(['value_stats'])
    
    # Calculate features
    feature_engine.calculate_features(sample_df)
    
    # Check feature results
    results = mock_session_state.state_manager.get_state('feature_results/value_stats')
    assert results is not None
    assert np.isclose(results['mean'], sample_df['value'].mean())
    assert np.isclose(results['median'], sample_df['value'].median())
    assert np.isclose(results['std'], sample_df['value'].std())

def test_calculate_date_features(mock_session_state, sample_df, column_types):
    """Test calculating date features."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Set active features
    feature_engine.set_active_features(['date_trends'])
    
    # Calculate features
    feature_engine.calculate_features(sample_df)
    
    # Check feature results
    results = mock_session_state.state_manager.get_state('feature_results/date_trends')
    assert results is not None
    assert 'daily' in results
    assert 'weekly' in results
    assert 'monthly' in results

def test_feature_state_persistence(mock_session_state, sample_df, column_types):
    """Test feature state persistence."""
    # Create feature engine
    feature_engine = FeatureEngine()
    
    # Set active features
    feature_engine.set_active_features(['category_counts', 'value_stats', 'date_trends'])
    
    # Calculate features
    feature_engine.calculate_features(sample_df)
    
    # Save state
    saved_state = mock_session_state.state_manager.save_state()
    
    # Clear state
    mock_session_state.state_manager.clear_state()
    assert mock_session_state.state_manager.get_state('feature_configs/category_counts') is None
    
    # Restore state
    mock_session_state.state_manager.load_state(saved_state)
    
    # Check restored state
    assert mock_session_state.state_manager.get_state('feature_configs/category_counts') is not None
    assert mock_session_state.state_manager.get_state('feature_configs/value_stats') is not None
    assert mock_session_state.state_manager.get_state('feature_configs/date_trends') is not None
    
    assert mock_session_state.state_manager.get_state('feature_results/category_counts') is not None
    assert mock_session_state.state_manager.get_state('feature_results/value_stats') is not None
    assert mock_session_state.state_manager.get_state('feature_results/date_trends') is not None