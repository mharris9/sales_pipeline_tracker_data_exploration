"""
Tests for Phase 2 - ReportEngine with StateManager integration.
"""
import pytest
import pandas as pd
import numpy as np
import streamlit as st
import logging

from core.state_manager import StateManager
from core.report_engine import ReportEngine
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

def test_report_engine_initialization(mock_session_state):
    """Test ReportEngine initialization with StateManager."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Check initial state
    report_configs = mock_session_state.state_manager.get_state('report_configs')
    assert isinstance(report_configs, dict)
    assert len(report_configs) > 0

def test_create_reports(mock_session_state, sample_df, column_types):
    """Test creating reports."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Check report configurations
    report_configs = mock_session_state.state_manager.get_state('report_configs')
    
    # Check categorical report
    assert mock_session_state.state_manager.get_state('report_configs/category_distribution') is not None
    config = mock_session_state.state_manager.get_state('report_configs/category_distribution')
    assert config['type'] == 'categorical'
    assert config['source_column'] == 'category'
    
    # Check numerical report
    assert mock_session_state.state_manager.get_state('report_configs/value_statistics') is not None
    config = mock_session_state.state_manager.get_state('report_configs/value_statistics')
    assert config['type'] == 'numerical'
    assert config['source_column'] == 'value'
    
    # Check date report
    assert mock_session_state.state_manager.get_state('report_configs/date_trends') is not None
    config = mock_session_state.state_manager.get_state('report_configs/date_trends')
    assert config['type'] == 'date'
    assert config['source_column'] == 'date'

def test_generate_categorical_report(mock_session_state, sample_df, column_types):
    """Test generating categorical report."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Set active reports
    report_engine.set_active_reports(['category_distribution'])
    
    # Generate reports
    report_engine.generate_reports(sample_df)
    
    # Check report results
    results = mock_session_state.state_manager.get_state('report_results/category_distribution')
    assert results is not None
    assert isinstance(results['value_counts'], dict)
    assert isinstance(results['percentages'], dict)
    assert len(results['value_counts']) == 3  # A, B, C
    assert results['value_counts']['A'] == 4
    assert results['value_counts']['B'] == 3
    assert results['value_counts']['C'] == 3

def test_generate_numerical_report(mock_session_state, sample_df, column_types):
    """Test generating numerical report."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Set active reports
    report_engine.set_active_reports(['value_statistics'])
    
    # Generate reports
    report_engine.generate_reports(sample_df)
    
    # Check report results
    results = mock_session_state.state_manager.get_state('report_results/value_statistics')
    assert results is not None
    assert isinstance(results['statistics'], dict)
    assert np.isclose(results['statistics']['mean'], sample_df['value'].mean())
    assert np.isclose(results['statistics']['median'], sample_df['value'].median())
    assert np.isclose(results['statistics']['std'], sample_df['value'].std())
    assert np.isclose(results['statistics']['min'], sample_df['value'].min())
    assert np.isclose(results['statistics']['max'], sample_df['value'].max())

def test_generate_date_report(mock_session_state, sample_df, column_types):
    """Test generating date report."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Set active reports
    report_engine.set_active_reports(['date_trends'])
    
    # Generate reports
    report_engine.generate_reports(sample_df)
    
    # Check report results
    results = mock_session_state.state_manager.get_state('report_results/date_trends')
    assert results is not None
    assert isinstance(results['daily_counts'], dict)
    assert isinstance(results['weekly_counts'], dict)
    assert isinstance(results['monthly_counts'], dict)
    assert isinstance(results['date_range'], dict)
    assert results['date_range']['start'] == sample_df['date'].min().isoformat()
    assert results['date_range']['end'] == sample_df['date'].max().isoformat()

def test_report_state_persistence(mock_session_state, sample_df, column_types):
    """Test report state persistence."""
    # Create report engine
    report_engine = ReportEngine()
    
    # Set active reports
    report_engine.set_active_reports(['category_distribution', 'value_statistics', 'date_trends'])
    
    # Generate reports
    report_engine.generate_reports(sample_df)
    
    # Save state
    saved_state = mock_session_state.state_manager.save_state()
    
    # Clear state
    mock_session_state.state_manager.clear_state()
    assert mock_session_state.state_manager.get_state('report_configs/category_distribution') is None
    
    # Restore state
    mock_session_state.state_manager.load_state(saved_state)
    
    # Check restored state
    assert mock_session_state.state_manager.get_state('report_configs/category_distribution') is not None
    assert mock_session_state.state_manager.get_state('report_configs/value_statistics') is not None
    assert mock_session_state.state_manager.get_state('report_configs/date_trends') is not None
    
    assert mock_session_state.state_manager.get_state('report_results/category_distribution') is not None
    assert mock_session_state.state_manager.get_state('report_results/value_statistics') is not None
    assert mock_session_state.state_manager.get_state('report_results/date_trends') is not None
