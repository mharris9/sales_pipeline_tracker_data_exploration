"""
Tests for Phase 2 - FilterManager with StateManager integration.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import logging

from core.state_manager import StateManager
from core.filter_manager import FilterManager
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

def test_filter_manager_initialization(mock_session_state):
    """Test FilterManager initialization with StateManager."""
    # Initialize state manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    
    # Get FilterManager instance
    filter_manager = state_manager.get_state('filters.filter_manager')
    assert isinstance(filter_manager, FilterManager)
    
    # Check initial state
    assert state_manager.get_state('filters.filter_configs') == {}
    assert state_manager.get_state('filters.active_filters') == {}

def test_create_filters(mock_session_state, sample_df, column_types):
    """Test creating filters from DataFrame."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
        # Check filter configurations
    filter_configs = state_manager.get_state('filters.filter_configs')
    assert 'category' in filter_configs
    assert filter_configs['category']['type'] == 'categorical'
    assert set(filter_configs['category']['unique_values']) == {'A', 'B', 'C'}
    
    assert 'value' in filter_configs
    assert filter_configs['value']['type'] == 'numerical'
    assert filter_configs['value']['min_value'] == 10.5
    assert filter_configs['value']['max_value'] == 35.0
    
    assert 'date' in filter_configs
    assert filter_configs['date']['type'] == 'date'
    
    assert 'active' in filter_configs
    assert filter_configs['active']['type'] == 'boolean'

def test_apply_categorical_filter(mock_session_state, sample_df, column_types):
    """Test applying categorical filter."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
    # Apply filter
    filter_manager.update_filter('category', {
        'type': 'categorical',
        'filter_type': 'include',
        'selected_values': ['A']
    })
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == 4  # Only 'A' values
    assert set(filtered_df['category'].unique()) == {'A'}
    
    # Check filter state
    active_filters = state_manager.get_state('filters.active_filters')
    assert active_filters['category'] is True
    
    filter_results = state_manager.get_state('filters.filter_results')
    assert filter_results['category']['filtered_count'] == 4
    assert filter_results['category']['total_count'] == 10

def test_apply_numerical_filter(mock_session_state, sample_df, column_types):
    """Test applying numerical filter."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
    # Apply filter
    filter_manager.update_filter('value', {
        'type': 'numerical',
        'filter_type': 'range',
        'selected_min': 15.0,
        'selected_max': 25.0
    })
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == 4  # Values between 15 and 25
    assert all(filtered_df['value'].between(15.0, 25.0))
    
    # Check filter state
    active_filters = state_manager.get_state('filters.active_filters')
    assert active_filters['value'] is True
    
    filter_results = state_manager.get_state('filters.filter_results')
    assert filter_results['value']['filtered_count'] == 4
    assert filter_results['value']['total_count'] == 10

def test_apply_date_filter(mock_session_state, sample_df, column_types):
    """Test applying date filter."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
    # Apply filter
    filter_manager.update_filter('date', {
        'type': 'date',
        'filter_type': 'range',
        'selected_min_date': pd.Timestamp('2024-01-03'),
        'selected_max_date': pd.Timestamp('2024-01-07')
    })
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == 5  # Dates between 01/03 and 01/07
    assert all(filtered_df['date'].between('2024-01-03', '2024-01-07'))
    
    # Check filter state
    active_filters = state_manager.get_state('filters.active_filters')
    assert active_filters['date'] is True
    
    filter_results = state_manager.get_state('filters.filter_results')
    assert filter_results['date']['filtered_count'] == 5
    assert filter_results['date']['total_count'] == 10

def test_apply_boolean_filter(mock_session_state, sample_df, column_types):
    """Test applying boolean filter."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
    # Apply filter
    filter_manager.update_filter('active', {
        'type': 'boolean',
        'filter_type': 'include',
        'selected_values': [True]
    })
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == 6  # Only True values
    assert all(filtered_df['active'])
    
    # Check filter state
    active_filters = state_manager.get_state('filters.active_filters')
    assert active_filters['active'] is True
    
    filter_results = state_manager.get_state('filters.filter_results')
    assert filter_results['active']['filtered_count'] == 6
    assert filter_results['active']['total_count'] == 10

def test_clear_filters(mock_session_state, sample_df, column_types):
    """Test clearing all filters."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters
    filter_manager.create_filters(sample_df, column_types)
    
    # Apply multiple filters
    filter_manager.update_filter('category', {
        'type': 'categorical',
        'filter_type': 'include',
        'selected_values': ['A']
    })
    filter_manager.update_filter('value', {
        'type': 'numerical',
        'filter_type': 'range',
        'selected_min': 15.0,
        'selected_max': 25.0
    })
    
    # Clear filters
    filter_manager.clear_all_filters()
    
    # Check filter state
    active_filters = state_manager.get_state('filters.active_filters')
    assert not any(active_filters.values())
    
    filter_results = state_manager.get_state('filters.filter_results')
    assert not filter_results
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == len(sample_df)  # No filters applied

def test_filter_state_persistence(mock_session_state, sample_df, column_types):
    """Test filter state persistence."""
    # Initialize state manager and filter manager
    state_manager = StateManager()
    state_manager.register_extension('filters', {
        'filter_manager': FilterManager(),
        'active_filters': {},
        'filter_configs': {},
        'filter_results': {},
        'filter_summary': {}
    })
    filter_manager = state_manager.get_state('filters.filter_manager')
    
    # Create filters and apply some
    filter_manager.create_filters(sample_df, column_types)
    filter_manager.update_filter('category', {
        'type': 'categorical',
        'filter_type': 'include',
        'selected_values': ['A']
    })
    
    # Save state
    saved_state = state_manager.save_state(['filters'])
    
    # Clear state
    state_manager.clear_state('filters')
    assert not state_manager.get_state('filters.active_filters')
    
    # Restore state
    state_manager.load_state(saved_state)
    
    # Check restored state
    active_filters = state_manager.get_state('filters.active_filters')
    assert active_filters['category'] is True
    
    filter_configs = state_manager.get_state('filters.filter_configs')
    assert filter_configs['category']['selected_values'] == ['A']
    
    # Check filtered data
    filtered_df = filter_manager.apply_filters(sample_df)
    assert len(filtered_df) == 4  # Only 'A' values
    assert set(filtered_df['category'].unique()) == {'A'}
