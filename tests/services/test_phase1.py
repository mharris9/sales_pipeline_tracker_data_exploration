"""
Tests for Phase 1 of state management transition.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import logging

logger = logging.getLogger(__name__)

from src.services.state_manager import StateManager
from src.services.data_handler import DataHandler
from src.services.filter_manager import FilterManager
from src.services.feature_engine import FeatureEngine
from src.services.report_engine import ReportEngine
from src.services.outlier_manager import OutlierManager
from src.utils.export_utils import ExportManager

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

def test_initialize_session_state(mock_session_state):
    """Test initialization of session state with StateManager."""
    from main import initialize_session_state
    
    # Call initialization
    initialize_session_state()
    
    # Verify state manager was created
    assert 'state_manager' in st.session_state
    assert isinstance(st.session_state.state_manager, StateManager)
    
    # Verify core state containers
    state_manager = st.session_state.state_manager
    
    # Check data state
    assert isinstance(state_manager.get_state('data.data_handler'), DataHandler)
    assert state_manager.get_state('data.current_df') is None
    assert state_manager.get_state('data.data_loaded') is False
    
    # Check filter state
    assert isinstance(state_manager.get_state('filters.filter_manager'), FilterManager)
    assert isinstance(state_manager.get_state('filters.active_filters'), dict)
    assert isinstance(state_manager.get_state('filters.filter_configs'), dict)
    
    # Check feature state
    assert isinstance(state_manager.get_state('features.feature_engine'), FeatureEngine)
    assert isinstance(state_manager.get_state('features.computed_features'), dict)
    
    # Check report state
    assert isinstance(state_manager.get_state('reports.report_engine'), ReportEngine)
    assert state_manager.get_state('reports.current_report') is None
    
    # Check export state
    assert isinstance(state_manager.get_state('exports.export_manager'), ExportManager)
    assert isinstance(state_manager.get_state('exports.export_history'), list)
    
    # Check outlier state
    assert isinstance(state_manager.get_state('outliers.outlier_manager'), OutlierManager)
    assert isinstance(state_manager.get_state('outliers.settings'), dict)
    assert state_manager.get_state('outliers.settings.outliers_enabled') is False
    
    # Verify backward compatibility
    assert isinstance(st.session_state.data_handler, DataHandler)
    assert isinstance(st.session_state.filter_manager, FilterManager)
    assert isinstance(st.session_state.feature_engine, FeatureEngine)
    assert isinstance(st.session_state.report_engine, ReportEngine)
    assert isinstance(st.session_state.export_manager, ExportManager)
    assert isinstance(st.session_state.outlier_manager, OutlierManager)
    assert st.session_state.current_df is None
    assert st.session_state.data_loaded is False
    assert isinstance(st.session_state.outlier_settings, dict)
    assert isinstance(st.session_state.exclusion_info, dict)

def test_state_validators(mock_session_state):
    """Test state validators are working correctly."""
    from main import initialize_session_state
    
    # Initialize state
    initialize_session_state()
    state_manager = st.session_state.state_manager
    
    # Test DataFrame validator
    with pytest.raises(ValueError):
        state_manager.update_state('data.current_df', "not a dataframe")
    
    valid_df = pd.DataFrame({'A': [1, 2, 3]})
    state_manager.update_state('data.current_df', valid_df)
    assert state_manager.get_state('data.current_df').equals(valid_df)
    
    # Test boolean validator
    with pytest.raises(ValueError):
        state_manager.update_state('data.data_loaded', "not a boolean")
    
    state_manager.update_state('data.data_loaded', True)
    assert state_manager.get_state('data.data_loaded') is True
    
    # Test outlier settings validator
    with pytest.raises(ValueError):
        state_manager.update_state('outliers.settings', {'wrong_key': True})
    
    valid_settings = {'outliers_enabled': True}
    state_manager.update_state('outliers.settings', valid_settings)
    assert state_manager.get_state('outliers.settings') == valid_settings

def test_state_watchers(mock_session_state, caplog):
    """Test state watchers are triggered correctly."""
    from main import initialize_session_state
    
    # Configure logging for test
    caplog.set_level(logging.INFO)
    
    # Initialize state
    initialize_session_state()
    state_manager = st.session_state.state_manager
    
    # Clear caplog to remove initialization messages
    caplog.clear()
    
    # Register watchers
    def df_watcher(old, new):
        if new is not None:
            logger.info(f"DataFrame updated: {len(new)} rows")
    
    def filter_watcher(old, new):
        logger.info(f"Active filters changed: {new}")
    
    state_manager.register_watcher('data.current_df', df_watcher)
    state_manager.register_watcher('filters.active_filters', filter_watcher)
    
    # Update DataFrame and check log
    df = pd.DataFrame({'A': range(10)})
    state_manager.update_state('data.current_df', df)
    assert "DataFrame updated: 10 rows" in caplog.text
    
    # Clear caplog for next test
    caplog.clear()
    
    # Update filters and check log
    filters = {'column1': True, 'column2': False}
    state_manager.update_state('filters.active_filters', filters)
    assert "Active filters changed:" in caplog.text
    assert str(filters) in caplog.text
