"""
Test suite for the state manager.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from core.state_manager import StateManager, StateChangeType, StateChange
from utils.data_types import DataType

# Mock Streamlit session state
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._state[name] = value
    
    def __getattr__(self, name):
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __contains__(self, key):
        return key in self._state

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit module."""
    with patch('core.state_manager.st') as mock_st:
        mock_st.session_state = MockSessionState()
        yield mock_st

@pytest.fixture
def state_manager(mock_streamlit):
    """Create a state manager instance with mocked Streamlit."""
    return StateManager()

class TestBasicFunctionality:
    """Test basic state manager functionality."""
    
    def test_initialization(self, state_manager):
        """Test state manager initialization."""
        assert state_manager._state is not None
        assert 'data' in state_manager._state
        assert 'filters' in state_manager._state
        assert 'view' in state_manager._state
    
    def test_update_and_get_state(self, state_manager):
        """Test updating and getting state."""
        # Test simple value
        state_manager.update_state('test.value', 42)
        assert state_manager.get_state('test.value') == 42
        
        # Test nested value
        state_manager.update_state('test.nested.value', 'test')
        assert state_manager.get_state('test.nested.value') == 'test'
        
        # Test default value
        assert state_manager.get_state('nonexistent.path', 'default') == 'default'
    
    def test_clear_state(self, state_manager):
        """Test clearing state."""
        # Set some state
        state_manager.update_state('test.value', 42)
        
        # Clear specific path
        state_manager.clear_state('test.value')
        assert state_manager.get_state('test.value') is None
        
        # Clear all state
        state_manager.update_state('test.value', 42)
        state_manager.clear_state()
        assert state_manager.get_state('test.value') is None

class TestExtensionMechanism:
    """Test extension registration and usage."""
    
    def test_register_extension(self, state_manager):
        """Test registering new state extensions."""
        # Register new extension
        state_manager.register_extension('test_extension', {'value': 42})
        assert state_manager.get_state('test_extension.value') == 42
        
        # Register without initial state
        state_manager.register_extension('empty_extension')
        assert state_manager.get_state('empty_extension') == {}
    
    def test_extension_updates(self, state_manager):
        """Test updating extension state."""
        state_manager.register_extension('test_extension', {'value': 42})
        state_manager.update_state('test_extension.new_value', 'test')
        assert state_manager.get_state('test_extension.new_value') == 'test'

class TestValidation:
    """Test state validation functionality."""
    
    def test_validator_registration(self, state_manager):
        """Test registering and using validators."""
        def validate_positive(value):
            return isinstance(value, (int, float)) and value > 0
        
        state_manager.register_validator('test.number', validate_positive)
        
        # Valid update
        state_manager.update_state('test.number', 42)
        assert state_manager.get_state('test.number') == 42
        
        # Invalid update
        with pytest.raises(ValueError):
            state_manager.update_state('test.number', -1)
    
    def test_multiple_validators(self, state_manager):
        """Test multiple validators on different paths."""
        def validate_string(value):
            return isinstance(value, str)
        
        def validate_list(value):
            return isinstance(value, list)
        
        state_manager.register_validator('test.string', validate_string)
        state_manager.register_validator('test.list', validate_list)
        
        state_manager.update_state('test.string', 'valid')
        state_manager.update_state('test.list', [1, 2, 3])
        
        with pytest.raises(ValueError):
            state_manager.update_state('test.string', 42)
        
        with pytest.raises(ValueError):
            state_manager.update_state('test.list', 'not a list')

class TestStateTracking:
    """Test state change tracking and watchers."""
    
    def test_state_history(self, state_manager):
        """Test state change history tracking."""
        state_manager.update_state('test.value', 42,
                                 StateChangeType.CUSTOM,
                                 'test',
                                 'Test update')
        
        history = state_manager._state_history
        assert len(history) == 1
        assert isinstance(history[0], StateChange)
        assert history[0].new_value == 42
    
    def test_state_watchers(self, state_manager):
        """Test state change watchers."""
        watcher_called = False
        watcher_old_value = None
        watcher_new_value = None
        
        def test_watcher(old, new):
            nonlocal watcher_called, watcher_old_value, watcher_new_value
            watcher_called = True
            watcher_old_value = old
            watcher_new_value = new
        
        state_manager.register_watcher('test.value', test_watcher)
        state_manager.update_state('test.value', 42)
        
        assert watcher_called
        assert watcher_old_value is None
        assert watcher_new_value == 42

class TestFilterFunctionality:
    """Test filter-specific functionality."""
    
    def test_filter_management(self, state_manager):
        """Test filter state management."""
        # Set up a test filter
        state_manager.update_filter('category', {
            'type': 'categorical',
            'values': ['A', 'B', 'C'],
            'selected': ['A']
        })
        
        assert state_manager.get_state('filters.filter_configs.category') is not None
        
        # Activate filter
        state_manager.set_filter_active('category', True)
        assert state_manager.get_state('filters.active_filters.category') is True

class TestDataManagement:
    """Test data management functionality."""
    
    def test_data_operations(self, state_manager):
        """Test data operations."""
        # Create test DataFrame
        df = pd.DataFrame({
            'A': range(10),
            'B': list('abcdefghij')
        })
        
        column_types = {
            'A': DataType.NUMERICAL,
            'B': DataType.CATEGORICAL
        }
        
        # Set data
        state_manager.set_data(df, column_types)
        
        # Check data is stored
        stored_df = state_manager.get_filtered_data()
        assert stored_df is not None
        assert len(stored_df) == len(df)
        
        # Check data info
        info = state_manager.get_data_info()
        assert info['total_records'] == len(df)
        assert info['columns'] == list(df.columns)

class TestDebugFeatures:
    """Test debugging and logging features."""
    
    def test_debug_info(self, state_manager):
        """Test debug information access."""
        # Make some state changes
        state_manager.update_state('test.value', 42)
        state_manager.register_extension('test_extension')
        
        debug_info = state_manager.get_debug_info()
        assert 'current_state' in debug_info
        assert 'state_history' in debug_info
        assert 'registered_validators' in debug_info
        assert 'registered_watchers' in debug_info

class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_invalid_path(self, state_manager):
        """Test handling invalid state paths."""
        # Get nonexistent path
        assert state_manager.get_state('invalid.path', default='default') == 'default'
        
        # Update invalid path should create it
        state_manager.update_state('new.path', 42)
        assert state_manager.get_state('new.path') == 42
    
    def test_validator_errors(self, state_manager):
        """Test validator error handling."""
        def failing_validator(value):
            raise Exception("Validator error")
        
        state_manager.register_validator('test.value', failing_validator)
        
        with pytest.raises(Exception):
            state_manager.update_state('test.value', 42)
    
    def test_watcher_errors(self, state_manager):
        """Test watcher error handling."""
        def failing_watcher(old, new):
            raise Exception("Watcher error")
        
        state_manager.register_watcher('test.value', failing_watcher)
        
        # Should not raise exception
        state_manager.update_state('test.value', 42)
        assert state_manager.get_state('test.value') == 42
