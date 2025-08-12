"""
Tests for the StateManager class.
"""
import pytest
from core.state_manager import StateManager

def test_initialization():
    """Test StateManager initialization."""
    state_manager = StateManager()
    
    # Check core categories exist
    assert isinstance(state_manager.get_state('data'), dict)
    assert isinstance(state_manager.get_state('filters'), dict)
    assert isinstance(state_manager.get_state('features'), dict)
    assert isinstance(state_manager.get_state('outliers'), dict)
    assert isinstance(state_manager.get_state('reports'), dict)
    assert isinstance(state_manager.get_state('ui'), dict)
    assert isinstance(state_manager.get_state('errors'), dict)
    
    # Check specific subcategories
    assert state_manager.get_state('features/configs') == {}
    assert state_manager.get_state('features/active') == {}
    assert state_manager.get_state('features/results') == {}

def test_set_and_get_state():
    """Test setting and getting state."""
    state_manager = StateManager()
    
    # Test setting and getting nested state
    state_manager.set_state('features/configs/test_feature', {'name': 'test'})
    assert state_manager.get_state('features/configs/test_feature') == {'name': 'test'}
    
    # Test setting and getting top-level state
    state_manager.set_state('features/active', {'all_features': True})
    assert state_manager.get_state('features/active') == {'all_features': True}
    
    # Test default value
    assert state_manager.get_state('nonexistent/path', 'default') == 'default'

def test_clear_state():
    """Test clearing state."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('features/configs/test_feature', {'name': 'test'})
    state_manager.set_state('features/active/test_feature', True)
    
    # Clear specific key
    state_manager.clear_state('features/configs/test_feature')
    assert state_manager.get_state('features/configs/test_feature') == {}
    assert state_manager.get_state('features/active/test_feature') is True
    
    # Clear subcategory
    state_manager.clear_state('features/active')
    assert state_manager.get_state('features/active') == {}
    
    # Clear category
    state_manager.clear_state('features')
    assert state_manager.get_state('features/configs') == {}
    assert state_manager.get_state('features/active') == {}
    assert state_manager.get_state('features/results') == {}

def test_save_and_load_state():
    """Test saving and loading state."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('features/configs/test_feature', {'name': 'test'})
    state_manager.set_state('features/active/test_feature', True)
    
    # Save state
    saved_state = state_manager.save_state()
    
    # Clear state
    state_manager.clear_state()
    
    # Load state
    state_manager.load_state(saved_state)
    
    # Verify state was restored
    assert state_manager.get_state('features/configs/test_feature') == {'name': 'test'}
    assert state_manager.get_state('features/active/test_feature') is True

def test_debug_info():
    """Test getting debug information."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('features/configs/test_feature', {'name': 'test'})
    
    # Get debug info
    debug_info = state_manager.get_debug_info()
    
    # Check debug info structure
    assert 'categories' in debug_info
    assert 'history_length' in debug_info
    assert isinstance(debug_info['categories'], list)
    assert isinstance(debug_info['history_length'], int)
    assert debug_info['history_length'] == 1