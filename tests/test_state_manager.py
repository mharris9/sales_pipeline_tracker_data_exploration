"""
Tests for the StateManager class.
"""
import pytest
from core.state_manager import StateManager

def test_initialization():
    """Test StateManager initialization."""
    state_manager = StateManager()
    assert state_manager.get_state('feature_configs') == {}
    assert state_manager.get_state('feature_active') == {}
    assert state_manager.get_state('feature_results') == {}

def test_set_and_get_state():
    """Test setting and getting state."""
    state_manager = StateManager()
    
    # Test setting state with category/key path
    state_manager.set_state('feature_configs/test_feature', {'name': 'test'})
    assert state_manager.get_state('feature_configs/test_feature') == {'name': 'test'}
    
    # Test setting state with category only
    state_manager.set_state('feature_configs', {'all_features': True})
    assert state_manager.get_state('feature_configs') == {'all_features': True}
    
    # Test default value
    assert state_manager.get_state('nonexistent/path', 'default') == 'default'

def test_clear_state():
    """Test clearing state."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('feature_configs/test_feature', {'name': 'test'})
    state_manager.set_state('feature_active/test_feature', True)
    
    # Clear specific key
    state_manager.clear_state('feature_configs/test_feature')
    assert state_manager.get_state('feature_configs/test_feature') is None
    assert state_manager.get_state('feature_active/test_feature') is True
    
    # Clear category
    state_manager.clear_state('feature_active')
    assert state_manager.get_state('feature_active') == {}
    
    # Clear all state
    state_manager.clear_state()
    assert state_manager.get_state('feature_configs') == {}
    assert state_manager.get_state('feature_active') == {}
    assert state_manager.get_state('feature_results') == {}

def test_save_and_load_state():
    """Test saving and loading state."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('feature_configs/test_feature', {'name': 'test'})
    state_manager.set_state('feature_active/test_feature', True)
    
    # Save state
    saved_state = state_manager.save_state()
    
    # Clear state
    state_manager.clear_state()
    
    # Load state
    state_manager.load_state(saved_state)
    
    # Verify state was restored
    assert state_manager.get_state('feature_configs/test_feature') == {'name': 'test'}
    assert state_manager.get_state('feature_active/test_feature') is True

def test_debug_info():
    """Test getting debug information."""
    state_manager = StateManager()
    
    # Set some state
    state_manager.set_state('feature_configs/test_feature', {'name': 'test'})
    
    # Get debug info
    debug_info = state_manager.get_debug_info()
    
    # Check debug info
    assert 'categories' in debug_info
    assert 'history_length' in debug_info
    assert debug_info['history_length'] == 1