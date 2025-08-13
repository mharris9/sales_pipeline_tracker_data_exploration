"""
Extended test suite for state manager covering edge cases and advanced functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import streamlit as st
from src.services.state_manager import StateManager
from src.utils.data_types import DataType
import time
import json

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
        """Mock Streamlit for testing."""
        with patch('src.services.state_manager.st') as mock_st:
            mock_st.session_state = MockSessionState()
            yield mock_st

@pytest.fixture
def state_manager(mock_streamlit):
    """Create a state manager instance with mocked Streamlit."""
    return StateManager()

class TestWidgetStateIntegration:
    """Test Streamlit widget state integration."""
    
    def test_checkbox_persistence(self, state_manager):
        """Test checkbox state persistence across reruns."""
        # Simulate checkbox state
        state_manager.update_state('widgets.checkbox1', True)
        assert state_manager.get_state('widgets.checkbox1') is True
        
        # Simulate rerun
        state_manager._initialize_state()
        assert state_manager.get_state('widgets.checkbox1', False) is False
    
    def test_multiselect_state(self, state_manager):
        """Test multiselect widget state handling."""
        options = ['A', 'B', 'C']
        selected = ['A', 'C']
        
        state_manager.update_state('widgets.multiselect1', {
            'options': options,
            'selected': selected
        })
        
        state = state_manager.get_state('widgets.multiselect1')
        assert state['options'] == options
        assert state['selected'] == selected
    
    def test_slider_range_state(self, state_manager):
        """Test slider widget with range values."""
        state_manager.update_state('widgets.slider1', {
            'min': 0,
            'max': 100,
            'value': (25, 75)
        })
        
        state = state_manager.get_state('widgets.slider1')
        assert state['min'] == 0
        assert state['max'] == 100
        assert state['value'] == (25, 75)
    
    def test_radio_button_groups(self, state_manager):
        """Test radio button group state consistency."""
        options = ['option1', 'option2', 'option3']
        state_manager.update_state('widgets.radio1', {
            'options': options,
            'selected': 'option2'
        })
        
        state = state_manager.get_state('widgets.radio1')
        assert state['selected'] in state['options']

class TestConcurrentOperations:
    """Test concurrent state operations."""
    
    def test_parallel_filter_updates(self, state_manager):
        """Test multiple filter updates happening close together."""
        updates = []
        
        def record_update(old, new):
            updates.append((old, new))
        
        state_manager.register_watcher('filters.active', record_update)
        
        # Simulate rapid updates
        for i in range(5):
            state_manager.update_state('filters.active', f'filter{i}')
        
        assert len(updates) == 5
        assert updates[-1][1] == 'filter4'
    
    def test_filter_while_loading(self, state_manager):
        """Test filter updates during data loading."""
        df = pd.DataFrame({'A': range(100)})
        
        # Start "loading" data
        state_manager.update_state('data.loading', True)
        state_manager.update_state('data.df', df)
        
        # Try to update filter during load
        state_manager.update_state('filters.A', {'min': 0, 'max': 50})
        
        # Complete loading
        state_manager.update_state('data.loading', False)
        
        # Check filter was applied
        filter_state = state_manager.get_state('filters.A')
        assert filter_state['min'] == 0
        assert filter_state['max'] == 50
    
    def test_state_consistency_check(self, state_manager):
        """Verify state consistency across all components."""
        # Set up some interrelated state
        df = pd.DataFrame({'A': range(100)})
        state_manager.update_state('data.df', df)
        state_manager.update_state('filters.A', {'min': 0, 'max': 50})
        state_manager.update_state('view.filtered_count', 51)
        
        # Check consistency
        data = state_manager.get_state('data.df')
        filters = state_manager.get_state('filters.A')
        view = state_manager.get_state('view.filtered_count')
        
        assert len(data) == 100
        assert filters['max'] == 50
        assert view == 51

class TestDataTypeHandling:
    """Test handling of various data types and edge cases."""
    
    def test_mixed_type_columns(self, state_manager):
        """Test columns with mixed data types."""
        mixed_data = pd.DataFrame({
            'mixed': ['1', 2, '3', 4.0, 'text'],
            'dates': ['2024-01-01', '2024-01-02', np.nan, 
                     datetime.now(), '2024-01-05']
        })
        
        state_manager.update_state('data.mixed_df', mixed_data)
        assert state_manager.get_state('data.mixed_df') is not None
    
    def test_null_handling(self, state_manager):
        """Test handling of null/NaN values."""
        null_data = pd.DataFrame({
            'nulls': [1, None, np.nan, pd.NA, 5],
            'complete': range(5)
        })
        
        state_manager.update_state('data.null_df', null_data)
        retrieved = state_manager.get_state('data.null_df')
        assert len(retrieved) == 5
    
    def test_date_format_variations(self, state_manager):
        """Test different date formats and timezone handling."""
        dates = pd.DataFrame({
            'dates': [
                '2024-01-01',
                '01/02/2024',
                datetime.now(timezone.utc),
                np.datetime64('2024-01-04'),
                pd.Timestamp('2024-01-05')
            ]
        })
        
        state_manager.update_state('data.dates_df', dates)
        assert state_manager.get_state('data.dates_df') is not None
    
    def test_large_categorical_values(self, state_manager):
        """Test categories with large number of unique values."""
        large_cats = pd.DataFrame({
            'category': [f'cat_{i}' for i in range(1000)]
        })
        
        state_manager.update_state('data.large_cats_df', large_cats)
        assert len(state_manager.get_state('data.large_cats_df')) == 1000

class TestStateHistory:
    """Test state history and rollback functionality."""
    
    def test_state_rollback(self, state_manager):
        """Test ability to rollback to previous state."""
        # Create some state changes
        state_manager.update_state('test.value', 1)
        state_manager.update_state('test.value', 2)
        state_manager.update_state('test.value', 3)
        
        # Rollback one step
        state_manager.rollback('test.value')
        assert state_manager.get_state('test.value') == 2
    
    def test_history_size_limits(self, state_manager):
        """Test history size management."""
        # Create many state changes
        for i in range(1000):
            state_manager.update_state('test.value', i)
        
        # Check history size is limited
        history = state_manager._state_history
        assert len(history) <= state_manager.MAX_HISTORY_SIZE
    
    def test_selective_rollback(self, state_manager):
        """Test rolling back specific changes."""
        state_manager.update_state('test.a', 1)
        state_manager.update_state('test.b', 2)
        state_manager.update_state('test.a', 3)
        
        # Rollback only 'test.a'
        state_manager.rollback('test.a')
        assert state_manager.get_state('test.a') == 1
        assert state_manager.get_state('test.b') == 2

class TestMemoryManagement:
    """Test memory management features."""
    
    def test_large_dataframe_handling(self, state_manager):
        """Test memory efficient handling of large DataFrames."""
        # Create a large DataFrame
        large_df = pd.DataFrame({
            'A': range(100000),
            'B': ['text'] * 100000
        })
        
        # Monitor memory usage
        initial_memory = large_df.memory_usage(deep=True).sum()
        
        state_manager.update_state('data.large_df', large_df)
        stored_df = state_manager.get_state('data.large_df')
        
        final_memory = stored_df.memory_usage(deep=True).sum()
        assert final_memory <= initial_memory * 1.1  # Allow 10% overhead
    
    def test_state_cleanup(self, state_manager):
        """Test proper cleanup of old state data."""
        # Create some temporary state
        state_manager.update_state('temp.data', pd.DataFrame({'A': range(1000)}))
        
        # Clear temporary state
        state_manager.clear_state('temp')
        assert state_manager.get_state('temp.data') is None
    
    def test_memory_thresholds(self, state_manager):
        """Test behavior when approaching memory limits."""
        large_data = [pd.DataFrame({'A': range(10000)}) for _ in range(10)]
        
        # Try to store increasingly more data
        for i, df in enumerate(large_data):
            try:
                state_manager.update_state(f'data.df_{i}', df)
            except MemoryError:
                # Should handle memory errors gracefully
                assert state_manager.get_state(f'data.df_{i}') is None
                break

class TestFilterDependencies:
    """Test filter dependency management."""
    
    def test_dependent_filters(self, state_manager):
        """Test filters that depend on other filter results."""
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100)
        })
        
        state_manager.update_state('data.df', df)
        
        # Set up dependent filters
        state_manager.update_state('filters.A', {'min': 0, 'max': 50})
        state_manager.update_state('filters.B', {'min': 25, 'max': 75})
        
        # Apply filters in sequence
        filtered_data = state_manager.apply_filters(['A', 'B'])
        assert len(filtered_data) < len(df)
    
    def test_filter_order_impact(self, state_manager):
        """Test if filter order affects results."""
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100)
        })
        
        state_manager.update_state('data.df', df)
        
        # Apply filters in different orders
        filters = {
            'A': {'min': 0, 'max': 50},
            'B': {'min': 25, 'max': 75}
        }
        
        result1 = state_manager.apply_filters(['A', 'B'])
        result2 = state_manager.apply_filters(['B', 'A'])
        
        # Results should be the same regardless of order
        assert result1.equals(result2)
    
    def test_circular_dependencies(self, state_manager):
        """Test handling of circular filter dependencies."""
        # Set up circular dependent filters
        with pytest.raises(ValueError):
            state_manager.add_filter_dependency('A', 'B')
            state_manager.add_filter_dependency('B', 'C')
            state_manager.add_filter_dependency('C', 'A')

class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_state_updates(self, state_manager):
        """Test performance with large state trees."""
        start_time = time.time()
        
        # Create a large state tree
        for i in range(1000):
            state_manager.update_state(f'large.tree.branch{i}', {
                'leaf1': i,
                'leaf2': str(i),
                'leaf3': [i, i+1, i+2]
            })
        
        end_time = time.time()
        assert end_time - start_time < 5  # Should complete within 5 seconds
    
    def test_many_watchers(self, state_manager):
        """Test performance with many state watchers."""
        counter = {'count': 0}
        
        # Register many watchers
        for i in range(100):
            def watcher(old, new, i=i):
                counter['count'] += 1
            state_manager.register_watcher(f'test.value{i}', watcher)
        
        start_time = time.time()
        
        # Trigger all watchers
        for i in range(100):
            state_manager.update_state(f'test.value{i}', i)
        
        end_time = time.time()
        assert end_time - start_time < 1  # Should complete within 1 second
        assert counter['count'] == 100
    
    def test_complex_filter_chains(self, state_manager):
        """Test performance of complex filter combinations."""
        df = pd.DataFrame({
            col: range(10000) for col in ['A', 'B', 'C', 'D', 'E']
        })
        
        state_manager.update_state('data.df', df)
        
        start_time = time.time()
        
        # Apply multiple complex filters
        for col in df.columns:
            state_manager.update_state(f'filters.{col}', {
                'min': len(df) // 4,
                'max': len(df) // 2
            })
        
        # Apply all filters
        filtered_df = state_manager.apply_filters(list(df.columns))
        
        end_time = time.time()
        assert end_time - start_time < 2  # Should complete within 2 seconds

class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_partial_state_corruption(self, state_manager):
        """Test recovery from partial state corruption."""
        # Set up some valid state
        state_manager.update_state('valid.data', {'key': 'value'})
        
        # Simulate corruption
        state_manager._state['corrupted'] = object()  # Un-serializable object
        
        # Should still be able to access valid state
        assert state_manager.get_state('valid.data')['key'] == 'value'
        
        # Should handle corrupted state gracefully
        assert state_manager.get_state('corrupted') is None
    
    def test_invalid_widget_state(self, state_manager):
        """Test recovery from invalid widget states."""
        # Set up valid widget state
        state_manager.update_state('widgets.slider', {
            'min': 0,
            'max': 100,
            'value': 50
        })
        
        # Corrupt widget state
        state_manager.update_state('widgets.slider.value', 150)  # Outside valid range
        
        # Should reset to valid state
        widget_state = state_manager.get_state('widgets.slider')
        assert widget_state['value'] <= widget_state['max']
    
    def test_inconsistent_state(self, state_manager):
        """Test detection and recovery from inconsistent states."""
        df = pd.DataFrame({'A': range(100)})
        
        # Create inconsistent state
        state_manager.update_state('data.df', df)
        state_manager.update_state('data.row_count', 200)  # Incorrect count
        
        # Should detect and fix inconsistency
        state_manager.validate_state_consistency()
        assert state_manager.get_state('data.row_count') == 100

class TestStatePersistence:
    """Test state persistence functionality."""
    
    def test_save_load_state(self, state_manager):
        """Test saving and loading complete state."""
        # Create some state
        state_manager.update_state('test.value', 42)
        state_manager.update_state('test.list', [1, 2, 3])
        
        # Save state
        saved_state = state_manager.save_state()
        
        # Clear state
        state_manager.clear_state()
        
        # Load state
        state_manager.load_state(saved_state)
        
        assert state_manager.get_state('test.value') == 42
        assert state_manager.get_state('test.list') == [1, 2, 3]
    
    def test_partial_state_restore(self, state_manager):
        """Test restoring partial state."""
        # Set up initial state
        state_manager.update_state('keep.value', 1)
        state_manager.update_state('restore.value', 2)
        
        # Save partial state
        saved_state = state_manager.save_state(['restore'])
        print("\nSaved State:", saved_state)  # Debug output
        
        # Modify all state
        state_manager.update_state('keep.value', 3)
        state_manager.update_state('restore.value', 4)
        
        # Restore partial state
        state_manager.load_state(saved_state, partial=True)
        print("\nState after restore:", state_manager._state)  # Debug output
        
        assert state_manager.get_state('keep.value') == 3  # Should not be restored
        assert state_manager.get_state('restore.value') == 2  # Should be restored
    
    def test_state_migration(self, state_manager):
        """Test handling state format changes."""
        # Create old format state
        old_state = {
            'version': 1,
            'data': {
                'old_key': 'value'
            }
        }
        
        # Register migration
        def migrate_v1_to_v2(state):
            state['data']['new_key'] = state['data'].pop('old_key')
            state['version'] = 2
            return state
        
        state_manager.register_migration(1, migrate_v1_to_v2)
        
        # Load and migrate state
        state_manager.load_state(old_state)
        
        assert state_manager.get_state('data.new_key') == 'value'
        assert 'old_key' not in state_manager.get_state('data')
