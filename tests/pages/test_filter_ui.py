"""
Test script for filter UI functionality using Streamlit test client.
"""
import pytest
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from core.filter_manager import FilterManager
from utils.data_types import DataType
from test_data_generator import generate_test_data

def test_filter_ui():
    """Test filter UI functionality."""
    # Generate test data
    df, expected_stats = generate_test_data()
    
    # Create filter manager
    filter_manager = FilterManager()
    
    # Create column types
    column_types = {
        'category': DataType.CATEGORICAL,
        'number': DataType.NUMERICAL,
        'date': DataType.DATE,
        'text': DataType.TEXT,
        'flag': DataType.BOOLEAN
    }
    
    # Initialize filters
    filter_manager.create_filters(df, column_types)
    
    # Test categorical filter
    filter_manager.active_filters['category'] = True
    filter_manager.filters['category']['selected_values'] = ['A']
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 20  # Category A has 20 records
    
    # Clear filters
    filter_manager.clear_all_filters()
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == len(df)  # All records returned
    
    # Test numerical filter
    filter_manager.active_filters['number'] = True
    filter_manager.filters['number']['filter_type'] = 'range'
    filter_manager.filters['number']['selected_min'] = 25
    filter_manager.filters['number']['selected_max'] = 75
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 51  # Numbers 25-75 (51 records)
    
    # Test combined filters
    filter_manager.active_filters['category'] = True
    filter_manager.filters['category']['selected_values'] = ['A', 'B']
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 16  # A & B categories between 25-75
    
    # Test clearing filters
    filter_manager.clear_all_filters()
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == len(df)  # All records returned
    
    # Verify filter state is properly reset
    assert not any(filter_manager.active_filters.values())
    for column, filter_config in filter_manager.filters.items():
        if filter_config['type'] == 'categorical':
            assert len(filter_config['selected_values']) == len(filter_config['unique_values'])
        elif filter_config['type'] == 'numerical':
            assert filter_config['selected_min'] == filter_config['min_value']
            assert filter_config['selected_max'] == filter_config['max_value']

def test_filter_ui_with_streamlit(monkeypatch):
    """Test filter UI functionality with Streamlit test client."""
    # Mock Streamlit session state
    session_state = {}
    def mock_session_state():
        return session_state
    monkeypatch.setattr(st, "session_state", mock_session_state())
    
    # Create test data
    df, _ = generate_test_data()
    
    # Create filter manager
    filter_manager = FilterManager()
    
    # Create column types
    column_types = {
        'category': DataType.CATEGORICAL,
        'number': DataType.NUMERICAL,
        'date': DataType.DATE,
        'text': DataType.TEXT,
        'flag': DataType.BOOLEAN
    }
    
    # Initialize filters
    filter_manager.create_filters(df, column_types)
    
    # Mock Streamlit UI elements
    def mock_checkbox(label, value=False, key=None):
        if key:
            session_state[key] = value
        return value
    monkeypatch.setattr(st, "checkbox", mock_checkbox)
    
    def mock_radio(label, options, index=0, key=None):
        if key:
            session_state[key] = options[index]
        return options[index]
    monkeypatch.setattr(st, "radio", mock_radio)
    
    def mock_multiselect(label, options, default=None, key=None):
        if key:
            session_state[key] = default or []
        return default or []
    monkeypatch.setattr(st, "multiselect", mock_multiselect)
    
    # Test categorical filter
    filter_manager.active_filters['category'] = True
    filter_manager.filters['category']['selected_values'] = ['A']
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 20  # Category A has 20 records
    
    # Clear filters
    filter_manager.clear_all_filters()
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == len(df)  # All records returned
    
    # Test numerical filter
    filter_manager.active_filters['number'] = True
    filter_manager.filters['number']['filter_type'] = 'range'
    filter_manager.filters['number']['selected_min'] = 25
    filter_manager.filters['number']['selected_max'] = 75
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 51  # Numbers 25-75 (51 records)
    
    # Test combined filters
    filter_manager.active_filters['category'] = True
    filter_manager.filters['category']['selected_values'] = ['A', 'B']
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == 16  # A & B categories between 25-75
    
    # Test clearing filters
    filter_manager.clear_all_filters()
    filtered_df = filter_manager.apply_filters(df)
    assert len(filtered_df) == len(df)  # All records returned
