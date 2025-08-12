"""
Comprehensive test suite for filter functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from core.filter_manager import FilterManager
from utils.data_types import DataType, detect_data_type
from test_data_generator import generate_test_data, get_expected_filter_results

class TestFilterFunctionality:
    @pytest.fixture(scope="class")
    def test_data(self):
        """Generate test data and expected statistics."""
        df, stats = generate_test_data()
        return df, stats
    
    @pytest.fixture(scope="class")
    def filter_manager(self):
        """Create and initialize FilterManager."""
        return FilterManager()
    
    @pytest.fixture(scope="class")
    def expected_results(self):
        """Get expected filter results."""
        return get_expected_filter_results()
    
    def test_data_generation(self, test_data):
        """Verify test data matches expected statistics."""
        df, expected_stats = test_data
        
        # Test categorical distribution
        cat_counts = df['category'].value_counts().to_dict()
        assert cat_counts == expected_stats['category']['value_counts']
        
        # Test numerical statistics
        assert df['number'].mean() == expected_stats['number']['mean']
        assert df['number'].median() == expected_stats['number']['median']
        assert df['number'].min() == expected_stats['number']['min']
        assert df['number'].max() == expected_stats['number']['max']
        
        # Test date range
        assert df['date'].min() == expected_stats['date']['min_date']
        assert df['date'].max() == expected_stats['date']['max_date']
        
        # Test boolean distribution
        assert df['flag'].sum() == expected_stats['flag']['true_count']
        assert (~df['flag']).sum() == expected_stats['flag']['false_count']
    
    def test_filter_creation(self, test_data, filter_manager):
        """Test filter creation for each data type."""
        df, _ = test_data
        
        # Create column types dictionary
        column_types = {
            'category': DataType.CATEGORICAL,
            'number': DataType.NUMERICAL,
            'date': DataType.DATE,
            'text': DataType.TEXT,
            'flag': DataType.BOOLEAN
        }
        
        # Create filters
        filter_manager.create_filters(df, column_types)
        
        # Verify filter configurations
        assert 'category' in filter_manager.filters
        assert filter_manager.filters['category']['type'] == 'categorical'
        assert len(filter_manager.filters['category']['unique_values']) == 5
        
        assert 'number' in filter_manager.filters
        assert filter_manager.filters['number']['type'] == 'numerical'
        assert filter_manager.filters['number']['min_value'] == 1.0
        assert filter_manager.filters['number']['max_value'] == 100.0
        
        assert 'date' in filter_manager.filters
        assert filter_manager.filters['date']['type'] == 'date'
        
        assert 'text' in filter_manager.filters
        assert filter_manager.filters['text']['type'] == 'text'
        
        assert 'flag' in filter_manager.filters
        assert filter_manager.filters['flag']['type'] == 'boolean'
    
    def test_categorical_filters(self, test_data, filter_manager, expected_results):
        """Test categorical filter functionality."""
        df, _ = test_data
        
        # Test include single category
        scenario = expected_results['category_filters']['include_single']
        filter_config = {
            'type': 'categorical',
            'selected_values': scenario['filter']['values'],
            'filter_type': scenario['filter']['type']
        }
        filtered_df = filter_manager._apply_categorical_filter(
            df, 'category', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
        
        # Test include multiple categories
        scenario = expected_results['category_filters']['include_multiple']
        filter_config['selected_values'] = scenario['filter']['values']
        filtered_df = filter_manager._apply_categorical_filter(
            df, 'category', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
        
        # Test exclude
        scenario = expected_results['category_filters']['exclude_single']
        filter_config['selected_values'] = scenario['filter']['values']
        filter_config['filter_type'] = scenario['filter']['type']
        filtered_df = filter_manager._apply_categorical_filter(
            df, 'category', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
    
    def test_numerical_filters(self, test_data, filter_manager, expected_results):
        """Test numerical filter functionality."""
        df, _ = test_data
        
        # Test range filter
        scenario = expected_results['numerical_filters']['range']
        filter_config = {
            'type': 'numerical',
            'filter_type': 'range',
            'selected_min': scenario['filter']['min'],
            'selected_max': scenario['filter']['max']
        }
        filtered_df = filter_manager._apply_numerical_filter(
            df, 'number', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
        
        # Test greater than
        scenario = expected_results['numerical_filters']['greater_than']
        filter_config['filter_type'] = 'greater_than'
        filter_config['selected_min'] = scenario['filter']['min']
        filtered_df = filter_manager._apply_numerical_filter(
            df, 'number', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
        
        # Test less than
        scenario = expected_results['numerical_filters']['less_than']
        filter_config['filter_type'] = 'less_than'
        filter_config['selected_max'] = scenario['filter']['max']
        filtered_df = filter_manager._apply_numerical_filter(
            df, 'number', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
    
    def test_date_filters(self, test_data, filter_manager, expected_results):
        """Test date filter functionality."""
        df, _ = test_data
        
        # Test date range
        scenario = expected_results['date_filters']['range']
        filter_config = {
            'type': 'date',
            'filter_type': 'range',
            'selected_min_date': scenario['filter']['start'],
            'selected_max_date': scenario['filter']['end']
        }
        filtered_df = filter_manager._apply_date_filter(
            df, 'date', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
    
    def test_text_filters(self, test_data, filter_manager, expected_results):
        """Test text filter functionality."""
        df, _ = test_data
        
        # Test contains
        scenario = expected_results['text_filters']['contains']
        filter_config = {
            'type': 'text',
            'filter_type': scenario['filter']['type'],
            'search_text': scenario['filter']['pattern'],
            'case_sensitive': False
        }
        filtered_df = filter_manager._apply_text_filter(
            df, 'text', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
        
        # Test starts with
        scenario = expected_results['text_filters']['starts_with']
        filter_config['filter_type'] = scenario['filter']['type']
        filtered_df = filter_manager._apply_text_filter(
            df, 'text', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
    
    def test_boolean_filters(self, test_data, filter_manager, expected_results):
        """Test boolean filter functionality."""
        df, _ = test_data
        
        # Test true only
        scenario = expected_results['boolean_filters']['true_only']
        filter_config = {
            'type': 'boolean',
            'selected_values': scenario['filter']['values']
        }
        filtered_df = filter_manager._apply_boolean_filter(
            df, 'flag', filter_config
        )
        assert len(filtered_df) == scenario['expected_count']
    
    def test_combined_filters(self, test_data, filter_manager, expected_results):
        """Test multiple filters applied together."""
        df, _ = test_data
        
        # Test category and number filters together
        scenario = expected_results['combined_filters']['category_and_number']
        
        # Apply first filter (category)
        filter1 = scenario['filters'][0]
        cat_config = {
            'type': 'categorical',
            'selected_values': filter1['values'],
            'filter_type': filter1['type']
        }
        filtered_df = filter_manager._apply_categorical_filter(
            df, 'category', cat_config
        )
        
        # Apply second filter (number)
        filter2 = scenario['filters'][1]
        num_config = {
            'type': 'numerical',
            'filter_type': 'range',
            'selected_min': filter2['min'],
            'selected_max': filter2['max']
        }
        filtered_df = filter_manager._apply_numerical_filter(
            filtered_df, 'number', num_config
        )
        
        assert len(filtered_df) == scenario['expected_count']
    
    def test_clear_filters(self, test_data, filter_manager):
        """Test clearing all filters."""
        df, _ = test_data
        original_count = len(df)
        
        # Create filters first
        column_types = {'category': DataType.CATEGORICAL}
        filter_manager.create_filters(df, column_types)
        
        # Then modify the filter
        filter_manager.filters['category']['selected_values'] = ['A']
        filter_manager.active_filters['category'] = True
        
        # Apply filters
        filtered_df = filter_manager.apply_filters(df)
        assert len(filtered_df) < original_count
        
        # Clear filters
        filter_manager.clear_all_filters()
        
        # Verify filters are cleared
        cleared_df = filter_manager.apply_filters(df)
        assert len(cleared_df) == original_count
        assert not any(filter_manager.active_filters.values())
        
        # Verify filter configurations are reset
        for column, filter_config in filter_manager.filters.items():
            if filter_config['type'] == 'categorical':
                assert len(filter_config['selected_values']) == len(filter_config['unique_values'])
            elif filter_config['type'] == 'numerical':
                assert filter_config['selected_min'] == filter_config['min_value']
                assert filter_config['selected_max'] == filter_config['max_value']
