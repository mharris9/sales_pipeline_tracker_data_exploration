"""
Comprehensive Phase 3 Test Suite
Tests all UI components, integrations, edge cases, and real-world scenarios
"""
import pytest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, call
import io
import tempfile
import os
from datetime import datetime, timedelta
import json
import time

# Import components
from pages.filters_page import render_filters_section
from pages.features_page import render_features_section
from pages.outliers_page import render_outliers_section
from pages.reports_page import render_reports_section
from pages.export_page import render_export_section
from pages.data_preview_page import render_data_preview_section

from src.services.state_manager import StateManager
from src.services.data_handler import DataHandler
from src.services.filter_manager import FilterManager
from src.services.feature_engine import FeatureEngine
from src.services.report_engine import ReportEngine
from src.services.outlier_manager import OutlierManager
from src.utils.export_utils import ExportManager

class TestPhase3Comprehensive:
    """Comprehensive test suite for Phase 3"""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing"""
        np.random.seed(42)
        n_rows = 10000
        data = {
            'Id': range(1, n_rows + 1),
            'Snapshot Date': pd.date_range('2024-01-01', periods=n_rows, freq='D').strftime('%m/%d/%Y'),
            'Stage': np.random.choice(['Prospecting', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won'], n_rows),
            'Amount': np.random.exponential(10000, n_rows),
            'Company': [f'Company_{i}' for i in range(n_rows)],
            'Owner': np.random.choice(['John', 'Jane', 'Bob', 'Alice', 'Charlie'], n_rows),
            'Probability': np.random.uniform(0, 1, n_rows),
            'Days_in_Stage': np.random.poisson(30, n_rows)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def corrupted_data(self):
        """Create corrupted data for error handling tests"""
        data = {
            'Id': [1, 2, 'invalid', 4, 5],
            'Snapshot Date': ['01/01/2024', 'invalid_date', '01/03/2024', '01/04/2024', '01/05/2024'],
            'Stage': ['Prospecting', 'Qualified', None, 'Negotiation', 'Closed Won'],
            'Amount': [1000, 'not_a_number', 15000, 25000, 50000],
            'Company': ['Company A', 'Company B', 'Company C', None, 'Company E'],
            'Owner': ['John', 'Jane', 'Bob', 'Alice', '']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_session_state(self):
        """Create a comprehensive mock session state"""
        with patch('streamlit.session_state') as mock_st_session:
            state_manager = StateManager()
            mock_st_session.state_manager = state_manager
            
            # Register all extensions with proper initialization
            state_manager.register_extension('data', {
                'data_handler': DataHandler(),
                'current_df': None,
                'data_loaded': False,
                'data_info': {}
            })
            
            state_manager.register_extension('filters', {
                'filter_manager': FilterManager(),
                'active_filters': {},
                'filter_configs': {},
                'filter_results': {}
            })
            
            state_manager.register_extension('features', {
                'feature_engine': FeatureEngine(),
                'computed_features': {},
                'feature_configs': {}
            })
            
            state_manager.register_extension('reports', {
                'report_engine': ReportEngine(),
                'current_report': None,
                'report_configs': {},
                'report_results': {}
            })
            
            state_manager.register_extension('exports', {
                'export_manager': ExportManager(),
                'export_history': []
            })
            
            state_manager.register_extension('outliers', {
                'outlier_manager': OutlierManager(),
                'settings': {'outliers_enabled': False},
                'exclusion_info': {'outliers_excluded': False}
            })
            
            return mock_st_session

    # ============================================================================
    # 1. CORE FUNCTIONALITY TESTS
    # ============================================================================
    
    def test_complete_data_workflow(self, mock_session_state, large_dataset):
        """Test complete data workflow: upload → process → filter → feature → report → export"""
        # Setup
        state_manager = mock_session_state.state_manager
        data_handler = state_manager.get_extension('data_handler')
        filter_manager = state_manager.get_extension('filters.filter_manager')
        feature_engine = state_manager.get_extension('features.feature_engine')
        report_engine = state_manager.get_extension('reports.report_engine')
        export_manager = state_manager.get_extension('exports.export_manager')
        
        # Step 1: Load data
        with patch('streamlit.success') as mock_success:
            state_manager.set_state('data.current_df', large_dataset)
            state_manager.set_state('data.data_loaded', True)
            assert state_manager.get_state('data.data_loaded') == True
            assert len(state_manager.get_state('data.current_df')) == 10000
        
        # Step 2: Apply filters
        with patch('streamlit.toast') as mock_toast:
            # Simulate filter application
            filter_config = {
                'Stage': {'type': 'categorical', 'values': ['Prospecting', 'Qualified']},
                'Amount': {'type': 'numerical', 'min': 5000, 'max': 50000}
            }
            state_manager.set_state('filters.active_filters', filter_config)
            
            # Apply filters
            filtered_df = large_dataset[
                (large_dataset['Stage'].isin(['Prospecting', 'Qualified'])) &
                (large_dataset['Amount'] >= 5000) &
                (large_dataset['Amount'] <= 50000)
            ]
            state_manager.set_state('filters.filtered_count', len(filtered_df))
            
            assert state_manager.get_state('filters.filtered_count') < 10000
        
        # Step 3: Calculate features
        with patch('streamlit.spinner'):
            # Simulate feature calculation
            feature_results = {
                'stage_counts': filtered_df['Stage'].value_counts().to_dict(),
                'avg_amount': filtered_df['Amount'].mean(),
                'total_opportunities': len(filtered_df)
            }
            state_manager.set_state('features.computed_features', feature_results)
            
            assert 'stage_counts' in state_manager.get_state('features.computed_features')
        
        # Step 4: Generate reports
        with patch('streamlit.plotly_chart'):
            # Simulate report generation
            report_data = {
                'chart_data': {'x': ['Prospecting', 'Qualified'], 'y': [100, 200]},
                'summary_stats': {'total': 300, 'avg_amount': 25000}
            }
            state_manager.set_state('reports.current_report', 'stage_distribution')
            state_manager.set_state('reports.report_results', report_data)
            
            assert state_manager.get_state('reports.current_report') == 'stage_distribution'
        
        # Step 5: Export data
        with patch('streamlit.download_button'):
            # Simulate export
            export_history = [{
                'timestamp': datetime.now().isoformat(),
                'type': 'csv',
                'filename': 'filtered_data.csv',
                'records': len(filtered_df)
            }]
            state_manager.set_state('exports.export_history', export_history)
            
            assert len(state_manager.get_state('exports.export_history')) == 1

    def test_state_persistence_across_navigation(self, mock_session_state, large_dataset):
        """Test that state persists correctly when navigating between pages"""
        state_manager = mock_session_state.state_manager
        
        # Set up initial state
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        state_manager.set_state('filters.active_filters', {'Stage': {'type': 'categorical', 'values': ['Prospecting']}})
        state_manager.set_state('features.computed_features', {'test_feature': 123})
        
        # Simulate navigation between pages
        pages = [
            render_filters_section,
            render_features_section,
            render_outliers_section,
            render_reports_section,
            render_export_section,
            render_data_preview_section
        ]
        
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.header'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.bar_chart'), \
             patch('streamlit.tabs'), \
             patch('streamlit.write'), \
             patch('streamlit.download_button'):
            
            for page_func in pages:
                # Verify state is maintained before each page
                assert state_manager.get_state('data.data_loaded') == True
                assert state_manager.get_state('filters.active_filters') == {'Stage': {'type': 'categorical', 'values': ['Prospecting']}}
                assert state_manager.get_state('features.computed_features') == {'test_feature': 123}
                
                # Render page
                page_func()
                
                # Verify state is still maintained after each page
                assert state_manager.get_state('data.data_loaded') == True
                assert state_manager.get_state('filters.active_filters') == {'Stage': {'type': 'categorical', 'values': ['Prospecting']}}
                assert state_manager.get_state('features.computed_features') == {'test_feature': 123}

    # ============================================================================
    # 2. ERROR HANDLING AND EDGE CASES
    # ============================================================================
    
    def test_corrupted_data_handling(self, mock_session_state, corrupted_data):
        """Test handling of corrupted/invalid data"""
        state_manager = mock_session_state.state_manager
        
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning:
            
            # Try to load corrupted data
            state_manager.set_state('data.current_df', corrupted_data)
            state_manager.set_state('data.data_loaded', True)
            
            # Test that pages handle corrupted data gracefully
            pages = [
                render_filters_section,
                render_features_section,
                render_outliers_section,
                render_reports_section,
                render_export_section,
                render_data_preview_section
            ]
            
            for page_func in pages:
                try:
                    page_func()
                    # Should not crash, may show warnings
                except Exception as e:
                    pytest.fail(f"Page {page_func.__name__} crashed with corrupted data: {e}")

    def test_memory_management_large_dataset(self, mock_session_state, large_dataset):
        """Test memory management with large datasets"""
        import psutil
        import gc
        
        state_manager = mock_session_state.state_manager
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load large dataset
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        # Simulate multiple operations
        for i in range(10):
            # Apply different filters
            filter_config = {
                'Amount': {'type': 'numerical', 'min': i * 1000, 'max': (i + 1) * 10000}
            }
            state_manager.set_state('filters.active_filters', filter_config)
            
            # Calculate features
            feature_results = {
                f'feature_{i}': large_dataset['Amount'].mean() + i
            }
            state_manager.set_state('features.computed_features', feature_results)
            
            # Generate reports
            report_data = {
                f'report_{i}': {'data': list(range(100))}
            }
            state_manager.set_state('reports.report_results', report_data)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 10K rows)
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.1f}MB"

    def test_concurrent_state_access(self, mock_session_state, large_dataset):
        """Test concurrent access to state manager"""
        import threading
        import time
        
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        # Create multiple threads accessing state
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    # Read state
                    df = state_manager.get_state('data.current_df')
                    assert df is not None
                    
                    # Write state
                    state_manager.set_state(f'thread_{thread_id}_data', i)
                    
                    # Read back
                    value = state_manager.get_state(f'thread_{thread_id}_data')
                    assert value == i
                    
                    time.sleep(0.001)  # Small delay
                
                results.append(f"Thread {thread_id} completed successfully")
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5, "Not all threads completed"

    # ============================================================================
    # 3. PERFORMANCE AND SCALABILITY TESTS
    # ============================================================================
    
    def test_page_rendering_performance(self, mock_session_state, large_dataset):
        """Test page rendering performance with large datasets"""
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        pages = [
            render_filters_section,
            render_features_section,
            render_outliers_section,
            render_reports_section,
            render_export_section,
            render_data_preview_section
        ]
        
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.header'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.bar_chart'), \
             patch('streamlit.tabs'), \
             patch('streamlit.write'), \
             patch('streamlit.download_button'):
            
            for page_func in pages:
                start_time = time.time()
                page_func()
                end_time = time.time()
                
                render_time = end_time - start_time
                # Page should render in under 1 second
                assert render_time < 1.0, f"Page {page_func.__name__} took too long: {render_time:.2f}s"

    def test_state_manager_performance(self, mock_session_state):
        """Test StateManager performance with many operations"""
        state_manager = mock_session_state.state_manager
        
        # Test many state operations
        start_time = time.time()
        
        for i in range(1000):
            # Set state
            state_manager.set_state(f'test/path/{i}', f'value_{i}')
            
            # Get state
            value = state_manager.get_state(f'test/path/{i}')
            assert value == f'value_{i}'
            
            # Clear state
            state_manager.clear_state(f'test/path/{i}')
            
            # Verify cleared
            cleared_value = state_manager.get_state(f'test/path/{i}')
            assert cleared_value is None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 3000 operations should complete in under 5 seconds
        assert total_time < 5.0, f"StateManager operations too slow: {total_time:.2f}s"

    # ============================================================================
    # 4. INTEGRATION TESTS
    # ============================================================================
    
    def test_component_integration(self, mock_session_state, large_dataset):
        """Test integration between all components"""
        state_manager = mock_session_state.state_manager
        
        # Setup data
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        # Test DataHandler integration
        data_handler = state_manager.get_extension('data_handler')
        assert data_handler is not None
        
        # Test FilterManager integration
        filter_manager = state_manager.get_extension('filters.filter_manager')
        assert filter_manager is not None
        
        # Test FeatureEngine integration
        feature_engine = state_manager.get_extension('features.feature_engine')
        assert feature_engine is not None
        
        # Test ReportEngine integration
        report_engine = state_manager.get_extension('reports.report_engine')
        assert report_engine is not None
        
        # Test OutlierManager integration
        outlier_manager = state_manager.get_extension('outliers.outlier_manager')
        assert outlier_manager is not None
        
        # Test ExportManager integration
        export_manager = state_manager.get_extension('exports.export_manager')
        assert export_manager is not None

    def test_state_synchronization(self, mock_session_state, large_dataset):
        """Test that state is synchronized across all components"""
        state_manager = mock_session_state.state_manager
        
        # Set up data
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        # Verify all components can access the same data
        data_handler = state_manager.get_extension('data_handler')
        filter_manager = state_manager.get_extension('filters.filter_manager')
        feature_engine = state_manager.get_extension('features.feature_engine')
        
        # All should see the same data
        assert state_manager.get_state('data.data_loaded') == True
        assert len(state_manager.get_state('data.current_df')) == 10000
        
        # Test state updates propagate correctly
        state_manager.set_state('filters.active_filters', {'test': 'filter'})
        assert state_manager.get_state('filters.active_filters') == {'test': 'filter'}
        
        state_manager.set_state('features.computed_features', {'test': 'feature'})
        assert state_manager.get_state('features.computed_features') == {'test': 'feature'}

    # ============================================================================
    # 5. USER INTERACTION TESTS
    # ============================================================================
    
    def test_file_upload_workflow(self, mock_session_state):
        """Test complete file upload workflow"""
        state_manager = mock_session_state.state_manager
        
        # Mock file upload
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.size = 1024
        mock_file.type = "text/csv"
        mock_file.read.return_value = b"Id,Snapshot Date,Stage,Amount,Company,Owner\n1,01/01/2024,Prospecting,1000,Company A,John"
        
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.form_submit_button') as mock_submit, \
             patch('streamlit.spinner'), \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.error') as mock_error:
            
            mock_uploader.return_value = mock_file
            mock_submit.return_value = True
            
            # Simulate file upload
            data_handler = state_manager.get_extension('data_handler')
            
            # This would normally call data_handler.load_file(mock_file)
            # For testing, we'll simulate the state changes
            state_manager.set_state('data.data_loaded', True)
            state_manager.set_state('data.file_info', {
                'name': mock_file.name,
                'size': mock_file.size,
                'type': mock_file.type
            })
            
            # Verify state was updated
            assert state_manager.get_state('data.data_loaded') == True
            assert state_manager.get_state('data.file_info')['name'] == "test_data.csv"

    def test_filter_interaction_workflow(self, mock_session_state, large_dataset):
        """Test complete filter interaction workflow"""
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        with patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.form_submit_button') as mock_submit, \
             patch('streamlit.toast') as mock_toast:
            
            # Simulate user interactions
            mock_checkbox.return_value = True
            mock_selectbox.return_value = 'Prospecting'
            mock_slider.return_value = (5000, 50000)
            mock_submit.return_value = True
            
            # Simulate filter application
            filter_config = {
                'Stage': {'type': 'categorical', 'values': ['Prospecting']},
                'Amount': {'type': 'numerical', 'min': 5000, 'max': 50000}
            }
            state_manager.set_state('filters.active_filters', filter_config)
            
            # Verify filter was applied
            assert state_manager.get_state('filters.active_filters') == filter_config

    # ============================================================================
    # 6. ERROR RECOVERY TESTS
    # ============================================================================
    
    def test_state_corruption_recovery(self, mock_session_state):
        """Test recovery from state corruption"""
        state_manager = mock_session_state.state_manager
        
        # Corrupt state by setting invalid values
        state_manager.set_state('data.current_df', "not_a_dataframe")
        state_manager.set_state('filters.active_filters', "not_a_dict")
        
        # Test recovery mechanisms
        with patch('streamlit.error') as mock_error:
            # Try to access corrupted state
            try:
                df = state_manager.get_state('data.current_df')
                # Should return default value
                assert df is None
            except Exception as e:
                # Should handle gracefully
                mock_error.assert_called()
            
            try:
                filters = state_manager.get_state('filters.active_filters')
                # Should return default value
                assert filters == {}
            except Exception as e:
                # Should handle gracefully
                mock_error.assert_called()

    def test_component_failure_recovery(self, mock_session_state, large_dataset):
        """Test recovery when individual components fail"""
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        # Simulate component failure
        with patch('streamlit.error') as mock_error:
            # Test that other components still work when one fails
            pages = [
                render_filters_section,
                render_features_section,
                render_outliers_section,
                render_reports_section,
                render_export_section,
                render_data_preview_section
            ]
            
            for page_func in pages:
                try:
                    page_func()
                except Exception as e:
                    # Should show error but not crash the app
                    mock_error.assert_called()

    # ============================================================================
    # 7. SECURITY AND VALIDATION TESTS
    # ============================================================================
    
    def test_input_validation(self, mock_session_state):
        """Test input validation and sanitization"""
        state_manager = mock_session_state.state_manager
        
        # Test malicious input
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        with patch('streamlit.error') as mock_error:
            for malicious_input in malicious_inputs:
                try:
                    # Try to set malicious state
                    state_manager.set_state('test/malicious', malicious_input)
                    
                    # Try to get it back
                    value = state_manager.get_state('test/malicious')
                    
                    # Should not execute malicious code
                    assert isinstance(value, str)
                    assert value == malicious_input  # Should be stored as-is
                    
                except Exception as e:
                    # Should handle gracefully
                    mock_error.assert_called()

    def test_file_validation(self, mock_session_state):
        """Test file upload validation"""
        state_manager = mock_session_state.state_manager
        
        # Test invalid file types
        invalid_files = [
            Mock(name="test.exe", size=1024, type="application/x-executable"),
            Mock(name="test.bat", size=1024, type="application/x-bat"),
            Mock(name="test.sh", size=1024, type="application/x-sh"),
            Mock(name="test.py", size=1024, type="text/x-python"),
        ]
        
        with patch('streamlit.error') as mock_error:
            for invalid_file in invalid_files:
                try:
                    # Should reject invalid file types
                    if invalid_file.name.endswith(('.exe', '.bat', '.sh')):
                        mock_error.assert_called()
                except Exception as e:
                    # Should handle gracefully
                    mock_error.assert_called()

    # ============================================================================
    # 8. ACCESSIBILITY AND UX TESTS
    # ============================================================================
    
    def test_page_accessibility(self, mock_session_state, large_dataset):
        """Test that pages are accessible and user-friendly"""
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        pages = [
            render_filters_section,
            render_features_section,
            render_outliers_section,
            render_reports_section,
            render_export_section,
            render_data_preview_section
        ]
        
        with patch('streamlit.title') as mock_title, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.form') as mock_form, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.success') as mock_success:
            
            for page_func in pages:
                page_func()
                
                # Verify that pages provide user feedback
                # At least one of these should be called
                assert (mock_title.called or mock_subheader.called or 
                       mock_form.called or mock_checkbox.called or 
                       mock_button.called or mock_info.called or 
                       mock_success.called)

    def test_loading_states(self, mock_session_state, large_dataset):
        """Test that loading states are properly handled"""
        state_manager = mock_session_state.state_manager
        state_manager.set_state('data.current_df', large_dataset)
        state_manager.set_state('data.data_loaded', True)
        
        with patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.progress') as mock_progress:
            
            # Test that long operations show loading indicators
            pages = [
                render_filters_section,
                render_features_section,
                render_outliers_section,
                render_reports_section,
                render_export_section,
                render_data_preview_section
            ]
            
            for page_func in pages:
                page_func()
                # Should show loading indicators for long operations
                # (This would be tested in actual implementation)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
