"""
Phase 3 UI Components Test Suite
Tests all UI components and their integration with StateManager
"""
import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import io
from datetime import datetime, timedelta

# Import the components to test
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

class TestPhase3UIComponents:
    """Test suite for Phase 3 UI components"""
    
    @pytest.fixture
    def mock_session_state(self):
        """Create a mock session state with StateManager"""
        # Create StateManager instance
        state_manager = StateManager()
        
        # Register all extensions with the paths that UI pages expect
        state_manager.register_extension('data', {
            'data_handler': DataHandler(state_manager),
            'current_df': None,
            'data_loaded': False,
            'data_info': {}
        })
        
        state_manager.register_extension('filters', {
            'filter_manager': FilterManager(state_manager),
            'active_filters': {},
            'filter_configs': {},
            'filter_results': {}
        })
        
        state_manager.register_extension('features', {
            'feature_engine': FeatureEngine(state_manager),
            'computed_features': {},
            'feature_configs': {}
        })
        
        state_manager.register_extension('reports', {
            'report_engine': ReportEngine(state_manager),
            'current_report': None,
            'report_configs': {},
            'report_results': {}
        })
        
        state_manager.register_extension('exports', {
            'export_manager': ExportManager(),
            'export_history': []
        })
        
        state_manager.register_extension('outliers', {
            'outlier_manager': OutlierManager(state_manager),
            'settings': {'outliers_enabled': False},
            'exclusion_info': {'outliers_excluded': False}
        })
        
        # Register individual extensions for direct access
        state_manager.register_extension('data_handler', {'handler': DataHandler(state_manager)})
        state_manager.register_extension('filter_manager', {'manager': FilterManager(state_manager)})
        state_manager.register_extension('feature_engine', {'engine': FeatureEngine(state_manager)})
        state_manager.register_extension('report_engine', {'engine': ReportEngine(state_manager)})
        state_manager.register_extension('export_manager', {'manager': ExportManager()})
        state_manager.register_extension('outlier_manager', {'manager': OutlierManager(state_manager)})
        
        # Set up detector configurations for outlier manager
        detector_configs = {
            'zscore': {
                'name': 'zscore',
                'description': 'Z-Score Outlier Detection',
                'active': False
            },
            'iqr': {
                'name': 'iqr',
                'description': 'IQR Outlier Detection',
                'active': False
            }
        }
        state_manager.set_state('outlier_configs', detector_configs)
        
        # Set up report configurations for report engine
        report_configs = {
            'descriptive_stats': {
                'name': 'descriptive_stats',
                'description': 'Descriptive Statistics',
                'active': False
            },
            'correlation': {
                'name': 'correlation',
                'description': 'Correlation Analysis',
                'active': False
            }
        }
        state_manager.set_state('report_configs', report_configs)
        
        # Create a mock session object
        mock_session = Mock()
        mock_session.state_manager = state_manager
        
        # Mock st.session_state to return our mock_session
        with patch('streamlit.session_state', mock_session):
            yield mock_session
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        data = {
            'Id': [1, 2, 3, 4, 5],
            'Snapshot Date': ['01/01/2024', '01/02/2024', '01/03/2024', '01/04/2024', '01/05/2024'],
            'Stage': ['Prospecting', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won'],
            'Amount': [1000, 5000, 15000, 25000, 50000],
            'Company': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
            'Owner': ['John', 'Jane', 'Bob', 'Alice', 'Charlie']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_uploaded_file(self):
        """Create a mock uploaded file"""
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.size = 1024  # 1KB
        mock_file.type = "text/csv"
        mock_file.read.return_value = b"Id,Snapshot Date,Stage,Amount,Company,Owner\n1,01/01/2024,Prospecting,1000,Company A,John"
        return mock_file

    def test_filters_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that filters page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock Streamlit components
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):
            
            # Test that the function runs without errors
            try:
                render_filters_section()
                assert True  # If we get here, no exceptions were raised
            except Exception as e:
                pytest.fail(f"Filters page rendering failed: {e}")

    def test_features_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that features page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock Streamlit components
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):
            
            try:
                render_features_section()
                assert True
            except Exception as e:
                pytest.fail(f"Features page rendering failed: {e}")

    def test_outliers_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that outliers page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock Streamlit components
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):
            
            try:
                render_outliers_section()
                assert True
            except Exception as e:
                pytest.fail(f"Outliers page rendering failed: {e}")

    def test_reports_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that reports page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock Streamlit components
        with patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):
            
            try:
                render_reports_section()
                assert True
            except Exception as e:
                pytest.fail(f"Reports page rendering failed: {e}")

    def test_export_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that export page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock Streamlit components
        with patch('streamlit.header'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.download_button'):
            
            try:
                render_export_section()
                assert True
            except Exception as e:
                pytest.fail(f"Export page rendering failed: {e}")

    def test_data_preview_page_rendering(self, mock_session_state, sample_dataframe):
        """Test that data preview page renders correctly"""
        # Set up data in state
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)

        # Mock Streamlit components
        with patch('streamlit.header'), \
             patch('streamlit.subheader'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.bar_chart'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.write'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):

            # Mock st.tabs to return 3 mock tab objects that support context manager
            mock_tab1 = Mock()
            mock_tab1.__enter__ = Mock(return_value=mock_tab1)
            mock_tab1.__exit__ = Mock(return_value=None)
            mock_tab2 = Mock()
            mock_tab2.__enter__ = Mock(return_value=mock_tab2)
            mock_tab2.__exit__ = Mock(return_value=None)
            mock_tab3 = Mock()
            mock_tab3.__enter__ = Mock(return_value=mock_tab3)
            mock_tab3.__exit__ = Mock(return_value=None)
            mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
            
            # Mock st.columns to return 4 mock column objects that support context manager
            mock_cols = []
            for i in range(4):
                mock_col = Mock()
                mock_col.__enter__ = Mock(return_value=mock_col)
                mock_col.__exit__ = Mock(return_value=None)
                mock_cols.append(mock_col)
            mock_columns.return_value = mock_cols

            try:
                render_data_preview_section()
                assert True
            except Exception as e:
                pytest.fail(f"Data preview page rendering failed: {e}")

    def test_no_data_handling(self, mock_session_state):
        """Test that all pages handle no data scenario gracefully"""
        # Ensure no data is loaded
        mock_session_state.state_manager.set_state('data.data_loaded', False)
        
        # Mock Streamlit components
        with patch('streamlit.warning') as mock_warning:
            
            # Test all pages
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
                    # Should show warning for no data
                    mock_warning.assert_called()
                except Exception as e:
                    pytest.fail(f"Page {page_func.__name__} failed to handle no data: {e}")

    def test_state_manager_integration(self, mock_session_state, sample_dataframe):
        """Test that UI components properly integrate with StateManager"""
        # Set up data
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Test that state is properly accessed
        assert mock_session_state.state_manager.get_state('data.data_loaded') == True
        assert mock_session_state.state_manager.get_state('data.current_df') is not None
        
        # Test that extensions are available
        assert mock_session_state.state_manager.get_extension('data_handler') is not None
        assert mock_session_state.state_manager.get_extension('filters.filter_manager') is not None
        assert mock_session_state.state_manager.get_extension('features.feature_engine') is not None

    def test_form_validation(self, mock_session_state, sample_dataframe):
        """Test form validation in UI components"""
        # Set up data
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Mock form components
        with patch('streamlit.form') as mock_form, \
             patch('streamlit.form_submit_button') as mock_submit, \
             patch('streamlit.error') as mock_error:
            
            mock_form.return_value.__enter__ = Mock()
            mock_form.return_value.__exit__ = Mock()
            mock_submit.return_value = True  # Simulate form submission
            
            # Test that forms are created and validation works
            try:
                render_filters_section()
                mock_form.assert_called()
            except Exception as e:
                pytest.fail(f"Form validation test failed: {e}")

    def test_error_handling(self, mock_session_state):
        """Test error handling in UI components"""
        # Mock Streamlit components
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning:

            # Test error handling when state manager is not available
            mock_session_state.state_manager = None
            
            # Mock hasattr to return False for state_manager
            with patch('builtins.hasattr', return_value=False):
                try:
                    render_filters_section()
                    mock_error.assert_called()
                except Exception as e:
                    pytest.fail(f"Error handling test failed: {e}")

    def test_widget_state_binding(self, mock_session_state, sample_dataframe):
        """Test that widgets properly bind to state"""
        # Set up data
        mock_session_state.state_manager.set_state('data.current_df', sample_dataframe)
        mock_session_state.state_manager.set_state('data.data_loaded', True)
        
        # Set up column types with correct enum values
        column_types = {
            'Stage': 'categorical',
            'Amount': 'numerical',
            'Company': 'categorical'
        }
        mock_session_state.state_manager.set_state('data.column_types', column_types)

        # Mock widget components
        with patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.title'), \
             patch('streamlit.subheader'), \
             patch('streamlit.form') as mock_form, \
             patch('streamlit.button'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):

            # Mock form context manager
            mock_form_instance = Mock()
            mock_form_instance.__enter__ = Mock(return_value=mock_form_instance)
            mock_form_instance.__exit__ = Mock(return_value=None)
            mock_form.return_value = mock_form_instance
            
            # Test that widgets are created with proper keys
            try:
                render_filters_section()
                # Verify that widgets are created (they should be called)
                assert mock_checkbox.called or mock_selectbox.called or mock_slider.called 
            except Exception as e:
                pytest.fail(f"Widget state binding test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
