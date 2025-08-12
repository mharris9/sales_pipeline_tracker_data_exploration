"""
Sales Pipeline Data Explorer - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import logging
import time # For anti-flicker

# Configure Streamlit page
from config.settings import APP_TITLE, APP_ICON, LAYOUT, MAX_FILE_SIZE_MB

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Import core modules from new structure
from src.services.data_handler import DataHandler
from src.services.filter_manager import FilterManager
from src.services.feature_engine import FeatureEngine
from src.services.report_engine import ReportEngine
from src.services.outlier_manager import OutlierManager
from src.services.state_manager import StateManager
from src.utils.export_utils import ExportManager
from src.utils.data_types import DataType
from src.utils.column_mapping import column_mapper

# Import page components
try:
    from pages.filters_page import render_filters_section
    from pages.features_page import render_features_section
    from pages.outliers_page import render_outliers_section
    from pages.reports_page import render_reports_section
    from pages.export_page import render_export_section
    from pages.data_preview_page import render_data_preview_section
    PAGES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some pages not available: {e}")
    PAGES_AVAILABLE = False

def load_custom_css():
    """Load custom CSS for styling"""
    try:
        with open('src/assets/style.css', 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Use default styling if custom CSS not found
        st.markdown("""
        <style>
        .main-header {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'state_manager' not in st.session_state:
        state_manager = StateManager()

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

        state_manager.register_validator('data.current_df', lambda df: isinstance(df, (pd.DataFrame, type(None))))
        state_manager.register_validator('data.data_loaded', lambda x: isinstance(x, bool))
        state_manager.register_validator('outliers.settings', lambda x: isinstance(x, dict) and 'outliers_enabled' in x)

        state_manager.register_watcher('data.current_df', lambda old, new: logger.info(f"DataFrame updated: {len(new)} rows"))
        state_manager.register_watcher('filters.active_filters', lambda old, new: logger.info(f"Active filters changed: {new}"))

        st.session_state.state_manager = state_manager

    # For backward compatibility during transition (these will be removed later)
    state_manager = st.session_state.state_manager
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = state_manager.get_extension('data_handler')
    if 'filter_manager' not in st.session_state:
        st.session_state.filter_manager = state_manager.get_extension('filters.filter_manager')
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = state_manager.get_extension('features.feature_engine')
    if 'report_engine' not in st.session_state:
        st.session_state.report_engine = state_manager.get_extension('reports.report_engine')
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = state_manager.get_extension('exports.export_manager')
    if 'outlier_manager' not in st.session_state:
        st.session_state.outlier_manager = state_manager.get_extension('outliers.outlier_manager')
    if 'current_df' not in st.session_state:
        st.session_state.current_df = state_manager.get_state('data.current_df')
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = state_manager.get_state('data.data_loaded')
    if 'outlier_settings' not in st.session_state:
        st.session_state.outlier_settings = state_manager.get_state('outliers.settings')
    if 'exclusion_info' not in st.session_state:
        st.session_state.exclusion_info = state_manager.get_state('outliers.exclusion_info')

def render_header():
    """Render the application header with metrics"""
    st.title("Sales Pipeline Data Explorer")
    
    # Get metrics from state
    data_loaded = st.session_state.state_manager.get_state('data.data_loaded', False)
    
    if data_loaded:
        data_handler = st.session_state.state_manager.get_extension('data_handler')
        df = data_handler.get_current_df()
        
        if df is not None:
            total_records = len(df)
            filtered_records = st.session_state.state_manager.get_state('filters.filtered_count', total_records)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", total_records)
            with col2:
                delta = filtered_records - total_records if total_records > 0 else 0
                st.metric("Filtered Records", filtered_records, delta=delta)
            with col3:
                memory_usage = f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
                st.metric("Memory Usage", memory_usage)
        else:
            st.info("Data loaded but not available for display.")
    else:
        st.info("Upload data to see metrics and begin analysis.")

def render_data_upload():
    """Render the data upload section with form validation"""
    st.header("Data Upload")

    with st.form("data_upload_form"):
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
            help="Upload your sales pipeline data file (CSV or Excel)"
        )

        form_valid = True
        validation_errors = []

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB: # Using MAX_FILE_SIZE_MB from settings
                validation_errors.append(f"File size ({file_size_mb:.1f} MB) exceeds {MAX_FILE_SIZE_MB} MB limit")
                form_valid = False

            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension not in ['csv', 'xlsx', 'xls']:
                validation_errors.append(f"Unsupported file type: {file_extension}")
                form_valid = False

        if validation_errors:
            for error in validation_errors:
                st.error(error)

        submit_button = st.form_submit_button(
            "Load Data",
            disabled=not uploaded_file or not form_valid,
            help="Click to load the uploaded file"
        )

        if submit_button and uploaded_file and form_valid:
            data_handler = st.session_state.state_manager.get_extension('data_handler')
            with st.spinner("Loading data..."):
                if data_handler.load_file(uploaded_file):
                    time.sleep(0.1)
                    st.session_state.state_manager.trigger_rerun()
                else:
                    st.error("Failed to load data. Please check the file format.")

def main():
    """Main application function."""
    initialize_session_state()
    render_header()

    # Render sidebar with data upload
    with st.sidebar:
        render_data_upload()
        # Data info (if loaded)
        if st.session_state.state_manager.get_state('data.data_loaded', False):
            st.markdown("---")
            st.header("📊 Data Info")
            data_handler = st.session_state.state_manager.get_extension('data_handler')
            file_info = data_handler.get_file_info()
            st.write(f"**File:** {file_info.get('name', 'N/A')}")
            st.write(f"**Size:** {file_info.get('size', 0) / 1024**2:.1f} MB")

            column_types = data_handler.get_column_types()
            type_counts = {}
            for dtype in column_types.values():
                type_counts[dtype.value] = type_counts.get(dtype.value, 0) + 1

            st.write("**Column Types:**")
            for dtype, count in type_counts.items():
                st.write(f"- {dtype.title()}: {count}")

    if st.session_state.state_manager.get_state('data.data_loaded', False):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Filters",
            "🎯 Outliers",
            "⚙️ Features",
            "📈 Reports",
            "💾 Export",
            "👀 Data Preview"
        ])

        with tab1:
            if PAGES_AVAILABLE:
                render_filters_section()
            else:
                st.error("Filters page not available")

        with tab2:
            if PAGES_AVAILABLE:
                render_outliers_section()
            else:
                st.error("Outliers page not available")

        with tab3:
            if PAGES_AVAILABLE:
                render_features_section()
            else:
                st.error("Features page not available")

        with tab4:
            if PAGES_AVAILABLE:
                render_reports_section()
            else:
                st.error("Reports page not available")

        with tab5:
            if PAGES_AVAILABLE:
                render_export_section()
            else:
                st.error("Export page not available")

        with tab6:
            if PAGES_AVAILABLE:
                render_data_preview_section()
            else:
                st.error("Data preview page not available")
    else:
        st.markdown("""
        ## Welcome to the Sales Pipeline Data Explorer! 🚀

        This application helps you analyze and explore your sales pipeline data with powerful filtering,
        feature engineering, and visualization capabilities.

        ### Getting Started:
        1. **Upload your data** using the file uploader in the sidebar
        2. **Apply filters** to focus on specific data segments
        3. **Add features** to derive new insights from your data
        4. **Generate reports** and visualizations
        5. **Export** your results for further analysis

        ### Supported File Formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)

        ### Expected Data Structure:
        - **Id**: Unique identifier for each opportunity (can have duplicates for different snapshots)
        - **Snapshot Date**: Date when the snapshot was taken (MM/DD/YYYY format)
        - **Stage**: Current stage of the opportunity
        - Additional columns with categorical, numerical, date, or text data

        Upload your file to get started! 📊
        """)

if __name__ == "__main__":
    main()
