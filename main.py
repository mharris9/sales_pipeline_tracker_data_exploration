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

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    # Keep only simple values in session_state to avoid pickle issues
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = None
    if 'filter_manager' not in st.session_state:
        st.session_state.filter_manager = None
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = None
    if 'report_engine' not in st.session_state:
        st.session_state.report_engine = None
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = None
    if 'outlier_manager' not in st.session_state:
        st.session_state.outlier_manager = None

    # Initialize data state
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_info' not in st.session_state:
        st.session_state.data_info = {}
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
    if 'column_info' not in st.session_state:
        st.session_state.column_info = {}

    # Filter state
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None

    # Form state for better drag and drop handling
    if 'file_upload_key' not in st.session_state:
        st.session_state.file_upload_key = 0

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
        /* Sticky marker makes the NEXT container sticky */
        .sticky-controls + div {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: white;
            padding: 0.5rem 0 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        /* Sticky header for metrics */
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 1500;
            background: white;
            padding: 0.75rem 0;
            border-bottom: 1px solid #eee;
            margin-bottom: 0.5rem;
        }
        /* Compact filters container */
        .filters-compact {
            max-height: 320px;
            overflow-y: auto;
            padding-right: 8px;
        }
        .filters-compact [data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.25rem;
        }
        .filters-compact [data-testid="stMultiSelect"],
        .filters-compact [data-testid="stDateInput"],
        .filters-compact [data-testid="stNumberInput"],
        .filters-compact [data-testid="stCheckbox"] {
            margin-bottom: 0.25rem;
        }
        .filters-compact label, .filters-compact p {
            margin-bottom: 0.1rem;
        }
        </style>
        """, unsafe_allow_html=True)

def render_header():
    """Render the application header with metrics"""
    st.title("Data Explorer")
    
    if st.session_state.data_loaded and st.session_state.current_df is not None:
        df = st.session_state.current_df
        total_records = len(df)
        filtered_records = len(st.session_state.filtered_df) if st.session_state.filtered_df is not None else total_records
        
        # Always-visible selection summary panel (sticky)
        st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("Selected Records", filtered_records)
        with col3:
            memory_usage = f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
            st.metric("Memory Usage", memory_usage)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload data to see metrics and begin analysis.")

def render_data_upload():
    """Render the data upload section with form validation"""
    st.header("Data Upload")

    # Add debugging information for drag and drop issues
    if st.checkbox("Debug Mode", key="debug_mode"):
        st.write("**Debug Info:**")
        st.write(f"File upload key: {st.session_state.file_upload_key}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")

    with st.form("data_upload_form"):
        # Use a dynamic key to force re-render on drag and drop
        upload_key = f"file_uploader_{st.session_state.file_upload_key}"
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv", "xlsx", "xls"],
            key=upload_key,
            help="Upload your sales pipeline data file (CSV or Excel). You can drag and drop files here."
        )

        form_valid = True
        validation_errors = []

        # Retrieve current uploaded file from session state to ensure availability within form submit
        current_file = st.session_state.get(upload_key)

        # Add file information display for debugging
        if current_file is not None:
            st.write(f"**File detected:** {current_file.name}")
            # UploadedFile has no .size attribute directly in some versions; compute from bytes
            try:
                current_bytes = current_file.getvalue()
                current_size = len(current_bytes)
            except Exception:
                # Fallback to attribute if available
                current_size = getattr(current_file, "size", 0)
                current_bytes = None
            st.write(f"**File size:** {current_size} bytes")
            st.write(f"**File type:** {current_file.type if hasattr(current_file, 'type') else 'unknown'}")
            
            file_size_mb = current_size / (1024 * 1024) if current_size else 0
            if file_size_mb > MAX_FILE_SIZE_MB:
                validation_errors.append(f"File size ({file_size_mb:.1f} MB) exceeds {MAX_FILE_SIZE_MB} MB limit")
                form_valid = False

            file_extension = current_file.name.lower().split('.')[-1]
            if file_extension not in ['csv', 'xlsx', 'xls']:
                validation_errors.append(f"Unsupported file type: {file_extension}")
                form_valid = False

        if validation_errors:
            for error in validation_errors:
                st.error(error)

        # Add a button to reset the form if needed
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button(
                "Load Data",
                disabled=False,
                help="Click to load the uploaded file"
            )
        with col2:
            if st.form_submit_button("Reset Form", help="Reset the upload form"):
                st.session_state.file_upload_key += 1
                st.rerun()

        if submit_button:
            # Re-fetch after submit; forms commit widget values on submit
            current_file = st.session_state.get(upload_key)
            if current_file is None:
                st.error("No file found. Please select a file and try again.")
                return
            if not form_valid:
                st.error("Fix validation errors above and try again.")
                return
            with st.spinner("Loading data..."):
                try:
                    # Use a local DataHandler for processing (avoid storing objects in session_state)
                    from src.services.data_handler import DataHandler
                    handler = DataHandler()
                    success = handler.load_file(current_file)
                    
                    if success:
                        # Update session state with loaded data
                        st.session_state.current_df = handler.get_current_df()
                        st.session_state.data_loaded = handler.is_data_loaded()
                        st.session_state.data_info = handler.get_file_info()
                        # Convert enum dict to plain strings for session_state
                        ct = handler.get_column_types()
                        st.session_state.column_types = {k: v.value if hasattr(v, 'value') else str(v) for k, v in ct.items()}
                        st.session_state.column_info = handler.get_column_info()
                        st.session_state.filtered_df = st.session_state.current_df.copy()
                        
                        st.success(f"✅ Successfully loaded {len(st.session_state.current_df)} rows and {len(st.session_state.current_df.columns)} columns")
                    else:
                        st.error("Failed to load data. Please check the file format and required columns.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    logger.error(f"File loading error: {str(e)}")

def main():
    """Main application function."""
    initialize_session_state()
    render_header()

    # Render sidebar with data upload
    with st.sidebar:
        render_data_upload()
        # Data info (if loaded)
        if st.session_state.data_loaded:
            st.markdown("---")
            st.header("📊 Data Info")
            st.write(f"**File:** {st.session_state.data_info.get('name', 'N/A')}")
            st.write(f"**Size:** {st.session_state.data_info.get('size', 0) / 1024**2:.1f} MB")

            type_counts = {}
            for dtype in st.session_state.column_types.values():
                dtype_key = dtype if isinstance(dtype, str) else getattr(dtype, 'value', str(dtype))
                type_counts[dtype_key] = type_counts.get(dtype_key, 0) + 1

            st.write("**Column Types:**")
            for dtype, count in type_counts.items():
                st.write(f"- {dtype.title()}: {count}")

    if st.session_state.data_loaded:
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
        ## Welcome to the Data Explorer! 🚀

        This application helps you analyze and explore any dataset with powerful filtering,
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

        ### How It Works:
        The application automatically:
        - **Detects column types** (text, numbers, dates, categories)
        - **Identifies data patterns** (ID columns, date columns, etc.)
        - **Creates appropriate filters** based on data types
        - **Generates relevant visualizations** for your data

        ### Tips for Best Results:
        - Include a mix of different data types (text, numbers, dates)
        - Use descriptive column names
        - Clean your data before uploading for better analysis

        Upload your file to get started! 📊
        """)

if __name__ == "__main__":
    main()
