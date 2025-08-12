"""
Sales Pipeline Tracker - Data Exploration Tool
Main application entry point
"""
import streamlit as st
import time
from pathlib import Path

# Import services
from src.services.state_manager import StateManager
from src.services.data_handler import DataHandler
from src.services.filter_manager import FilterManager
from src.services.feature_engine import FeatureEngine
from src.services.outlier_manager import OutlierManager
from src.services.report_engine import ReportEngine

# Import utilities
from src.utils.data_types import DataType
from src.utils.column_mapping import column_mapper

def load_custom_css():
    """Load custom CSS styling"""
    css_file = Path("src/assets/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize the application state using StateManager"""
    if not hasattr(st.session_state, 'state_manager'):
        st.session_state.state_manager = StateManager()
        
        # Register core components as extensions
        st.session_state.state_manager.register_extension('data_handler', DataHandler())
        st.session_state.state_manager.register_extension('filter_manager', FilterManager())
        st.session_state.state_manager.register_extension('feature_engine', FeatureEngine())
        st.session_state.state_manager.register_extension('outlier_manager', OutlierManager())
        st.session_state.state_manager.register_extension('report_engine', ReportEngine())

def render_header():
    """Render the application header with metrics"""
    st.title("Sales Pipeline Data Explorer")
    
    # Get metrics from state
    total_records = st.session_state.state_manager.get_state('data.total_records', 0)
    filtered_records = st.session_state.state_manager.get_state('filters.filtered_count', 0)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", total_records)
    with col2:
        delta = filtered_records - total_records if total_records > 0 else 0
        st.metric("Filtered Records", filtered_records, delta=delta)
    with col3:
        memory_usage = st.session_state.state_manager.get_state('data.memory_usage', '0 MB')
        st.metric("Memory Usage", memory_usage)

def render_data_upload():
    """Render the data upload section"""
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        key="file_uploader"
    )
    
    if uploaded_file:
        data_handler = st.session_state.state_manager.get_extension('data_handler')
        with st.spinner("Loading data..."):
            if data_handler.load_file(uploaded_file):
                st.success("Data loaded successfully!")
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()
            else:
                st.error("Failed to load data. Please check the file format.")

def main():
    """Main application entry point"""
    # Initialize state and load styling
    initialize_session_state()
    load_custom_css()
    
    # Render header
    render_header()
    
    # Main content
    render_data_upload()
    
    # Display data preview if data is loaded
    if st.session_state.state_manager.get_state('data.data_loaded', False):
        st.header("Data Preview")
        data_handler = st.session_state.state_manager.get_extension('data_handler')
        df = data_handler.get_current_df()
        if df is not None:
            st.dataframe(df.head())

if __name__ == "__main__":
    main()
