"""
Sales Pipeline Data Explorer - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go

# Configure Streamlit page
from config.settings import APP_TITLE, APP_ICON, LAYOUT

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Import core modules
from core.data_handler import DataHandler
from core.filter_manager import FilterManager
from core.feature_engine import FeatureEngine
from core.report_engine import ReportEngine
from core.outlier_manager import OutlierManager
from utils.export_utils import ExportManager
from utils.data_types import DataType

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    if 'filter_manager' not in st.session_state:
        st.session_state.filter_manager = FilterManager()
    
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = FeatureEngine()
    
    if 'report_engine' not in st.session_state:
        st.session_state.report_engine = ReportEngine()
    
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = ExportManager()
    
    if 'outlier_manager' not in st.session_state:
        st.session_state.outlier_manager = OutlierManager()
    
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'outlier_settings' not in st.session_state:
        st.session_state.outlier_settings = {'outliers_enabled': False}
    
    if 'exclusion_info' not in st.session_state:
        st.session_state.exclusion_info = {'outliers_excluded': False}

def render_header():
    """Render the application header."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")
    
    # Quick stats if data is loaded
    if st.session_state.data_loaded and st.session_state.current_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(st.session_state.current_df):,}")
        
        with col2:
            st.metric("Total Columns", len(st.session_state.current_df.columns))
        
        with col3:
            memory_mb = st.session_state.current_df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        with col4:
            # Count active filters
            active_filters = sum(st.session_state.filter_manager.active_filters.values())
            st.metric("Active Filters", active_filters)

def render_sidebar():
    """Render the sidebar with data loading and configuration options."""
    st.sidebar.header("📁 Data Management")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or XLSX file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales pipeline data file"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Load Data", type="primary"):
            with st.spinner("Loading data..."):
                success = st.session_state.data_handler.load_file(uploaded_file)
                
                if success:
                    # Get processed data
                    df = st.session_state.data_handler.get_data(processed=True)
                    st.session_state.current_df = df
                    st.session_state.data_loaded = True
                    
                    # Initialize filters
                    column_types = st.session_state.data_handler.column_types
                    st.session_state.filter_manager.create_filters(df, column_types)
                    
                    # Show validation results
                    validation = st.session_state.data_handler.validate_sales_pipeline_data()
                    
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.sidebar.warning(warning)
                    
                    if validation['suggestions']:
                        for suggestion in validation['suggestions']:
                            st.sidebar.info(suggestion)
                    
                    st.sidebar.success("Data loaded successfully!")
                    st.rerun()
    
    # Data info (if loaded)
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Data Info")
        
        file_info = st.session_state.data_handler.get_file_info()
        st.sidebar.write(f"**File:** {file_info.get('name', 'N/A')}")
        st.sidebar.write(f"**Size:** {file_info.get('size', 0) / 1024**2:.1f} MB")
        
        # Column type summary
        column_types = st.session_state.data_handler.column_types
        type_counts = {}
        for dtype in column_types.values():
            type_counts[dtype.value] = type_counts.get(dtype.value, 0) + 1
        
        st.sidebar.write("**Column Types:**")
        for dtype, count in type_counts.items():
            st.sidebar.write(f"- {dtype.title()}: {count}")

def render_filters_section():
    """Render the data filtering section."""
    if not st.session_state.data_loaded:
        st.info("Please load data to access filtering options.")
        return
    
    st.header("🔍 Data Filters")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Clear All Filters", help="Remove all active filters"):
            st.session_state.filter_manager.clear_all_filters()
            st.rerun()
    
    # Get column information
    column_types = st.session_state.data_handler.column_types
    
    # Create filter columns
    num_columns = min(3, len(column_types))
    if num_columns > 0:
        filter_cols = st.columns(num_columns)
        
        for i, (column, data_type) in enumerate(column_types.items()):
            with filter_cols[i % num_columns]:
                with st.expander(f"{column} ({data_type.value})", expanded=False):
                    st.session_state.filter_manager.render_filter_ui(column, data_type)
    
    # Apply filters and update current dataframe
    if st.session_state.data_loaded:
        base_df = st.session_state.data_handler.get_data(processed=True)
        filtered_df = st.session_state.filter_manager.apply_filters(base_df)
        
        # Apply outlier exclusion if enabled
        if st.session_state.outlier_settings.get('outliers_enabled', False):
            enabled_columns = st.session_state.outlier_settings.get('enabled_columns', [])
            combination_method = st.session_state.outlier_settings.get('combination_method', 'any')
            
            filtered_df, exclusion_info = st.session_state.outlier_manager.apply_outlier_exclusion(
                filtered_df, enabled_columns, combination_method
            )
            st.session_state.exclusion_info = exclusion_info
        else:
            st.session_state.exclusion_info = {'outliers_excluded': False}
        
        st.session_state.current_df = filtered_df
        
        # Show filter summary
        filter_summary = st.session_state.filter_manager.get_active_filters_summary()
        if filter_summary:
            st.info(f"**Active Filters:** {', '.join([f'{k}: {v}' for k, v in filter_summary.items()])}")
        
        # Show outlier exclusion summary
        if st.session_state.exclusion_info.get('outliers_excluded', False):
            exclusion_info = st.session_state.exclusion_info
            st.warning(f"**Outliers Excluded:** {exclusion_info['excluded_rows']:,} rows "
                      f"({exclusion_info['exclusion_percentage']:.1f}%) removed from analysis")

def render_outliers_section():
    """Render the outlier detection section."""
    if not st.session_state.data_loaded:
        st.info("Please load data to access outlier detection options.")
        return
    
    # Get column information
    column_types = st.session_state.data_handler.column_types
    
    # Render outlier detection UI
    outlier_settings = st.session_state.outlier_manager.render_outlier_ui(
        st.session_state.current_df, 
        column_types
    )
    
    # Update session state
    st.session_state.outlier_settings = outlier_settings

def render_features_section():
    """Render the feature engineering section."""
    if not st.session_state.data_loaded:
        st.info("Please load data to access feature engineering options.")
        return
    
    st.header("⚙️ Feature Engineering")
    
    # Get available features
    df_columns = list(st.session_state.current_df.columns)
    available_features = st.session_state.feature_engine.get_available_features(df_columns)
    
    if not available_features:
        st.warning("No additional features can be calculated with the current data columns.")
        return
    
    # Feature selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_features = st.multiselect(
            "Select features to add:",
            options=list(available_features.keys()),
            default=[],
            format_func=lambda x: f"{x}: {available_features[x]['description']}",
            help="Select which derived features to add to your dataset"
        )
    
    with col2:
        if st.button("Add Features", type="primary", disabled=not selected_features):
            with st.spinner("Calculating features..."):
                try:
                    # Add features to current dataframe
                    df_with_features = st.session_state.feature_engine.add_features(
                        st.session_state.current_df, 
                        selected_features
                    )
                    st.session_state.current_df = df_with_features
                    
                    # Update column types for new features
                    for feature_name in selected_features:
                        if feature_name in df_with_features.columns:
                            feature_info = available_features[feature_name]
                            st.session_state.data_handler.column_types[feature_name] = feature_info['data_type']
                    
                    st.success(f"Added {len(selected_features)} feature(s) successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error adding features: {str(e)}")
    
    # Show feature descriptions
    if available_features:
        with st.expander("📖 Feature Descriptions", expanded=False):
            for name, info in available_features.items():
                st.write(f"**{name}:** {info['description']}")
                if info.get('requirements'):
                    st.caption(f"Requires: {', '.join(info['requirements'])}")

def render_groupby_section():
    """Render the group by functionality section."""
    if not st.session_state.data_loaded:
        return None
    
    st.subheader("👥 Group By Analysis")
    
    # Get categorical columns for grouping
    categorical_columns = st.session_state.data_handler.get_categorical_columns()
    text_columns = st.session_state.data_handler.get_text_columns()
    groupby_columns = categorical_columns + text_columns
    
    if not groupby_columns:
        st.info("No categorical columns available for grouping.")
        return None
    
    selected_groupby = st.selectbox(
        "Group data by:",
        options=[None] + groupby_columns,
        index=0,
        help="Select a column to group the analysis by"
    )
    
    return selected_groupby

def render_reports_section():
    """Render the reports and visualization section."""
    if not st.session_state.data_loaded:
        st.info("Please load data to access reporting options.")
        return
    
    st.header("📈 Reports & Visualizations")
    
    # Group by selection
    group_by_column = render_groupby_section()
    
    # Report type selection
    available_reports = st.session_state.report_engine.get_available_reports()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_report = st.selectbox(
            "Select Report Type:",
            options=list(available_reports.keys()),
            format_func=lambda x: available_reports[x]['name'],
            help="Choose the type of report or visualization to generate"
        )
    
    report_config = available_reports[selected_report]
    config = {'group_by_column': group_by_column}
    
    # Axis selection for reports that require it
    if report_config.get('requires_axes', False):
        st.subheader("📊 Axis Configuration")
        
        axis_cols = st.columns(2)
        column_types = st.session_state.data_handler.column_types
        
        # X-axis selection
        if 'x_axis' in report_config.get('axis_requirements', {}):
            with axis_cols[0]:
                compatible_x = st.session_state.report_engine.get_compatible_columns_for_report(
                    selected_report, 'x_axis', column_types
                )
                
                if compatible_x:
                    config['x_axis'] = st.selectbox(
                        "X-Axis:",
                        options=compatible_x,
                        help="Select the column for X-axis"
                    )
                else:
                    st.warning("No compatible columns found for X-axis")
        
        # Y-axis selection
        if 'y_axis' in report_config.get('axis_requirements', {}):
            with axis_cols[1]:
                compatible_y = st.session_state.report_engine.get_compatible_columns_for_report(
                    selected_report, 'y_axis', column_types
                )
                
                if compatible_y:
                    config['y_axis'] = st.selectbox(
                        "Y-Axis:",
                        options=compatible_y,
                        help="Select the column for Y-axis"
                    )
                else:
                    st.warning("No compatible columns found for Y-axis")
        
        # Additional options
        if selected_report in ['bar_chart', 'line_chart', 'time_series']:
            config['aggregation'] = st.selectbox(
                "Aggregation Method:",
                options=['sum', 'mean', 'count', 'median', 'min', 'max'],
                index=0,
                help="How to aggregate the data"
            )
        
        if selected_report == 'histogram':
            config['bins'] = st.slider(
                "Number of Bins:",
                min_value=10,
                max_value=100,
                value=30,
                help="Number of bins for the histogram"
            )
        
        if selected_report == 'scatter_plot':
            numerical_columns = st.session_state.data_handler.get_numerical_columns()
            if numerical_columns:
                config['size_column'] = st.selectbox(
                    "Size Column (Optional):",
                    options=[None] + numerical_columns,
                    index=0,
                    help="Column to determine point sizes"
                )
        
        if selected_report == 'time_series':
            config['time_period'] = st.selectbox(
                "Time Period:",
                options=['D', 'W', 'M', 'Q', 'Y'],
                format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}[x],
                index=2,  # Default to Monthly
                help="Time period for aggregation"
            )
    
    # Column selection for reports that need it
    if selected_report in ['descriptive_statistics', 'correlation_heatmap']:
        all_columns = list(st.session_state.current_df.columns)
        
        if selected_report == 'correlation_heatmap':
            # Only show numerical columns for correlation
            numerical_columns = st.session_state.data_handler.get_numerical_columns()
            default_columns = [col for col in numerical_columns if col in all_columns]
        else:
            default_columns = all_columns
        
        config['selected_columns'] = st.multiselect(
            "Select Columns:",
            options=all_columns,
            default=default_columns[:10] if len(default_columns) > 10 else default_columns,
            help="Choose which columns to include in the analysis"
        )
    
    # Generate report button
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                figure, data_table = st.session_state.report_engine.generate_report(
                    selected_report,
                    st.session_state.current_df,
                    config,
                    st.session_state.exclusion_info
                )
                
                # Display results
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True)
                    
                    # Add download buttons for the chart
                    st.session_state.export_manager.create_chart_download_buttons(
                        figure, 
                        f"{selected_report}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                
                if data_table is not None:
                    st.subheader("📋 Report Data")
                    st.dataframe(data_table, use_container_width=True)
                    
                    # Add download button for the data
                    csv_data = st.session_state.export_manager.export_data_to_csv(
                        data_table,
                        f"{selected_report}_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                    
                    st.download_button(
                        label="📥 Download Report Data",
                        data=csv_data,
                        file_name=f"{selected_report}_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def render_export_section():
    """Render the data export section."""
    if not st.session_state.data_loaded:
        return
    
    st.header("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export current filtered data
        st.session_state.export_manager.create_download_link_csv(
            st.session_state.current_df,
            f"filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    with col2:
        # Export analysis summary
        filter_summary = st.session_state.filter_manager.get_active_filters_summary()
        column_info = st.session_state.data_handler.get_column_info()
        
        st.session_state.export_manager.create_summary_download_button(
            st.session_state.current_df,
            filter_summary,
            column_info,
            st.session_state.exclusion_info
        )
    
    with col3:
        # Show export info
        export_info = st.session_state.export_manager.get_export_info(st.session_state.current_df)
        
        st.metric("Export Size", f"{export_info['estimated_csv_size_mb']:.1f} MB")
        st.caption(f"{export_info['row_count']:,} rows × {export_info['column_count']} columns")

def render_data_preview():
    """Render a preview of the current data."""
    if not st.session_state.data_loaded:
        return
    
    st.header("👀 Data Preview")
    
    # Show sample of current data
    preview_rows = min(100, len(st.session_state.current_df))
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"Showing first {preview_rows} rows of {len(st.session_state.current_df):,} total rows")
    
    with col2:
        show_all_columns = st.checkbox("Show all columns", value=False)
    
    # Display data
    if show_all_columns:
        st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)
    else:
        # Show first 10 columns
        display_columns = list(st.session_state.current_df.columns)[:10]
        st.dataframe(
            st.session_state.current_df[display_columns].head(preview_rows), 
            use_container_width=True
        )
        
        if len(st.session_state.current_df.columns) > 10:
            st.caption(f"Showing 10 of {len(st.session_state.current_df.columns)} columns. Check 'Show all columns' to see more.")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.data_loaded:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Filters", 
            "🎯 Outliers",
            "⚙️ Features", 
            "📈 Reports", 
            "💾 Export", 
            "👀 Data Preview"
        ])
        
        with tab1:
            render_filters_section()
        
        with tab2:
            render_outliers_section()
        
        with tab3:
            render_features_section()
        
        with tab4:
            render_reports_section()
        
        with tab5:
            render_export_section()
        
        with tab6:
            render_data_preview()
    
    else:
        # Welcome message and instructions
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

