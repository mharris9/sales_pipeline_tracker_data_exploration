"""
Data Preview Page - Preview and explore data
"""
import streamlit as st
import pandas as pd
from src.services.data_handler import DataHandler

def render_data_preview_section():
    st.title("Data Preview")
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.current_df is None:
        st.warning("No data loaded. Please upload data first.")
        return
    
    # Display data preview
    df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.current_df
    
    # Ensure Arrow compatibility defensively
    df_display = DataHandler.ensure_arrow_compatible(df)
    
    st.subheader("Data Overview")
    st.write(f"**Total Records:** {len(df_display)}")
    st.write(f"**Total Columns:** {len(df_display.columns)}")
    
    # Show first few rows
    st.subheader("First 10 Rows")
    st.dataframe(df_display.head(10), use_container_width=True)
    
    # Show data types
    st.subheader("Column Information")
    st.write(df_display.dtypes)
    
    # Show basic statistics for numerical columns
    numerical_cols = df_display.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        st.subheader("Numerical Column Statistics")
        st.write(df_display[numerical_cols].describe())

if __name__ == "__main__":
    render_data_preview_section()
