"""
Filters Page - Manage and apply data filters
"""
import streamlit as st
import time

def render_filters_section():
    st.title("Data Filters")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    filter_manager = st.session_state.state_manager.get_extension('filter_manager')
    
    # Check if data is loaded
    if not st.session_state.state_manager.get_state('data.data_loaded', False):
        st.warning("No data loaded. Please upload data first.")
        return
    
    # Render filter UI with form validation
    with st.form("filter_form"):
        st.subheader("Configure Filters")
        
        # Get current data for validation
        data_handler = st.session_state.state_manager.get_extension('data_handler')
        df = data_handler.get_current_df()
        column_types = data_handler.get_column_types()
        
        if df is not None and column_types:
            filter_manager.render_filter_ui()
            
            # Form validation
            form_valid = True
            validation_errors = []
            
            # Add any filter-specific validation here
            # For example, check if date ranges are valid
            
            # Show validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                form_valid = False
            
            # Submit button
            submitted = st.form_submit_button(
                "Apply Filters",
                disabled=not form_valid,
                help="Apply the configured filters to the data"
            )
            
            if submitted and form_valid:
                with st.spinner("Applying filters..."):
                    try:
                        filter_manager.apply_filters()
                        st.toast("✅ Filters applied successfully!", icon="✅")
                        time.sleep(0.1)  # Anti-flicker
                        st.session_state.state_manager.trigger_rerun()
                    except Exception as e:
                        st.toast(f"❌ Error applying filters: {str(e)}", icon="❌")
        else:
            st.error("No data available for filtering.")

    # Clear filters button with confirmation
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Clear All Filters", help="Remove all active filters"):
            try:
                filter_manager.clear_all_filters()
                st.toast("✅ All filters cleared!", icon="✅")
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()
            except Exception as e:
                st.toast(f"❌ Error clearing filters: {str(e)}", icon="❌")

if __name__ == "__main__":
    render_filters_section()
