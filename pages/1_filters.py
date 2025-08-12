"""
Filters Page - Manage and apply data filters
"""
import streamlit as st
import time

def render_filters_page():
    st.title("Data Filters")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    filter_manager = st.session_state.state_manager.get_extension('filter_manager')
    
    # Render filter UI
    with st.form("filter_form"):
        filter_manager.render_filter_ui()
        submitted = st.form_submit_button("Apply Filters")
        
        if submitted:
            with st.spinner("Applying filters..."):
                filter_manager.apply_filters()
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()

    # Clear filters button
    if st.button("Clear All Filters"):
        filter_manager.clear_all_filters()
        time.sleep(0.1)  # Anti-flicker
        st.session_state.state_manager.trigger_rerun()

if __name__ == "__main__":
    render_filters_page()
