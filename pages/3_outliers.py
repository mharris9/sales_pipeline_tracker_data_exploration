"""
Outliers Page - Detect and manage outliers
"""
import streamlit as st
import time

def render_outliers_page():
    st.title("Outlier Detection")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    outlier_manager = st.session_state.state_manager.get_extension('outlier_manager')
    
    # Outlier detection settings
    st.subheader("Detection Methods")
    detectors = outlier_manager.get_available_detectors()
    
    with st.form("outlier_form"):
        for detector in detectors:
            st.checkbox(
                detector['description'],
                key=f"detector_{detector['name']}",
                value=detector['active']
            )
            
            # Method-specific settings
            if detector['active']:
                outlier_manager.render_detector_settings(detector['name'])
        
        submitted = st.form_submit_button("Detect Outliers")
        
        if submitted:
            with st.spinner("Detecting outliers..."):
                active_detectors = [
                    d['name'] for d in detectors 
                    if st.session_state[f"detector_{d['name']}"]
                ]
                outlier_manager.set_active_detectors(active_detectors)
                outlier_manager.detect_outliers()
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()

    # Display outlier results
    if outlier_manager.get_active_detectors():
        st.subheader("Outlier Results")
        outlier_manager.display_results()

if __name__ == "__main__":
    render_outliers_page()
