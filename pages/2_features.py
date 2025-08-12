"""
Features Page - Create and manage derived features
"""
import streamlit as st
import time

def render_features_page():
    st.title("Feature Engineering")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    feature_engine = st.session_state.state_manager.get_extension('feature_engine')
    
    # Feature selection
    st.subheader("Available Features")
    features = feature_engine.get_available_features()
    
    with st.form("feature_form"):
        for feature in features:
            st.checkbox(
                feature['description'],
                key=f"feature_{feature['name']}",
                value=feature['active']
            )
        
        submitted = st.form_submit_button("Update Features")
        
        if submitted:
            with st.spinner("Updating features..."):
                active_features = [
                    f['name'] for f in features 
                    if st.session_state[f"feature_{f['name']}"]
                ]
                feature_engine.set_active_features(active_features)
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()

    # Display feature results
    if feature_engine.get_active_features():
        st.subheader("Feature Results")
        feature_engine.display_results()

if __name__ == "__main__":
    render_features_page()
