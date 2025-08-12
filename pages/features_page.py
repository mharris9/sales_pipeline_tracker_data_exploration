"""
Features Page - Create and manage derived features
"""
import streamlit as st
import time

def render_features_section():
    st.title("Feature Engineering")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    feature_engine = st.session_state.state_manager.get_extension('feature_engine')
    
    # Check if data is loaded
    if not st.session_state.state_manager.get_state('data.data_loaded', False):
        st.warning("No data loaded. Please upload data first.")
        return
    
    # Get available features
    data_handler = st.session_state.state_manager.get_extension('data_handler')
    df = data_handler.get_current_df()
    features = feature_engine.get_available_features(df.columns.tolist()) if df is not None else {}
    
    if not features:
        st.info("No features available for the current data columns.")
        return
    
    # Feature selection with form validation
    st.subheader("Available Features")
    
    with st.form("feature_form"):
        st.write("Select features to calculate:")
        
        # Feature checkboxes
        selected_features = []
        for feature_name, feature_info in features.items():
            if st.checkbox(
                f"{feature_name}: {feature_info['description']}",
                key=f"feature_{feature_name}",
                help=f"Requires: {', '.join(feature_info['requirements'])}"
            ):
                selected_features.append(feature_name)
        
        # Form validation
        form_valid = True
        validation_errors = []
        
        # Check if selected features have required columns
        if df is not None:
            available_columns = df.columns.tolist()
            for feature_name in selected_features:
                if feature_name in features:
                    required_cols = features[feature_name]['requirements']
                    missing_cols = [col for col in required_cols if col not in available_columns]
                    if missing_cols:
                        validation_errors.append(f"Feature '{feature_name}' requires columns: {', '.join(missing_cols)}")
                        form_valid = False
        
        # Show validation errors
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        
        # Submit button
        submitted = st.form_submit_button(
            "Calculate Features",
            disabled=not form_valid,
            help="Calculate the selected features"
        )
        
        if submitted and form_valid:
            with st.spinner("Calculating features..."):
                try:
                    feature_engine.set_active_features(selected_features)
                    
                    # Calculate features
                    df_with_features = feature_engine.calculate_features(df)
                    
                    # Update state with new dataframe
                    st.session_state.state_manager.set_state('data.current_df', df_with_features)
                    
                    st.toast(f"✅ Calculated {len(selected_features)} feature(s) successfully!", icon="✅")
                    time.sleep(0.1)  # Anti-flicker
                    st.session_state.state_manager.trigger_rerun()
                    
                except Exception as e:
                    st.toast(f"❌ Error calculating features: {str(e)}", icon="❌")

    # Display feature results
    active_features = feature_engine.get_active_features()
    if active_features:
        st.subheader("Feature Results")
        
        # Get feature results from state
        feature_results = st.session_state.state_manager.get_state('feature_results', {})
        
        for feature_name in active_features:
            if feature_name in feature_results:
                st.write(f"**{feature_name}:**")
                st.json(feature_results[feature_name])
            else:
                st.info(f"No results available for {feature_name}")

if __name__ == "__main__":
    render_features_section()
