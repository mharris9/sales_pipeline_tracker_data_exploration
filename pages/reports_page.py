"""
Reports Page - Generate and display data reports
"""
import streamlit as st
import time

def render_reports_section():
    st.title("Data Reports")
    
    if not hasattr(st.session_state, 'state_manager'):
        st.error("Please upload data first!")
        return
        
    report_engine = st.session_state.state_manager.get_extension('report_engine')
    
    # Report selection
    st.subheader("Available Reports")
    reports = report_engine.get_available_reports()
    
    with st.form("report_form"):
        for report in reports:
            st.checkbox(
                report['description'],
                key=f"report_{report['name']}",
                value=report['active']
            )
            
            # Report-specific settings
            if report['active']:
                report_engine.render_report_settings(report['name'])
        
        submitted = st.form_submit_button("Generate Reports")
        
        if submitted:
            with st.spinner("Generating reports..."):
                active_reports = [
                    r['name'] for r in reports 
                    if st.session_state[f"report_{r['name']}"]
                ]
                report_engine.set_active_reports(active_reports)
                report_engine.generate_reports()
                time.sleep(0.1)  # Anti-flicker
                st.session_state.state_manager.trigger_rerun()

    # Display reports
    if report_engine.get_active_reports():
        st.subheader("Report Results")
        report_engine.display_reports()

if __name__ == "__main__":
    render_reports_section()
