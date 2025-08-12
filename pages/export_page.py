import streamlit as st
import time

def render_export_section():
    st.header("💾 Export Data & Charts")
    export_manager = st.session_state.state_manager.get_extension('exports.export_manager')

    if not st.session_state.state_manager.get_state('data.data_loaded', False):
        st.warning("No data loaded. Please upload data first.")
        return

    data_handler = st.session_state.state_manager.get_extension('data_handler')
    df = data_handler.get_current_df()

    if df is None:
        st.error("No data available for export.")
        return

    st.subheader("Export Options")

    with st.form("export_form"):
        st.write("Select export options:")

        # Data export options
        export_data = st.checkbox(
            "Export filtered data as CSV",
            value=True,
            help="Download the current filtered dataset"
        )

        # Chart export options
        export_charts = st.checkbox(
            "Export charts as images",
            help="Download charts as PNG or SVG files"
        )

        # Report export options
        export_report = st.checkbox(
            "Export analysis summary",
            help="Generate a text summary of the analysis"
        )

        form_valid = True
        validation_errors = []

        if export_charts and not st.session_state.state_manager.get_state('reports.current_report'):
            validation_errors.append("No charts available. Generate a report first to export charts.")
            form_valid = False

        if validation_errors:
            for error in validation_errors:
                st.error(error)

        submitted = st.form_submit_button(
            "Generate Exports",
            disabled=not form_valid,
            help="Generate the selected exports"
        )

        if submitted and form_valid:
            with st.spinner("Generating exports..."):
                try:
                    # Export data
                    if export_data:
                        csv_data = export_manager.export_data_to_csv(df)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name="filtered_data.csv",
                            mime="text/csv",
                            help="Download filtered data as CSV"
                        )

                    # Export charts
                    if export_charts:
                        current_report = st.session_state.state_manager.get_state('reports.current_report')
                        if current_report:
                            report_engine = st.session_state.state_manager.get_extension('reports.report_engine')
                            chart = report_engine.get_current_chart()
                            if chart:
                                # PNG export
                                png_data = export_manager.export_chart_as_png(chart)
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=png_data,
                                    file_name=f"{current_report}_chart.png",
                                    mime="image/png",
                                    help="Download chart as PNG"
                                )

                                # SVG export
                                svg_data = export_manager.export_chart_as_svg(chart)
                                st.download_button(
                                    label="📥 Download SVG",
                                    data=svg_data,
                                    file_name=f"{current_report}_chart.svg",
                                    mime="image/svg+xml",
                                    help="Download chart as SVG"
                                )

                    # Export summary
                    if export_report:
                        summary = export_manager.create_summary_report(df)
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name="analysis_summary.txt",
                            mime="text/plain",
                            help="Download analysis summary"
                        )

                    st.toast("✅ Exports generated successfully!", icon="✅")
                    time.sleep(0.1)
                    st.session_state.state_manager.trigger_rerun()

                except Exception as e:
                    st.toast(f"❌ Error generating exports: {str(e)}", icon="❌")

    # Export history
    export_history = st.session_state.state_manager.get_state('exports.export_history', [])
    if export_history:
        st.subheader("Export History")
        for export in export_history[-5:]:  # Show last 5 exports
            st.write(f"**{export['timestamp']}**: {export['type']} - {export['filename']}")

if __name__ == "__main__":
    render_export_section()
