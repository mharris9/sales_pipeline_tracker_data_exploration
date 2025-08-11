"""
Export utilities for data and charts.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List
import streamlit as st
import io
from datetime import datetime
import base64

from config.settings import EXPORT_DPI, EXPORT_FORMATS

class ExportManager:
    """
    Manages export functionality for data and visualizations.
    """
    
    def __init__(self):
        """Initialize the ExportManager."""
        pass
    
    def export_data_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> bytes:
        """
        Export DataFrame to CSV format.
        
        Args:
            df: DataFrame to export
            filename: Optional filename (for display purposes)
            
        Returns:
            CSV data as bytes
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sales_pipeline_data_{timestamp}.csv"
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')
        
        return csv_data
    
    def create_download_link_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> None:
        """
        Create a Streamlit download button for CSV data.
        
        Args:
            df: DataFrame to export
            filename: Optional filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sales_pipeline_data_{timestamp}.csv"
        
        csv_data = self.export_data_to_csv(df, filename)
        
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download the current filtered dataset as CSV"
        )
    
    def export_chart_as_png(self, fig: go.Figure, filename: Optional[str] = None, 
                           width: int = 1200, height: int = 800) -> bytes:
        """
        Export Plotly figure as PNG.
        
        Args:
            fig: Plotly figure
            filename: Optional filename
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            PNG data as bytes
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
        
        # Export as PNG with high DPI
        png_data = fig.to_image(
            format="png", 
            width=width, 
            height=height,
            scale=EXPORT_DPI / 72  # Convert DPI to scale factor
        )
        
        return png_data
    
    def export_chart_as_svg(self, fig: go.Figure, filename: Optional[str] = None) -> str:
        """
        Export Plotly figure as SVG.
        
        Args:
            fig: Plotly figure
            filename: Optional filename
            
        Returns:
            SVG data as string
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.svg"
        
        # Export as SVG
        svg_data = fig.to_image(format="svg").decode('utf-8')
        
        return svg_data
    
    def create_chart_download_buttons(self, fig: go.Figure, base_filename: Optional[str] = None) -> None:
        """
        Create download buttons for chart in multiple formats.
        
        Args:
            fig: Plotly figure
            base_filename: Base filename (without extension)
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"chart_{timestamp}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PNG download
            try:
                png_data = self.export_chart_as_png(fig, f"{base_filename}.png")
                st.download_button(
                    label="📊 Download as PNG",
                    data=png_data,
                    file_name=f"{base_filename}.png",
                    mime="image/png",
                    help="Download chart as high-resolution PNG image"
                )
            except Exception as e:
                st.error(f"Error creating PNG export: {str(e)}")
        
        with col2:
            # SVG download
            try:
                svg_data = self.export_chart_as_svg(fig, f"{base_filename}.svg")
                st.download_button(
                    label="🎨 Download as SVG",
                    data=svg_data,
                    file_name=f"{base_filename}.svg",
                    mime="image/svg+xml",
                    help="Download chart as scalable SVG image"
                )
            except Exception as e:
                st.error(f"Error creating SVG export: {str(e)}")
    
    def create_summary_report(self, df: pd.DataFrame, filter_summary: Dict[str, str], 
                             column_info: Dict[str, Dict[str, Any]], 
                             exclusion_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a text summary report of the current analysis.
        
        Args:
            df: Current DataFrame
            filter_summary: Summary of active filters
            column_info: Information about columns
            
        Returns:
            Summary report as string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Sales Pipeline Data Analysis Report
Generated on: {timestamp}

## Data Summary
- Total Records: {len(df):,}
- Total Columns: {len(df.columns)}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## Active Filters
"""
        
        if filter_summary:
            for column, summary in filter_summary.items():
                report += f"- {column}: {summary}\n"
        else:
            report += "- No filters applied\n"
        
        ## Outlier Exclusion
        report += "\n## Outlier Exclusion\n"
        
        if exclusion_info and exclusion_info.get('outliers_excluded', False):
            report += f"- Outliers excluded: {exclusion_info['excluded_rows']:,} rows ({exclusion_info['exclusion_percentage']:.1f}%)\n"
            report += f"- Detection method: {exclusion_info['detection_info']['method'].replace('_', ' ').title()}\n"
            report += f"- Sensitivity level: {exclusion_info['detection_info']['sensitivity'].replace('_', ' ').title()}\n"
            report += f"- Columns analyzed: {', '.join(exclusion_info['excluded_columns'])}\n"
            report += f"- Combination method: {exclusion_info['combination_method']}\n"
            
            # Per-column details
            for col, col_info in exclusion_info['detection_info']['column_results'].items():
                report += f"  - {col}: {col_info['outlier_count']} outliers ({col_info['outlier_percentage']:.1f}%)\n"
        else:
            report += "- No outlier exclusion applied\n"
        
        report += "\n## Column Information\n"
        
        for column, info in column_info.items():
            data_type = info['type'].value if hasattr(info['type'], 'value') else str(info['type'])
            stats = info.get('stats', {})
            
            report += f"\n### {column} ({data_type})\n"
            
            # Add relevant statistics based on data type
            if 'count' in stats:
                report += f"- Count: {stats['count']:,}\n"
            if 'null_count' in stats:
                report += f"- Missing Values: {stats['null_count']:,} ({stats.get('null_percentage', 0):.1f}%)\n"
            
            if data_type == 'numerical':
                if 'mean' in stats:
                    report += f"- Mean: {stats['mean']:.2f}\n"
                if 'median' in stats:
                    report += f"- Median: {stats['median']:.2f}\n"
                if 'std' in stats:
                    report += f"- Standard Deviation: {stats['std']:.2f}\n"
                if 'min' in stats and 'max' in stats:
                    report += f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
            
            elif data_type == 'categorical':
                if 'unique_count' in stats:
                    report += f"- Unique Values: {stats['unique_count']}\n"
                if 'most_frequent' in stats:
                    report += f"- Most Frequent: {stats['most_frequent']} ({stats.get('most_frequent_count', 0)} times)\n"
            
            elif data_type == 'date':
                if 'earliest' in stats and 'latest' in stats:
                    report += f"- Date Range: {stats['earliest']} to {stats['latest']}\n"
                if 'range_days' in stats:
                    report += f"- Total Days: {stats['range_days']}\n"
        
        return report
    
    def create_summary_download_button(self, df: pd.DataFrame, filter_summary: Dict[str, str], 
                                     column_info: Dict[str, Dict[str, Any]], 
                                     exclusion_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a download button for the summary report.
        
        Args:
            df: Current DataFrame
            filter_summary: Summary of active filters
            column_info: Information about columns
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_summary_{timestamp}.txt"
        
        report = self.create_summary_report(df, filter_summary, column_info, exclusion_info)
        
        st.download_button(
            label="📋 Download Analysis Summary",
            data=report.encode('utf-8'),
            file_name=filename,
            mime="text/plain",
            help="Download a summary report of the current analysis"
        )
    
    def export_filtered_data_with_features(self, df: pd.DataFrame, 
                                         original_columns: List[str],
                                         feature_columns: List[str]) -> pd.DataFrame:
        """
        Prepare data for export with clear separation of original and feature columns.
        
        Args:
            df: DataFrame with all columns
            original_columns: List of original column names
            feature_columns: List of feature column names
            
        Returns:
            DataFrame organized for export
        """
        # Reorder columns: original columns first, then features
        export_columns = []
        
        # Add original columns that exist in the DataFrame
        for col in original_columns:
            if col in df.columns:
                export_columns.append(col)
        
        # Add feature columns that exist in the DataFrame
        for col in feature_columns:
            if col in df.columns and col not in export_columns:
                export_columns.append(col)
        
        # Add any remaining columns
        for col in df.columns:
            if col not in export_columns:
                export_columns.append(col)
        
        return df[export_columns].copy()
    
    def get_export_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about the data being exported.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Dictionary with export information
        """
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'estimated_csv_size_mb': len(df.to_csv(index=False).encode('utf-8')) / 1024**2,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        }
