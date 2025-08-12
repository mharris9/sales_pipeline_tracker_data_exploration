# Sales Pipeline Data Explorer 📊

A comprehensive Streamlit application for exploring and analyzing sales pipeline data with advanced filtering, feature engineering, and visualization capabilities.

## Features 🚀

### Core Functionality
- **Data Import**: Support for CSV and XLSX files up to 200MB
- **Smart Data Type Detection**: Automatically detects and converts categorical, numerical, date, text, and boolean data
- **Advanced Filtering**: Customizable filters for every column type with various filter modes
- **Outlier Detection & Exclusion**: Multiple algorithms with adjustable sensitivity for data cleaning
- **Feature Engineering**: Derive new insights with calculated features
- **Interactive Visualizations**: Multiple chart types with group-by capabilities
- **Data Export**: Export filtered data and high-resolution charts

### Supported Report Types
- **Descriptive Statistics**: Summary statistics for all data types
- **Histogram**: Distribution analysis for numerical and date data
- **Bar Chart**: Categorical data visualization with aggregation options
- **Scatter Plot**: Relationship analysis between numerical variables
- **Line Chart**: Trend analysis over time or numerical progression
- **Correlation Heatmap**: Correlation matrix for numerical variables
- **Box Plot**: Distribution and outlier analysis
- **Time Series**: Time-based trend analysis with various aggregation periods

### Advanced Features
- **Pipeline-Specific Features**:
  - Days in pipeline calculation
  - Time to close (won) analysis
  - Starting and final stage tracking
  - User win rate calculations
  - User activity rating (Low/Medium/High based on volume)
  - Time spent in each stage
  - Stage progression tracking

- **Outlier Detection Methods**:
  - Interquartile Range (IQR) - Classic statistical method
  - Z-Score - Standard deviation based detection
  - Modified Z-Score (MAD) - Robust median-based method
  - Isolation Forest - Machine learning approach

- **Group-By Analysis**: Group all reports by categorical columns
- **Smart Filtering**: Context-aware filters that adapt to data types
- **Export Options**: CSV data export, PNG/SVG chart export, analysis summaries

## Installation 💻

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run main.py
```

## Usage Guide 📖

### 1. Data Upload
- Click "Browse files" in the sidebar
- Select your CSV or XLSX file
- Click "Load Data" to process the file

### 2. Expected Data Structure
Your data should include these recommended columns:
- **Id**: Unique identifier for opportunities (duplicates expected for snapshots)
- **Snapshot Date**: Date of snapshot (MM/DD/YYYY format)
- **Stage**: Current opportunity stage

Supported stages:
- Lead
- Budget
- Proposal Development
- Proposal Submitted
- Negotiation
- Closed - WON
- Closed - LOST
- Canceled/deferred

Additional columns can contain any combination of:
- **Categorical data**: Text with limited unique values
- **Numerical data**: Numbers for calculations and aggregations
- **Date data**: Dates in various formats
- **Text data**: Free-form text fields

### 3. Filtering Data
- Navigate to the "Filters" tab
- Each column gets an appropriate filter type:
  - **Categorical**: Include/exclude specific values
  - **Numerical**: Range, greater than, or less than filters
  - **Date**: Date range, before, or after filters
  - **Text**: Contains, starts with, ends with, or exact match
  - **Boolean**: True/false selection
- Apply multiple filters simultaneously
- Use "Clear All Filters" to reset

### 4. Outlier Detection & Exclusion
- Navigate to the "Outliers" tab
- Enable outlier detection globally
- Select columns for outlier analysis
- Choose detection method:
  - **IQR (Interquartile Range)**: Best for normally distributed data
  - **Z-Score**: Good for large datasets with normal distribution
  - **Modified Z-Score**: Robust against extreme outliers
  - **Isolation Forest**: Machine learning approach for complex patterns
- Adjust sensitivity:
  - **Conservative**: Fewer outliers detected
  - **Moderate**: Balanced approach (default)
  - **Aggressive**: More outliers detected
  - **Very Aggressive**: Maximum outlier detection
- Preview outliers before applying
- Choose combination method for multiple columns

### 5. Feature Engineering
- Navigate to the "Features" tab
- Select from available calculated features:
  - **days_in_pipeline**: Time between first and latest snapshot
  - **days_to_close_won**: Time from start to Closed - WON
  - **starting_stage**: First recorded stage
  - **final_stage**: Most recent stage
  - **user_win_rate**: Win percentage by user/owner
  - **user_activity_rating**: Activity level (Low/Medium/High)
  - **days_in_current_stage**: Time spent in current stage
  - **opportunity_age_days**: Age from first snapshot
  - **stage_progression_count**: Number of stages experienced

### 6. Generating Reports
- Navigate to the "Reports" tab
- Select a report type
- Choose group-by column (optional)
- Configure axes for chart-based reports
- Set aggregation methods and other parameters
- Click "Generate Report"

### 7. Exporting Data
- Navigate to the "Export" tab
- Download filtered dataset as CSV
- Download analysis summary as text
- Download charts as PNG or SVG from individual reports

## Architecture 🏗️

### Core Modules

#### `core/data_handler.py`
- **DataHandler**: Manages data import, validation, and type conversion
- Features:
  - Automatic data type detection
  - Missing data handling
  - Sales pipeline data validation
  - Memory-efficient processing

#### `core/filter_manager.py`
- **FilterManager**: Creates and manages data filters
- Features:
  - Dynamic filter generation based on data types
  - Multiple filter modes per data type
  - Filter state management
  - Batch filter application

#### `core/feature_engine.py`
- **FeatureEngine**: Handles derived feature calculations
- Features:
  - Extensible feature framework
  - Sales pipeline-specific features
  - Automatic column detection
  - Error handling and validation

#### `core/report_engine.py`
- **ReportEngine**: Generates reports and visualizations
- Features:
  - Multiple chart types
  - Group-by functionality
  - Dynamic axis selection
  - Statistical calculations

#### `utils/data_types.py`
- Data type detection and utilities
- Column compatibility checking
- Statistical calculations

#### `utils/export_utils.py`
- **ExportManager**: Handles all export functionality
- Features:
  - Multiple export formats
  - High-resolution chart export
  - Summary report generation

### Configuration
- `config/settings.py`: Application settings and constants
- Easily customizable parameters
- Chart styling and color schemes

## Performance Considerations 🚄

The application is optimized for large datasets:

- **Memory Efficient**: Processes data in chunks where possible
- **Lazy Loading**: Features calculated only when requested
- **Caching**: Streamlit caching for expensive operations
- **Optimized Filtering**: Efficient pandas operations
- **Progress Indicators**: User feedback for long operations

**Tested with**: 300,000 rows × 30 columns (≈20MB files)

## Customization 🎨

### Adding New Features
1. Create a feature calculation function in `FeatureEngine`
2. Register the feature with requirements and metadata
3. The UI will automatically detect and offer the new feature

### Adding New Report Types
1. Add report configuration to `ReportEngine._register_default_reports()`
2. Implement the report generation function
3. Define axis requirements and compatibility
4. The UI will automatically include the new report type

### Styling
- Modify `config/settings.py` for colors and themes
- Chart styling uses Plotly themes
- Streamlit components follow the configured layout

## Troubleshooting 🔧

### Common Issues

**File Upload Errors**:
- Ensure file size is under 200MB
- Check file format (CSV, XLSX, XLS only)
- Verify file is not corrupted

**Data Type Detection Issues**:
- Check date formats (expected: MM/DD/YYYY)
- Ensure numerical columns don't contain text
- Review column names for special characters

**Feature Calculation Errors**:
- Verify required columns exist
- Check for sufficient data (some features need multiple records per ID)
- Review date column formatting

**Performance Issues**:
- Consider filtering data before adding features
- Use smaller date ranges for time-series analysis
- Reduce the number of groups in group-by analysis

### Getting Help
- Check the application logs in the terminal
- Review error messages in the Streamlit interface
- Verify your data structure matches expectations

## Technical Requirements 📋

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM recommended for large datasets
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Dependencies
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.15.0
- numpy >= 1.24.0
- openpyxl >= 3.1.0 (for Excel support)
- Additional dependencies in `requirements.txt`

## License 📄

This project is provided as-is for data analysis purposes. Feel free to modify and extend according to your needs.

## Current Development 🚧

### State Management Enhancement
We are transitioning to a centralized state management system to improve:
- Data consistency and reliability
- Debugging and monitoring capabilities
- Feature extensibility
- Performance optimization
- Error recovery

See [State Management Transition](docs/STATE_MANAGEMENT_TRANSITION.md) for details about this major enhancement.

## Future Enhancements 🔮

Potential improvements and extensions:
- Database connectivity (SQL, MongoDB)
- Real-time data refresh
- Advanced statistical analysis
- Machine learning predictions
- Custom dashboard creation
- Multi-user support
- API integration capabilities

---

**Happy Data Exploring!** 🎉

