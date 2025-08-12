# Quick Start Guide 🚀

Get up and running with the Sales Pipeline Data Explorer in minutes!

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
streamlit run main.py
```

3. **Open your browser** to the URL shown (typically `http://localhost:8501`)

## Generate Sample Data

If you don't have your own data yet, create sample data:

```bash
# Generate 200 opportunities with default settings
python create_sample_data.py

# Generate more data with custom settings
python create_sample_data.py --opportunities 500 --snapshots 10 --output my_data.csv

# Generate Excel format
python create_sample_data.py --format xlsx --output sample_data.xlsx
```

## Quick Test

Run the test script to verify everything works:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/services/test_data_handler.py -v

# Run with coverage
pytest --cov=src
```

## First Steps

1. **Upload Data**: Click "Browse files" in the sidebar and select your CSV/XLSX file
   - File size limit: 200MB
   - Supported formats: CSV, XLSX, XLS
   - Required columns: Id, Snapshot Date, Stage

2. **Explore Filters**: Go to the "Filters" tab and try filtering by different columns
   - Categorical filters: Include/exclude specific values
   - Numerical filters: Range, greater than, less than
   - Date filters: Date ranges and comparisons
   - Text filters: Contains, starts with, ends with, exact match

3. **Add Features**: Visit the "Features" tab and add calculated features like "days_in_pipeline"
   - Days in pipeline calculation
   - Win rate analysis
   - User activity ratings
   - Stage progression tracking

4. **Detect Outliers**: Use the "Outliers" tab to identify and exclude outliers
   - Multiple detection algorithms (IQR, Z-Score, Isolation Forest)
   - Configurable sensitivity levels
   - Preview before exclusion

5. **Create Reports**: Generate your first chart in the "Reports" tab
   - Descriptive statistics
   - Histograms and bar charts
   - Scatter plots and line charts
   - Correlation heatmaps and box plots

6. **Export Results**: Download your filtered data and charts from the "Export" tab
   - CSV data export
   - High-resolution chart export (PNG/SVG)
   - Analysis summaries

## Sample Data Structure

Your CSV/XLSX should have columns like:

| Id | Snapshot Date | Stage | Owner | Deal Value | ... |
|----|---------------|-------|-------|------------|-----|
| OPP-00001 | 01/15/2023 | Lead | Alice Johnson | 50000 | ... |
| OPP-00001 | 01/22/2023 | Budget | Alice Johnson | 50000 | ... |
| OPP-00002 | 01/20/2023 | Proposal Development | Bob Smith | 75000 | ... |

### Supported Stages
- Lead
- Budget
- Proposal Development
- Proposal Submitted
- Negotiation
- Closed - WON
- Closed - LOST
- Canceled/deferred

## Key Features to Try

### Filtering
- **Filter by Stage**: See only "Closed - WON" opportunities
- **Date Range Filter**: Focus on specific time periods
- **Value Range Filter**: Filter by deal value ranges
- **Owner Filter**: Compare performance across sales reps

### Feature Engineering
- **Add Win Rate Feature**: Calculate success rates by user
- **Days in Pipeline**: Track how long opportunities stay in pipeline
- **User Activity Rating**: Identify high-activity sales reps
- **Stage Progression**: Track movement through sales stages

### Outlier Detection
- **IQR Method**: Classic statistical outlier detection
- **Z-Score Method**: Standard deviation based detection
- **Isolation Forest**: Machine learning approach
- **Preview Outliers**: See detected outliers before exclusion

### Analysis & Reporting
- **Time Series Analysis**: Track pipeline changes over time
- **Group by Owner**: Compare performance across sales reps
- **Correlation Analysis**: Find relationships between variables
- **Distribution Analysis**: Understand data distributions

### Export Everything
- **Download Filtered Data**: CSV export of current dataset
- **High-Resolution Charts**: PNG/SVG export of visualizations
- **Analysis Summaries**: Text reports of key findings

## Advanced Features

### State Management
The application uses a centralized state management system that provides:
- **Data consistency**: Reliable state across all components
- **Debugging capabilities**: Comprehensive logging and error tracking
- **Performance optimization**: Memory management and caching
- **Error recovery**: Automatic state validation and recovery

### Multi-Page Application
The app is organized into focused pages:
- **Filters**: Data filtering and selection
- **Features**: Feature engineering and calculation
- **Outliers**: Outlier detection and management
- **Reports**: Visualization and analysis
- **Export**: Data and chart export
- **Data Preview**: Raw data exploration

### Performance Optimizations
- **Caching**: Expensive operations are cached for performance
- **Memory Management**: Efficient data processing and storage
- **Anti-flicker**: Protection against UI flickering during updates
- **Progress Indicators**: User feedback for long operations

## Troubleshooting

### Common Issues

**File Upload Problems**:
- Check file size (max 200MB)
- Verify file format (CSV, XLSX, XLS)
- Ensure required columns exist (Id, Snapshot Date, Stage)

**Data Type Issues**:
- Verify date format (MM/DD/YYYY)
- Check for text in numerical columns
- Review column names for special characters

**Performance Issues**:
- Use filters to reduce dataset size
- Limit date ranges for time-series analysis
- Reduce group-by categories

**State Management Issues**:
- Check application logs for errors
- Use debug information in StateManager
- Verify state consistency across components

### Getting Help

- **Check Logs**: Review terminal output for error messages
- **Debug Info**: Use StateManager debug information
- **Test Data**: Try with sample data to isolate issues
- **Documentation**: Review full [README.md](README.md) for detailed information

## Project Structure

```
sales_pipeline_tracker_data_exploration/
├── main.py                    # Main application
├── src/services/              # Core services
├── src/utils/                 # Utilities
├── pages/                     # Streamlit pages
├── tests/                     # Test suite
├── config/                    # Configuration
└── docs/                      # Documentation
```

## Next Steps

1. **Explore the Documentation**:
   - [README.md](README.md) - Comprehensive guide
   - [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture details
   - [docs/STATE_MANAGEMENT_TRANSITION.md](docs/STATE_MANAGEMENT_TRANSITION.md) - State management details

2. **Run the Test Suite**:
   ```bash
   pytest -v
   ```

3. **Customize the Application**:
   - Modify `config/settings.py` for customization
   - Add new features in `src/services/feature_engine.py`
   - Create new reports in `src/services/report_engine.py`

4. **Extend Functionality**:
   - Add new outlier detection methods
   - Create custom export formats
   - Implement new data types

**Happy exploring!** 📊

