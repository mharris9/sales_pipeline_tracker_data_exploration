# Project Structure 📁

```
sales_pipeline_tracker_data_exploration/
├── 📋 main.py                      # Main Streamlit application entry point
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # Comprehensive documentation
├── 🚀 QUICKSTART.md               # Quick start guide
├── 📊 PROJECT_STRUCTURE.md        # This file - project overview
├── 🧪 test_app.py                 # Test script for application components
├── 🎲 create_sample_data.py       # Sample data generator
│
├── 🏗️ core/                       # Core application modules
│   ├── __init__.py
│   ├── 📊 data_handler.py         # DataHandler - data import and processing
│   ├── 🔍 filter_manager.py       # FilterManager - data filtering system
│   ├── ⚙️ feature_engine.py       # FeatureEngine - derived feature calculations
│   └── 📈 report_engine.py        # ReportEngine - reports and visualizations
│
├── 🛠️ utils/                       # Utility modules
│   ├── __init__.py
│   ├── 🏷️ data_types.py           # Data type detection and utilities
│   └── 💾 export_utils.py         # Export functionality
│
└── ⚙️ config/                      # Configuration
    ├── __init__.py
    └── 🔧 settings.py              # Application settings and constants
```

## Module Descriptions

### Core Modules

#### `main.py` 🎯
- **Purpose**: Main Streamlit application
- **Features**: UI layout, session state management, tab organization
- **Key Functions**: 
  - `initialize_session_state()` - Setup application state
  - `render_*_section()` - UI rendering functions
  - `main()` - Application entry point

#### `core/data_handler.py` 📊
- **Class**: `DataHandler`
- **Purpose**: Data import, validation, and preprocessing
- **Key Methods**:
  - `load_file()` - Import CSV/XLSX files
  - `_process_data()` - Clean and convert data types
  - `validate_sales_pipeline_data()` - Pipeline-specific validation
  - `get_*_columns()` - Column type getters

#### `core/filter_manager.py` 🔍
- **Class**: `FilterManager`
- **Purpose**: Dynamic data filtering system
- **Key Methods**:
  - `create_filters()` - Generate filters based on data types
  - `render_filter_ui()` - Streamlit filter UI components
  - `apply_filters()` - Apply active filters to data
  - `get_active_filters_summary()` - Filter state summary

#### `core/feature_engine.py` ⚙️
- **Class**: `FeatureEngine`
- **Purpose**: Calculate derived features
- **Key Methods**:
  - `register_feature()` - Add new feature calculations
  - `add_features()` - Calculate selected features
  - `get_available_features()` - List compatible features
- **Built-in Features**:
  - Days in pipeline, time to close, stage progression
  - Win rates, user activity ratings, time in stages

#### `core/report_engine.py` 📈
- **Class**: `ReportEngine`
- **Purpose**: Generate reports and visualizations
- **Key Methods**:
  - `generate_report()` - Create charts and tables
  - `get_compatible_columns_for_report()` - Column compatibility
- **Report Types**:
  - Descriptive statistics, histograms, bar charts
  - Scatter plots, line charts, correlation heatmaps
  - Box plots, time series analysis

### Utility Modules

#### `utils/data_types.py` 🏷️
- **Purpose**: Data type detection and management
- **Key Functions**:
  - `detect_data_type()` - Automatic type detection
  - `convert_to_proper_type()` - Type conversion
  - `calculate_statistics()` - Type-appropriate statistics
- **Types**: Categorical, Numerical, Date, Text, Boolean

#### `utils/export_utils.py` 💾
- **Class**: `ExportManager`
- **Purpose**: Data and chart export functionality
- **Key Methods**:
  - `export_data_to_csv()` - CSV data export
  - `export_chart_as_png/svg()` - High-resolution chart export
  - `create_summary_report()` - Analysis summary generation

#### `config/settings.py` 🔧
- **Purpose**: Application configuration
- **Contains**:
  - UI settings (colors, themes, layout)
  - Data processing parameters
  - Chart styling options
  - Performance tuning parameters

## Data Flow 🔄

1. **Data Import** (`DataHandler`)
   - User uploads CSV/XLSX file
   - Automatic type detection and conversion
   - Data validation and cleaning

2. **Filtering** (`FilterManager`)
   - Dynamic filter creation based on column types
   - User applies filters through UI
   - Filtered dataset maintained in session state

3. **Feature Engineering** (`FeatureEngine`)
   - User selects features to calculate
   - Features added to filtered dataset
   - Column types updated for new features

4. **Analysis & Reporting** (`ReportEngine`)
   - User selects report type and configuration
   - Charts and tables generated from current dataset
   - Group-by analysis supported

5. **Export** (`ExportManager`)
   - Filtered data exported as CSV
   - Charts exported as PNG/SVG
   - Summary reports generated

## Extension Points 🔌

### Adding New Features
1. Create calculation function in `FeatureEngine`
2. Register with `register_feature()`
3. UI automatically detects and offers new feature

### Adding New Report Types
1. Add configuration to `ReportEngine._register_default_reports()`
2. Implement generation function
3. Define axis requirements and compatibility

### Adding New Data Types
1. Add enum value to `DataType` in `data_types.py`
2. Implement detection logic in `detect_data_type()`
3. Add filter support in `FilterManager`
4. Update statistics calculation

### Adding New Export Formats
1. Extend `ExportManager` with new format methods
2. Update `create_*_download_buttons()` functions
3. Add format options to UI

## Performance Considerations 🚄

- **Memory Management**: Efficient pandas operations, data copying minimized
- **Large Dataset Support**: Chunked processing where applicable
- **Caching**: Streamlit caching for expensive operations
- **UI Responsiveness**: Progress indicators for long operations
- **Optimized Filtering**: Vectorized pandas operations

## Testing 🧪

- **Unit Tests**: `test_app.py` - Component-level testing
- **Sample Data**: `create_sample_data.py` - Realistic test data generation
- **Manual Testing**: UI interaction testing through main application

## Deployment 🚀

**Local Development**:
```bash
pip install -r requirements.txt
streamlit run main.py
```

**Production Considerations**:
- File upload size limits
- Memory usage monitoring
- Error logging and handling
- User session management
