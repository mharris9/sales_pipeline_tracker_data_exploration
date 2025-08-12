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
├── 🏗️ src/                        # Source code directory
│   ├── services/                   # Core application services
│   │   ├── __init__.py
│   │   ├── 📊 data_handler.py     # DataHandler - data import and processing
│   │   ├── 🔍 filter_manager.py   # FilterManager - data filtering system
│   │   ├── ⚙️ feature_engine.py   # FeatureEngine - derived feature calculations
│   │   ├── 📈 report_engine.py    # ReportEngine - reports and visualizations
│   │   ├── 🎯 outlier_manager.py  # OutlierManager - outlier detection and handling
│   │   └── 🧠 state_manager.py    # StateManager - centralized state management
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── 🏷️ data_types.py       # Data type detection and utilities
│   │   ├── 💾 export_utils.py     # Export functionality
│   │   └── 🔗 column_mapping.py   # Column mapping utilities
│   └── assets/                    # Static assets
│       └── style.css              # Custom CSS styles
│
├── 📄 pages/                      # Streamlit pages for multi-page app
│   ├── 1_filters.py              # Filters page
│   ├── 2_features.py             # Features page
│   ├── 3_outliers.py             # Outliers page
│   ├── 4_reports.py              # Reports page
│   ├── 5_export.py               # Export page
│   └── 6_data_preview.py         # Data preview page
│
├── ⚙️ config/                     # Configuration
│   ├── __init__.py
│   └── 🔧 settings.py             # Application settings and constants
│
├── 🧪 tests/                     # Test suite
│   ├── services/                 # Service layer tests
│   │   ├── test_data_handler.py     # DataHandler tests
│   │   ├── test_filter_manager.py   # FilterManager tests
│   │   ├── test_feature_engine.py   # FeatureEngine tests
│   │   ├── test_report_engine.py    # ReportEngine tests
│   │   ├── test_outlier_manager.py  # OutlierManager tests
│   │   ├── test_state_manager.py    # StateManager tests
│   │   ├── test_state_manager_extended.py  # Extended StateManager tests
│   │   ├── test_phase1.py           # Phase 1 transition tests
│   │   ├── test_phase2_data_handler.py  # Phase 2 DataHandler tests
│   │   ├── test_phase2_filter_manager.py # Phase 2 FilterManager tests
│   │   ├── test_phase2_feature_engine.py # Phase 2 FeatureEngine tests
│   │   ├── test_phase2_outlier_manager.py # Phase 2 OutlierManager tests
│   │   └── test_phase2_report_engine.py  # Phase 2 ReportEngine tests
│   ├── pages/                    # Page component tests
│   │   └── test_filter_ui.py        # Filter UI tests
│   ├── conftest.py              # Test configuration and fixtures
│   └── test_data_generator.py   # Test data generation utilities
│
├── 📚 docs/                      # Documentation
│   ├── STATE_MANAGEMENT_TRANSITION.md  # State management transition strategy
│   └── LESSONS_LEARNED.md            # Lessons learned from implementation
│
├── ⚙️ .streamlit/                # Streamlit configuration
│   ├── config.toml              # Streamlit app configuration
│   └── secrets.toml.template    # Secrets template (gitignored)
│
└── 🗂️ Legacy/                    # Legacy files (for reference)
    ├── core/                     # Old core modules (migrated to src/services)
    └── utils/                    # Old utils (migrated to src/utils)
```

## Module Descriptions

### Main Application

#### `main.py` 🎯
- **Purpose**: Main Streamlit application entry point
- **Features**: 
  - Application initialization and configuration
  - Session state management with StateManager
  - Multi-page application structure
  - Data upload and validation
  - Tab organization and navigation
- **Key Functions**: 
  - `initialize_session_state()` - Setup StateManager and core components
  - `render_data_upload()` - Data upload form with validation
  - `main()` - Application entry point and layout

### Core Services (`src/services/`)

#### `src/services/data_handler.py` 📊
- **Class**: `DataHandler`
- **Purpose**: Data import, validation, and preprocessing
- **Key Methods**:
  - `load_file()` - Import CSV/XLSX files with caching
  - `_process_data()` - Clean and convert data types
  - `validate_sales_pipeline_data()` - Pipeline-specific validation
  - `get_*_columns()` - Column type getters
  - `get_file_info()` - File metadata retrieval
- **State Integration**: Uses StateManager for data persistence and caching

#### `src/services/filter_manager.py` 🔍
- **Class**: `FilterManager`
- **Purpose**: Dynamic data filtering system
- **Key Methods**:
  - `create_filters()` - Generate filters based on data types
  - `render_filter_ui()` - Streamlit filter UI components with callbacks
  - `apply_filters()` - Apply active filters to data
  - `get_active_filters_summary()` - Filter state summary
  - `clear_all_filters()` - Reset all filter configurations
- **Features**: Widget callbacks, state binding, form validation

#### `src/services/feature_engine.py` ⚙️
- **Class**: `FeatureEngine`
- **Purpose**: Calculate derived features
- **Key Methods**:
  - `register_feature()` - Add new feature calculations
  - `calculate_features()` - Calculate selected features with caching
  - `get_available_features()` - List compatible features
  - `set_active_features()` - Manage active feature selection
- **Built-in Features**:
  - Days in pipeline, time to close, stage progression
  - Win rates, user activity ratings, time in stages
- **Performance**: Caching for expensive calculations

#### `src/services/report_engine.py` 📈
- **Class**: `ReportEngine`
- **Purpose**: Generate reports and visualizations
- **Key Methods**:
  - `generate_report()` - Create charts and tables
  - `get_available_reports()` - List available report types
  - `render_report_ui()` - Report configuration UI
- **Report Types**:
  - Descriptive statistics, histograms, bar charts
  - Scatter plots, line charts, correlation heatmaps
  - Box plots, time series analysis

#### `src/services/outlier_manager.py` 🎯
- **Class**: `OutlierManager`
- **Purpose**: Outlier detection and exclusion
- **Key Methods**:
  - `register_detector()` - Add new detection algorithms
  - `detect_outliers()` - Apply active detectors
  - `create_outlier_settings()` - Configuration UI
  - `render_outlier_ui()` - Outlier management interface
- **Detection Methods**:
  - IQR, Z-Score, Modified Z-Score, Isolation Forest
- **Features**: Sensitivity levels, preview capabilities, exclusion management

#### `src/services/state_manager.py` 🧠
- **Class**: `StateManager`
- **Purpose**: Centralized state management system
- **Key Methods**:
  - `get_state()` / `set_state()` - State access and modification
  - `update_state()` - Partial state updates
  - `clear_state()` - State cleanup
  - `register_extension()` - Component registration
  - `register_validator()` - State validation
  - `register_watcher()` - State change monitoring
  - `get_debug_info()` - Debugging information
- **Features**:
  - Hierarchical state organization
  - State validation and error recovery
  - History tracking and debugging
  - Extension system for components
  - Memory management and optimization

### Utility Modules (`src/utils/`)

#### `src/utils/data_types.py` 🏷️
- **Purpose**: Data type detection and management
- **Key Functions**:
  - `detect_data_type()` - Automatic type detection
  - `convert_to_proper_type()` - Type conversion
  - `calculate_statistics()` - Type-appropriate statistics
- **Types**: Categorical, Numerical, Date, Text, Boolean
- **Features**: Column compatibility checking, statistical calculations

#### `src/utils/export_utils.py` 💾
- **Class**: `ExportManager`
- **Purpose**: Data and chart export functionality
- **Key Methods**:
  - `export_data_to_csv()` - CSV data export
  - `export_chart_as_png/svg()` - High-resolution chart export
  - `create_summary_report()` - Analysis summary generation
- **Features**: Multiple export formats, high-resolution output

#### `src/utils/column_mapping.py` 🔗
- **Purpose**: Column mapping and transformation utilities
- **Key Functions**:
  - `column_mapper()` - Column name mapping
  - Column transformation helpers
- **Features**: Flexible column mapping, transformation utilities

### Pages (`pages/`)

#### `pages/1_filters.py` 🔍
- **Purpose**: Filter management page
- **Features**: Filter configuration, form validation, state management
- **Integration**: Uses FilterManager service

#### `pages/2_features.py` ⚙️
- **Purpose**: Feature engineering page
- **Features**: Feature selection, calculation, caching
- **Integration**: Uses FeatureEngine service

#### `pages/3_outliers.py` 🎯
- **Purpose**: Outlier detection and management page
- **Features**: Detection configuration, preview, exclusion
- **Integration**: Uses OutlierManager service

#### `pages/4_reports.py` 📈
- **Purpose**: Report generation page
- **Features**: Report selection, configuration, visualization
- **Integration**: Uses ReportEngine service

#### `pages/5_export.py` 💾
- **Purpose**: Data and chart export page
- **Features**: Export options, format selection, download
- **Integration**: Uses ExportManager utility

#### `pages/6_data_preview.py` 👀
- **Purpose**: Data preview and exploration page
- **Features**: Data display, column information, statistics
- **Integration**: Uses DataHandler service

### Configuration

#### `config/settings.py` 🔧
- **Purpose**: Application configuration
- **Contains**:
  - UI settings (colors, themes, layout)
  - Data processing parameters
  - Chart styling options
  - Performance tuning parameters
  - File size limits and validation rules

#### `.streamlit/config.toml` ⚙️
- **Purpose**: Streamlit application configuration
- **Contains**:
  - Page configuration
  - Theme settings
  - Performance options
  - Serialization settings

## Data Flow 🔄

1. **Data Import** (`DataHandler`)
   - User uploads CSV/XLSX file through form
   - File validation and size checking
   - Automatic type detection and conversion
   - Data validation and cleaning
   - State storage in StateManager

2. **State Management** (`StateManager`)
   - Centralized state organization
   - Component registration and validation
   - State history tracking
   - Error recovery and debugging

3. **Filtering** (`FilterManager`)
   - Dynamic filter creation based on column types
   - User applies filters through UI with callbacks
   - Filtered dataset maintained in state
   - Form validation and error handling

4. **Feature Engineering** (`FeatureEngine`)
   - User selects features to calculate
   - Features added to filtered dataset with caching
   - Column types updated for new features
   - State integration for feature results

5. **Outlier Management** (`OutlierManager`)
   - User configures outlier detection
   - Multiple algorithms with sensitivity levels
   - Preview and exclusion capabilities
   - Integration with filtering system

6. **Analysis & Reporting** (`ReportEngine`)
   - User selects report type and configuration
   - Charts and tables generated from current dataset
   - Group-by analysis supported
   - State management for report results

7. **Export** (`ExportManager`)
   - Filtered data exported as CSV
   - Charts exported as PNG/SVG
   - Summary reports generated
   - Multiple format options

## Extension Points 🔌

### Adding New Services
1. Create service class in `src/services/`
2. Register with StateManager in `main.py`
3. Add to session state initialization
4. Create corresponding page in `pages/`

### Adding New Features
1. Create calculation function in `FeatureEngine`
2. Register with `register_feature()`
3. Add caching for performance
4. UI automatically detects and offers new feature

### Adding New Report Types
1. Add configuration to `ReportEngine._register_default_reports()`
2. Implement generation function
3. Define axis requirements and compatibility
4. Add state management integration

### Adding New Outlier Detection Methods
1. Implement detection function in `OutlierManager`
2. Register with `register_detector()`
3. Add UI configuration in `create_outlier_settings()`
4. Integrate with state management

### Adding New Data Types
1. Add enum value to `DataType` in `data_types.py`
2. Implement detection logic in `detect_data_type()`
3. Add filter support in `FilterManager`
4. Update statistics calculation

### Adding New Export Formats
1. Extend `ExportManager` with new format methods
2. Update export UI components
3. Add format options to configuration

## Performance Considerations 🚄

- **Memory Management**: Efficient pandas operations, data copying minimized
- **Large Dataset Support**: Chunked processing where applicable
- **Caching**: Streamlit caching for expensive operations
- **UI Responsiveness**: Progress indicators for long operations
- **Optimized Filtering**: Vectorized pandas operations
- **State Management**: Centralized state with memory optimization
- **Anti-flicker**: Protection against UI flickering during updates

## Testing 🧪

### Test Structure
```
tests/
├── services/                     # Service layer tests
│   ├── test_data_handler.py     # DataHandler tests
│   ├── test_filter_manager.py   # FilterManager tests
│   ├── test_feature_engine.py   # FeatureEngine tests
│   ├── test_report_engine.py    # ReportEngine tests
│   ├── test_outlier_manager.py  # OutlierManager tests
│   ├── test_state_manager.py    # StateManager tests
│   └── test_phase*.py           # Phase transition tests
├── pages/                        # Page component tests
│   └── test_filter_ui.py        # Filter UI tests
└── conftest.py                  # Test configuration and fixtures
```

### Testing Strategy
- **Unit Tests**: Component-level testing with mocking
- **Integration Tests**: Service interaction testing
- **State Management Tests**: State consistency and validation
- **UI Tests**: Streamlit component testing
- **Performance Tests**: Memory and speed optimization

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
- State persistence and recovery
- Performance monitoring

## Migration Notes 📝

### From Legacy Structure
- `core/` modules moved to `src/services/`
- `utils/` modules moved to `src/utils/`
- State management centralized in `StateManager`
- Multi-page application structure implemented
- Comprehensive test suite added

### State Management Transition
- Distributed session state → Centralized StateManager
- Manual state tracking → Automated state management
- Basic error handling → Comprehensive error recovery
- Simple debugging → Advanced debugging and monitoring

## Best Practices ✅

### Code Organization
- Modular service architecture
- Clear separation of concerns
- Consistent naming conventions
- Comprehensive documentation

### State Management
- Centralized state organization
- Hierarchical state structure
- State validation and error recovery
- History tracking and debugging

### Performance
- Caching for expensive operations
- Memory optimization
- Anti-flicker protection
- Efficient data processing

### Testing
- Comprehensive test coverage
- Systematic testing approach
- Integration testing
- Performance benchmarking

### Documentation
- Updated project structure
- Clear module descriptions
- Extension point documentation
- Migration guides
