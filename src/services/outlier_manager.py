"""
OutlierManager class for detecting and handling outliers in data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import streamlit as st
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

from src.utils.data_types import DataType

logger = logging.getLogger(__name__)

class OutlierManager:
    """
    Manages outlier detection and exclusion for data analysis.
    """
    
    def __init__(self, state_manager=None):
        """Initialize the OutlierManager."""
        # Get state manager instance
        if state_manager is not None:
            self.state_manager = state_manager
        elif hasattr(st.session_state, 'state_manager'):
            self.state_manager = st.session_state.state_manager
        else:
            # Create a temporary state manager for testing
            from src.services.state_manager import StateManager
            self.state_manager = StateManager()
        
        # Store detector functions separately (not in state)
        self.detector_functions: Dict[str, Callable] = {}
        self.detection_methods = {
            'iqr': 'Interquartile Range (IQR)',
            'z_score': 'Z-Score',
            'modified_z_score': 'Modified Z-Score (MAD)',
            'isolation_forest': 'Isolation Forest'
        }
        self.sensitivity_levels = {
            'conservative': {'iqr_multiplier': 3.0, 'z_threshold': 3.5, 'mad_threshold': 3.5, 'contamination': 0.05},
            'moderate': {'iqr_multiplier': 2.0, 'z_threshold': 3.0, 'mad_threshold': 3.0, 'contamination': 0.1},
            'aggressive': {'iqr_multiplier': 1.5, 'z_threshold': 2.5, 'mad_threshold': 2.5, 'contamination': 0.15},
            'very_aggressive': {'iqr_multiplier': 1.2, 'z_threshold': 2.0, 'mad_threshold': 2.0, 'contamination': 0.2}
        }
        
        # Initialize state if needed
        if not self.state_manager.get_state('outlier_configs'):
            self.state_manager.set_state('outlier_configs', {})
        if not self.state_manager.get_state('outlier_active'):
            self.state_manager.set_state('outlier_active', {})
        if not self.state_manager.get_state('outlier_results'):
            self.state_manager.set_state('outlier_results', {})
        
        # Register test detectors
        self._register_test_detectors()
        
        logger.info("OutlierManager initialized")
    
    def _register_test_detectors(self) -> None:
        """Register test outlier detectors."""
        # Z-score detector for numerical values
        self.register_detector(
            name="value_zscore",
            description="Detect outliers using z-score method",
            requirements=["value"],
            data_type=DataType.NUMERICAL,
            method="zscore",
            function=self.detect_outliers_z_score,
            source_column="value"
        )
        
        # Range detector for dates
        self.register_detector(
            name="date_range",
            description="Detect outliers using date range",
            requirements=["date"],
            data_type=DataType.DATE,
            method="range",
            function=self.detect_outliers_iqr,
            source_column="date"
        )
        
        logger.debug("Test detectors registered")
    
    def register_detector(self, name: str, description: str, requirements: List[str],
                         data_type: DataType, method: str, function: Callable,
                         source_column: str) -> None:
        """
        Register a new outlier detector.
        
        Args:
            name: Detector name
            description: Detector description
            requirements: Required columns for the detector
            data_type: Expected data type to analyze
            method: Detection method name
            function: Function to detect outliers
            source_column: Column to analyze
        """
        # Create detector config
        detector_config = {
            'name': name,
            'description': description,
            'requirements': requirements,
            'data_type': data_type.value,  # Store as string for serialization
            'method': method,
            'source_column': source_column,
            'type': data_type.value  # For compatibility with test expectations
        }
        
        # Store function separately
        self.detector_functions[name] = function
        
        # Update state
        self.state_manager.set_state(f'outlier_configs/{name}', detector_config)
        self.state_manager.set_state(f'outlier_active/{name}', False)
        
        logger.debug("Detector registered: %s", name)
    
    def set_active_detectors(self, detectors: List[str]) -> None:
        """Set which detectors are active."""
        # Get all detector names
        detector_configs = self.state_manager.get_state('outlier_configs', {})
        
        # Reset all detectors
        for name in detector_configs:
            self.state_manager.set_state(f'outlier_active/{name}', False)
        
        # Activate selected detectors
        for detector in detectors:
            if detector in detector_configs:
                self.state_manager.set_state(f'outlier_active/{detector}', True)
        
        logger.debug("Active detectors updated: %s", detectors)
    
    def get_active_detectors(self) -> List[str]:
        """Get list of active detectors."""
        outlier_active = self.state_manager.get_state('outlier_active', {})
        return [name for name, active in outlier_active.items() if active]

    def get_available_detectors(self) -> List[Dict[str, Any]]:
        """Get list of all available detectors with their configurations."""
        outlier_configs = self.state_manager.get_state('outlier_configs', {})
        outlier_active = self.state_manager.get_state('outlier_active', {})
        
        detectors = []
        for name, config in outlier_configs.items():
            detectors.append({
                'name': name,
                'description': config.get('description', name),
                'active': outlier_active.get(name, False)
            })
        
        return detectors

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using all active detectors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outlier detection results
        """
        if df.empty:
            return df
        
        df_with_outliers = df.copy()
        active_detectors = self.get_active_detectors()
        detector_configs = self.state_manager.get_state('outlier_configs', {})
        
        for detector_name in active_detectors:
            if detector_name not in detector_configs:
                logger.warning("Detector '%s' not found", detector_name)
                continue
            
            try:
                # Get detector configuration
                config = detector_configs[detector_name]
                source_column = config['source_column']
                
                # Check if source column exists
                if source_column not in df.columns:
                    logger.warning("Source column '%s' not found for detector '%s'", source_column, detector_name)
                    continue
                
                # Get detector function
                detector_function = self.detector_functions[detector_name]
                
                # Detect outliers
                if detector_name == 'value_zscore':
                    # Calculate z-scores
                    series = df[source_column]
                    z_scores = np.abs((series - series.mean()) / series.std())
                    threshold = 2.0  # Standard threshold for z-score method
                    outliers = z_scores > threshold
                    
                    # Store results with z-scores
                    results = {
                        'outlier_indices': list(df[outliers].index),
                        'outlier_count': int(outliers.sum()),
                        'outlier_percentage': float((outliers.sum() / len(df)) * 100),
                        'zscore_values': {int(i): float(z) for i, z in z_scores.items()},
                        'threshold': float(threshold)
                    }
                    
                elif detector_name == 'date_range':
                    # Calculate date range
                    series = df[source_column]
                    min_date = series.min()
                    max_date = series.max()
                    date_range = max_date - min_date
                    threshold_days = date_range.days * 0.1  # 10% of total range
                    
                    # Find outliers
                    outliers = (series < min_date + pd.Timedelta(days=threshold_days)) | \
                              (series > max_date - pd.Timedelta(days=threshold_days))
                    
                    # Store results with date info
                    results = {
                        'outlier_indices': list(df[outliers].index),
                        'outlier_count': int(outliers.sum()),
                        'outlier_percentage': float((outliers.sum() / len(df)) * 100),
                        'min_date': min_date.isoformat(),
                        'max_date': max_date.isoformat(),
                        'threshold_days': int(threshold_days)
                    }
                    
                else:
                    # Generic outlier detection
                    outliers = detector_function(df[source_column])
                    results = {
                        'outlier_indices': list(df[outliers].index),
                        'outlier_count': int(outliers.sum()),
                        'outlier_percentage': float((outliers.sum() / len(df)) * 100)
                    }
                
                self.state_manager.set_state(f'outlier_results/{detector_name}', results)
                logger.debug("Outliers detected: %s", detector_name)
                
            except Exception as e:
                logger.error("Error detecting outliers with '%s': %s", detector_name, str(e))
        
        return df_with_outliers
    
    def detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using Interquartile Range method.
        
        Args:
            series: Pandas series to analyze
            multiplier: IQR multiplier for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_outliers_z_score(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Pandas series to analyze
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return z_scores > threshold
    
    def detect_outliers_modified_z_score(self, series: pd.Series, threshold: float = 3.5) -> np.ndarray:
        """
        Detect outliers using Modified Z-score (Median Absolute Deviation).
        
        Args:
            series: Pandas series to analyze
            threshold: Modified Z-score threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            # If MAD is 0, use standard deviation as fallback
            mad = series.std()
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def detect_outliers_isolation_forest(self, series: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """
        Detect outliers using Isolation Forest method.
        
        Args:
            series: Pandas series to analyze
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean array indicating outliers
        """
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Remove NaN values for training
        mask_valid = ~np.isnan(X.flatten())
        if mask_valid.sum() < 10:  # Need minimum samples
            return np.zeros(len(series), dtype=bool)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict on all data (including NaN)
        outliers = np.zeros(len(series), dtype=bool)
        if mask_valid.sum() > 0:
            iso_forest.fit(X[mask_valid])
            predictions = iso_forest.predict(X)
            outliers = predictions == -1
        
        return outliers
    
    def detect_outliers_column(self, df: pd.DataFrame, column: str, 
                              method: str = 'iqr', 
                              sensitivity: str = 'moderate') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers in a specific column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to analyze
            method: Detection method to use
            sensitivity: Sensitivity level
            
        Returns:
            Tuple of (outlier_mask, detection_info)
        """
        if column not in df.columns:
            return np.zeros(len(df), dtype=bool), {}
        
        series = df[column].copy()
        
        # Only work with numeric data
        if not pd.api.types.is_numeric_dtype(series):
            return np.zeros(len(df), dtype=bool), {}
        
        # Remove NaN values for detection
        valid_mask = series.notna()
        if valid_mask.sum() < 10:  # Need minimum samples
            return np.zeros(len(df), dtype=bool), {}
        
        # Get sensitivity parameters
        params = self.sensitivity_levels[sensitivity]
        
        # Detect outliers based on method
        outliers = np.zeros(len(df), dtype=bool)
        detection_info = {
            'method': method,
            'sensitivity': sensitivity,
            'column': column,
            'total_values': len(series),
            'valid_values': valid_mask.sum(),
            'parameters': {}
        }
        
        try:
            if method == 'iqr':
                multiplier = params['iqr_multiplier']
                outliers[valid_mask] = self.detect_outliers_iqr(series[valid_mask], multiplier)
                detection_info['parameters'] = {'iqr_multiplier': multiplier}
                
                # Add bounds info
                Q1 = series[valid_mask].quantile(0.25)
                Q3 = series[valid_mask].quantile(0.75)
                IQR = Q3 - Q1
                detection_info['bounds'] = {
                    'lower': Q1 - multiplier * IQR,
                    'upper': Q3 + multiplier * IQR,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
                
            elif method == 'z_score':
                threshold = params['z_threshold']
                outliers[valid_mask] = self.detect_outliers_z_score(series[valid_mask], threshold)
                detection_info['parameters'] = {'z_threshold': threshold}
                detection_info['bounds'] = {
                    'mean': series[valid_mask].mean(),
                    'std': series[valid_mask].std(),
                    'threshold': threshold
                }
                
            elif method == 'modified_z_score':
                threshold = params['mad_threshold']
                outliers[valid_mask] = self.detect_outliers_modified_z_score(series[valid_mask], threshold)
                detection_info['parameters'] = {'mad_threshold': threshold}
                detection_info['bounds'] = {
                    'median': series[valid_mask].median(),
                    'mad': np.median(np.abs(series[valid_mask] - series[valid_mask].median())),
                    'threshold': threshold
                }
                
            elif method == 'isolation_forest':
                contamination = params['contamination']
                outliers[valid_mask] = self.detect_outliers_isolation_forest(series[valid_mask], contamination)
                detection_info['parameters'] = {'contamination': contamination}
                
        except Exception as e:
            st.warning(f"Error detecting outliers in column '{column}': {str(e)}")
            return np.zeros(len(df), dtype=bool), detection_info
        
        detection_info['outlier_count'] = outliers.sum()
        detection_info['outlier_percentage'] = (outliers.sum() / len(df)) * 100
        
        return outliers, detection_info
    
    def detect_outliers_multiple_columns(self, df: pd.DataFrame, 
                                       columns: List[str],
                                       method: str = 'iqr',
                                       sensitivity: str = 'moderate',
                                       combination_method: str = 'any') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers across multiple columns.
        
        Args:
            df: DataFrame containing the data
            columns: List of column names to analyze
            method: Detection method to use
            sensitivity: Sensitivity level
            combination_method: How to combine outliers ('any', 'all', 'majority')
            
        Returns:
            Tuple of (combined_outlier_mask, detection_info)
        """
        column_outliers = {}
        detection_info = {
            'method': method,
            'sensitivity': sensitivity,
            'columns': columns,
            'combination_method': combination_method,
            'column_results': {}
        }
        
        # Detect outliers for each column
        for column in columns:
            outliers, col_info = self.detect_outliers_column(df, column, method, sensitivity)
            column_outliers[column] = outliers
            detection_info['column_results'][column] = col_info
        
        if not column_outliers:
            return np.zeros(len(df), dtype=bool), detection_info
        
        # Combine outlier masks
        outlier_arrays = list(column_outliers.values())
        
        if combination_method == 'any':
            # Row is outlier if it's an outlier in ANY column
            combined_outliers = np.any(outlier_arrays, axis=0)
        elif combination_method == 'all':
            # Row is outlier if it's an outlier in ALL columns
            combined_outliers = np.all(outlier_arrays, axis=0)
        elif combination_method == 'majority':
            # Row is outlier if it's an outlier in majority of columns
            outlier_sum = np.sum(outlier_arrays, axis=0)
            combined_outliers = outlier_sum > (len(columns) / 2)
        else:
            combined_outliers = np.any(outlier_arrays, axis=0)
        
        detection_info['total_outliers'] = combined_outliers.sum()
        detection_info['outlier_percentage'] = (combined_outliers.sum() / len(df)) * 100
        
        return combined_outliers, detection_info
    
    def create_outlier_settings(self, df: pd.DataFrame, column_types: Dict[str, DataType]) -> None:
        """
        Create outlier detection settings for all numerical columns.
        
        Args:
            df: DataFrame to analyze
            column_types: Dictionary mapping column names to data types
        """
        outlier_configs = {}
        outlier_active = {}
        
        # Only create settings for numerical columns
        for column, data_type in column_types.items():
            if data_type == DataType.NUMERICAL and column in df.columns:
                outlier_configs[column] = {
                    'method': 'iqr',
                    'sensitivity': 'moderate',
                    'detection_info': None
                }
                outlier_active[column] = False
        
        # Update state
        self.state_manager.set_state('outlier_configs', outlier_configs)
        self.state_manager.set_state('outlier_active', outlier_active)
        self.state_manager.set_state('outlier_results', {})
    
    def apply_outlier_exclusion(self, df: pd.DataFrame, 
                               selected_columns: List[str] = None,
                               combination_method: str = 'any') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply outlier exclusion to the DataFrame.
        
        Args:
            df: DataFrame to filter
            selected_columns: Columns to consider for outlier detection
            combination_method: How to combine outliers from multiple columns
            
        Returns:
            Tuple of (filtered_dataframe, exclusion_info)
        """
        if not selected_columns:
            # Use all enabled columns
            selected_columns = [
                col for col, settings in self.outlier_settings.items()
                if settings.get('enabled', False)
            ]
        
        if not selected_columns:
            return df.copy(), {'outliers_excluded': False, 'message': 'No outlier detection enabled'}
        
        # Get the first enabled column's settings for method and sensitivity
        first_col = selected_columns[0]
        method = self.outlier_settings[first_col]['method']
        sensitivity = self.outlier_settings[first_col]['sensitivity']
        
        # Detect outliers
        outlier_mask, detection_info = self.detect_outliers_multiple_columns(
            df, selected_columns, method, sensitivity, combination_method
        )
        
        # Filter DataFrame
        filtered_df = df[~outlier_mask].copy()
        
        exclusion_info = {
            'outliers_excluded': True,
            'original_rows': len(df),
            'filtered_rows': len(filtered_df),
            'excluded_rows': outlier_mask.sum(),
            'exclusion_percentage': (outlier_mask.sum() / len(df)) * 100,
            'detection_info': detection_info,
            'excluded_columns': selected_columns,
            'combination_method': combination_method
        }
        
        return filtered_df, exclusion_info
    
    def get_outlier_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of outlier detection settings and results.
        
        Args:
            df: DataFrame being analyzed
            
        Returns:
            Dictionary with outlier summary information
        """
        enabled_columns = [
            col for col, settings in self.outlier_settings.items()
            if settings.get('enabled', False)
        ]
        
        summary = {
            'total_columns': len(self.outlier_settings),
            'enabled_columns': enabled_columns,
            'enabled_count': len(enabled_columns),
            'methods_used': {},
            'sensitivity_levels': {}
        }
        
        # Count methods and sensitivity levels
        for col in enabled_columns:
            settings = self.outlier_settings[col]
            method = settings['method']
            sensitivity = settings['sensitivity']
            
            summary['methods_used'][method] = summary['methods_used'].get(method, 0) + 1
            summary['sensitivity_levels'][sensitivity] = summary['sensitivity_levels'].get(sensitivity, 0) + 1
        
        return summary
    
    def render_outlier_ui(self, df: pd.DataFrame, column_types: Dict[str, DataType]) -> Dict[str, Any]:
        """
        Render the outlier detection UI.
        
        Args:
            df: DataFrame being analyzed
            column_types: Dictionary mapping column names to data types
            
        Returns:
            Dictionary with outlier exclusion settings
        """
        # Create settings if not exists
        outlier_configs = self.state_manager.get_state('outlier_configs')
        if not outlier_configs:
            self.create_outlier_settings(df, column_types)
            outlier_configs = self.state_manager.get_state('outlier_configs')
        
        st.subheader("🎯 Outlier Detection & Exclusion")
        
        # Get numerical columns
        numerical_columns = [
            col for col, dtype in column_types.items()
            if dtype == DataType.NUMERICAL and col in df.columns
        ]
        
        if not numerical_columns:
            st.info("No numerical columns available for outlier detection.")
            return {'outliers_enabled': False}
        
        # Global outlier settings
        col1, col2 = st.columns(2)
        
        outlier_active = self.state_manager.get_state('outlier_active', {})
        with col1:
            global_enable = st.checkbox(
                "Enable Outlier Exclusion",
                value=any(outlier_active.values()),
                help="Enable outlier detection and exclusion for analysis"
            )
        
        with col2:
            combination_method = st.selectbox(
                "Multi-column Combination",
                options=['any', 'all', 'majority'],
                format_func=lambda x: {
                    'any': 'Any column (most inclusive)',
                    'all': 'All columns (most restrictive)', 
                    'majority': 'Majority of columns'
                }[x],
                help="How to combine outliers when multiple columns are selected"
            )
        
        if not global_enable:
            # Disable all columns
            for col in outlier_configs:
                self.state_manager.set_state(f'outlier_active/{col}', False)
            return {'outliers_enabled': False}
        
        # Column-specific settings
        st.write("**Column-specific Settings:**")
        
        # Create columns for layout
        n_cols = min(3, len(numerical_columns))
        cols = st.columns(n_cols)
        
        for i, column in enumerate(numerical_columns):
            with cols[i % n_cols]:
                with st.expander(f"📊 {column}", expanded=False):
                    # Enable checkbox
                    enabled = st.checkbox(
                        f"Include {column}",
                        value=outlier_active.get(column, False),
                        key=f"outlier_enable_{column}"
                    )
                    self.state_manager.set_state(f'outlier_active/{column}', enabled)
                    
                    if enabled:
                        # Detection method
                        method = st.selectbox(
                            "Detection Method",
                            options=list(self.detection_methods.keys()),
                            format_func=lambda x: self.detection_methods[x],
                            index=list(self.detection_methods.keys()).index(
                                outlier_configs[column].get('method', 'iqr')
                            ),
                            key=f"outlier_method_{column}"
                        )
                        outlier_configs[column]['method'] = method
                        
                        # Sensitivity level
                        sensitivity = st.selectbox(
                            "Sensitivity",
                            options=list(self.sensitivity_levels.keys()),
                            format_func=lambda x: x.replace('_', ' ').title(),
                            index=list(self.sensitivity_levels.keys()).index(
                                outlier_configs[column].get('sensitivity', 'moderate')
                            ),
                            key=f"outlier_sensitivity_{column}"
                        )
                        outlier_configs[column]['sensitivity'] = sensitivity
                        
                        # Update state
                        self.state_manager.set_state(f'outlier_configs/{column}', outlier_configs[column])
                        
                        # Show preview
                        if st.button(f"Preview Outliers", key=f"preview_{column}"):
                            outliers, info = self.detect_outliers_column(df, column, method, sensitivity)
                            st.write(f"**Outliers detected:** {outliers.sum()} ({info['outlier_percentage']:.1f}%)")
                            
                            if 'bounds' in info:
                                bounds = info['bounds']
                                if method == 'iqr':
                                    st.write(f"**Valid range:** {bounds['lower']:.2f} to {bounds['upper']:.2f}")
                                elif method in ['z_score', 'modified_z_score']:
                                    st.write(f"**Threshold:** ±{info['parameters'][list(info['parameters'].keys())[0]]}")
        
        # Show global preview
        enabled_columns = [
            col for col, active in outlier_active.items()
            if active
        ]
        
        if enabled_columns:
            st.write("---")
            
            if st.button("🔍 Preview Combined Outlier Exclusion"):
                filtered_df, exclusion_info = self.apply_outlier_exclusion(df, enabled_columns, combination_method)
                
                st.write("**Exclusion Preview:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Rows", exclusion_info['original_rows'])
                
                with col2:
                    st.metric("Excluded Rows", exclusion_info['excluded_rows'])
                
                with col3:
                    st.metric("Exclusion %", f"{exclusion_info['exclusion_percentage']:.1f}%")
                
                # Show per-column breakdown
                st.write("**Per-column Breakdown:**")
                breakdown_data = []
                for col, col_info in exclusion_info['detection_info']['column_results'].items():
                    breakdown_data.append({
                        'Column': col,
                        'Method': col_info['method'].replace('_', ' ').title(),
                        'Sensitivity': col_info['sensitivity'].replace('_', ' ').title(),
                        'Outliers': col_info['outlier_count'],
                        'Percentage': f"{col_info['outlier_percentage']:.1f}%"
                    })
                
                breakdown_df = pd.DataFrame(breakdown_data)
                
                # Configure column widths based on content
                column_config = {}
                for col in breakdown_df.columns:
                    try:
                        col_name_length = len(str(col))
                        
                        if breakdown_df[col].empty:
                            content_length = 0
                        else:
                            content_lengths = breakdown_df[col].astype(str).str.len()
                            content_length = content_lengths.max() if not content_lengths.empty else 0
                            if pd.isna(content_length):
                                content_length = 0
                        
                        max_content_length = max(col_name_length, int(content_length))
                        
                        # Convert to regular Python int to avoid JSON serialization issues
                        width = int(min(max(max_content_length * 8 + 20, 100), 200))
                        column_config[col] = st.column_config.Column(width=width)
                        
                    except Exception as e:
                        # Fallback to default width if calculation fails
                        column_config[col] = st.column_config.Column(width=100)
                
                st.dataframe(breakdown_df, use_container_width=True, column_config=column_config)
        
        return {
            'outliers_enabled': global_enable and len(enabled_columns) > 0,
            'enabled_columns': enabled_columns,
            'combination_method': combination_method
        }
    
    def get_exclusion_note(self, exclusion_info: Dict[str, Any]) -> str:
        """
        Generate a note about outlier exclusion for charts and tables.
        
        Args:
            exclusion_info: Information about outlier exclusion
            
        Returns:
            String note about exclusion
        """
        if not exclusion_info.get('outliers_excluded', False):
            return ""
        
        excluded_rows = exclusion_info['excluded_rows']
        total_rows = exclusion_info['original_rows']
        percentage = exclusion_info['exclusion_percentage']
        columns = exclusion_info['excluded_columns']
        method = exclusion_info['detection_info']['method'].replace('_', ' ').title()
        sensitivity = exclusion_info['detection_info']['sensitivity'].replace('_', ' ').title()
        combination = exclusion_info['combination_method']
        
        note = f"Outliers Excluded: {excluded_rows:,} rows ({percentage:.1f}%) removed using {method} method "
        note += f"with {sensitivity} sensitivity on columns: {', '.join(columns)}. "
        note += f"Combination method: {combination}."
        
        return note
