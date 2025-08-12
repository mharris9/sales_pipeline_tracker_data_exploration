"""
DataHandler class for importing and managing sales pipeline data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from pathlib import Path
from io import StringIO
import logging

logger = logging.getLogger(__name__)

from config.settings import (
    DATE_FORMAT, SNAPSHOT_DATE_COLUMN, ID_COLUMN, 
    SALES_STAGES, MAX_FILE_SIZE_MB
)
from src.utils.data_types import (
    DataType, detect_data_type, convert_to_proper_type, 
    calculate_statistics
)

class DataHandler:
    """
    Handles data import, validation, and basic processing for sales pipeline data.
    """
    
    def __init__(self):
        """Initialize the DataHandler."""
        # Get state manager instance
        if not hasattr(st.session_state, 'state_manager'):
            raise RuntimeError("StateManager not initialized")
        self.state_manager = st.session_state.state_manager
        
        # Initialize state if needed
        if not self.state_manager.get_state('data.data_info'):
            self.state_manager.update_state('data.data_info', {})
        
        # Local cache for performance
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, DataType] = {}
        
    def load_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Load data from a pandas DataFrame (for testing purposes).
        
        Args:
            df: Pandas DataFrame to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or df.empty:
                st.error("DataFrame is empty or None")
                return False
            
            # Store raw data
            self.df_raw = df.copy()
            
            # Validate required columns
            required_columns = [ID_COLUMN, SNAPSHOT_DATE_COLUMN]
            missing_columns = [col for col in required_columns if col not in self.df_raw.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return False
            
            # Update file info
            file_info = {
                'name': 'test_data.csv',
                'size': len(df),
                'type': 'DataFrame',
                'columns': len(df.columns)
            }
            self.state_manager.update_state('data.data_info', file_info)
            
            # Process the data
            success = self._process_data()
            if not success:
                return False
            
            # Update state
            self.state_manager.update_state('data.current_df', self.df_processed)
            self.state_manager.update_state('data.data_loaded', True)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading DataFrame: {str(e)}")
            return False
    
    @st.cache_data(ttl=3600, show_spinner="Loading data...")
    def _load_file_cached(_uploaded_file) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Cached version of file loading for performance.
        
        Args:
            _uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (success, dataframe, file_info)
        """
        try:
            # Check file size
            if _uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, {'error': f'File too large. Maximum size is {MAX_FILE_SIZE_MB}MB'}
            
            # Read file based on type
            if _uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(_uploaded_file)
            elif _uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(_uploaded_file)
            else:
                return False, None, {'error': 'Unsupported file format. Please upload CSV or Excel file.'}
            
            # Validate required columns
            required_columns = [ID_COLUMN, SNAPSHOT_DATE_COLUMN]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, None, {'error': f'Missing required columns: {", ".join(missing_columns)}'}
            
            # Create file info
            file_info = {
                'name': _uploaded_file.name,
                'size': _uploaded_file.size,
                'type': _uploaded_file.type,
                'columns': len(df.columns)
            }
            
            return True, df, file_info
            
        except Exception as e:
            return False, None, {'error': f'Error reading file: {str(e)}'}

    def load_file(self, uploaded_file) -> bool:
        """
        Load data from uploaded CSV or XLSX file with caching and validation.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use cached file loading for performance
            success, df, file_info = self._load_file_cached(uploaded_file)
            
            if not success:
                error_msg = file_info.get('error', 'Unknown error occurred')
                st.toast(f"❌ {error_msg}", icon="❌")
                return False
            
            # Store raw data
            self.df_raw = df.copy()
            
            # Update file info in state
            self.state_manager.update_state('data.data_info', file_info)
            
            # Process the data
            success = self._process_data()
            if not success:
                return False
            
            # Update state
            self.state_manager.update_state('data.current_df', self.df_processed)
            self.state_manager.update_state('data.data_loaded', True)
            
            # Show success message with toast
            st.toast(f"✅ Successfully loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns", icon="✅")
            return True
            
        except Exception as e:
            st.toast(f"❌ Error loading file: {str(e)}", icon="❌")
            return False
    
    def _process_data(self) -> bool:
        """
        Process the raw data: detect types, convert formats, handle missing data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.df_raw is None:
                return False
            
            # Create a copy for processing
            self.df_processed = self.df_raw.copy()
            
            # Detect and store column types
            self.column_types = {}
            column_info = {}
            
            for column in self.df_processed.columns:
                # Detect data type
                data_type = detect_data_type(self.df_processed[column], column)
                self.column_types[column] = data_type
                
                # Convert to proper type first
                try:
                    self.df_processed[column] = convert_to_proper_type(
                        self.df_processed[column], 
                        data_type,
                        DATE_FORMAT if column == SNAPSHOT_DATE_COLUMN else None
                    )
                except Exception as e:
                    logger.warning(f"Error converting column {column}: {str(e)}")
                
                # Calculate column statistics
                stats = calculate_statistics(self.df_processed[column], data_type)
                # Convert numpy types to Python types for serialization
                serializable_stats = {}
                for k, v in stats.items():
                    if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                        serializable_stats[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        serializable_stats[k] = float(v)
                    elif isinstance(v, pd.Timestamp):
                        serializable_stats[k] = v.isoformat()
                    else:
                        serializable_stats[k] = v
                
                column_info[column] = {
                    'type': data_type.value,  # Store enum value instead of enum
                    **serializable_stats
                }
            
            # Update state
            self.state_manager.update_state('data.column_types', self.column_types)
            self.state_manager.update_state('data.column_info', column_info)
            
            # Update state
            self.state_manager.update_state('data.current_df', self.df_processed)
            self.state_manager.update_state('data.data_loaded', True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return False
    
    def get_data(self, processed: bool = True) -> Optional[pd.DataFrame]:
        """
        Get the loaded data.
        
        Args:
            processed: If True, return processed data; if False, return raw data
            
        Returns:
            DataFrame or None if no data is loaded
        """
        if processed:
            return self.state_manager.get_state('data.current_df')
        else:
            return self.df_raw.copy() if self.df_raw is not None else None
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all columns.
        
        Returns:
            Dictionary with column information including type and statistics
        """
        return self.state_manager.get_state('data.column_info', {})
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns that exist in the current dataframe."""
        df = self.state_manager.get_state('data.current_df')
        if df is None:
            return []
        
        column_types = self.state_manager.get_state('data.column_types', {})
        current_columns = set(df.columns)
        return [
            col for col, dtype in column_types.items() 
            if dtype == DataType.CATEGORICAL and col in current_columns
        ]
    
    def get_numerical_columns(self) -> List[str]:
        """Get list of numerical columns that exist in the current dataframe."""
        df = self.state_manager.get_state('data.current_df')
        if df is None:
            return []
        
        column_types = self.state_manager.get_state('data.column_types', {})
        current_columns = set(df.columns)
        return [
            col for col, dtype in column_types.items() 
            if dtype == DataType.NUMERICAL and col in current_columns
        ]
    
    def get_date_columns(self) -> List[str]:
        """Get list of date columns that exist in the current dataframe."""
        df = self.state_manager.get_state('data.current_df')
        if df is None:
            return []
        
        column_types = self.state_manager.get_state('data.column_types', {})
        current_columns = set(df.columns)
        return [
            col for col, dtype in column_types.items() 
            if dtype == DataType.DATE and col in current_columns
        ]
    
    def get_text_columns(self) -> List[str]:
        """Get list of text columns that exist in the current dataframe."""
        df = self.state_manager.get_state('data.current_df')
        if df is None:
            return []
        
        column_types = self.state_manager.get_state('data.column_types', {})
        current_columns = set(df.columns)
        return [
            col for col, dtype in column_types.items() 
            if dtype == DataType.TEXT and col in current_columns
        ]
    
    def validate_sales_pipeline_data(self) -> Dict[str, Any]:
        """
        Validate that the data looks like sales pipeline data.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        df = self.state_manager.get_state('data.current_df')
        if df is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No data loaded")
            return validation_results
        
        # Check for ID column
        if ID_COLUMN not in df.columns:
            validation_results['warnings'].append(f"No '{ID_COLUMN}' column found")
        
        # Check for Snapshot Date column
        if SNAPSHOT_DATE_COLUMN not in df.columns:
            validation_results['warnings'].append(f"No '{SNAPSHOT_DATE_COLUMN}' column found")
        
        # Check for Stage column
        stage_columns = [col for col in df.columns if 'stage' in col.lower()]
        if not stage_columns:
            validation_results['warnings'].append("No 'Stage' column found")
        else:
            # Check if stage values match expected values
            stage_col = stage_columns[0]
            unique_stages = set(df[stage_col].dropna().unique())
            expected_stages = set(SALES_STAGES)
            
            unexpected_stages = unique_stages - expected_stages
            if unexpected_stages:
                validation_results['suggestions'].append(
                    f"Unexpected stage values found: {', '.join(unexpected_stages)}"
                )
        
        # Check for duplicate handling capability
        if ID_COLUMN in df.columns and SNAPSHOT_DATE_COLUMN in df.columns:
            duplicate_ids = df[ID_COLUMN].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['suggestions'].append(
                    f"Found {duplicate_ids} duplicate IDs - this is expected for pipeline snapshots"
                )
        
        return validation_results
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the loaded file."""
        return self.state_manager.get_state('data.data_info', {})
    
    def is_data_loaded(self) -> bool:
        """Check if data is loaded."""
        return self.state_manager.get_state('data.data_loaded', False)