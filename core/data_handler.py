"""
DataHandler class for importing and managing sales pipeline data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from pathlib import Path
import io

from config.settings import (
    DATE_FORMAT, SNAPSHOT_DATE_COLUMN, ID_COLUMN, 
    SALES_STAGES, MAX_FILE_SIZE_MB
)
from utils.data_types import (
    DataType, detect_data_type, convert_to_proper_type, 
    calculate_statistics
)

class DataHandler:
    """
    Handles data import, validation, and basic processing for sales pipeline data.
    """
    
    def __init__(self):
        """Initialize the DataHandler."""
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, DataType] = {}
        self.column_stats: Dict[str, Dict[str, Any]] = {}
        self.file_info: Dict[str, Any] = {}
        
    def load_file(self, uploaded_file) -> bool:
        """
        Load data from uploaded CSV or XLSX file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check file size
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size ({uploaded_file.size / (1024*1024):.1f} MB) exceeds limit of {MAX_FILE_SIZE_MB} MB")
                return False
            
            # Store file info
            self.file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type
            }
            
            # Read file based on extension
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.csv':
                self.df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            elif file_extension in ['.xlsx', '.xls']:
                self.df_raw = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return False
            
            # Basic validation
            if self.df_raw.empty:
                st.error("The uploaded file is empty.")
                return False
            
            # Process the data
            self._process_data()
            
            st.success(f"Successfully loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def _process_data(self) -> None:
        """
        Process the raw data: detect types, convert formats, handle missing data.
        """
        if self.df_raw is None:
            return
        
        # Create a copy for processing
        self.df_processed = self.df_raw.copy()
        
        # Detect and convert data types
        self._detect_and_convert_types()
        
        # Calculate statistics for each column
        self._calculate_column_statistics()
        
        # Validate required columns
        self._validate_required_columns()
        
        # Handle missing data
        self._handle_missing_data()
    
    def _detect_and_convert_types(self) -> None:
        """Detect and convert data types for all columns."""
        for column in self.df_processed.columns:
            # Detect data type
            detected_type = detect_data_type(self.df_processed[column], column)
            self.column_types[column] = detected_type
            
            # Convert to proper type
            try:
                if detected_type == DataType.DATE:
                    # Special handling for snapshot date column
                    if column.lower() == SNAPSHOT_DATE_COLUMN.lower():
                        self.df_processed[column] = pd.to_datetime(
                            self.df_processed[column], 
                            format=DATE_FORMAT, 
                            errors='coerce'
                        )
                    else:
                        self.df_processed[column] = pd.to_datetime(
                            self.df_processed[column], 
                            errors='coerce'
                        )
                elif detected_type == DataType.NUMERICAL:
                    self.df_processed[column] = pd.to_numeric(
                        self.df_processed[column], 
                        errors='coerce'
                    )
                elif detected_type == DataType.CATEGORICAL:
                    # Keep as object type but clean up
                    self.df_processed[column] = self.df_processed[column].astype(str)
                    self.df_processed[column] = self.df_processed[column].replace('nan', np.nan)
                
            except Exception as e:
                st.warning(f"Could not convert column '{column}' to {detected_type.value}: {str(e)}")
    
    def _calculate_column_statistics(self) -> None:
        """Calculate statistics for each column."""
        for column, data_type in self.column_types.items():
            self.column_stats[column] = calculate_statistics(
                self.df_processed[column], 
                data_type
            )
    
    def _validate_required_columns(self) -> None:
        """Validate that required columns exist."""
        required_columns = [ID_COLUMN, SNAPSHOT_DATE_COLUMN]
        missing_columns = []
        
        for req_col in required_columns:
            # Case-insensitive search for required columns
            found = False
            for col in self.df_processed.columns:
                if col.lower() == req_col.lower():
                    found = True
                    # Rename to standard format if needed
                    if col != req_col:
                        self.df_processed = self.df_processed.rename(columns={col: req_col})
                        self.column_types[req_col] = self.column_types.pop(col)
                        self.column_stats[req_col] = self.column_stats.pop(col)
                    break
            
            if not found:
                missing_columns.append(req_col)
        
        if missing_columns:
            st.warning(f"Missing recommended columns: {', '.join(missing_columns)}")
    
    def _handle_missing_data(self) -> None:
        """Handle missing data appropriately for each column type."""
        for column, data_type in self.column_types.items():
            if data_type == DataType.CATEGORICAL:
                # Fill categorical missing values with 'Unknown'
                self.df_processed[column] = self.df_processed[column].fillna('Unknown')
            elif data_type == DataType.TEXT:
                # Fill text missing values with empty string
                self.df_processed[column] = self.df_processed[column].fillna('')
            # For numerical and date columns, leave NaN values as they are
            # They will be handled appropriately in analysis
    
    def get_data(self, processed: bool = True) -> Optional[pd.DataFrame]:
        """
        Get the loaded data.
        
        Args:
            processed: If True, return processed data; if False, return raw data
            
        Returns:
            DataFrame or None if no data is loaded
        """
        if processed:
            return self.df_processed.copy() if self.df_processed is not None else None
        else:
            return self.df_raw.copy() if self.df_raw is not None else None
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all columns.
        
        Returns:
            Dictionary with column information including type and statistics
        """
        column_info = {}
        for column in self.column_types:
            column_info[column] = {
                'type': self.column_types[column],
                'stats': self.column_stats.get(column, {})
            }
        return column_info
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns."""
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.CATEGORICAL
        ]
    
    def get_numerical_columns(self) -> List[str]:
        """Get list of numerical columns."""
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.NUMERICAL
        ]
    
    def get_date_columns(self) -> List[str]:
        """Get list of date columns."""
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.DATE
        ]
    
    def get_text_columns(self) -> List[str]:
        """Get list of text columns."""
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.TEXT
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
        
        if self.df_processed is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No data loaded")
            return validation_results
        
        # Check for ID column
        if ID_COLUMN not in self.df_processed.columns:
            validation_results['warnings'].append(f"No '{ID_COLUMN}' column found")
        
        # Check for Snapshot Date column
        if SNAPSHOT_DATE_COLUMN not in self.df_processed.columns:
            validation_results['warnings'].append(f"No '{SNAPSHOT_DATE_COLUMN}' column found")
        
        # Check for Stage column
        stage_columns = [col for col in self.df_processed.columns if 'stage' in col.lower()]
        if not stage_columns:
            validation_results['warnings'].append("No 'Stage' column found")
        else:
            # Check if stage values match expected values
            stage_col = stage_columns[0]
            unique_stages = set(self.df_processed[stage_col].dropna().unique())
            expected_stages = set(SALES_STAGES)
            
            unexpected_stages = unique_stages - expected_stages
            if unexpected_stages:
                validation_results['suggestions'].append(
                    f"Unexpected stage values found: {', '.join(unexpected_stages)}"
                )
        
        # Check for duplicate handling capability
        if ID_COLUMN in self.df_processed.columns and SNAPSHOT_DATE_COLUMN in self.df_processed.columns:
            duplicate_ids = self.df_processed[ID_COLUMN].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['suggestions'].append(
                    f"Found {duplicate_ids} duplicate IDs - this is expected for pipeline snapshots"
                )
        
        return validation_results
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the loaded file."""
        return self.file_info.copy()
    
    def is_data_loaded(self) -> bool:
        """Check if data is loaded."""
        return self.df_processed is not None and not self.df_processed.empty

