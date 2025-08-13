"""
DataHandler class for importing and managing sales pipeline data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from pathlib import Path
from io import StringIO, BytesIO
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
        # Local cache for performance
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, DataType] = {}
        self.column_info: Dict[str, Dict[str, Any]] = {}
        # Maintain separate column_stats for tests that expect it
        self.column_stats: Dict[str, Dict[str, Any]] = {}
        self.data_info: Dict[str, Any] = {}
        # Back-compat alias expected by some tests
        self.file_info: Dict[str, Any] = {}
        self.data_loaded: bool = False
        
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
            
            # Update file info
            self.data_info = {
                'name': 'test_data.csv',
                'size': len(df),
                'type': 'DataFrame',
                'columns': len(df.columns)
            }
            self.file_info = dict(self.data_info)
            
            # Process the data
            success = self._process_data()
            if not success:
                return False
            
            # Update state
            self.data_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading DataFrame: {str(e)}")
            return False
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Loading data...")
    def _load_file_cached(file_name: str, file_bytes: bytes) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Cached version of file loading for performance.
        
        Args:
            file_name: Name of the uploaded file
            file_bytes: Raw bytes of the uploaded file
            
        Returns:
            Tuple of (success, dataframe, file_info)
        """
        try:
            # Check file size
            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, {'error': f'File too large. Maximum size is {MAX_FILE_SIZE_MB}MB'}
            
            # Read file based on type
            if file_name.lower().endswith('.csv'):
                df = pd.read_csv(BytesIO(file_bytes))
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                return False, None, {'error': 'Unsupported file format. Please upload a CSV or Excel file'}
            
            # Basic validation
            if df is None or df.empty:
                return False, None, {'error': 'File is empty or could not be read'}
            
            # Create file info
            file_info = {
                'name': file_name,
                'size': len(file_bytes),
                'type': 'csv' if file_name.lower().endswith('.csv') else 'excel',
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
            # Extract file info and bytes
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            
            # Use cached file loading for performance
            success, df, file_info = DataHandler._load_file_cached(file_name, file_bytes)
            
            if not success:
                error_msg = file_info.get('error', 'Unknown error occurred')
                try:
                    st.toast(f"❌ {error_msg}", icon="❌")
                except:
                    print(f"❌ {error_msg}")
                return False
            
            # Store raw data
            self.df_raw = df.copy()
            
            # Update file info in state
            self.data_info = file_info
            self.file_info = dict(file_info)
            
            # Process the data
            success = self._process_data()
            if not success:
                return False
            
            # Update state
            self.data_loaded = True
            
            # Show success message with toast
            try:
                st.toast(f"✅ Successfully loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns", icon="✅")
            except:
                print(f"✅ Successfully loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns")
            return True
            
        except Exception as e:
            try:
                st.toast(f"❌ Error loading file: {str(e)}", icon="❌")
            except:
                print(f"❌ Error loading file: {str(e)}")
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
            
            # After conversion, compute stats structure expected by tests
            self._calculate_column_statistics()
            # Ensure Arrow compatibility for downstream rendering
            self.df_processed = DataHandler.ensure_arrow_compatible(self.df_processed)
            
            # Update state
            self.column_info = column_info
            
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
            return self.df_processed.copy() if self.df_processed is not None else None
        else:
            return self.df_raw.copy() if self.df_raw is not None else None
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all columns.
        
        Returns:
            Dictionary with column information including type and statistics
        """
        return self.column_info
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns that exist in the current dataframe."""
        if self.df_processed is None or len(self.df_processed) == 0:
            return []
        
        current_columns = set(self.df_processed.columns)
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.CATEGORICAL and col in current_columns
        ]
    
    def get_numerical_columns(self) -> List[str]:
        """Get list of numerical columns that exist in the current dataframe."""
        if self.df_processed is None or len(self.df_processed) == 0:
            return []
        
        current_columns = set(self.df_processed.columns)
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.NUMERICAL and col in current_columns
        ]
    
    def get_date_columns(self) -> List[str]:
        """Get list of date columns that exist in the current dataframe."""
        if self.df_processed is None or len(self.df_processed) == 0:
            return []
        
        current_columns = set(self.df_processed.columns)
        return [
            col for col, dtype in self.column_types.items() 
            if dtype == DataType.DATE and col in current_columns
        ]
    
    def get_text_columns(self) -> List[str]:
        """Get list of text columns that exist in the current dataframe."""
        if self.df_processed is None or len(self.df_processed) == 0:
            return []
        
        current_columns = set(self.df_processed.columns)
        return [
            col for col, dtype in self.column_types.items() 
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
        
        if self.df_processed is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No data loaded")
            return validation_results
        
        # Check for potential ID columns (flexible naming)
        id_columns = [col for col in self.df_processed.columns if any(keyword in col.lower() for keyword in ['id', 'identifier', 'key', 'primary'])]
        if not id_columns:
            validation_results['warnings'].append("No obvious ID column found (look for columns with 'id', 'identifier', 'key', or 'primary' in the name)")
        else:
            validation_results['suggestions'].append(f"Potential ID columns found: {', '.join(id_columns)}")
        
        # Check for potential date columns (flexible naming)
        date_columns = [col for col in self.df_processed.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp'])]
        if not date_columns:
            validation_results['warnings'].append("No obvious date columns found (look for columns with 'date', 'time', 'created', 'updated', 'modified', or 'timestamp' in the name)")
        else:
            validation_results['suggestions'].append(f"Potential date columns found: {', '.join(date_columns)}")
        
        # Check for potential stage/status columns (flexible naming)
        stage_columns = [col for col in self.df_processed.columns if any(keyword in col.lower() for keyword in ['stage', 'status', 'phase', 'state', 'step'])]
        if not stage_columns:
            validation_results['warnings'].append("No obvious stage/status columns found (look for columns with 'stage', 'status', 'phase', 'state', or 'step' in the name)")
        else:
            validation_results['suggestions'].append(f"Potential stage/status columns found: {', '.join(stage_columns)}")
        
        # Check for potential amount/price columns (flexible naming)
        amount_columns = [col for col in self.df_processed.columns if any(keyword in col.lower() for keyword in ['amount', 'price', 'value', 'cost', 'revenue', 'sales'])]
        if not amount_columns:
            validation_results['warnings'].append("No obvious amount/price columns found (look for columns with 'amount', 'price', 'value', 'cost', 'revenue', or 'sales' in the name)")
        else:
            validation_results['suggestions'].append(f"Potential amount/price columns found: {', '.join(amount_columns)}")
        
        # Check for duplicate handling capability if we have ID columns
        if id_columns:
            id_col = id_columns[0]  # Use the first potential ID column
            duplicate_ids = self.df_processed[id_col].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['suggestions'].append(
                    f"Found {duplicate_ids} duplicate values in '{id_col}' - this might be expected for time-series data"
                )
        
        # Check data quality
        total_rows = len(self.df_processed)
        total_columns = len(self.df_processed.columns)
        
        if total_rows == 0:
            validation_results['errors'].append("Dataset is empty")
        elif total_rows < 10:
            validation_results['warnings'].append(f"Dataset has only {total_rows} rows - consider using a larger dataset for meaningful analysis")
        
        if total_columns < 3:
            validation_results['warnings'].append(f"Dataset has only {total_columns} columns - consider including more variables for comprehensive analysis")
        
        # Check for missing data
        missing_data = self.df_processed.isnull().sum().sum()
        if missing_data > 0:
            missing_percentage = (missing_data / (total_rows * total_columns)) * 100
            validation_results['suggestions'].append(f"Dataset contains {missing_data} missing values ({missing_percentage:.1f}% of all data)")
        
        return validation_results
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the loaded file."""
        return self.data_info
    
    def is_data_loaded(self) -> bool:
        """Check if data is loaded."""
        return self.data_loaded

    def get_current_df(self) -> Optional[pd.DataFrame]:
        """
        Get the current processed DataFrame.
        
        Returns:
            Current DataFrame or None if no data is loaded
        """
        return self.df_processed

    def get_column_types(self) -> Dict[str, DataType]:
        """
        Get the detected column types.
        
        Returns:
            Dictionary mapping column names to DataType enums
        """
        return self.column_types

    @staticmethod
    def ensure_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce a DataFrame to be Arrow-friendly for Streamlit rendering.
        - Convert object columns containing datetime-like values to datetime64[ns]
        - Convert numpy datetime64 in object arrays to pandas datetime
        - Leave other columns unchanged
        """
        if df is None or df.empty:
            return df
        result = df.copy()
        for col in result.columns:
            s = result[col]
            # Already datetime – continue
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            # Handle object columns that contain datetime-like values
            if s.dtype == object:
                # Sample a few non-null values to detect datetime-like content
                sample = s.dropna().head(20).tolist()
                has_dt_like = False
                for v in sample:
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        has_dt_like = True
                        break
                    # Fallback: try parsing a couple values cheaply
                    try:
                        pd.to_datetime([v], errors='raise')
                        has_dt_like = True
                        break
                    except Exception:
                        continue
                if has_dt_like:
                    # Convert entire column to datetime; non-parsable entries -> NaT
                    result[col] = pd.to_datetime(s, errors='coerce')
                    continue
            # Numpy datetime64 dtype strings
            if 'datetime64' in str(s.dtype):
                try:
                    result[col] = pd.to_datetime(s, errors='coerce')
                except Exception:
                    pass
        return result

    def _detect_and_convert_types(self) -> None:
        """
        Detect and convert data types for all columns in the current dataframe.
        This method is called after data is loaded to ensure proper type conversion.
        """
        if self.df_processed is None:
            logger.warning("No data available for type detection")
            return
        
        try:
            logger.info("Starting type detection and conversion")
            
            # Process each column
            for column in self.df_processed.columns:
                # Detect data type
                data_type = detect_data_type(self.df_processed[column], column)
                self.column_types[column] = data_type
                
                # Convert to proper type - use flexible date format detection
                try:
                    # Try to detect if this is a date column and use appropriate format
                    date_format = None
                    if data_type == DataType.DATE:
                        # Try common date formats
                        sample_values = self.df_processed[column].dropna().head(10)
                        if len(sample_values) > 0:
                            # Check if values match common date patterns
                            sample_str = str(sample_values.iloc[0])
                            if '/' in sample_str and len(sample_str.split('/')) == 3:
                                date_format = "%m/%d/%Y"  # Common US format
                            elif '-' in sample_str and len(sample_str.split('-')) == 3:
                                date_format = "%Y-%m-%d"  # ISO format
                            elif len(sample_str) == 8 and sample_str.isdigit():
                                date_format = "%Y%m%d"  # YYYYMMDD format
                    
                    self.df_processed[column] = convert_to_proper_type(
                        self.df_processed[column], 
                        data_type,
                        date_format
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
                
                self.column_info[column] = {
                    'type': data_type.value,  # Store enum value instead of enum
                    **serializable_stats
                }
            
            logger.info(f"Type detection and conversion completed for {len(self.df_processed.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error during type detection and conversion: {str(e)}")
            raise

    def _calculate_column_statistics(self) -> None:
        """Calculate and store per-column statistics in self.column_info and column_stats."""
        if self.df_processed is None:
            self.column_info = {}
            self.column_stats = {}
            return
        info: Dict[str, Dict[str, Any]] = {}
        stats_map: Dict[str, Dict[str, Any]] = {}
        for column in self.df_processed.columns:
            data_type = self.column_types.get(column, detect_data_type(self.df_processed[column], column))
            stats = calculate_statistics(self.df_processed[column], data_type)
            # Convert numpy/pandas types to built-in for serialization
            serializable_stats: Dict[str, Any] = {}
            for k, v in stats.items():
                if isinstance(v, (np.integer,)):
                    serializable_stats[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    serializable_stats[k] = float(v)
                elif isinstance(v, pd.Timestamp):
                    serializable_stats[k] = v.isoformat()
                else:
                    serializable_stats[k] = v
            info[column] = {"type": data_type, "stats": serializable_stats}
            stats_map[column] = serializable_stats
        self.column_info = info
        self.column_stats = stats_map