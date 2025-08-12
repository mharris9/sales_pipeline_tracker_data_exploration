"""
FeatureEngine class for creating derived features from data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import streamlit as st
import logging

from utils.data_types import DataType

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Creates and manages derived features using centralized state management.
    """
    
    def __init__(self):
        """Initialize the FeatureEngine."""
        # Get state manager instance
        if not hasattr(st.session_state, 'state_manager'):
            raise RuntimeError("StateManager not initialized")
        self.state_manager = st.session_state.state_manager
        
        # Store feature functions separately (not in state)
        self.feature_functions: Dict[str, Callable] = {}
        
        # Register test features
        self._register_test_features()
        
        logger.info("FeatureEngine initialized")
    
    def _register_test_features(self) -> None:
        """Register test features."""
        # Category counts feature
        self.register_feature(
            name="category_counts",
            description="Count of values in each category",
            requirements=["category"],
            data_type=DataType.CATEGORICAL,
            function=self._calculate_category_counts,
            group_by_column="category"
        )
        
        # Value stats feature
        self.register_feature(
            name="value_stats",
            description="Statistical metrics for numerical values",
            requirements=["value"],
            data_type=DataType.NUMERICAL,
            function=self._calculate_value_stats,
            group_by_column="value"
        )
        
        # Date trends feature
        self.register_feature(
            name="date_trends",
            description="Trends over time",
            requirements=["date"],
            data_type=DataType.DATE,
            function=self._calculate_date_trends,
            group_by_column="date"
        )
        
        logger.debug("Test features registered")
    
    def register_feature(self, name: str, description: str, requirements: List[str],
                        data_type: DataType, function: Callable, 
                        group_by_column: Optional[str] = None) -> None:
        """
        Register a new feature calculation.
        
        Args:
            name: Feature name
            description: Feature description
            requirements: Required columns for the feature
            data_type: Expected data type of the feature
            function: Function to calculate the feature
            group_by_column: Optional column to group by for calculation
        """
        # Create feature config
        feature_config = {
            'name': name,
            'description': description,
            'requirements': requirements,
            'data_type': data_type.value,  # Store as string for serialization
            'group_by_column': group_by_column,
            'type': data_type.value,  # For compatibility with test expectations
            'source_column': group_by_column  # For compatibility with test expectations
        }
        
        # Store function separately
        self.feature_functions[name] = function
        
        # Update state
        self.state_manager.set_state(f'feature_configs/{name}', feature_config)
        self.state_manager.set_state(f'feature_active/{name}', False)
        
        logger.debug("Feature registered: %s", name)
    
    def get_available_features(self, df_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get features that can be calculated with the available columns.
        
        Args:
            df_columns: Available columns in the DataFrame
            
        Returns:
            Dictionary of available features
        """
        available = {}
        feature_configs = self.state_manager.get_state('feature_configs', {})
        
        for name, info in feature_configs.items():
            # Check if all required columns are available
            requirements_met = all(
                any(col.lower() == req.lower() for col in df_columns) 
                for req in info['requirements']
            )
            
            # Check if group_by_column exists (if specified)
            if info.get('group_by_column'):
                group_col_exists = any(
                    col.lower() == info['group_by_column'].lower() 
                    for col in df_columns
                )
                requirements_met = requirements_met and group_col_exists
            
            if requirements_met:
                available[name] = info
        
        return available
    
    def set_active_features(self, features: List[str]) -> None:
        """Set which features are active."""
        # Get all feature names
        feature_configs = self.state_manager.get_state('feature_configs', {})
        
        # Reset all features
        for name in feature_configs:
            self.state_manager.set_state(f'feature_active/{name}', False)
        
        # Activate selected features
        for feature in features:
            if feature in feature_configs:
                self.state_manager.set_state(f'feature_active/{feature}', True)
        
        logger.debug("Active features updated: %s", features)
    
    def get_active_features(self) -> List[str]:
        """Get list of active features."""
        feature_active = self.state_manager.get_state('feature_active', {})
        return [name for name, active in feature_active.items() if active]
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all active features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with calculated features
        """
        if df.empty:
            return df
        
        df_with_features = df.copy()
        active_features = self.get_active_features()
        available_features = self.get_available_features(df.columns.tolist())
        
        for feature_name in active_features:
            if feature_name not in available_features:
                logger.warning("Feature '%s' cannot be calculated with available columns", feature_name)
                continue
            
            try:
                # Calculate the feature
                feature_function = self.feature_functions[feature_name]
                df_with_features = feature_function(df_with_features)
                logger.debug("Feature calculated: %s", feature_name)
                
            except Exception as e:
                logger.error("Error calculating feature '%s': %s", feature_name, str(e))
        
        return df_with_features
    
    def _calculate_category_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate counts for each category."""
        df = df.copy()
        value_counts = df['category'].value_counts().to_dict()
        self.state_manager.set_state(f'feature_results/category_counts', value_counts)
        return df
    
    def _calculate_value_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical metrics for values."""
        df = df.copy()
        stats = {
            'mean': float(df['value'].mean()),
            'median': float(df['value'].median()),
            'std': float(df['value'].std())
        }
        self.state_manager.set_state(f'feature_results/value_stats', stats)
        return df
    
    def _calculate_date_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trends over time."""
        df = df.copy()
        
        # Convert numpy types to Python types
        daily_counts = df['date'].dt.day.value_counts()
        weekly_counts = df['date'].dt.isocalendar().week.value_counts()
        monthly_counts = df['date'].dt.month.value_counts()
        
        trends = {
            'daily': {int(k): int(v) for k, v in daily_counts.items()},
            'weekly': {int(k): int(v) for k, v in weekly_counts.items()},
            'monthly': {int(k): int(v) for k, v in monthly_counts.items()}
        }
        
        self.state_manager.set_state(f'feature_results/date_trends', trends)
        return df