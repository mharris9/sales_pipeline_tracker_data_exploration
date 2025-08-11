"""
FeatureEngine class for creating derived features from sales pipeline data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import streamlit as st
from datetime import datetime

from config.settings import SNAPSHOT_DATE_COLUMN, ID_COLUMN, SALES_STAGES
from utils.data_types import DataType

class FeatureEngine:
    """
    Creates and manages derived features for sales pipeline analysis.
    """
    
    def __init__(self):
        """Initialize the FeatureEngine."""
        self.available_features: Dict[str, Dict[str, Any]] = {}
        self.active_features: Dict[str, bool] = {}
        self.feature_functions: Dict[str, Callable] = {}
        self._register_default_features()
    
    def _register_default_features(self) -> None:
        """Register default feature calculations."""
        
        # Time in pipeline feature
        self.register_feature(
            name="days_in_pipeline",
            description="Days between first and most recent snapshot for each opportunity",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN],
            data_type=DataType.NUMERICAL,
            function=self._calculate_days_in_pipeline
        )
        
        # Time to close won feature
        self.register_feature(
            name="days_to_close_won",
            description="Days from first snapshot to Closed - WON stage",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN, "Stage"],
            data_type=DataType.NUMERICAL,
            function=self._calculate_days_to_close_won
        )
        
        # Starting stage feature
        self.register_feature(
            name="starting_stage",
            description="First stage recorded for each opportunity",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN, "Stage"],
            data_type=DataType.CATEGORICAL,
            function=self._calculate_starting_stage
        )
        
        # Final stage feature
        self.register_feature(
            name="final_stage",
            description="Most recent stage for each opportunity",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN, "Stage"],
            data_type=DataType.CATEGORICAL,
            function=self._calculate_final_stage
        )
        
        # Win rate by user feature
        self.register_feature(
            name="user_win_rate",
            description="Win rate percentage for each user/owner",
            requirements=[ID_COLUMN, "Stage"],
            data_type=DataType.NUMERICAL,
            function=self._calculate_user_win_rate,
            group_by_column="Owner"  # This will be detected automatically
        )
        
        # User activity rating feature
        self.register_feature(
            name="user_activity_rating",
            description="User rating based on monthly opportunity volume (Low/Medium/High)",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN],
            data_type=DataType.CATEGORICAL,
            function=self._calculate_user_activity_rating,
            group_by_column="Owner"  # This will be detected automatically
        )
        
        # Time in each stage feature
        self.register_feature(
            name="time_in_stages",
            description="Days spent in each stage for each opportunity",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN, "Stage"],
            data_type=DataType.NUMERICAL,
            function=self._calculate_time_in_stages
        )
        
        # Opportunity age feature
        self.register_feature(
            name="opportunity_age_days",
            description="Age of opportunity in days from first snapshot",
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN],
            data_type=DataType.NUMERICAL,
            function=self._calculate_opportunity_age
        )
        
        # Stage progression count
        self.register_feature(
            name="stage_progression_count",
            description="Number of different stages the opportunity has been through",
            requirements=[ID_COLUMN, "Stage"],
            data_type=DataType.NUMERICAL,
            function=self._calculate_stage_progression_count
        )
    
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
        self.available_features[name] = {
            'description': description,
            'requirements': requirements,
            'data_type': data_type,
            'group_by_column': group_by_column
        }
        self.feature_functions[name] = function
        self.active_features[name] = False
    
    def get_available_features(self, df_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get features that can be calculated with the available columns.
        
        Args:
            df_columns: Available columns in the DataFrame
            
        Returns:
            Dictionary of available features
        """
        available = {}
        for name, info in self.available_features.items():
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
    
    def add_features(self, df: pd.DataFrame, selected_features: List[str] = None) -> pd.DataFrame:
        """
        Add selected features to the DataFrame.
        
        Args:
            df: Input DataFrame
            selected_features: List of features to add (if None, add all active features)
            
        Returns:
            DataFrame with added features
        """
        if df.empty:
            return df
        
        df_with_features = df.copy()
        
        # Determine which features to add
        features_to_add = selected_features or [
            name for name, active in self.active_features.items() if active
        ]
        
        # Get available features for this DataFrame
        available_features = self.get_available_features(df.columns.tolist())
        
        for feature_name in features_to_add:
            if feature_name not in available_features:
                st.warning(f"Feature '{feature_name}' cannot be calculated with available columns")
                continue
            
            try:
                # Calculate the feature
                feature_function = self.feature_functions[feature_name]
                df_with_features = feature_function(df_with_features)
                
            except Exception as e:
                st.error(f"Error calculating feature '{feature_name}': {str(e)}")
        
        return df_with_features
    
    def _find_column(self, df: pd.DataFrame, target_column: str) -> Optional[str]:
        """Find column by case-insensitive search."""
        for col in df.columns:
            if col.lower() == target_column.lower():
                return col
        return None
    
    def _calculate_days_in_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days between first and most recent snapshot for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        
        if not id_col or not date_col:
            return df
        
        # Calculate min and max dates for each ID
        date_stats = df.groupby(id_col)[date_col].agg(['min', 'max']).reset_index()
        date_stats['days_in_pipeline'] = (date_stats['max'] - date_stats['min']).dt.days
        
        # Merge back to original DataFrame
        df = df.merge(
            date_stats[[id_col, 'days_in_pipeline']], 
            on=id_col, 
            how='left'
        )
        
        return df
    
    def _calculate_days_to_close_won(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days from first snapshot to Closed - WON stage."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Find first snapshot date for each ID
        first_dates = df.groupby(id_col)[date_col].min().reset_index()
        first_dates.columns = [id_col, 'first_date']
        
        # Find Closed - WON date for each ID
        won_records = df[df[stage_col] == "Closed - WON"].copy()
        if not won_records.empty:
            won_dates = won_records.groupby(id_col)[date_col].min().reset_index()
            won_dates.columns = [id_col, 'won_date']
            
            # Merge and calculate days
            time_to_won = first_dates.merge(won_dates, on=id_col, how='inner')
            time_to_won['days_to_close_won'] = (
                time_to_won['won_date'] - time_to_won['first_date']
            ).dt.days
            
            # Merge back to original DataFrame
            df = df.merge(
                time_to_won[[id_col, 'days_to_close_won']], 
                on=id_col, 
                how='left'
            )
        else:
            df['days_to_close_won'] = np.nan
        
        return df
    
    def _calculate_starting_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the first stage recorded for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Find the earliest record for each ID
        earliest_records = df.loc[df.groupby(id_col)[date_col].idxmin()]
        starting_stages = earliest_records[[id_col, stage_col]].copy()
        starting_stages.columns = [id_col, 'starting_stage']
        
        # Merge back to original DataFrame
        df = df.merge(starting_stages, on=id_col, how='left')
        
        return df
    
    def _calculate_final_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the most recent stage for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Find the latest record for each ID
        latest_records = df.loc[df.groupby(id_col)[date_col].idxmax()]
        final_stages = latest_records[[id_col, stage_col]].copy()
        final_stages.columns = [id_col, 'final_stage']
        
        # Merge back to original DataFrame
        df = df.merge(final_stages, on=id_col, how='left')
        
        return df
    
    def _calculate_user_win_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate win rate for each user/owner."""
        id_col = self._find_column(df, ID_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        # Try to find owner/user column
        owner_col = None
        for possible_name in ['Owner', 'User', 'Sales Rep', 'Salesperson', 'Rep']:
            owner_col = self._find_column(df, possible_name)
            if owner_col:
                break
        
        if not id_col or not stage_col or not owner_col:
            return df
        
        # Get final stage for each opportunity
        df_temp = self._calculate_final_stage(df)
        
        # Calculate win rates by owner
        final_stages = df_temp[[id_col, owner_col, 'final_stage']].drop_duplicates(subset=[id_col])
        
        win_rates = []
        for owner in final_stages[owner_col].unique():
            if pd.isna(owner):
                continue
            
            owner_opps = final_stages[final_stages[owner_col] == owner]
            total_closed = len(owner_opps[owner_opps['final_stage'].isin(['Closed - WON', 'Closed - LOST'])])
            won_count = len(owner_opps[owner_opps['final_stage'] == 'Closed - WON'])
            
            win_rate = (won_count / total_closed * 100) if total_closed > 0 else 0
            win_rates.append({owner_col: owner, 'user_win_rate': win_rate})
        
        win_rate_df = pd.DataFrame(win_rates)
        
        # Merge back to original DataFrame
        if not win_rate_df.empty:
            df = df.merge(win_rate_df, on=owner_col, how='left')
        else:
            df['user_win_rate'] = np.nan
        
        return df
    
    def _calculate_user_activity_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate user activity rating based on monthly opportunity volume."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        
        # Try to find owner/user column
        owner_col = None
        for possible_name in ['Owner', 'User', 'Sales Rep', 'Salesperson', 'Rep']:
            owner_col = self._find_column(df, possible_name)
            if owner_col:
                break
        
        if not id_col or not date_col or not owner_col:
            return df
        
        # Get unique opportunities per user per month
        df_temp = df.copy()
        df_temp['year_month'] = df_temp[date_col].dt.to_period('M')
        
        monthly_activity = (
            df_temp.groupby([owner_col, 'year_month'])[id_col]
            .nunique()
            .reset_index()
            .groupby(owner_col)[id_col]
            .mean()
            .reset_index()
        )
        monthly_activity.columns = [owner_col, 'avg_monthly_opps']
        
        # Calculate terciles for rating
        if len(monthly_activity) > 0:
            terciles = monthly_activity['avg_monthly_opps'].quantile([0.33, 0.67]).values
            
            def get_rating(avg_opps):
                if avg_opps <= terciles[0]:
                    return 'Low'
                elif avg_opps <= terciles[1]:
                    return 'Medium'
                else:
                    return 'High'
            
            monthly_activity['user_activity_rating'] = monthly_activity['avg_monthly_opps'].apply(get_rating)
            
            # Merge back to original DataFrame
            df = df.merge(
                monthly_activity[[owner_col, 'user_activity_rating']], 
                on=owner_col, 
                how='left'
            )
        else:
            df['user_activity_rating'] = 'Unknown'
        
        return df
    
    def _calculate_time_in_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time spent in each stage for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Sort by ID and date
        df_sorted = df.sort_values([id_col, date_col])
        
        # Calculate time in current stage for each record
        time_in_stage = []
        
        for opp_id in df_sorted[id_col].unique():
            opp_data = df_sorted[df_sorted[id_col] == opp_id].copy()
            
            for i, (idx, row) in enumerate(opp_data.iterrows()):
                if i == 0:
                    # First record - time in stage is 0
                    time_in_stage.append({'index': idx, 'days_in_current_stage': 0})
                else:
                    prev_row = opp_data.iloc[i-1]
                    if row[stage_col] == prev_row[stage_col]:
                        # Same stage - calculate days since previous snapshot
                        days = (row[date_col] - prev_row[date_col]).days
                        time_in_stage.append({'index': idx, 'days_in_current_stage': days})
                    else:
                        # Stage changed - reset to 0
                        time_in_stage.append({'index': idx, 'days_in_current_stage': 0})
        
        # Convert to DataFrame and merge
        time_df = pd.DataFrame(time_in_stage).set_index('index')
        df.loc[time_df.index, 'days_in_current_stage'] = time_df['days_in_current_stage']
        
        return df
    
    def _calculate_opportunity_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate age of opportunity in days from first snapshot."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        
        if not id_col or not date_col:
            return df
        
        # Find first date for each opportunity
        first_dates = df.groupby(id_col)[date_col].min().reset_index()
        first_dates.columns = [id_col, 'first_snapshot_date']
        
        # Merge back and calculate age
        df = df.merge(first_dates, on=id_col, how='left')
        df['opportunity_age_days'] = (df[date_col] - df['first_snapshot_date']).dt.days
        
        # Clean up temporary column
        df = df.drop('first_snapshot_date', axis=1)
        
        return df
    
    def _calculate_stage_progression_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate number of different stages the opportunity has been through."""
        id_col = self._find_column(df, ID_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not stage_col:
            return df
        
        # Count unique stages per opportunity
        stage_counts = (
            df.groupby(id_col)[stage_col]
            .nunique()
            .reset_index()
        )
        stage_counts.columns = [id_col, 'stage_progression_count']
        
        # Merge back to original DataFrame
        df = df.merge(stage_counts, on=id_col, how='left')
        
        return df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available features."""
        return {
            name: info['description'] 
            for name, info in self.available_features.items()
        }
    
    def set_active_features(self, features: List[str]) -> None:
        """Set which features are active."""
        # Reset all features
        for name in self.active_features:
            self.active_features[name] = False
        
        # Activate selected features
        for feature in features:
            if feature in self.active_features:
                self.active_features[feature] = True
    
    def get_active_features(self) -> List[str]:
        """Get list of active features."""
        return [name for name, active in self.active_features.items() if active]

