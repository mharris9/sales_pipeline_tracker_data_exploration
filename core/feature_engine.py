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
            requirements=[ID_COLUMN, SNAPSHOT_DATE_COLUMN, "Stage", "Owner"],
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
        
        # Remove rows with NaN in critical columns
        clean_df = df[[id_col, date_col]].dropna()
        
        if clean_df.empty:
            df['days_in_pipeline'] = np.nan
            return df
        
        try:
            # Calculate min and max dates for each ID
            date_stats = clean_df.groupby(id_col)[date_col].agg(['min', 'max']).reset_index()
            
            # Calculate days, handling any remaining NaN values
            date_stats['days_in_pipeline'] = (date_stats['max'] - date_stats['min']).dt.days
            
            # Replace any infinite or invalid values with NaN
            date_stats['days_in_pipeline'] = date_stats['days_in_pipeline'].replace([np.inf, -np.inf], np.nan)
            
            # Merge back to original DataFrame
            df = df.merge(
                date_stats[[id_col, 'days_in_pipeline']], 
                on=id_col, 
                how='left'
            )
            
        except Exception as e:
            st.warning(f"Error calculating days in pipeline: {str(e)}")
            df['days_in_pipeline'] = np.nan
        
        return df
    
    def _calculate_days_to_close_won(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days from first snapshot to Closed - WON stage."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            df['days_to_close_won'] = np.nan
            return df
        
        try:
            # Remove rows with NaN in critical columns
            clean_df = df[[id_col, date_col, stage_col]].dropna()
            
            if clean_df.empty:
                df['days_to_close_won'] = np.nan
                return df
            
            # Find first snapshot date for each ID
            first_dates = clean_df.groupby(id_col)[date_col].min().reset_index()
            first_dates.columns = [id_col, 'first_date']
            
            # Find Closed - WON date for each ID (handle different naming conventions)
            won_stages = ['Closed - WON', 'Won', 'Closed Won', 'WON']
            won_records = clean_df[clean_df[stage_col].isin(won_stages)].copy()
            
            if not won_records.empty:
                won_dates = won_records.groupby(id_col)[date_col].min().reset_index()
                won_dates.columns = [id_col, 'won_date']
                
                # Merge and calculate days
                time_to_won = first_dates.merge(won_dates, on=id_col, how='inner')
                time_to_won['days_to_close_won'] = (
                    time_to_won['won_date'] - time_to_won['first_date']
                ).dt.days
                
                # Replace any infinite or invalid values with NaN
                time_to_won['days_to_close_won'] = time_to_won['days_to_close_won'].replace([np.inf, -np.inf], np.nan)
                
                # Merge back to original DataFrame
                df = df.merge(
                    time_to_won[[id_col, 'days_to_close_won']], 
                    on=id_col, 
                    how='left'
                )
            else:
                df['days_to_close_won'] = np.nan
                
        except Exception as e:
            st.warning(f"Error calculating days to close won: {str(e)}")
            df['days_to_close_won'] = np.nan
        
        return df
    
    def _calculate_starting_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the first stage recorded for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Remove rows with NaN in critical columns
        clean_df = df[[id_col, date_col, stage_col]].dropna()
        
        if clean_df.empty:
            df['starting_stage'] = np.nan
            return df
        
        try:
            # Find the earliest record for each ID
            earliest_indices = clean_df.groupby(id_col)[date_col].idxmin()
            
            # Handle case where idxmin returns NaN (shouldn't happen with clean data, but safety check)
            valid_indices = earliest_indices.dropna()
            
            if valid_indices.empty:
                df['starting_stage'] = np.nan
                return df
            
            earliest_records = clean_df.loc[valid_indices]
            starting_stages = earliest_records[[id_col, stage_col]].copy()
            starting_stages.columns = [id_col, 'starting_stage']
            
            # Merge back to original DataFrame
            df = df.merge(starting_stages, on=id_col, how='left')
            
        except Exception as e:
            st.warning(f"Error calculating starting stage: {str(e)}")
            df['starting_stage'] = np.nan
        
        return df
    
    def _calculate_final_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the most recent stage for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            return df
        
        # Remove rows with NaN in critical columns
        clean_df = df[[id_col, date_col, stage_col]].dropna()
        
        if clean_df.empty:
            df['final_stage'] = np.nan
            return df
        
        try:
            # Find the latest record for each ID
            latest_indices = clean_df.groupby(id_col)[date_col].idxmax()
            
            # Handle case where idxmax returns NaN
            valid_indices = latest_indices.dropna()
            
            if valid_indices.empty:
                df['final_stage'] = np.nan
                return df
            
            latest_records = clean_df.loc[valid_indices]
            final_stages = latest_records[[id_col, stage_col]].copy()
            final_stages.columns = [id_col, 'final_stage']
            
            # Merge back to original DataFrame
            df = df.merge(final_stages, on=id_col, how='left')
            
        except Exception as e:
            st.warning(f"Error calculating final stage: {str(e)}")
            df['final_stage'] = np.nan
        
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
            df['user_win_rate'] = np.nan
            return df
        
        try:
            # Check if final_stage already exists, if not calculate it
            if 'final_stage' not in df.columns:
                df_temp = self._calculate_final_stage(df.copy())
            else:
                df_temp = df.copy()
            
            # Check if final_stage column was successfully created
            if 'final_stage' not in df_temp.columns:
                st.warning("Could not calculate final stage - required for user win rate calculation")
                df['user_win_rate'] = np.nan
                return df
            
            # Remove rows with NaN in critical columns
            clean_final_stages = df_temp[[id_col, owner_col, 'final_stage']].dropna()
            
            if clean_final_stages.empty:
                df['user_win_rate'] = np.nan
                return df
            
            # Get unique opportunities only
            final_stages = clean_final_stages.drop_duplicates(subset=[id_col])
            
            win_rates = []
            for owner in final_stages[owner_col].unique():
                if pd.isna(owner) or str(owner).strip() == '':
                    continue
                
                owner_opps = final_stages[final_stages[owner_col] == owner]
                
                # Handle different stage naming conventions
                won_stages = ['Closed - WON', 'Won', 'Closed Won', 'WON']
                lost_stages = ['Closed - LOST', 'Lost', 'Closed Lost', 'LOST']
                closed_stages = won_stages + lost_stages
                
                total_closed = len(owner_opps[owner_opps['final_stage'].isin(closed_stages)])
                won_count = len(owner_opps[owner_opps['final_stage'].isin(won_stages)])
                
                win_rate = (won_count / total_closed * 100) if total_closed > 0 else 0
                win_rates.append({owner_col: owner, 'user_win_rate': win_rate})
            
            # Merge back to working DataFrame
            if win_rates:
                win_rate_df = pd.DataFrame(win_rates)
                df_temp = df_temp.merge(win_rate_df, on=owner_col, how='left')
                # Copy the user_win_rate column back to original DataFrame
                df['user_win_rate'] = df_temp['user_win_rate']
            else:
                df['user_win_rate'] = np.nan
                
        except Exception as e:
            st.warning(f"Error calculating user win rate: {str(e)}")
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
            df['user_activity_rating'] = 'Unknown'
            return df
        
        try:
            # Remove rows with NaN in critical columns
            clean_df = df[[id_col, date_col, owner_col]].dropna()
            
            if clean_df.empty:
                df['user_activity_rating'] = 'Unknown'
                return df
            
            # Get unique opportunities per user per month
            df_temp = clean_df.copy()
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
            if len(monthly_activity) > 2:  # Need at least 3 users for terciles
                terciles = monthly_activity['avg_monthly_opps'].quantile([0.33, 0.67]).values
                
                def get_rating(avg_opps):
                    if pd.isna(avg_opps):
                        return 'Unknown'
                    elif avg_opps <= terciles[0]:
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
                
                # Fill any remaining NaN values
                df['user_activity_rating'] = df['user_activity_rating'].fillna('Unknown')
            else:
                df['user_activity_rating'] = 'Unknown'
                
        except Exception as e:
            st.warning(f"Error calculating user activity rating: {str(e)}")
            df['user_activity_rating'] = 'Unknown'
        
        return df
    
    def _calculate_time_in_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time spent in each stage for each opportunity."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not date_col or not stage_col:
            df['days_in_current_stage'] = np.nan
            return df
        
        try:
            # Initialize the column with NaN
            df['days_in_current_stage'] = np.nan
            
            # Remove rows with NaN in critical columns for processing
            clean_df = df[[id_col, date_col, stage_col]].dropna()
            
            if clean_df.empty:
                return df
            
            # Sort by ID and date
            df_sorted = df.sort_values([id_col, date_col])
            
            # Calculate time in current stage for each record
            time_in_stage = []
            
            for opp_id in df_sorted[id_col].unique():
                if pd.isna(opp_id):
                    continue
                    
                opp_data = df_sorted[df_sorted[id_col] == opp_id].copy()
                
                # Skip if no valid data for this opportunity
                if opp_data[[date_col, stage_col]].isna().all(axis=1).all():
                    continue
                
                for i, (idx, row) in enumerate(opp_data.iterrows()):
                    # Skip if critical data is missing
                    if pd.isna(row[date_col]) or pd.isna(row[stage_col]):
                        time_in_stage.append({'index': idx, 'days_in_current_stage': np.nan})
                        continue
                    
                    if i == 0:
                        # First record - time in stage is 0
                        time_in_stage.append({'index': idx, 'days_in_current_stage': 0})
                    else:
                        prev_row = opp_data.iloc[i-1]
                        
                        # Skip if previous row has missing data
                        if pd.isna(prev_row[date_col]) or pd.isna(prev_row[stage_col]):
                            time_in_stage.append({'index': idx, 'days_in_current_stage': np.nan})
                            continue
                        
                        if row[stage_col] == prev_row[stage_col]:
                            # Same stage - calculate days since previous snapshot
                            try:
                                days = (row[date_col] - prev_row[date_col]).days
                                time_in_stage.append({'index': idx, 'days_in_current_stage': max(0, days)})
                            except:
                                time_in_stage.append({'index': idx, 'days_in_current_stage': np.nan})
                        else:
                            # Stage changed - reset to 0
                            time_in_stage.append({'index': idx, 'days_in_current_stage': 0})
            
            # Convert to DataFrame and merge
            if time_in_stage:
                time_df = pd.DataFrame(time_in_stage).set_index('index')
                df.loc[time_df.index, 'days_in_current_stage'] = time_df['days_in_current_stage']
                
        except Exception as e:
            st.warning(f"Error calculating time in stages: {str(e)}")
            df['days_in_current_stage'] = np.nan
        
        return df
    
    def _calculate_opportunity_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate age of opportunity in days from first snapshot."""
        id_col = self._find_column(df, ID_COLUMN)
        date_col = self._find_column(df, SNAPSHOT_DATE_COLUMN)
        
        if not id_col or not date_col:
            df['opportunity_age_days'] = np.nan
            return df
        
        try:
            # Remove rows with NaN in critical columns
            clean_df = df[[id_col, date_col]].dropna()
            
            if clean_df.empty:
                df['opportunity_age_days'] = np.nan
                return df
            
            # Find first date for each opportunity
            first_dates = clean_df.groupby(id_col)[date_col].min().reset_index()
            first_dates.columns = [id_col, 'first_snapshot_date']
            
            # Merge back and calculate age
            df = df.merge(first_dates, on=id_col, how='left')
            
            # Calculate age, handling NaN values
            df['opportunity_age_days'] = (df[date_col] - df['first_snapshot_date']).dt.days
            
            # Replace any infinite or invalid values with NaN
            df['opportunity_age_days'] = df['opportunity_age_days'].replace([np.inf, -np.inf], np.nan)
            
            # Clean up temporary column
            df = df.drop('first_snapshot_date', axis=1)
            
        except Exception as e:
            st.warning(f"Error calculating opportunity age: {str(e)}")
            df['opportunity_age_days'] = np.nan
        
        return df
    
    def _calculate_stage_progression_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate number of different stages the opportunity has been through."""
        id_col = self._find_column(df, ID_COLUMN)
        stage_col = self._find_column(df, "Stage")
        
        if not id_col or not stage_col:
            df['stage_progression_count'] = np.nan
            return df
        
        try:
            # Remove rows with NaN in critical columns
            clean_df = df[[id_col, stage_col]].dropna()
            
            if clean_df.empty:
                df['stage_progression_count'] = np.nan
                return df
            
            # Count unique stages per opportunity
            stage_counts = (
                clean_df.groupby(id_col)[stage_col]
                .nunique()
                .reset_index()
            )
            stage_counts.columns = [id_col, 'stage_progression_count']
            
            # Merge back to original DataFrame
            df = df.merge(stage_counts, on=id_col, how='left')
            
        except Exception as e:
            st.warning(f"Error calculating stage progression count: {str(e)}")
            df['stage_progression_count'] = np.nan
        
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

