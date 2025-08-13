"""
Comprehensive validation test using controlled dataset.

This test demonstrates that our controlled dataset approach works
and validates the core functionality of the application.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from tests.fixtures.controlled_test_data import (
    create_controlled_dataset, 
    get_expected_results,
    validate_dataset_integrity
)
from src.services.data_handler import DataHandler
from src.services.feature_engine import FeatureEngine
from src.services.report_engine import ReportEngine


class TestComprehensiveValidation:
    """Comprehensive validation of the application using controlled data."""
    
    def test_end_to_end_workflow(self, mock_streamlit):
        """Test the complete workflow from data loading to report generation."""
        
        # Step 1: Create and validate controlled dataset
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        assert validate_dataset_integrity(controlled_data), "Dataset integrity check failed"
        assert len(controlled_data) == expected_results['total_rows']
        assert controlled_data['Id'].nunique() == expected_results['unique_opportunities']
        
        # Step 2: Load data through DataHandler
        data_handler = DataHandler()
        success = data_handler.load_dataframe(controlled_data)
        assert success, "Failed to load controlled dataset"
        
        processed_data = data_handler.get_data()
        assert processed_data is not None, "Processed data is None"
        assert len(processed_data) > 0, "Processed data is empty"
        
        # Step 3: Verify column type detection
        numerical_cols = data_handler.get_numerical_columns()
        categorical_cols = data_handler.get_categorical_columns()
        date_cols = data_handler.get_date_columns()
        
        assert 'SellPrice' in numerical_cols, "SellPrice not detected as numerical"
        assert 'GM%' in numerical_cols, "GM% not detected as numerical"
        assert 'Id' in categorical_cols, "Id not detected as categorical"
        assert 'Owner' in categorical_cols, "Owner not detected as categorical"
        assert 'Snapshot Date' in date_cols, "Snapshot Date not detected as date"
        assert 'Created' in date_cols, "Created not detected as date"
        
        # Step 4: Test feature calculations
        feature_engine = FeatureEngine()
        
        # Test days_in_pipeline calculation
        df_with_days = feature_engine._calculate_days_in_pipeline(processed_data)
        assert 'days_in_pipeline' in df_with_days.columns, "days_in_pipeline not calculated"
        
        # Verify specific calculations
        opp_001_data = df_with_days[df_with_days['Id'] == 'OPP-001']
        if len(opp_001_data) > 0:
            days_in_pipeline = opp_001_data['days_in_pipeline'].iloc[-1]  # Most recent
            assert days_in_pipeline == expected_results['opp_001_days_in_pipeline'], \
                f"OPP-001 days in pipeline mismatch: {days_in_pipeline}"
        
        # Test final_stage calculation
        df_with_final = feature_engine._calculate_final_stage(processed_data)
        assert 'final_stage' in df_with_final.columns, "final_stage not calculated"
        
        # Verify specific final stages
        for opp_id in ['OPP-001', 'OPP-002', 'OPP-003', 'OPP-004']:
            opp_data = df_with_final[df_with_final['Id'] == opp_id]
            if len(opp_data) > 0:
                final_stage = opp_data['final_stage'].iloc[0]
                expected_stage = expected_results[f'{opp_id.lower().replace("-", "_")}_final_stage']
                assert final_stage == expected_stage, \
                    f"{opp_id} final stage mismatch: {final_stage} != {expected_stage}"
        
        # Step 5: Test report generation with known results
        report_engine = ReportEngine()
        
        # Test deduplication accuracy
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_data)
        assert len(deduplicated_df) == expected_results['unique_opportunities'], \
            "Deduplication didn't produce expected number of opportunities"
        
        total_sellprice = deduplicated_df['SellPrice'].sum()
        assert total_sellprice == expected_results['total_sellprice_deduplicated'], \
            f"Total SellPrice mismatch: {total_sellprice} != {expected_results['total_sellprice_deduplicated']}"
        
        # Test bar chart with known results
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_data, config)
        
        # Verify bar chart totals match expected values
        bar_total = sum(fig.data[0].y)
        assert bar_total == expected_results['total_sellprice_deduplicated'], \
            f"Bar chart total mismatch: {bar_total}"
        
        # Verify individual owner values
        expected_owner_values = {
            'Alice': expected_results['alice_deals'] * 150000,  # OPP-001 final value
            'Bob': 0,  # OPP-002 lost
            'Carol': 350000,  # OPP-003 current value
            'David': 500000   # OPP-004 final value
        }
        
        for i, owner in enumerate(fig.data[0].x):
            actual_value = fig.data[0].y[i]
            expected_value = expected_owner_values.get(owner, 0)
            assert actual_value == expected_value, \
                f"Owner {owner} value mismatch: {actual_value} != {expected_value}"
        
        # Test time series with known results
        config = {
            'x_axis': 'Created',
            'y_axis': 'SellPrice', 
            'aggregation': 'sum',
            'time_period': 'M'
        }
        fig, _ = report_engine.generate_time_series(processed_data, config)
        
        # Should have 1 point (all opportunities created in January)
        assert len(fig.data[0].x) == 1, "Time series should have 1 monthly point"
        
        ts_total = sum(fig.data[0].y)
        assert ts_total == expected_results['total_sellprice_deduplicated'], \
            f"Time series total mismatch: {ts_total}"
        
        # Test scatter plot
        config = {'x_axis': 'SellPrice', 'y_axis': 'GM%'}
        fig, _ = report_engine.generate_scatter_plot(processed_data, config)
        
        # Should have 4 points (one per opportunity after deduplication)
        assert len(fig.data[0].x) == expected_results['unique_opportunities'], \
            "Scatter plot should have 4 points"
        
        # Verify we have the expected final values
        x_values = list(fig.data[0].x)
        expected_sellprices = [150000, 0, 350000, 500000]  # Final values
        
        for expected_price in expected_sellprices:
            assert expected_price in x_values, \
                f"Expected SellPrice {expected_price} not found in scatter plot"
    
    def test_data_consistency_across_reports(self, mock_streamlit):
        """Test that all reports use consistent deduplicated data."""
        
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        # Load data
        data_handler = DataHandler()
        data_handler.load_dataframe(controlled_data)
        processed_data = data_handler.get_data()
        
        report_engine = ReportEngine()
        
        # Generate multiple reports and verify they all use the same total
        expected_total = expected_results['total_sellprice_deduplicated']
        
        # Bar chart
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(processed_data, config)
        bar_total = sum(fig.data[0].y)
        assert bar_total == expected_total, f"Bar chart total inconsistent: {bar_total}"
        
        # Time series
        config = {'x_axis': 'Created', 'y_axis': 'SellPrice', 'aggregation': 'sum', 'time_period': 'M'}
        fig, _ = report_engine.generate_time_series(processed_data, config)
        ts_total = sum(fig.data[0].y)
        assert ts_total == expected_total, f"Time series total inconsistent: {ts_total}"
        
        # Line chart
        config = {'x_axis': 'Snapshot Date', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_line_chart(processed_data, config)
        line_total = sum(fig.data[0].y)
        assert line_total == expected_total, f"Line chart total inconsistent: {line_total}"
        
        # All totals should be identical
        assert bar_total == ts_total == line_total == expected_total, \
            "Report totals are inconsistent across different chart types"
    
    def test_owner_performance_accuracy(self, mock_streamlit):
        """Test that owner performance calculations are accurate."""
        
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        # Load data
        data_handler = DataHandler()
        data_handler.load_dataframe(controlled_data)
        processed_data = data_handler.get_data()
        
        report_engine = ReportEngine()
        
        # Get deduplicated data for manual verification
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_data)
        
        # Test each owner's performance
        owners = ['Alice', 'Bob', 'Carol', 'David']
        
        for owner in owners:
            owner_data = deduplicated_df[deduplicated_df['Owner'] == owner]
            
            # Verify deal count
            actual_deals = len(owner_data)
            expected_deals = expected_results[f'{owner.lower()}_deals']
            assert actual_deals == expected_deals, \
                f"{owner} deal count mismatch: {actual_deals} != {expected_deals}"
            
            # Verify won deals count
            actual_won = (owner_data['Stage'] == 'Closed - WON').sum()
            expected_won = expected_results[f'{owner.lower()}_won_deals']
            assert actual_won == expected_won, \
                f"{owner} won deals mismatch: {actual_won} != {expected_won}"
            
            # Verify win rate calculation
            actual_win_rate = (actual_won / actual_deals * 100) if actual_deals > 0 else 0
            expected_win_rate = expected_results[f'{owner.lower()}_win_rate']
            assert abs(actual_win_rate - expected_win_rate) < 0.01, \
                f"{owner} win rate mismatch: {actual_win_rate} != {expected_win_rate}"
    
    def test_business_unit_breakdown_accuracy(self, mock_streamlit):
        """Test that business unit breakdowns are accurate."""
        
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        # Load data
        data_handler = DataHandler()
        data_handler.load_dataframe(controlled_data)
        processed_data = data_handler.get_data()
        
        report_engine = ReportEngine()
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_data)
        
        # Test each business unit
        business_units = ['Enterprise', 'SMB', 'Government']
        
        for bu in business_units:
            bu_data = deduplicated_df[deduplicated_df['BusinessUnit'] == bu]
            
            # Verify deal count
            actual_deals = len(bu_data)
            expected_deals = expected_results[f'{bu.lower()}_deals']
            assert actual_deals == expected_deals, \
                f"{bu} deal count mismatch: {actual_deals} != {expected_deals}"
            
            # Verify total value
            actual_value = bu_data['SellPrice'].sum()
            expected_value = expected_results[f'{bu.lower()}_value']
            assert actual_value == expected_value, \
                f"{bu} value mismatch: {actual_value} != {expected_value}"
    
    def test_statistical_calculations_accuracy(self, mock_streamlit):
        """Test that statistical calculations are accurate."""
        
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        # Load data
        data_handler = DataHandler()
        data_handler.load_dataframe(controlled_data)
        processed_data = data_handler.get_data()
        
        report_engine = ReportEngine()
        deduplicated_df = report_engine._get_most_recent_snapshots(processed_data)
        
        # Test SellPrice statistics
        sellprice_values = deduplicated_df['SellPrice']
        
        assert sellprice_values.min() == expected_results['sellprice_min']
        assert sellprice_values.max() == expected_results['sellprice_max']
        assert sellprice_values.median() == expected_results['sellprice_median']
        assert sellprice_values.mean() == expected_results['avg_sellprice_deduplicated']
        
        # Test GM% statistics  
        gm_values = deduplicated_df['GM%']
        
        assert gm_values.min() == expected_results['gm_min']
        assert gm_values.max() == expected_results['gm_max']
        assert abs(gm_values.mean() - expected_results['gm_avg']) < 0.001
    
    def test_feature_calculation_pipeline(self, mock_streamlit):
        """Test the complete feature calculation pipeline."""
        
        controlled_data = create_controlled_dataset()
        expected_results = get_expected_results()
        
        # Load data
        data_handler = DataHandler()
        data_handler.load_dataframe(controlled_data)
        processed_data = data_handler.get_data()
        
        feature_engine = FeatureEngine()
        
        # Test adding multiple features
        features_to_add = ['days_in_pipeline', 'final_stage', 'starting_stage', 'stage_progression_count']
        
        df_with_features = processed_data.copy()
        for feature in features_to_add:
            df_with_features = feature_engine.add_features(df_with_features, [feature])
            assert feature in df_with_features.columns, f"Feature {feature} not added"
        
        # Verify specific feature calculations for each opportunity
        opportunities = ['OPP-001', 'OPP-002', 'OPP-003', 'OPP-004']
        
        for opp_id in opportunities:
            opp_data = df_with_features[df_with_features['Id'] == opp_id]
            if len(opp_data) > 0:
                # Test days_in_pipeline
                expected_days = expected_results[f'{opp_id.lower().replace("-", "_")}_days_in_pipeline']
                actual_days = opp_data['days_in_pipeline'].iloc[-1]  # Most recent
                assert actual_days == expected_days, \
                    f"{opp_id} days in pipeline: {actual_days} != {expected_days}"
                
                # Test final_stage
                expected_final = expected_results[f'{opp_id.lower().replace("-", "_")}_final_stage']
                actual_final = opp_data['final_stage'].iloc[0]  # Same for all rows of same opportunity
                assert actual_final == expected_final, \
                    f"{opp_id} final stage: {actual_final} != {expected_final}"
                
                # Test starting_stage
                expected_starting = expected_results[f'{opp_id.lower().replace("-", "_")}_starting_stage']
                actual_starting = opp_data['starting_stage'].iloc[0]
                assert actual_starting == expected_starting, \
                    f"{opp_id} starting stage: {actual_starting} != {expected_starting}"
                
                # Test stage_progression_count
                expected_count = expected_results[f'{opp_id.lower().replace("-", "_")}_stage_count']
                actual_count = opp_data['stage_progression_count'].iloc[0]
                assert actual_count == expected_count, \
                    f"{opp_id} stage count: {actual_count} != {expected_count}"
