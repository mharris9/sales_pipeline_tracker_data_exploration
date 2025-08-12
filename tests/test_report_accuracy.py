"""
Test report generation accuracy using controlled dataset.

This module focuses specifically on testing all report types
to ensure they produce accurate visualizations and calculations.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from tests.fixtures.controlled_test_data import (
    create_controlled_dataset, 
    get_expected_results, 
    get_test_scenarios
)
from core.report_engine import ReportEngine


class TestReportAccuracy:
    """Test all report types for accuracy using controlled dataset."""
    
    @pytest.fixture
    def controlled_data(self):
        """Create the controlled test dataset."""
        return create_controlled_dataset()
    
    @pytest.fixture
    def expected_results(self):
        """Get expected results for all calculations."""
        return get_expected_results()
    
    @pytest.fixture
    def test_scenarios(self):
        """Get test scenarios for report configurations."""
        return get_test_scenarios()
    
    @pytest.fixture
    def report_engine(self):
        """Create ReportEngine for testing."""
        return ReportEngine()
    
    def test_bar_chart_all_aggregations(self, report_engine, controlled_data, mock_streamlit):
        """Test bar chart with different aggregation functions."""
        
        base_config = {
            'x_axis': 'Owner',
            'y_axis': 'SellPrice'
        }
        
        # Test SUM aggregation
        config = {**base_config, 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(controlled_data, config)
        
        # Expected sums (most recent values): Alice=150k, Bob=0, Carol=350k, David=500k
        expected_sums = {'Alice': 150000, 'Bob': 0, 'Carol': 350000, 'David': 500000}
        
        for i, owner in enumerate(fig.data[0].x):
            assert fig.data[0].y[i] == expected_sums[owner], f"Sum mismatch for {owner}"
        
        # Test MEAN aggregation
        config = {**base_config, 'aggregation': 'mean'}
        fig, _ = report_engine.generate_bar_chart(controlled_data, config)
        
        # Expected means (most recent values): same as sums since one value per owner
        for i, owner in enumerate(fig.data[0].x):
            assert fig.data[0].y[i] == expected_sums[owner], f"Mean mismatch for {owner}"
        
        # Test COUNT aggregation
        config = {**base_config, 'aggregation': 'count'}
        fig, _ = report_engine.generate_bar_chart(controlled_data, config)
        
        # Expected counts: 1 deal per owner (after deduplication)
        for i, owner in enumerate(fig.data[0].x):
            assert fig.data[0].y[i] == 1, f"Count mismatch for {owner}"
    
    def test_bar_chart_with_grouping(self, report_engine, controlled_data, mock_streamlit):
        """Test bar chart with group_by_column functionality."""
        
        config = {
            'x_axis': 'BusinessUnit',
            'y_axis': 'SellPrice',
            'aggregation': 'sum',
            'group_by_column': 'Stage'
        }
        
        fig, _ = report_engine.generate_bar_chart(controlled_data, config)
        
        # Should have multiple traces (one per stage)
        assert len(fig.data) > 1, "Should have multiple traces for grouped bar chart"
        
        # Verify we have the expected stages
        stage_names = [trace.name for trace in fig.data]
        expected_stages = ['Won', 'Lost', 'Negotiation']  # Final stages in our data
        
        for stage in expected_stages:
            assert stage in stage_names, f"Missing stage {stage} in grouped bar chart"
    
    def test_time_series_different_periods(self, report_engine, controlled_data, mock_streamlit):
        """Test time series with different time periods."""
        
        base_config = {
            'x_axis': 'Created',
            'y_axis': 'SellPrice',
            'aggregation': 'sum'
        }
        
        # Test Daily aggregation
        config = {**base_config, 'time_period': 'D'}
        fig, _ = report_engine.generate_time_series(controlled_data, config)
        
        # Should have multiple daily points
        assert len(fig.data[0].x) > 1, "Daily time series should have multiple points"
        
        # Total should equal our expected deduplicated sum
        total_value = sum(fig.data[0].y)
        assert total_value == 1000000, "Daily time series total mismatch"
        
        # Test Monthly aggregation
        config = {**base_config, 'time_period': 'M'}
        fig, _ = report_engine.generate_time_series(controlled_data, config)
        
        # Should have 1 monthly point (all deals created in January)
        assert len(fig.data[0].x) == 1, "Monthly time series should have 1 point"
        
        # Total should equal our expected deduplicated sum
        total_value = sum(fig.data[0].y)
        assert total_value == 1000000, "Monthly time series total mismatch"
    
    def test_scatter_plot_different_axes(self, report_engine, controlled_data, mock_streamlit):
        """Test scatter plot with different axis combinations."""
        
        # Test SellPrice vs GM%
        config = {
            'x_axis': 'SellPrice',
            'y_axis': 'GM%'
        }
        
        fig, _ = report_engine.generate_scatter_plot(controlled_data, config)
        
        # Should have 4 points (one per opportunity, deduplicated)
        assert len(fig.data[0].x) == 4
        assert len(fig.data[0].y) == 4
        
        # Verify specific points exist
        x_values = list(fig.data[0].x)
        y_values = list(fig.data[0].y)
        
        # Check for our known final values
        assert 150000 in x_values, "OPP-001 final SellPrice not found"
        assert 0 in x_values, "OPP-002 final SellPrice not found"
        assert 350000 in x_values, "OPP-003 final SellPrice not found"
        assert 500000 in x_values, "OPP-004 final SellPrice not found"
    
    def test_scatter_plot_with_grouping(self, report_engine, controlled_data, mock_streamlit):
        """Test scatter plot with color grouping."""
        
        config = {
            'x_axis': 'SellPrice',
            'y_axis': 'GM%',
            'group_by_column': 'BusinessUnit'
        }
        
        fig, _ = report_engine.generate_scatter_plot(controlled_data, config)
        
        # Should have multiple traces (one per business unit)
        assert len(fig.data) == 3, "Should have 3 traces for 3 business units"
        
        # Verify business unit names
        bu_names = [trace.name for trace in fig.data]
        expected_bus = ['Enterprise', 'SMB', 'Government']
        
        for bu in expected_bus:
            assert bu in bu_names, f"Missing business unit {bu} in scatter plot"
    
    def test_line_chart_accuracy(self, report_engine, controlled_data, mock_streamlit):
        """Test line chart calculations in detail."""
        
        config = {
            'x_axis': 'Snapshot Date',
            'y_axis': 'SellPrice',
            'aggregation': 'sum'
        }
        
        fig, _ = report_engine.generate_line_chart(controlled_data, config)
        
        # Should have data points
        assert len(fig.data[0].x) > 0
        assert len(fig.data[0].y) > 0
        
        # Verify the line chart uses deduplication
        # Total should be the sum of most recent values
        total_value = sum(fig.data[0].y)
        assert total_value == 1000000, "Line chart total should match deduplicated sum"
    
    def test_histogram_accuracy(self, report_engine, controlled_data, mock_streamlit):
        """Test histogram generation and binning."""
        
        config = {
            'column': 'SellPrice',
            'bins': 5
        }
        
        fig, _ = report_engine.generate_histogram(controlled_data, config)
        
        # Should have histogram data
        assert len(fig.data) == 1
        assert hasattr(fig.data[0], 'x'), "Histogram should have x data"
        
        # Verify we're using deduplicated data (4 values)
        # The histogram should reflect our 4 final values: 150k, 0, 350k, 500k
        x_data = fig.data[0].x
        assert len(x_data) == 4, "Histogram should use deduplicated data (4 values)"
    
    def test_box_plot_accuracy(self, report_engine, controlled_data, mock_streamlit):
        """Test box plot statistical calculations."""
        
        config = {
            'y_axis': 'SellPrice',
            'group_by_column': 'BusinessUnit'
        }
        
        fig, _ = report_engine.generate_box_plot(controlled_data, config)
        
        # Should have box plot data for each business unit
        assert len(fig.data) == 3, "Should have 3 box plots for 3 business units"
        
        # Verify business unit names
        bu_names = [trace.name for trace in fig.data]
        expected_bus = ['Enterprise', 'SMB', 'Government']
        
        for bu in expected_bus:
            assert bu in bu_names, f"Missing business unit {bu} in box plot"
    
    def test_correlation_heatmap_accuracy(self, report_engine, controlled_data, mock_streamlit):
        """Test correlation heatmap calculations."""
        
        config = {
            'columns': ['SellPrice', 'GM%']
        }
        
        fig, data_table = report_engine.generate_correlation_heatmap(controlled_data, config)
        
        # Should return correlation matrix
        assert data_table is not None
        assert data_table.shape == (2, 2), "Should be 2x2 correlation matrix"
        
        # Diagonal should be 1.0
        assert data_table.loc['SellPrice', 'SellPrice'] == 1.0
        assert data_table.loc['GM%', 'GM%'] == 1.0
        
        # Off-diagonal should be the same (symmetric)
        corr_value = data_table.loc['SellPrice', 'GM%']
        assert data_table.loc['GM%', 'SellPrice'] == corr_value
        
        # Correlation should be valid
        assert not pd.isna(corr_value), "Correlation should not be NaN"
        assert abs(corr_value) <= 1.0, "Correlation should be between -1 and 1"
    
    def test_descriptive_statistics_completeness(self, report_engine, controlled_data, expected_results, mock_streamlit):
        """Test that descriptive statistics include all expected metrics."""
        
        config = {
            'columns': ['SellPrice', 'GM%']
        }
        
        fig, data_table = report_engine.generate_descriptive_statistics(controlled_data, config)
        
        # Should return statistics table
        assert data_table is not None
        
        # Should have expected statistics
        expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        for stat in expected_stats:
            assert stat in data_table.index, f"Missing statistic: {stat}"
        
        # Verify specific values for SellPrice
        sellprice_stats = data_table['SellPrice']
        assert sellprice_stats['count'] == 4, "Should have 4 values (deduplicated)"
        assert sellprice_stats['min'] == expected_results['sellprice_min']
        assert sellprice_stats['max'] == expected_results['sellprice_max']
        assert sellprice_stats['50%'] == expected_results['sellprice_median']
        
        # Verify specific values for GM%
        gm_stats = data_table['GM%']
        assert gm_stats['count'] == 4, "Should have 4 values (deduplicated)"
        assert gm_stats['min'] == expected_results['gm_min']
        assert gm_stats['max'] == expected_results['gm_max']
        assert abs(gm_stats['mean'] - expected_results['gm_avg']) < 0.001
    
    def test_report_data_consistency(self, report_engine, controlled_data, mock_streamlit):
        """Test that all reports use consistent data (deduplication)."""
        
        # Generate multiple reports and verify they all use deduplicated data
        
        # Bar chart total
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_bar_chart(controlled_data, config)
        bar_total = sum(fig.data[0].y)
        
        # Time series total
        config = {'x_axis': 'Created', 'y_axis': 'SellPrice', 'aggregation': 'sum', 'time_period': 'M'}
        fig, _ = report_engine.generate_time_series(controlled_data, config)
        ts_total = sum(fig.data[0].y)
        
        # Line chart total
        config = {'x_axis': 'Snapshot Date', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        fig, _ = report_engine.generate_line_chart(controlled_data, config)
        line_total = sum(fig.data[0].y)
        
        # Descriptive statistics mean * count
        config = {'columns': ['SellPrice']}
        fig, data_table = report_engine.generate_descriptive_statistics(controlled_data, config)
        desc_total = data_table.loc['mean', 'SellPrice'] * data_table.loc['count', 'SellPrice']
        
        # All should equal the same deduplicated total
        expected_total = 1000000  # Sum of final values: 150k + 0 + 350k + 500k
        
        assert bar_total == expected_total, f"Bar chart total mismatch: {bar_total}"
        assert ts_total == expected_total, f"Time series total mismatch: {ts_total}"
        assert line_total == expected_total, f"Line chart total mismatch: {line_total}"
        assert abs(desc_total - expected_total) < 1, f"Descriptive stats total mismatch: {desc_total}"
    
    def test_empty_result_handling(self, report_engine, mock_streamlit):
        """Test how reports handle empty or filtered datasets."""
        
        # Create empty dataset
        empty_df = pd.DataFrame(columns=['Id', 'Owner', 'SellPrice', 'Snapshot Date'])
        
        config = {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'}
        
        # Should not crash on empty data
        try:
            fig, _ = report_engine.generate_bar_chart(empty_df, config)
            # If it doesn't crash, verify it returns empty or minimal data
            if fig.data:
                assert len(fig.data[0].x) == 0 or fig.data[0].x is None
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_single_value_handling(self, report_engine, mock_streamlit):
        """Test how reports handle datasets with single values."""
        
        # Create single-row dataset
        single_df = pd.DataFrame([{
            'Id': 'OPP-001',
            'Owner': 'Alice', 
            'SellPrice': 100000,
            'Snapshot Date': '2024-01-01',
            'GM%': 0.25,
            'BusinessUnit': 'Enterprise',
            'Stage': 'Won'
        }])
        
        # Test various report types
        configs = [
            {'x_axis': 'Owner', 'y_axis': 'SellPrice', 'aggregation': 'sum'},
            {'columns': ['SellPrice']},
            {'x_axis': 'SellPrice', 'y_axis': 'GM%'},
            {'column': 'SellPrice', 'bins': 5}
        ]
        
        methods = [
            report_engine.generate_bar_chart,
            report_engine.generate_descriptive_statistics,
            report_engine.generate_scatter_plot,
            report_engine.generate_histogram
        ]
        
        for method, config in zip(methods, configs):
            try:
                fig, data_table = method(single_df, config)
                # Should handle single value gracefully
                assert fig is not None or data_table is not None
            except Exception as e:
                # If it raises an exception, it should be meaningful
                assert len(str(e)) > 0, f"Empty error message from {method.__name__}"
