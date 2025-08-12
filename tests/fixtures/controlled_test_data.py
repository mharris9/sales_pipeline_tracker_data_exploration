"""
Controlled test dataset with known expected results for validation.

This module creates a simple, predictable dataset where we know exactly
what every calculation, aggregation, and feature should produce.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any


def create_controlled_dataset() -> pd.DataFrame:
    """
    Create a controlled dataset with known expected results.
    
    Dataset Design:
    - 4 opportunities (OPP-001 to OPP-004)
    - 3 snapshots each (12 total rows)
    - Simple, round numbers for easy calculation verification
    - Clear stage progression patterns
    - Predictable date patterns
    
    Returns:
        pd.DataFrame: Controlled test dataset
    """
    
    # Define the controlled data
    data = [
        # OPP-001: Won deal, clear progression
        {'Id': 'OPP-001', 'OpportunityName': 'Deal Alpha', 'Snapshot Date': '2024-01-01', 'Created': '2024-01-01', 'Stage': 'Lead', 'SellPrice': 100000, 'GM%': 0.20, 'BusinessUnit': 'Enterprise', 'Owner': 'Alice'},
        {'Id': 'OPP-001', 'OpportunityName': 'Deal Alpha', 'Snapshot Date': '2024-01-15', 'Created': '2024-01-01', 'Stage': 'Budget', 'SellPrice': 120000, 'GM%': 0.25, 'BusinessUnit': 'Enterprise', 'Owner': 'Alice'},
        {'Id': 'OPP-001', 'OpportunityName': 'Deal Alpha', 'Snapshot Date': '2024-01-30', 'Created': '2024-01-01', 'Stage': 'Closed - WON', 'SellPrice': 150000, 'GM%': 0.30, 'BusinessUnit': 'Enterprise', 'Owner': 'Alice'},
        
        # OPP-002: Lost deal
        {'Id': 'OPP-002', 'OpportunityName': 'Deal Beta', 'Snapshot Date': '2024-01-05', 'Created': '2024-01-05', 'Stage': 'Lead', 'SellPrice': 200000, 'GM%': 0.15, 'BusinessUnit': 'SMB', 'Owner': 'Bob'},
        {'Id': 'OPP-002', 'OpportunityName': 'Deal Beta', 'Snapshot Date': '2024-01-20', 'Created': '2024-01-05', 'Stage': 'Budget', 'SellPrice': 180000, 'GM%': 0.18, 'BusinessUnit': 'SMB', 'Owner': 'Bob'},
        {'Id': 'OPP-002', 'OpportunityName': 'Deal Beta', 'Snapshot Date': '2024-02-05', 'Created': '2024-01-05', 'Stage': 'Closed - LOST', 'SellPrice': 0, 'GM%': 0.00, 'BusinessUnit': 'SMB', 'Owner': 'Bob'},
        
        # OPP-003: Active deal (no final outcome)
        {'Id': 'OPP-003', 'OpportunityName': 'Deal Gamma', 'Snapshot Date': '2024-01-10', 'Created': '2024-01-10', 'Stage': 'Lead', 'SellPrice': 300000, 'GM%': 0.25, 'BusinessUnit': 'Government', 'Owner': 'Carol'},
        {'Id': 'OPP-003', 'OpportunityName': 'Deal Gamma', 'Snapshot Date': '2024-01-25', 'Created': '2024-01-10', 'Stage': 'Budget', 'SellPrice': 320000, 'GM%': 0.28, 'BusinessUnit': 'Government', 'Owner': 'Carol'},
        {'Id': 'OPP-003', 'OpportunityName': 'Deal Gamma', 'Snapshot Date': '2024-02-10', 'Created': '2024-01-10', 'Stage': 'Negotiation', 'SellPrice': 350000, 'GM%': 0.30, 'BusinessUnit': 'Government', 'Owner': 'Carol'},
        
        # OPP-004: Won deal, different owner
        {'Id': 'OPP-004', 'OpportunityName': 'Deal Delta', 'Snapshot Date': '2024-01-15', 'Created': '2024-01-15', 'Stage': 'Lead', 'SellPrice': 400000, 'GM%': 0.10, 'BusinessUnit': 'Enterprise', 'Owner': 'David'},
        {'Id': 'OPP-004', 'OpportunityName': 'Deal Delta', 'Snapshot Date': '2024-02-01', 'Created': '2024-01-15', 'Stage': 'Budget', 'SellPrice': 450000, 'GM%': 0.12, 'BusinessUnit': 'Enterprise', 'Owner': 'David'},
        {'Id': 'OPP-004', 'OpportunityName': 'Deal Delta', 'Snapshot Date': '2024-02-15', 'Created': '2024-01-15', 'Stage': 'Closed - WON', 'SellPrice': 500000, 'GM%': 0.15, 'BusinessUnit': 'Enterprise', 'Owner': 'David'},
    ]
    
    df = pd.DataFrame(data)
    
    # Convert date columns to proper datetime format
    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'])
    df['Created'] = pd.to_datetime(df['Created'])
    
    return df


def get_expected_results() -> Dict[str, Any]:
    """
    Get the expected results for all calculations on the controlled dataset.
    
    Returns:
        Dict[str, Any]: Dictionary of expected results for various calculations
    """
    
    expected = {
        # Basic counts and sums
        'total_rows': 12,
        'unique_opportunities': 4,
        'unique_owners': 4,
        'unique_business_units': 3,
        
        # Most recent snapshot sums (deduplicated)
        'total_sellprice_deduplicated': 1000000,  # 150k + 0 + 350k + 500k
        'avg_sellprice_deduplicated': 250000,     # 1000k / 4
        'won_deals_count': 2,                     # OPP-001, OPP-004
        'lost_deals_count': 1,                    # OPP-002
        'active_deals_count': 1,                  # OPP-003
        
        # Owner performance (using most recent snapshots)
        'alice_deals': 1,
        'alice_won_deals': 1,
        'alice_win_rate': 100.0,  # 1/1 * 100
        'bob_deals': 1,
        'bob_won_deals': 0,
        'bob_win_rate': 0.0,      # 0/1 * 100
        'carol_deals': 1,
        'carol_won_deals': 0,     # Still active
        'carol_win_rate': 0.0,    # 0/1 * 100 (no wins yet)
        'david_deals': 1,
        'david_won_deals': 1,
        'david_win_rate': 100.0,  # 1/1 * 100
        
        # Business unit breakdown (most recent snapshots)
        'enterprise_deals': 2,    # OPP-001, OPP-004
        'enterprise_value': 650000,  # 150k + 500k
        'smb_deals': 1,           # OPP-002
        'smb_value': 0,           # Lost deal
        'government_deals': 1,    # OPP-003
        'government_value': 350000,
        
        # Time-based calculations
        'january_created_deals': 4,   # All deals created in January
        'february_created_deals': 0,
        
        # Feature calculations
        'opp_001_days_in_pipeline': 29,  # Jan 1 to Jan 30
        'opp_002_days_in_pipeline': 31,  # Jan 5 to Feb 5
        'opp_003_days_in_pipeline': 31,  # Jan 10 to Feb 10
        'opp_004_days_in_pipeline': 31,  # Jan 15 to Feb 15
        
        'opp_001_final_stage': 'Closed - WON',
        'opp_002_final_stage': 'Closed - LOST',
        'opp_003_final_stage': 'Negotiation',
        'opp_004_final_stage': 'Closed - WON',
        
        'opp_001_starting_stage': 'Lead',
        'opp_002_starting_stage': 'Lead',
        'opp_003_starting_stage': 'Lead',
        'opp_004_starting_stage': 'Lead',
        
        # Stage progression counts
        'opp_001_stage_count': 3,  # Lead -> Budget -> Won
        'opp_002_stage_count': 3,  # Lead -> Budget -> Lost
        'opp_003_stage_count': 3,  # Lead -> Budget -> Negotiation
        'opp_004_stage_count': 3,  # Lead -> Budget -> Won
        
        # Monthly time series (using creation date, most recent values)
        'jan_2024_deals': 4,
        'jan_2024_total_value': 1000000,  # All deals created in Jan
        'feb_2024_deals': 0,
        'feb_2024_total_value': 0,
        
        # Correlation expectations
        'sellprice_gm_correlation_sign': 'positive',  # Generally higher prices have higher GM%
        
        # Statistical measures (most recent snapshots)
        'sellprice_min': 0,        # Lost deal
        'sellprice_max': 500000,   # OPP-004
        'sellprice_median': 250000, # Between 150k and 350k
        'gm_min': 0.0,             # Lost deal
        'gm_max': 0.30,            # OPP-001 and OPP-003
        'gm_avg': 0.1875,          # (0.30 + 0.0 + 0.30 + 0.15) / 4
    }
    
    return expected


def get_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Get specific test scenarios for various report configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Test scenarios with expected outcomes
    """
    
    scenarios = {
        'bar_chart_by_owner': {
            'config': {
                'x_axis': 'Owner',
                'y_axis': 'SellPrice',
                'aggregation': 'sum'
            },
            'expected_bars': 4,
            'expected_values': {
                'Alice': 150000,
                'Bob': 0,
                'Carol': 350000,
                'David': 500000
            }
        },
        
        'bar_chart_by_business_unit': {
            'config': {
                'x_axis': 'BusinessUnit',
                'y_axis': 'SellPrice',
                'aggregation': 'sum'
            },
            'expected_bars': 3,
            'expected_values': {
                'Enterprise': 650000,
                'SMB': 0,
                'Government': 350000
            }
        },
        
        'time_series_monthly': {
            'config': {
                'x_axis': 'Created',
                'y_axis': 'SellPrice',
                'aggregation': 'sum',
                'time_period': 'M'
            },
            'expected_points': 1,  # Only January 2024
            'expected_values': {
                '2024-01': 1000000
            }
        },
        
        'scatter_plot_price_vs_gm': {
            'config': {
                'x_axis': 'SellPrice',
                'y_axis': 'GM%'
            },
            'expected_points': 4,  # One per opportunity (most recent)
            'expected_correlation': 'positive'
        },
        
        'line_chart_gm_over_time': {
            'config': {
                'x_axis': 'Snapshot Date',
                'y_axis': 'GM%',
                'aggregation': 'mean'
            },
            'expected_trend': 'varies'  # Mix of increases and decreases
        }
    }
    
    return scenarios


def validate_dataset_integrity(df: pd.DataFrame) -> bool:
    """
    Validate that the controlled dataset has the expected structure.
    
    Args:
        df: The dataset to validate
        
    Returns:
        bool: True if dataset passes all integrity checks
    """
    
    expected = get_expected_results()
    
    # Basic structure checks
    if len(df) != expected['total_rows']:
        return False
    
    if df['Id'].nunique() != expected['unique_opportunities']:
        return False
    
    if df['Owner'].nunique() != expected['unique_owners']:
        return False
    
    # Check required columns exist
    required_columns = ['Id', 'OpportunityName', 'Snapshot Date', 'Created', 
                       'Stage', 'SellPrice', 'GM%', 'BusinessUnit', 'Owner']
    
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Check data types can be converted properly
    try:
        pd.to_datetime(df['Snapshot Date'])
        pd.to_datetime(df['Created'])
        pd.to_numeric(df['SellPrice'])
        pd.to_numeric(df['GM%'])
    except:
        return False
    
    return True
