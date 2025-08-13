"""
Pytest configuration and shared fixtures for the Sales Pipeline Data Explorer tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.data_handler import DataHandler
from src.services.feature_engine import FeatureEngine
from src.services.report_engine import ReportEngine
from src.services.filter_manager import FilterManager
from src.services.outlier_manager import OutlierManager
from src.utils.data_types import DataType


@pytest.fixture
def sample_pipeline_data():
    """Create a realistic sales pipeline dataset for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 100 opportunities with multiple snapshots each
    opportunities = []
    base_date = datetime(2024, 1, 1)
    
    for opp_id in range(1, 51):  # 50 unique opportunities
        # Each opportunity has 3-8 snapshots over time
        num_snapshots = np.random.randint(3, 9)
        
        # Generate stages progression
        stages = ["Lead", "Budget", "Proposal Development", "Proposal Submitted", "Negotiation"]
        final_stages = ["Closed - WON", "Closed - LOST", "Canceled/deferred"]
        
        # Random progression through stages
        stage_progression = stages[:np.random.randint(2, len(stages))]
        if np.random.random() > 0.3:  # 70% chance to close
            stage_progression.append(np.random.choice(final_stages))
        
        # Generate snapshots
        for i in range(num_snapshots):
            snapshot_date = base_date + timedelta(days=i*7 + np.random.randint(0, 7))
            stage_idx = min(i, len(stage_progression) - 1)
            
            opportunities.append({
                'Id': f'OPP-{opp_id:03d}',
                'OpportunityName': f'Deal {opp_id}',
                'Snapshot Date': snapshot_date,
                'Created': base_date + timedelta(days=np.random.randint(0, 30)),
                'Stage': stage_progression[stage_idx],
                'SellPrice': np.random.randint(10000, 500000),
                'GM%': np.random.uniform(0.2, 0.4),
                'BusinessUnit': np.random.choice(['Enterprise', 'SMB', 'Government']),
                'Owner': np.random.choice(['Alice Smith', 'Bob Johnson', 'Carol Davis', 'Dave Wilson'])
            })
    
    return pd.DataFrame(opportunities)


@pytest.fixture
def empty_dataframe():
    """Create an empty dataframe with correct columns."""
    return pd.DataFrame(columns=['Id', 'OpportunityName', 'Snapshot Date', 'Created', 'Stage', 'SellPrice', 'GM%', 'BusinessUnit', 'Owner'])


@pytest.fixture
def malformed_data():
    """Create a dataframe with various data quality issues."""
    return pd.DataFrame({
        'Id': ['OPP-001', None, 'OPP-003', ''],
        'OpportunityName': ['Deal 1', 'Deal 2', None, 'Deal 4'],
        'Snapshot Date': ['2024-01-01', 'invalid_date', '2024-01-03', None],
        'Created': ['2024-01-01', '2024-01-02', None, '2024-01-04'],
        'Stage': ['Lead', None, 'Budget', ''],
        'SellPrice': [100000, 'not_a_number', 250000, -50000],
        'GM%': [0.3, None, 'invalid', 0.25],
        'BusinessUnit': ['Enterprise', '', None, 'SMB'],
        'Owner': ['Alice', None, 'Bob', '']
    })


@pytest.fixture
def data_handler():
    """Create a DataHandler instance."""
    return DataHandler()


@pytest.fixture
def loaded_data_handler(sample_pipeline_data):
    """Create a DataHandler with sample data loaded."""
    handler = DataHandler()
    # Simulate loading the data
    handler.df_raw = sample_pipeline_data.copy()
    handler.df_processed = sample_pipeline_data.copy()
    
    # Detect column types
    handler._detect_and_convert_types()
    handler._calculate_column_statistics()
    
    return handler


@pytest.fixture
def feature_engine():
    """Create a FeatureEngine instance."""
    return FeatureEngine()


@pytest.fixture
def report_engine():
    """Create a ReportEngine instance."""
    return ReportEngine()


@pytest.fixture
def filter_manager():
    """Create a FilterManager instance."""
    return FilterManager()


@pytest.fixture
def outlier_manager():
    """Create an OutlierManager instance."""
    return OutlierManager()


@pytest.fixture
def controlled_dataset():
    """Create the controlled test dataset with known expected results."""
    return create_controlled_dataset()


@pytest.fixture
def controlled_expected_results():
    """Get the expected results for the controlled dataset."""
    return get_expected_results()


@pytest.fixture
def mock_streamlit(mocker):
    """Mock streamlit functions to avoid UI dependencies."""
    # Mock specific streamlit functions that are used in the code
    mocker.patch('streamlit.warning')
    mocker.patch('streamlit.error')
    mocker.patch('streamlit.info')
    mocker.patch('streamlit.success')
    
    # Also patch the module imports in our code
    mocker.patch('src.services.data_handler.st.warning')
    mocker.patch('src.services.data_handler.st.error')
    mocker.patch('src.services.data_handler.st.info')
    mocker.patch('src.services.feature_engine.st.warning')
    mocker.patch('src.services.feature_engine.st.error')
    mocker.patch('src.services.feature_engine.st.info')
    mocker.patch('src.services.report_engine.st.warning')
    mocker.patch('src.services.report_engine.st.error')
    mocker.patch('src.services.report_engine.st.info')
    
    return mocker
