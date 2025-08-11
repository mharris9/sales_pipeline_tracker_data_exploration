"""
Test script to verify the Sales Pipeline Data Explorer application.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data(n_records=1000, n_opportunities=100):
    """
    Create sample sales pipeline data for testing.
    
    Args:
        n_records: Total number of records (snapshots)
        n_opportunities: Number of unique opportunities
        
    Returns:
        DataFrame with sample data
    """
    # Set random seed for reproducible data
    np.random.seed(42)
    random.seed(42)
    
    # Sales stages
    stages = [
        "Lead", "Budget", "Proposal Development", "Proposal Submitted", 
        "Negotiation", "Closed - WON", "Closed - LOST", "Canceled/deferred"
    ]
    
    # Sample owners/users
    owners = ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown"]
    
    # Sample business units
    business_units = ["Enterprise", "SMB", "Government", "Healthcare", "Education"]
    
    # Sample regions
    regions = ["North", "South", "East", "West", "Central"]
    
    # Generate data
    data = []
    
    # Create opportunities with multiple snapshots
    for opp_id in range(1, n_opportunities + 1):
        # Random number of snapshots per opportunity (1-10)
        n_snapshots = random.randint(1, 10)
        
        # Random owner and other attributes (consistent for the opportunity)
        owner = random.choice(owners)
        business_unit = random.choice(business_units)
        region = random.choice(regions)
        deal_value = random.randint(10000, 500000)
        
        # Generate snapshots
        start_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        current_stage_idx = 0
        
        for snapshot in range(n_snapshots):
            # Snapshot date (weekly intervals)
            snapshot_date = start_date + timedelta(days=snapshot * 7)
            
            # Stage progression (can stay in same stage or advance)
            if random.random() > 0.3 and current_stage_idx < len(stages) - 1:
                # Advance stage
                if current_stage_idx < len(stages) - 3:  # Not in final stages
                    current_stage_idx += random.randint(0, 1)
                elif random.random() > 0.5:  # In later stages, might close
                    current_stage_idx = random.choice([len(stages) - 3, len(stages) - 2, len(stages) - 1])
            
            current_stage = stages[min(current_stage_idx, len(stages) - 1)]
            
            # Probability score (decreases over time if not progressing)
            if current_stage in ["Closed - WON", "Closed - LOST", "Canceled/deferred"]:
                probability = 100 if current_stage == "Closed - WON" else 0
            else:
                base_prob = {"Lead": 10, "Budget": 25, "Proposal Development": 40, 
                           "Proposal Submitted": 60, "Negotiation": 80}.get(current_stage, 50)
                probability = max(0, min(100, base_prob + random.randint(-20, 20)))
            
            # Create record
            record = {
                "Id": f"OPP-{opp_id:04d}",
                "Snapshot Date": snapshot_date.strftime("%m/%d/%Y"),
                "Stage": current_stage,
                "Owner": owner,
                "Business Unit": business_unit,
                "Region": region,
                "Deal Value": deal_value,
                "Probability": probability,
                "Product Category": random.choice(["Software", "Hardware", "Services", "Consulting"]),
                "Lead Source": random.choice(["Website", "Referral", "Cold Call", "Trade Show", "Partner"]),
                "Customer Type": random.choice(["New", "Existing", "Expansion"]),
                "Priority": random.choice(["High", "Medium", "Low"]),
                "Competition": random.choice(["None", "Low", "Medium", "High"]),
                "Budget Confirmed": random.choice([True, False]),
                "Decision Maker Identified": random.choice([True, False]),
                "Notes": f"Sample note for opportunity {opp_id} snapshot {snapshot + 1}"
            }
            
            data.append(record)
            
            # Stop if opportunity is closed
            if current_stage in ["Closed - WON", "Closed - LOST", "Canceled/deferred"]:
                break
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure we have the right number of records
    if len(df) > n_records:
        df = df.sample(n=n_records, random_state=42)
    
    return df

def test_data_handler():
    """Test the DataHandler class."""
    print("Testing DataHandler...")
    
    from core.data_handler import DataHandler
    
    # Create sample data
    df = create_sample_data(500, 50)
    
    # Save to CSV for testing
    df.to_csv("sample_data.csv", index=False)
    
    # Test data handler (would normally use uploaded file)
    handler = DataHandler()
    
    # Simulate file loading
    handler.df_raw = df
    handler._process_data()
    
    print(f"✓ Loaded {len(handler.df_processed)} rows and {len(handler.df_processed.columns)} columns")
    print(f"✓ Detected {len(handler.column_types)} column types")
    print(f"✓ Validation results: {handler.validate_sales_pipeline_data()}")
    
    return handler

def test_feature_engine():
    """Test the FeatureEngine class."""
    print("\nTesting FeatureEngine...")
    
    from core.feature_engine import FeatureEngine
    
    # Create sample data
    df = create_sample_data(500, 50)
    
    engine = FeatureEngine()
    available_features = engine.get_available_features(df.columns.tolist())
    
    print(f"✓ Found {len(available_features)} available features")
    
    # Test adding features
    features_to_add = ["days_in_pipeline", "starting_stage", "final_stage"]
    df_with_features = engine.add_features(df, features_to_add)
    
    print(f"✓ Added features, DataFrame now has {len(df_with_features.columns)} columns")
    
    return engine, df_with_features

def test_filter_manager():
    """Test the FilterManager class."""
    print("\nTesting FilterManager...")
    
    from core.filter_manager import FilterManager
    from utils.data_types import detect_data_type
    
    # Create sample data
    df = create_sample_data(500, 50)
    
    # Detect column types
    column_types = {}
    for col in df.columns:
        column_types[col] = detect_data_type(df[col], col)
    
    manager = FilterManager()
    manager.create_filters(df, column_types)
    
    print(f"✓ Created filters for {len(manager.filters)} columns")
    
    # Test applying filters (no filters active, should return same data)
    filtered_df = manager.apply_filters(df)
    print(f"✓ Filter application works, returned {len(filtered_df)} rows")
    
    return manager

def test_report_engine():
    """Test the ReportEngine class."""
    print("\nTesting ReportEngine...")
    
    from core.report_engine import ReportEngine
    
    # Create sample data
    df = create_sample_data(500, 50)
    
    engine = ReportEngine()
    available_reports = engine.get_available_reports()
    
    print(f"✓ Found {len(available_reports)} available report types")
    
    # Test descriptive statistics
    config = {'selected_columns': ['Deal Value', 'Probability', 'Stage']}
    figure, data_table = engine.generate_report('descriptive_statistics', df, config)
    
    print(f"✓ Generated descriptive statistics with {len(data_table)} rows")
    
    return engine

def test_outlier_manager():
    """Test the OutlierManager class."""
    print("\nTesting OutlierManager...")
    
    from core.outlier_manager import OutlierManager
    from utils.data_types import detect_data_type, DataType
    
    # Create sample data with some outliers
    df = create_sample_data(500, 50)
    
    # Add some artificial outliers
    df.loc[df.index[:5], 'Deal Value'] = 10000000  # Very high values
    df.loc[df.index[5:10], 'Probability'] = -50    # Invalid probabilities
    
    manager = OutlierManager()
    
    # Create column types
    column_types = {}
    for col in df.columns:
        column_types[col] = detect_data_type(df[col], col)
    
    # Create outlier settings
    manager.create_outlier_settings(df, column_types)
    
    print(f"✓ Created outlier settings for {len(manager.outlier_settings)} numerical columns")
    
    # Test outlier detection on Deal Value
    outliers, info = manager.detect_outliers_column(df, 'Deal Value', 'iqr', 'moderate')
    print(f"✓ Detected {outliers.sum()} outliers in Deal Value using IQR method")
    
    # Test multiple column detection
    numerical_columns = ['Deal Value', 'Probability']
    combined_outliers, combined_info = manager.detect_outliers_multiple_columns(
        df, numerical_columns, 'iqr', 'moderate', 'any'
    )
    
    print(f"✓ Combined outlier detection found {combined_outliers.sum()} outliers across {len(numerical_columns)} columns")
    
    return manager

def main():
    """Run all tests."""
    print("🧪 Testing Sales Pipeline Data Explorer Components\n")
    
    try:
        # Test individual components
        handler = test_data_handler()
        engine, df_with_features = test_feature_engine()
        filter_manager = test_filter_manager()
        report_engine = test_report_engine()
        outlier_manager = test_outlier_manager()
        
        print("\n✅ All tests passed! The application components are working correctly.")
        print("\n🚀 You can now run the main application with: streamlit run main.py")
        
        # Clean up
        import os
        if os.path.exists("sample_data.csv"):
            os.remove("sample_data.csv")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

