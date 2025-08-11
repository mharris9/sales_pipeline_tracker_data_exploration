"""
Sample data generator for the Sales Pipeline Data Explorer.
Run this script to create sample data for testing the application.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

def create_sales_pipeline_data(n_opportunities=200, max_snapshots_per_opp=8, start_date="2023-01-01"):
    """
    Create realistic sales pipeline data.
    
    Args:
        n_opportunities: Number of unique opportunities to create
        max_snapshots_per_opp: Maximum snapshots per opportunity
        start_date: Start date for data generation
        
    Returns:
        DataFrame with sales pipeline data
    """
    # Set random seed for reproducible data
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    stages = [
        "Lead", "Budget", "Proposal Development", "Proposal Submitted", 
        "Negotiation", "Closed - WON", "Closed - LOST", "Canceled/deferred"
    ]
    
    owners = [
        "Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown",
        "Frank Miller", "Grace Lee", "Henry Taylor", "Iris Chen", "Jack Anderson"
    ]
    
    business_units = ["Enterprise", "SMB", "Government", "Healthcare", "Education", "Financial Services"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"]
    industries = ["Technology", "Manufacturing", "Healthcare", "Financial", "Retail", "Government", "Education"]
    products = ["CRM Software", "ERP System", "Analytics Platform", "Security Suite", "Collaboration Tools"]
    lead_sources = ["Website", "Referral", "Cold Call", "Trade Show", "Partner", "Marketing Campaign"]
    
    data = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    for opp_id in range(1, n_opportunities + 1):
        # Opportunity characteristics (remain constant)
        owner = random.choice(owners)
        business_unit = random.choice(business_units)
        region = random.choice(regions)
        industry = random.choice(industries)
        product = random.choice(products)
        lead_source = random.choice(lead_sources)
        
        # Deal value based on business unit
        if business_unit == "Enterprise":
            deal_value = random.randint(100000, 1000000)
        elif business_unit == "SMB":
            deal_value = random.randint(10000, 100000)
        else:
            deal_value = random.randint(50000, 500000)
        
        # Customer type
        customer_type = random.choice(["New Customer", "Existing Customer", "Expansion"])
        
        # Opportunity start date
        opp_start_date = start_dt + timedelta(days=random.randint(0, 300))
        
        # Generate snapshots for this opportunity
        n_snapshots = random.randint(1, max_snapshots_per_opp)
        current_stage_idx = 0
        current_date = opp_start_date
        
        for snapshot_num in range(n_snapshots):
            # Stage progression logic
            if snapshot_num > 0:
                # Opportunity might advance, stay same, or regress slightly
                progression = random.choices(
                    [0, 1, -1],  # stay, advance, regress
                    weights=[0.4, 0.5, 0.1],
                    k=1
                )[0]
                
                current_stage_idx = max(0, min(len(stages) - 1, current_stage_idx + progression))
                
                # If in negotiation, higher chance of closing
                if current_stage_idx == 4 and random.random() > 0.6:  # Negotiation stage
                    current_stage_idx = random.choice([5, 6])  # Close won or lost
            
            current_stage = stages[current_stage_idx]
            
            # Probability based on stage and other factors
            stage_probabilities = {
                "Lead": 10, "Budget": 25, "Proposal Development": 40,
                "Proposal Submitted": 60, "Negotiation": 80,
                "Closed - WON": 100, "Closed - LOST": 0, "Canceled/deferred": 0
            }
            
            base_prob = stage_probabilities[current_stage]
            
            # Adjust probability based on factors
            prob_adjustment = 0
            if business_unit == "Enterprise":
                prob_adjustment += 5
            if customer_type == "Existing Customer":
                prob_adjustment += 10
            if deal_value > 500000:
                prob_adjustment -= 5
            
            probability = max(0, min(100, base_prob + prob_adjustment + random.randint(-10, 10)))
            
            # Other fields
            priority = random.choices(
                ["High", "Medium", "Low"],
                weights=[0.2, 0.6, 0.2],
                k=1
            )[0]
            
            competition_level = random.choice(["None", "Low", "Medium", "High"])
            budget_confirmed = random.choice([True, False]) if current_stage_idx >= 1 else False
            decision_maker_identified = random.choice([True, False]) if current_stage_idx >= 2 else False
            
            # Create record
            record = {
                "Id": f"OPP-{opp_id:05d}",
                "Snapshot Date": current_date.strftime("%m/%d/%Y"),
                "Stage": current_stage,
                "Owner": owner,
                "Business Unit": business_unit,
                "Region": region,
                "Industry": industry,
                "Product": product,
                "Lead Source": lead_source,
                "Customer Type": customer_type,
                "Deal Value": deal_value,
                "Probability": probability,
                "Priority": priority,
                "Competition Level": competition_level,
                "Budget Confirmed": budget_confirmed,
                "Decision Maker Identified": decision_maker_identified,
                "Days Since Created": (current_date - opp_start_date).days,
                "Quarter": f"Q{((current_date.month - 1) // 3) + 1} {current_date.year}",
                "Created Date": opp_start_date.strftime("%m/%d/%Y"),
                "Expected Close Date": (current_date + timedelta(days=random.randint(30, 120))).strftime("%m/%d/%Y"),
                "Last Activity": (current_date - timedelta(days=random.randint(0, 14))).strftime("%m/%d/%Y")
            }
            
            data.append(record)
            
            # Move to next snapshot date (1-4 weeks later)
            current_date += timedelta(days=random.randint(7, 28))
            
            # Stop if opportunity is closed
            if current_stage in ["Closed - WON", "Closed - LOST", "Canceled/deferred"]:
                break
    
    return pd.DataFrame(data)

def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description="Generate sample sales pipeline data")
    parser.add_argument("--opportunities", type=int, default=200, help="Number of opportunities")
    parser.add_argument("--snapshots", type=int, default=8, help="Max snapshots per opportunity")
    parser.add_argument("--output", type=str, default="sample_sales_data.csv", help="Output filename")
    parser.add_argument("--format", type=str, choices=["csv", "xlsx"], default="csv", help="Output format")
    
    args = parser.parse_args()
    
    print(f"🔧 Generating sample sales pipeline data...")
    print(f"   - Opportunities: {args.opportunities}")
    print(f"   - Max snapshots per opportunity: {args.snapshots}")
    print(f"   - Output file: {args.output}")
    
    # Generate data
    df = create_sales_pipeline_data(
        n_opportunities=args.opportunities,
        max_snapshots_per_opp=args.snapshots
    )
    
    # Save data
    if args.format == "csv":
        df.to_csv(args.output, index=False)
    else:
        df.to_excel(args.output, index=False)
    
    print(f"✅ Generated {len(df)} records for {args.opportunities} opportunities")
    print(f"📊 Data saved to: {args.output}")
    
    # Show sample statistics
    print(f"\n📈 Data Summary:")
    print(f"   - Date range: {df['Snapshot Date'].min()} to {df['Snapshot Date'].max()}")
    print(f"   - Stages: {', '.join(df['Stage'].unique())}")
    print(f"   - Owners: {len(df['Owner'].unique())} unique")
    print(f"   - Average deal value: ${df['Deal Value'].mean():,.0f}")
    print(f"   - Records per opportunity: {len(df) / args.opportunities:.1f} average")
    
    print(f"\n🚀 Ready to use with Sales Pipeline Data Explorer!")
    print(f"   Run: streamlit run main.py")
    print(f"   Then upload: {args.output}")

if __name__ == "__main__":
    main()

