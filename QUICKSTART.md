# Quick Start Guide 🚀

Get up and running with the Sales Pipeline Data Explorer in minutes!

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
streamlit run main.py
```

3. **Open your browser** to the URL shown (typically `http://localhost:8501`)

## Generate Sample Data

If you don't have your own data yet, create sample data:

```bash
# Generate 200 opportunities with default settings
python create_sample_data.py

# Generate more data with custom settings
python create_sample_data.py --opportunities 500 --snapshots 10 --output my_data.csv

# Generate Excel format
python create_sample_data.py --format xlsx --output sample_data.xlsx
```

## Quick Test

Run the test script to verify everything works:

```bash
python test_app.py
```

## First Steps

1. **Upload Data**: Click "Browse files" in the sidebar and select your CSV/XLSX file
2. **Explore Filters**: Go to the "Filters" tab and try filtering by different columns
3. **Add Features**: Visit the "Features" tab and add calculated features like "days_in_pipeline"
4. **Create Reports**: Generate your first chart in the "Reports" tab
5. **Export Results**: Download your filtered data and charts from the "Export" tab

## Sample Data Structure

Your CSV/XLSX should have columns like:

| Id | Snapshot Date | Stage | Owner | Deal Value | ... |
|----|---------------|-------|-------|------------|-----|
| OPP-00001 | 01/15/2023 | Lead | Alice Johnson | 50000 | ... |
| OPP-00001 | 01/22/2023 | Budget | Alice Johnson | 50000 | ... |
| OPP-00002 | 01/20/2023 | Proposal Development | Bob Smith | 75000 | ... |

## Key Features to Try

- **Filter by Stage**: See only "Closed - WON" opportunities
- **Group by Owner**: Compare performance across sales reps
- **Add Win Rate Feature**: Calculate success rates by user
- **Time Series Analysis**: Track pipeline changes over time
- **Export Everything**: Download filtered data and high-res charts

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review error messages in the browser console
- Ensure your data has the expected column structure

**Happy exploring!** 📊

