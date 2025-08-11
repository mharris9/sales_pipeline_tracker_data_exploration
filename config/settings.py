"""
Configuration settings for the Sales Pipeline Explorer application.
"""
from typing import List, Dict, Any

# App Configuration
APP_TITLE = "Sales Pipeline Data Explorer"
APP_ICON = "📊"
LAYOUT = "wide"

# Data Configuration
MAX_FILE_SIZE_MB = 200
SUPPORTED_FILE_TYPES = ['csv', 'xlsx']
DATE_FORMAT = "%m/%d/%Y"
SNAPSHOT_DATE_COLUMN = "Snapshot Date"
ID_COLUMN = "Id"

# Sales Pipeline Stages
SALES_STAGES = [
    "Lead",
    "Budget", 
    "Proposal Development",
    "Proposal Submitted",
    "Negotiation",
    "Closed - WON",
    "Closed - LOST",
    "Canceled/deferred"
]

# Chart Configuration
CHART_HEIGHT = 500
CHART_THEME = "plotly_white"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Export Configuration
EXPORT_DPI = 300
EXPORT_FORMATS = {
    'data': ['csv'],
    'charts': ['png', 'svg']
}

# Performance Configuration
CHUNK_SIZE = 10000  # For processing large datasets
CACHE_TTL = 3600  # Cache time-to-live in seconds

# UI Configuration
SIDEBAR_WIDTH = 300
FILTER_CONTAINER_HEIGHT = 400

