"""
Column name mapping utilities for converting technical names to user-friendly display names.
"""
import re
from typing import Dict, Any


class ColumnMapper:
    """
    Handles mapping of technical column names to user-friendly display names.
    """
    
    def __init__(self):
        """Initialize the column mapper with default mappings."""
        self.custom_mappings = {
            # Common sales pipeline fields
            'SellPrice': 'Sell Price',
            'BusinessUnit': 'Business Unit',
            'OpportunityName': 'Opportunity Name',
            'OpportunityId': 'Opportunity ID',
            'SnapshotDate': 'Snapshot Date',
            'CloseDate': 'Close Date',
            'CreateDate': 'Create Date',
            'LastModified': 'Last Modified',
            'LastActivity': 'Last Activity',
            'DaysInPipeline': 'Days in Pipeline',
            'DaysToCloseWon': 'Days to Close Won',
            'StartingStage': 'Starting Stage',
            'UserWinRate': 'User Win Rate',
            'UserActivityRating': 'User Activity Rating',
            'TimeInStages': 'Time in Stages',
            'OpportunityAge': 'Opportunity Age',
            'StageProgressionCount': 'Stage Progression Count',
            
            # Common aggregation prefixes
            'sum_of_': 'Total ',
            'mean_of_': 'Average ',
            'count_of_': 'Count of ',
            'median_of_': 'Median ',
            'min_of_': 'Minimum ',
            'max_of_': 'Maximum ',
        }
    
    def add_custom_mapping(self, technical_name: str, display_name: str) -> None:
        """
        Add a custom mapping for a specific column name.
        
        Args:
            technical_name: The technical/original column name
            display_name: The user-friendly display name
        """
        self.custom_mappings[technical_name] = display_name
    
    def _split_camel_case(self, text: str) -> str:
        """
        Split camelCase or PascalCase text into separate words.
        
        Args:
            text: The camelCase text to split
            
        Returns:
            Text with spaces between words
        """
        # Insert space before uppercase letters (but not at the start)
        spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        
        # Handle sequences of uppercase letters (like ID, API, etc.)
        spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
        
        return spaced
    
    def _clean_aggregation_prefix(self, text: str) -> str:
        """
        Clean up aggregation prefixes in column names.
        
        Args:
            text: The text that may contain aggregation prefixes
            
        Returns:
            Cleaned text with proper aggregation labels
        """
        text_lower = text.lower()
        
        # Check for aggregation patterns
        for prefix, replacement in self.custom_mappings.items():
            if prefix.endswith('_of_') and text_lower.startswith(prefix):
                # Remove the prefix and apply the replacement
                remaining = text[len(prefix):]
                return replacement + self.map_column_name(remaining)
        
        # Check for "Sum of", "Mean of", etc. patterns
        agg_patterns = {
            r'^sum of (.+)': r'Total \1',
            r'^mean of (.+)': r'Average \1', 
            r'^count of (.+)': r'Count of \1',
            r'^median of (.+)': r'Median \1',
            r'^min of (.+)': r'Minimum \1',
            r'^max of (.+)': r'Maximum \1',
        }
        
        for pattern, replacement in agg_patterns.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                column_part = match.group(1)
                mapped_column = self.map_column_name(column_part)
                return replacement.replace(r'\1', mapped_column)
        
        return text
    
    def map_column_name(self, column_name: str) -> str:
        """
        Convert a technical column name to a user-friendly display name.
        
        Args:
            column_name: The technical column name to convert
            
        Returns:
            User-friendly display name
        """
        if not column_name:
            return column_name
        
        # Check for exact matches in custom mappings first
        if column_name in self.custom_mappings:
            return self.custom_mappings[column_name]
        
        # Clean aggregation prefixes
        cleaned = self._clean_aggregation_prefix(column_name)
        if cleaned != column_name:
            return cleaned
        
        # Split camelCase/PascalCase
        spaced = self._split_camel_case(column_name)
        
        # Convert to title case for better readability
        title_cased = spaced.title()
        
        # Handle common abbreviations and special cases
        replacements = {
            ' Id ': ' ID ',
            ' Id$': ' ID',
            '^Id ': 'ID ',
            ' Url ': ' URL ',
            ' Api ': ' API ',
            ' Crm ': ' CRM ',
            ' Roi ': ' ROI ',
            ' Kpi ': ' KPI ',
            ' Ceo ': ' CEO ',
            ' Cfo ': ' CFO ',
            ' Vp ': ' VP ',
            ' Usa ': ' USA ',
            ' Usd ': ' USD ',
            ' Gbp ': ' GBP ',
            ' Eur ': ' EUR ',
        }
        
        for pattern, replacement in replacements.items():
            title_cased = re.sub(pattern, replacement, title_cased)
        
        return title_cased
    
    def map_chart_title(self, title: str) -> str:
        """
        Convert chart titles to use friendly column names.
        
        Args:
            title: The original chart title
            
        Returns:
            Chart title with friendly column names
        """
        # Common patterns in chart titles
        patterns = [
            (r'Sum of (\w+)', lambda m: f"Total {self.map_column_name(m.group(1))}"),
            (r'Mean of (\w+)', lambda m: f"Average {self.map_column_name(m.group(1))}"),
            (r'Count of (\w+)', lambda m: f"Count of {self.map_column_name(m.group(1))}"),
            (r'Median of (\w+)', lambda m: f"Median {self.map_column_name(m.group(1))}"),
            (r'Distribution of (\w+)', lambda m: f"Distribution of {self.map_column_name(m.group(1))}"),
            (r'(\w+) vs (\w+)', lambda m: f"{self.map_column_name(m.group(1))} vs {self.map_column_name(m.group(2))}"),
            (r'(\w+) by (\w+)', lambda m: f"{self.map_column_name(m.group(1))} by {self.map_column_name(m.group(2))}"),
        ]
        
        mapped_title = title
        for pattern, replacement_func in patterns:
            mapped_title = re.sub(pattern, replacement_func, mapped_title)
        
        return mapped_title
    
    def get_display_columns(self, columns: list) -> Dict[str, str]:
        """
        Get a mapping of technical column names to display names.
        
        Args:
            columns: List of technical column names
            
        Returns:
            Dictionary mapping technical names to display names
        """
        return {col: self.map_column_name(col) for col in columns}


# Global instance for easy access
column_mapper = ColumnMapper()
