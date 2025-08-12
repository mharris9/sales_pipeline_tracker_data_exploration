"""
Test data generator with controlled values for filter testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data():
    """
    Generate test data with known values for each data type.
    
    The data is carefully constructed to have known:
    - Value counts for categorical data
    - Statistical measures for numerical data
    - Date ranges and distributions
    - Text patterns
    - Boolean distributions
    
    Returns:
        DataFrame, dict with expected statistics
    """
    # Create a DataFrame with 100 rows (10 rows per test case)
    n_rows = 100
    
    # Generate controlled categorical data
    categories = ['A', 'B', 'C', 'D', 'E']
    category_data = np.repeat(categories, n_rows // len(categories))
    # Add known distribution: A:20, B:20, C:20, D:20, E:20
    
    # Generate controlled numerical data
    # Numbers 1-100 to make statistics easily verifiable
    numerical_data = np.arange(1, n_rows + 1, dtype=float)
    # Known stats: mean=50.5, median=50.5, min=1, max=100
    
    # Generate controlled date data
    base_date = datetime(2024, 1, 1)
    date_data = [base_date + timedelta(days=i) for i in range(n_rows)]
    # Known range: 2024-01-01 to 2024-04-09
    
    # Generate controlled text data
    text_patterns = [
        'test_a_', 'test_b_', 'test_c_', 'test_d_', 'test_e_'
    ]
    text_data = []
    for i in range(n_rows):
        pattern = text_patterns[i % len(text_patterns)]
        text_data.append(f"{pattern}{i//len(text_patterns)}")
    # Known patterns for testing contains/startswith/endswith
    
    # Generate controlled boolean data
    boolean_data = np.repeat([True, False], n_rows // 2)
    # Known distribution: True:50, False:50
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(1, n_rows + 1),
        'category': category_data,
        'number': numerical_data,
        'date': date_data,
        'text': text_data,
        'flag': boolean_data
    })
    
    # Expected statistics for validation
    expected_stats = {
        'category': {
            'value_counts': {cat: n_rows // len(categories) 
                           for cat in categories},
            'unique_count': len(categories)
        },
        'number': {
            'mean': 50.5,
            'median': 50.5,
            'min': 1.0,
            'max': 100.0,
            'std': np.std(numerical_data),
            'quartiles': {
                '25': np.percentile(numerical_data, 25),
                '50': np.percentile(numerical_data, 50),
                '75': np.percentile(numerical_data, 75)
            }
        },
        'date': {
            'min_date': base_date,
            'max_date': base_date + timedelta(days=n_rows-1),
            'unique_count': n_rows
        },
        'text': {
            'patterns': text_patterns,
            'unique_count': n_rows,
            'prefix_counts': {pattern: n_rows // len(text_patterns) 
                            for pattern in text_patterns}
        },
        'flag': {
            'true_count': n_rows // 2,
            'false_count': n_rows // 2
        }
    }
    
    return df, expected_stats

def get_expected_filter_results():
    """
    Get expected results for various filter combinations.
    
    Returns:
        dict with filter scenarios and expected outcomes
    """
    return {
        'category_filters': {
            'include_single': {
                'filter': {'column': 'category', 'values': ['A'], 'type': 'include'},
                'expected_count': 20
            },
            'include_multiple': {
                'filter': {'column': 'category', 'values': ['A', 'B'], 'type': 'include'},
                'expected_count': 40
            },
            'exclude_single': {
                'filter': {'column': 'category', 'values': ['A'], 'type': 'exclude'},
                'expected_count': 80
            }
        },
        'numerical_filters': {
            'range': {
                'filter': {'column': 'number', 'min': 25, 'max': 75},
                'expected_count': 51
            },
            'greater_than': {
                'filter': {'column': 'number', 'min': 75},
                'expected_count': 25
            },
            'less_than': {
                'filter': {'column': 'number', 'max': 25},
                'expected_count': 24  # Numbers 1-24 (less than 25)
            }
        },
        'date_filters': {
            'range': {
                'filter': {
                    'column': 'date',
                    'start': datetime(2024, 2, 1),
                    'end': datetime(2024, 3, 1)
                },
                'expected_count': 30
            }
        },
        'text_filters': {
            'contains': {
                'filter': {'column': 'text', 'pattern': 'test_a', 'type': 'contains'},
                'expected_count': 20
            },
            'starts_with': {
                'filter': {'column': 'text', 'pattern': 'test_a', 'type': 'starts_with'},
                'expected_count': 20
            }
        },
        'boolean_filters': {
            'true_only': {
                'filter': {'column': 'flag', 'values': [True]},
                'expected_count': 50
            }
        },
        'combined_filters': {
            'category_and_number': {
                'filters': [
                    {'column': 'category', 'values': ['A', 'B'], 'type': 'include'},
                    {'column': 'number', 'min': 25, 'max': 75}
                ],
                'expected_count': 16  # Intersection of both filters (A & B categories between 25-75)
            }
        }
    }