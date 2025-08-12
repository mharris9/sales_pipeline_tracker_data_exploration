"""
Centralized state management for the Sales Pipeline Data Explorer.

This module provides a flexible and extensible state management system that:
1. Maintains a single source of truth for application state
2. Provides easy extension points for new features
3. Includes comprehensive debugging and logging
4. Supports state validation and consistency checks
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import logging
import json
import psutil
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from utils.data_types import DataType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateChangeType(Enum):
    """Types of state changes for tracking and debugging."""
    FILTER = "filter"
    DATA = "data"
    VIEW = "view"
    FEATURE = "feature"
    REPORT = "report"
    CUSTOM = "custom"

@dataclass
class StateChange:
    """Tracks individual state changes for debugging."""
    change_type: StateChangeType
    component: str
    description: str
    timestamp: datetime = datetime.now()
    old_value: Any = None
    new_value: Any = None

class StateManager:
    """
    Centralized state manager with extension capabilities.
    
    Features:
    - Modular state management
    - State change tracking
    - Easy extension points
    - Debug logging
    - State validation
    - State persistence
    - Memory management
    - Error recovery
    - Performance optimization
    """
    
    VERSION = 2  # Current state format version
    MAX_HISTORY_SIZE = 1000  # Maximum number of history entries to keep
    MEMORY_THRESHOLD = 0.8  # Memory usage threshold (80% of available memory)
    
    def __init__(self):
        """Initialize the state manager."""
        # Initialize core state containers
        if 'state_manager' not in st.session_state:
            st.session_state.state_manager = self
            self._initialize_state()
    
    def _initialize_state(self):
        """Initialize all state containers."""
        # Core state containers
        self._state = {
            'data': {
                'original_df': None,
                'filtered_df': None,
                'column_types': {},
                'data_info': {},
                'data_loaded': False
            },
            'filters': {
                'active_filters': {},
                'filter_configs': {},
                'filter_results': {}
            },
            'view': {
                'current_tab': None,
                'display_options': {},
                'ui_settings': {}
            }
        }
        
        # State tracking for debugging
        self._state_history: List[StateChange] = []
        self._state_validators: Dict[str, Callable] = {}
        self._state_watchers: Dict[str, List[Callable]] = {}
        
        # Store class instances separately
        self._instances = {}
        
        logger.info("State manager initialized")
    
    def register_extension(self, name: str, initial_state: Dict[str, Any] = None) -> None:
        """
        Register a new state extension.
        
        Args:
            name: Name of the extension
            initial_state: Initial state for the extension
        """
        if initial_state is None:
            initial_state = {}
        
        # Extract class instances and store them separately
        instances = {}
        serializable_state = {}
        
        for key, value in initial_state.items():
            if hasattr(value, '__class__') and value.__class__.__name__ in [
                'DataHandler', 'FilterManager', 'FeatureEngine', 
                'ReportEngine', 'OutlierManager', 'ExportManager'
            ]:
                instances[key] = value
            else:
                serializable_state[key] = value
        
        # Store instances
        if instances:
            self._instances[name] = instances
        
        # Store serializable state
        if name in self._state:
            logger.warning(f"Extension {name} already exists. Updating state.")
            self._state[name].update(serializable_state)
        else:
            self._state[name] = serializable_state
            
        logger.info(f"Registered extension: {name}")
    
    def register_validator(self, state_path: str, validator: Callable) -> None:
        """
        Register a validation function for a specific state path.
        
        Args:
            state_path: Path to state (e.g., 'filters.active_filters')
            validator: Function that validates state changes
        """
        self._state_validators[state_path] = validator
        logger.info(f"Registered validator for {state_path}")
    
    def register_watcher(self, state_path: str, watcher: Callable) -> None:
        """
        Register a watcher function for state changes.
        
        Args:
            state_path: Path to watch (e.g., 'filters.active_filters')
            watcher: Function to call on state changes
        """
        if state_path not in self._state_watchers:
            self._state_watchers[state_path] = []
        self._state_watchers[state_path].append(watcher)
        logger.info(f"Registered watcher for {state_path}")
    
    def update_state(self, path: str, value: Any, 
                    change_type: StateChangeType = StateChangeType.CUSTOM,
                    component: str = None, description: str = None) -> None:
        """
        Update state at a specific path.
        
        Args:
            path: Path to state (e.g., 'filters.active_filters')
            value: New value
            change_type: Type of state change
            component: Component making the change
            description: Description of the change
        """
        try:
            # Get current value
            old_value = self.get_state(path)
            
            # Validate if validator exists
            if path in self._state_validators:
                if not self._state_validators[path](value):
                    raise ValueError(f"Invalid state update for {path}")
            
            # Validate widget state
            if path.startswith('widgets.'):
                value = self._validate_widget_state(path, value, old_value)
            
            # Handle special types
            if isinstance(value, pd.DataFrame):
                value = value.copy()  # Store a copy to prevent modifications
            
            # Update state
            parts = path.split('.')
            current = self._state
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
            
            # Record change
            change = StateChange(
                change_type=change_type,
                component=component or path,
                description=description or f"Updated {path}",
                old_value=old_value,
                new_value=value
            )
            self._state_history.append(change)
            
            # Notify watchers
            if path in self._state_watchers:
                for watcher in self._state_watchers[path]:
                    try:
                        watcher(old_value, value)
                    except Exception as e:
                        logger.error(f"Error in watcher for {path}: {str(e)}")
                        # Don't re-raise watcher exceptions
            
            logger.info(f"State updated: {path}")
            
        except Exception as e:
            logger.error(f"Error updating state at {path}: {str(e)}")
            raise
    
    def get_state(self, path: str, default: Any = None) -> Any:
        """
        Get state from a specific path.
        
        Args:
            path: Path to state
            default: Default value if path doesn't exist
            
        Returns:
            State value or default
        """
        try:
            # Check for class instance first
            parts = path.split('.')
            if len(parts) >= 2:
                container = parts[0]
                key = parts[1]
                if container in self._instances and key in self._instances[container]:
                    return self._instances[container][key]
            
            # Get from regular state
            current = self._state
            for part in parts:
                current = current[part]
            
            # Handle special types
            if isinstance(current, pd.DataFrame):
                return current.copy()  # Return a copy to prevent modifications
            
            # Handle un-serializable objects
            try:
                json.dumps(current)
                return current
            except (TypeError, OverflowError):
                logger.warning(f"Found un-serializable state at {path}")
                return default
                
        except (KeyError, TypeError):
            return default
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about current state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'current_state': self._state,
            'state_history': [asdict(change) for change in self._state_history[-10:]],  # Last 10 changes
            'registered_validators': list(self._state_validators.keys()),
            'registered_watchers': {k: len(v) for k, v in self._state_watchers.items()},
            'memory_usage': self._get_memory_usage(),
            'version': self.VERSION
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as a percentage."""
        import psutil
        process = psutil.Process()
        return process.memory_percent()
    
    def _check_memory_threshold(self) -> bool:
        """Check if memory usage is below threshold."""
        return self._get_memory_usage() < self.MEMORY_THRESHOLD
    
    def rollback(self, path: str, steps: int = 1) -> None:
        """
        Rollback state at path to a previous version.
        
        Args:
            path: Path to state to rollback
            steps: Number of steps to rollback (default: 1)
        """
        relevant_history = [
            change for change in reversed(self._state_history)
            if change.component == path
        ]
        
        if len(relevant_history) >= steps:
            target_state = relevant_history[steps - 1].old_value
            self.update_state(
                path, target_state,
                StateChangeType.CUSTOM,
                'state_manager',
                f"Rolled back {path} {steps} steps"
            )
    
    def save_state(self, paths: List[str] = None) -> Dict[str, Any]:
        """
        Save current state to a serializable format.
        
        Args:
            paths: Optional list of paths to save (saves all if None)
            
        Returns:
            Dictionary with saved state
        """
        state_to_save = {}
        
        if paths:
            # Save specific paths
            for path in paths:
                # Get all nested paths
                flattened = self._flatten_dict(self._state[path] if path in self._state else {}, prefix=path)
                for nested_path, value in flattened.items():
                    try:
                        # Handle special types
                        if isinstance(value, pd.DataFrame):
                            value = value.to_dict('records')
                        elif isinstance(value, DataType):
                            value = value.value
                        elif isinstance(value, pd.Timestamp):
                            value = value.isoformat()
                        elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                            value = int(value)
                        elif isinstance(value, (np.float64, np.float32)):
                            value = float(value)
                        elif isinstance(value, pd.Series):
                            value = value.tolist()
                        elif isinstance(value, np.ndarray):
                            value = value.tolist()
                        elif isinstance(value, (set, frozenset)):
                            value = list(value)
                        elif isinstance(value, (type, object)):
                            # Skip class instances and types
                            continue
                        
                        # Ensure value is serializable
                        json.dumps(value)
                        
                        # Save value with full path
                        state_to_save[nested_path] = value
                        
                        # Save path in metadata
                        if '_paths' not in state_to_save:
                            state_to_save['_paths'] = []
                        state_to_save['_paths'].append(nested_path)
                    except (TypeError, OverflowError):
                        logger.warning(f"Skipping un-serializable state at {nested_path}")
                        
                # Save top-level path
                if path not in state_to_save:
                    state_to_save[path] = {}
                
                # Save data_loaded state
                if path == 'data':
                    state_to_save['data.data_loaded'] = self._state['data'].get('data_loaded', False)
                    
                    # Save DataFrame as records
                    df = self._state['data'].get('current_df')
                    if isinstance(df, pd.DataFrame):
                        state_to_save['data.current_df'] = df.to_dict('records')
                elif path == 'filters':
                    # Save active filters
                    active_filters = self._state['filters'].get('active_filters', {})
                    state_to_save['filters.active_filters'] = active_filters
                    
                    # Save filter configs
                    filter_configs = self._state['filters'].get('filter_configs', {})
                    state_to_save['filters.filter_configs'] = filter_configs
                    
                    # Save filter results
                    filter_results = self._state['filters'].get('filter_results', {})
                    state_to_save['filters.filter_results'] = filter_results
        else:
            # Save all state
            flattened = self._flatten_dict(self._state)
            for path, value in flattened.items():
                try:
                    # Handle special types
                    if isinstance(value, pd.DataFrame):
                        value = value.to_dict('records')
                    
                    # Ensure value is serializable
                    json.dumps(value)
                    
                    # Save value with full path
                    state_to_save[path] = value
                except (TypeError, OverflowError):
                    logger.warning(f"Skipping un-serializable state at {path}")
        
        return {
            'version': self.VERSION,
            'state': state_to_save,
            'timestamp': datetime.now().isoformat()
        }
    
    def load_state(self, saved_state: Dict[str, Any], partial: bool = False) -> None:
        """
        Load state from saved format.
        
        Args:
            saved_state: State dictionary from save_state()
            partial: If True, only update specified paths
        """
        version = saved_state.get('version', 1)
        
        # Apply migrations if needed
        while version < self.VERSION:
            if version in self._migrations:
                saved_state = self._migrations[version](saved_state)
                version = saved_state['version']
            else:
                raise ValueError(f"No migration path from version {version}")
        
        # Extract state data
        if 'state' in saved_state:
            state_data = saved_state['state']
        else:
            # Handle old format or direct state data
            state_data = {k: v for k, v in saved_state.items() 
                         if k not in ['version', 'timestamp']}
        
        # Get paths to restore
        paths = state_data.get('_paths', [])
        if not paths:
            # If no paths specified, use all non-underscore paths
            paths = [p for p in state_data.keys() if not p.startswith('_')]
        
        # Update state
        for path in paths:
            value = state_data.get(path)
            if value is not None:
                # Handle special types
                if isinstance(value, list):
                    if path.endswith('_df') or path.endswith('current_df'):
                        # Convert list back to DataFrame
                        value = pd.DataFrame(value)
                    elif path.endswith('_series'):
                        # Convert list back to Series
                        value = pd.Series(value)
                    elif path.endswith('_array'):
                        # Convert list back to ndarray
                        value = np.array(value)
                elif isinstance(value, str):
                    if path.endswith('.type'):
                        # Convert string back to DataType enum
                        value = DataType(value)
                    elif path.endswith('_date') or path.endswith('earliest') or path.endswith('latest'):
                        # Convert ISO string back to Timestamp
                        try:
                            value = pd.Timestamp(value)
                        except:
                            pass
                elif isinstance(value, dict) and path.endswith('_df'):
                    # Handle nested DataFrame
                    value = pd.DataFrame(value)
                
                # Update state
                parts = path.split('.')
                current = self._state
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
                
                # Handle special cases
                if path == 'data.data_loaded':
                    self._state['data']['data_loaded'] = bool(value)
                elif path == 'data.current_df':
                    self._state['data']['data_loaded'] = True
                elif path == 'data':
                    # Ensure data_loaded is properly set
                    if 'data_loaded' in value:
                        self._state['data']['data_loaded'] = bool(value['data_loaded'])
                    if 'current_df' in value:
                        self._state['data']['data_loaded'] = True
                
                logger.info(f"Restored state at {path}")
        
        # Initialize any missing top-level paths
        for path in ['data', 'filters', 'view']:
            if path not in self._state:
                self._state[path] = {}
            
            # Initialize default values
            if path == 'data':
                if 'data_loaded' not in self._state[path]:
                    self._state[path]['data_loaded'] = False
                if 'current_df' not in self._state[path]:
                    self._state[path]['current_df'] = None
                if 'data_info' not in self._state[path]:
                    self._state[path]['data_info'] = {}
                if 'column_types' not in self._state[path]:
                    self._state[path]['column_types'] = {}
            elif path == 'filters':
                if 'active_filters' not in self._state[path]:
                    self._state[path]['active_filters'] = {}
                if 'filter_configs' not in self._state[path]:
                    self._state[path]['filter_configs'] = {}
                if 'filter_results' not in self._state[path]:
                    self._state[path]['filter_results'] = {}
            elif path == 'view':
                if 'current_tab' not in self._state[path]:
                    self._state[path]['current_tab'] = None
                if 'display_options' not in self._state[path]:
                    self._state[path]['display_options'] = {}
                if 'ui_settings' not in self._state[path]:
                    self._state[path]['ui_settings'] = {}
        
        logger.info("Restored state")
    
    def _validate_widget_state(self, path: str, value: Any, old_value: Any) -> Any:
        """
        Validate and normalize widget state.
        
        Args:
            path: State path
            value: New value
            old_value: Current value
            
        Returns:
            Validated value
        """
        parts = path.split('.')
        widget_type = parts[-1] if len(parts) > 1 else path
        parent_path = '.'.join(parts[:-1]) if len(parts) > 1 else None
        
        if parent_path:
            parent_state = self.get_state(parent_path)
            if isinstance(parent_state, dict):
                # Handle range widgets (slider, number input)
                if 'min' in parent_state and 'max' in parent_state:
                    if widget_type == 'value':
                        # Ensure value is within bounds
                        min_val = parent_state.get('min', float('-inf'))
                        max_val = parent_state.get('max', float('inf'))
                        if isinstance(value, (int, float)):
                            return max(min_val, min(max_val, value))
                    return value
                
                # Handle selection widgets (dropdown, multiselect)
                if 'options' in parent_state:
                    if widget_type == 'selected':
                        # Ensure selected values are in options
                        options = parent_state.get('options', [])
                        if isinstance(value, list):
                            return [v for v in value if v in options]
                        elif value in options:
                            return value
                        return options[0] if options else None
                    return value
        
        # Handle direct widget state updates
        if isinstance(value, dict):
            if 'min' in value and 'max' in value and 'value' in value:
                # Handle both single value and range value
                if isinstance(value['value'], (tuple, list)):
                    # Range value
                    min_val, max_val = value['value']
                    value['value'] = (
                        max(value['min'], min(value['max'], min_val)),
                        max(value['min'], min(value['max'], max_val))
                    )
                else:
                    # Single value
                    value['value'] = max(value['min'], min(value['max'], value['value']))
            elif 'options' in value and 'selected' in value:
                # Ensure selected values are in options
                if isinstance(value['selected'], list):
                    value['selected'] = [v for v in value['selected'] if v in value['options']]
                elif value['selected'] not in value['options']:
                    value['selected'] = value['options'][0] if value['options'] else None
        
        return value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.', prefix: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary with dot notation.
        
        Args:
            d: Dictionary to flatten
            parent_key: Current parent key
            sep: Separator to use between keys
            prefix: Optional prefix to add to all keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if prefix:
                new_key = f"{prefix}{sep}{new_key}"
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def register_migration(self, from_version: int, migration_func: Callable) -> None:
        """
        Register a migration function for a specific version.
        
        Args:
            from_version: Version to migrate from
            migration_func: Function that takes old state and returns new state
        """
        if not hasattr(self, '_migrations'):
            self._migrations = {}
        self._migrations[from_version] = migration_func
        logger.info(f"Registered migration from version {from_version}")
    
    def validate_state_consistency(self) -> bool:
        """
        Check and fix state consistency.
        
        Returns:
            True if state was consistent, False if fixes were needed
        """
        was_consistent = True
        
        # Check data consistency
        df = self.get_state('data.df')
        if df is not None:
            # Update row count
            row_count = self.get_state('data.row_count')
            if row_count != len(df):
                self.update_state('data.row_count', len(df),
                                StateChangeType.CUSTOM,
                                'state_manager',
                                "Fixed inconsistent row count")
                was_consistent = False
            
            # Update filtered data
            active_filters = self.get_state('filters', {})
            if active_filters:
                filtered_df = self.apply_filters(list(active_filters.keys()))
                if filtered_df is not None:
                    self.update_state('data.filtered_df', filtered_df,
                                    StateChangeType.CUSTOM,
                                    'state_manager',
                                    "Updated filtered data")
                    
                    # Update filtered count
                    filtered_count = self.get_state('view.filtered_count')
                    if filtered_count != len(filtered_df):
                        self.update_state('view.filtered_count', len(filtered_df),
                                        StateChangeType.CUSTOM,
                                        'state_manager',
                                        "Fixed inconsistent filtered count")
                        was_consistent = False
        
        # Check widget state consistency
        for path, value in self._flatten_dict(self.get_state('widgets', {})).items():
            if isinstance(value, dict):
                # Check range widgets
                if 'min' in value and 'max' in value:
                    if 'value' in value:
                        if value['value'] < value['min'] or value['value'] > value['max']:
                            # Reset to middle of range
                            new_value = (value['min'] + value['max']) / 2
                            self.update_state(f"{path}.value", new_value,
                                            StateChangeType.CUSTOM,
                                            'state_manager',
                                            f"Reset {path} to valid range")
                            was_consistent = False
                
                # Check selection widgets
                if 'options' in value and 'selected' in value:
                    if isinstance(value['selected'], list):
                        valid_selected = [v for v in value['selected'] if v in value['options']]
                        if len(valid_selected) != len(value['selected']):
                            self.update_state(f"{path}.selected", valid_selected,
                                            StateChangeType.CUSTOM,
                                            'state_manager',
                                            f"Removed invalid selections from {path}")
                            was_consistent = False
                    elif value['selected'] not in value['options']:
                        self.update_state(f"{path}.selected", value['options'][0] if value['options'] else None,
                                        StateChangeType.CUSTOM,
                                        'state_manager',
                                        f"Reset invalid selection in {path}")
                        was_consistent = False
        
        return was_consistent
    
    def apply_filters(self, filter_order: List[str]) -> pd.DataFrame:
        """
        Apply filters in specified order.
        
        Args:
            filter_order: List of filter names in application order
            
        Returns:
            Filtered DataFrame
        """
        df = self.get_state('data.df')
        if df is None:
            return None
        
        result = df.copy()
        
        for filter_name in filter_order:
            filter_config = self.get_state(f'filters.{filter_name}')
            if filter_config:
                result = self._apply_filter(result, filter_name, filter_config)
        
        return result
    
    def _apply_filter(self, df: pd.DataFrame, column: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single filter."""
        if 'min' in config and 'max' in config:
            return df[(df[column] >= config['min']) & (df[column] <= config['max'])]
        elif 'values' in config:
            return df[df[column].isin(config['values'])]
        return df
    
    def add_filter_dependency(self, filter_name: str, depends_on: str) -> None:
        """
        Add a filter dependency.
        
        Args:
            filter_name: Name of the filter
            depends_on: Name of the filter it depends on
        """
        if not hasattr(self, '_filter_dependencies'):
            self._filter_dependencies = {}
        
        # Check for circular dependencies
        def check_circular(current: str, target: str, visited: set) -> bool:
            if current == target:
                return True
            if current in self._filter_dependencies:
                for dep in self._filter_dependencies[current]:
                    if dep not in visited:
                        visited.add(dep)
                        if check_circular(dep, target, visited):
                            return True
            return False
        
        if check_circular(depends_on, filter_name, {depends_on}):
            raise ValueError(f"Adding dependency from {filter_name} to {depends_on} would create a circular dependency")
        
        if filter_name not in self._filter_dependencies:
            self._filter_dependencies[filter_name] = set()
        self._filter_dependencies[filter_name].add(depends_on)
    
    def clear_state(self, path: str = None) -> None:
        """
        Clear state at a specific path or all state if no path provided.
        
        Args:
            path: Optional path to clear
        """
        if path:
            parts = path.split('.')
            if len(parts) == 1:
                # Clear top-level path
                if path in self._state:
                    self._state[path] = {
                        'data_loaded': False,
                        'current_df': None,
                        'data_info': {},
                        'column_types': {},
                        'column_info': {}
                    }
                if path in self._instances:
                    del self._instances[path]
            else:
                # Clear nested path
                current = self._state
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                if parts[-1] in current:
                    current[parts[-1]] = None
            
            logger.info(f"Cleared state at {path}")
        else:
            self._initialize_state()
            self._instances = {}
            logger.info("All state cleared")
    
    # Filter-specific methods
    def update_filter(self, column: str, config: Dict[str, Any]) -> None:
        """Update filter configuration and trigger recalculation."""
        self.update_state(
            f'filters.filter_configs.{column}',
            config,
            StateChangeType.FILTER,
            'filter_manager',
            f"Updated filter config for {column}"
        )
        self._recalculate_filters()
    
    def set_filter_active(self, column: str, active: bool) -> None:
        """Set filter active state."""
        self.update_state(
            f'filters.active_filters.{column}',
            active,
            StateChangeType.FILTER,
            'filter_manager',
            f"Set filter {column} active state to {active}"
        )
        self._recalculate_filters()
    
    def _recalculate_filters(self) -> None:
        """Recalculate filtered data based on current filters."""
        df = self.get_state('data.original_df')
        if df is None:
            return
        
        filtered_df = df.copy()
        active_filters = self.get_state('filters.active_filters', {})
        filter_configs = self.get_state('filters.filter_configs', {})
        
        for column, is_active in active_filters.items():
            if not is_active or column not in filter_configs:
                continue
                
            config = filter_configs[column]
            # Apply filter based on config
            # This is a placeholder - actual filter logic would go here
            
        self.update_state(
            'data.filtered_df',
            filtered_df,
            StateChangeType.FILTER,
            'filter_manager',
            "Updated filtered data"
        )
    
    # Data-specific methods
    def set_data(self, df: pd.DataFrame, column_types: Dict[str, Any]) -> None:
        """Set the current DataFrame and column types."""
        self.update_state('data.original_df', df,
                         StateChangeType.DATA, 'data_handler',
                         "Updated original DataFrame")
        self.update_state('data.filtered_df', df.copy(),
                         StateChangeType.DATA, 'data_handler',
                         "Updated filtered DataFrame")
        self.update_state('data.column_types', column_types,
                         StateChangeType.DATA, 'data_handler',
                         "Updated column types")
    
    def get_filtered_data(self) -> Optional[pd.DataFrame]:
        """Get the current filtered DataFrame."""
        return self.get_state('data.filtered_df')
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the current data state."""
        filtered_df = self.get_state('data.filtered_df')
        original_df = self.get_state('data.original_df')
        
        if filtered_df is None or original_df is None:
            return {}
            
        return {
            'total_records': len(original_df),
            'filtered_records': len(filtered_df),
            'columns': list(filtered_df.columns),
            'memory_usage': filtered_df.memory_usage(deep=True).sum() / 1024**2
        }

# Example usage:
"""
# Initialize state manager
state_manager = StateManager()

# Register a custom extension
state_manager.register_extension('reports', {
    'current_report': None,
    'report_settings': {}
})

# Register a validator
def validate_filter_config(config):
    return isinstance(config, dict) and 'type' in config

state_manager.register_validator('filters.filter_configs', validate_filter_config)

# Register a watcher
def on_filter_change(old_value, new_value):
    logger.info(f"Filter changed from {old_value} to {new_value}")

state_manager.register_watcher('filters.active_filters', on_filter_change)

# Update state
state_manager.update_state('filters.active_filters.category', True,
                          StateChangeType.FILTER, 'filter_manager',
                          "Activated category filter")
"""
