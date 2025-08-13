"""
StateManager class for centralized state management.
"""
import logging
from typing import Any, Dict, Optional
import json
from copy import deepcopy
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages centralized state with a simplified path structure.
    """
    VERSION = 1

    def __init__(self):
        """Initialize the state manager with basic state categories."""
        self._state = {}
        self._history = []
        self._initialize_state()
        logger.info("StateManager initialized with categories: %s", list(self._state.keys()))
    
    def _initialize_state(self) -> None:
        """Initialize the core state structure."""
        initial_state = {
            'data': {
                'current_df': None,
                'loaded': False,
                'info': {},
                'stats': {},
                'column_types': {}
            },
            'filters': {
                'configs': {},
                'active': {},
                'results': {},
                'stats': {
                    'filtered_count': 0,
                    'total_count': 0
                }
            },
            'features': {
                'configs': {},
                'active': {},
                'results': {},
                'stats': {}
            },
            'outliers': {
                'configs': {},
                'active': {},
                'results': {},
                'stats': {}
            },
            'reports': {
                'configs': {},
                'active': {},
                'results': {},
                'stats': {}
            },
            'ui': {
                'widgets': {},
                'expanded': {},
                'selected': {},
                'page': 'main'
            },
            'errors': {
                'last_error': None,
                'count': 0,
                'history': []
            }
        }
        
        # Initialize each category with a deep copy
        self._state = deepcopy(initial_state)
    
    def _get_nested_dict(self, path: str) -> tuple[Dict, str]:
        """
        Get the nested dictionary and final key for a path.
        
        Args:
            path: Path in format 'category/subcategory/key'
            
        Returns:
            Tuple of (parent_dict, final_key)
        """
        parts = path.split('/')
        current = self._state
        
        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
            # Ensure we maintain dict structure
            if not isinstance(current, dict):
                current = {}
        
        return current, parts[-1] if parts else ''

    def get_state(self, path: str, default: Any = None) -> Any:
        """
        Get state using path like 'features/configs/category_counts'.
        
        Args:
            path: Path in format 'category/subcategory/key'
            default: Default value if path doesn't exist
            
        Returns:
            State value or default
        """
        try:
            if not path:
                return default
            
            parts = path.split('/')
            current = self._state
            
            # Navigate through path
            for part in parts[:-1]:
                if part not in current:
                    return default if not isinstance(default, dict) else {}
                current = current[part]
                if not isinstance(current, dict):
                    return default
            
            # Handle final key
            final_key = parts[-1]
            if final_key not in current:
                return default if not isinstance(default, dict) else {}
            
            # Return value
            value = current[final_key]
            if isinstance(value, dict):
                # Return empty dict for dictionary paths that exist
                return value if value else {}
            return value
            
        except Exception as e:
            logger.error("Error getting state at path %s: %s", path, str(e))
            return default

    def set_state(self, path: str, value: Any) -> None:
        """
        Set state using path like 'features/configs/category_counts'.
        
        Args:
            path: Path in format 'category/subcategory/key'
            value: Value to store
        """
        try:
            if not path:
                return
            
            current, final_key = self._get_nested_dict(path)
            
            # Set the value
            current[final_key] = value
            
            # Save state in history
            self._history.append({
                'path': path,
                'value': value
            })
            logger.debug("State updated at path %s", path)
            
            # Save to session state if available
            try:
                import streamlit as st
                if hasattr(st, 'session_state'):
                    st.session_state.state_data = self.save_state()
            except ImportError:
                pass  # Not in Streamlit context
        except Exception as e:
            logger.error("Error setting state at path %s: %s", path, str(e))

    def update_state(self, path: str, value: Any) -> None:
        """
        Update state using path like 'features/configs/category_counts'.
        This is an alias for set_state for backward compatibility.
        
        Args:
            path: Path in format 'category/subcategory/key'
            value: Value to store
        """
        self.set_state(path, value)

    def clear_state(self, path: Optional[str] = None) -> None:
        """
        Clear state at path or all state if no path provided.
        
        Args:
            path: Optional path to clear (format: 'category/subcategory/key')
        """
        try:
            if path is None:
                # Reset to initial structure
                self._initialize_state()
                return
            
            parts = path.split('/')
            current = self._state
            
            # Navigate to parent
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                if not isinstance(current, dict):
                    current = {}
            
            # Clear the target
            if parts:
                final_key = parts[-1]
                if final_key in current:
                    if isinstance(current[final_key], dict):
                        # Preserve structure for dictionary paths
                        if final_key in self._state:
                            # Top-level category - preserve structure
                            current[final_key] = {
                                k: {} for k in self._state[final_key].keys()
                            }
                        else:
                            # Nested path - empty dict
                            current[final_key] = {}
                    else:
                        # Remove leaf values
                        del current[final_key]
            
            logger.info("State cleared at path: %s", path or 'all')
        except Exception as e:
            logger.error("Error clearing state at path %s: %s", path, str(e))

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable version of the object
        """
        
        try:
            # Handle pandas DataFrames
            if isinstance(obj, pd.DataFrame):
                return {
                    '_type': 'DataFrame',
                    'columns': obj.columns.tolist(),
                    'index': obj.index.tolist(),
                    'data': obj.values.tolist(),
                    'shape': obj.shape
                }
            
            # Handle pandas Series
            elif isinstance(obj, pd.Series):
                return {
                    '_type': 'Series',
                    'index': obj.index.tolist(),
                    'data': obj.values.tolist(),
                    'name': obj.name
                }
            
            # Handle numpy arrays
            elif isinstance(obj, np.ndarray):
                return {
                    '_type': 'ndarray',
                    'data': obj.tolist(),
                    'shape': obj.shape,
                    'dtype': str(obj.dtype)
                }
            
            # Handle numpy scalars
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            
            # Handle datetime objects
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return {
                    '_type': 'datetime',
                    'value': obj.isoformat()
                }
            
            # Handle dictionaries recursively
            elif isinstance(obj, dict):
                return {str(k): self._make_serializable(v) for k, v in obj.items()}
            
            # Handle lists recursively
            elif isinstance(obj, list):
                return [self._make_serializable(item) for item in obj]
            
            # Handle tuples
            elif isinstance(obj, tuple):
                return tuple(self._make_serializable(item) for item in obj)
            
            # Handle basic types that are already serializable
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            
            # For other objects, try to convert to string or skip
            else:
                logger.warning(f"Cannot serialize object of type {type(obj)}: {obj}")
                return None
                
        except Exception as e:
            logger.error(f"Error serializing object {type(obj)}: {str(e)}")
            return None

    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to a serializable format.
        
        Returns:
            Dictionary with saved state
        """
        try:
            # Convert state to serializable format
            serializable_state = {}
            
            # Process all state entries and create proper nested structure
            for category, values in self._state.items():
                if isinstance(values, dict):
                    # Initialize category if not exists
                    if category not in serializable_state:
                        serializable_state[category] = {}
                    
                    # Process each key-value pair in the category
                    for key, value in values.items():
                        serializable_value = self._make_serializable(value)
                        if serializable_value is not None:
                            serializable_state[category][key] = serializable_value
                elif isinstance(values, list):
                    # List category - serialize the entire list
                    serializable_value = self._make_serializable(values)
                    if serializable_value is not None:
                        serializable_state[category] = serializable_value
                else:
                    # Other types - serialize directly
                    serializable_value = self._make_serializable(values)
                    if serializable_value is not None:
                        serializable_state[category] = serializable_value
            
            return {
                'version': self.VERSION,
                'state': serializable_state,
                'timestamp': json.dumps(self._history[-1] if self._history else {})
            }
        except Exception as e:
            logger.error("Error saving state: %s", str(e))
            return {}

    def _restore_from_serializable(self, obj: Any) -> Any:
        """
        Restore an object from serializable format.
        
        Args:
            obj: Serializable object to restore
            
        Returns:
            Restored object
        """
        import pandas as pd
        import numpy as np
        
        try:
            # Handle dictionaries with type information
            if isinstance(obj, dict) and '_type' in obj:
                obj_type = obj['_type']
                
                if obj_type == 'DataFrame':
                    # Restore DataFrame
                    df = pd.DataFrame(obj['data'], columns=obj['columns'], index=obj['index'])
                    return df
                
                elif obj_type == 'Series':
                    # Restore Series
                    series = pd.Series(obj['data'], index=obj['index'], name=obj['name'])
                    return series
                
                elif obj_type == 'ndarray':
                    # Restore numpy array
                    arr = np.array(obj['data'], dtype=obj['dtype'])
                    return arr
                
                elif obj_type == 'datetime':
                    # Restore datetime
                    return pd.to_datetime(obj['value'])
                
                else:
                    # Unknown type, return as-is
                    return obj
            
            # Handle dictionaries recursively
            elif isinstance(obj, dict):
                return {k: self._restore_from_serializable(v) for k, v in obj.items()}
            
            # Handle lists recursively
            elif isinstance(obj, list):
                return [self._restore_from_serializable(item) for item in obj]
            
            # Handle tuples
            elif isinstance(obj, tuple):
                return tuple(self._restore_from_serializable(item) for item in obj)
            
            # Handle basic types
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            
            # For other objects, return as-is
            else:
                return obj
                
        except Exception as e:
            logger.error(f"Error restoring object {type(obj)}: {str(e)}")
            return obj

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Load state from saved format.
        
        Args:
            saved_state: State dictionary from save_state()
        """
        try:
            if not isinstance(saved_state, dict):
                raise ValueError("Invalid state format")
            
            version = saved_state.get('version', 1)
            if version != self.VERSION:
                logger.warning("Loading state from different version: %d -> %d", 
                             version, self.VERSION)
            
            state_data = saved_state.get('state', {})
            if not isinstance(state_data, dict):
                raise ValueError("Invalid state data format")
            
            # Initialize empty state
            self._initialize_state()
            
            # Update state categories
            for category, values in state_data.items():
                if category in self._state:
                    # Handle different types of category values
                    if isinstance(values, dict):
                        # Dictionary category - update key-value pairs
                        for key, value in values.items():
                            # Restore complex objects
                            restored_value = self._restore_from_serializable(value)
                            
                            if key in self._state[category]:
                                if isinstance(self._state[category][key], dict):
                                    if isinstance(restored_value, dict):
                                        self._state[category][key].update(restored_value)
                                else:
                                    self._state[category][key] = restored_value
                            else:
                                # New key, add it
                                self._state[category][key] = restored_value
                    elif isinstance(values, list):
                        # List category - replace entire list
                        restored_value = self._restore_from_serializable(values)
                        self._state[category] = restored_value
                    else:
                        # Other types - replace directly
                        restored_value = self._restore_from_serializable(values)
                        self._state[category] = restored_value
            
            logger.info("State loaded successfully")
        except Exception as e:
            logger.error("Error loading state: %s", str(e))

    def register_extension(self, name: str, extension_data: Dict[str, Any]) -> None:
        """
        Register an extension with the state manager.
        
        Args:
            name: Name of the extension
            extension_data: Dictionary containing extension data
        """
        try:
            if not isinstance(extension_data, dict):
                raise ValueError(f"Extension data must be a dictionary, got {type(extension_data)}")
            
            # Store extension data in state
            self.set_state(f'extensions/{name}', extension_data)
            logger.info("Extension registered: %s", name)
        except Exception as e:
            logger.error("Error registering extension %s: %s", name, str(e))

    def get_extension(self, path: str) -> Any:
        """
        Get an extension by path.
        
        Args:
            path: Path to the extension (e.g., 'data_handler' or 'filters.filter_manager')
            
        Returns:
            Extension object or None if not found
        """
        try:
            # Handle nested paths like 'filters.filter_manager'
            if '.' in path:
                category, extension_name = path.split('.', 1)
                extension_data = self.get_state(f'extensions/{category}')
                if extension_data and extension_name in extension_data:
                    return extension_data[extension_name]
            else:
                # Direct extension name
                extension_data = self.get_state(f'extensions/{path}')
                if extension_data:
                    # Return the first item if it's a dictionary
                    if isinstance(extension_data, dict) and len(extension_data) == 1:
                        return list(extension_data.values())[0]
                    return extension_data
            
            return None
        except Exception as e:
            logger.error("Error getting extension %s: %s", path, str(e))
            return None

    def register_validator(self, path: str, validator_func) -> None:
        """
        Register a validator function for a state path.
        
        Args:
            path: State path to validate
            validator_func: Function that takes a value and returns bool
        """
        try:
            validators = self.get_state('validators', {})
            validators[path] = validator_func
            self.set_state('validators', validators)
            logger.info("Validator registered for path: %s", path)
        except Exception as e:
            logger.error("Error registering validator for %s: %s", path, str(e))

    def register_watcher(self, path: str, watcher_func) -> None:
        """
        Register a watcher function for a state path.
        
        Args:
            path: State path to watch
            watcher_func: Function that takes (old_value, new_value)
        """
        try:
            watchers = self.get_state('watchers', {})
            watchers[path] = watcher_func
            self.set_state('watchers', watchers)
            logger.info("Watcher registered for path: %s", path)
        except Exception as e:
            logger.error("Error registering watcher for %s: %s", path, str(e))

    def trigger_rerun(self) -> None:
        """
        Trigger a Streamlit rerun.
        This is a placeholder for integration with Streamlit.
        """
        try:
            # In a real Streamlit context, this would call st.rerun()
            logger.info("Rerun triggered")
        except Exception as e:
            logger.error("Error triggering rerun: %s", str(e))

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about current state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'categories': list(self._state.keys()),
            'history_length': len(self._history),
            'last_update': self._history[-1] if self._history else None,
            'errors': self.get_state('errors'),
            'extensions': list(self.get_state('extensions', {}).keys())
        }