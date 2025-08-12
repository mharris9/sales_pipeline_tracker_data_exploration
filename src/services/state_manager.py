"""
StateManager class for centralized state management.
"""
import logging
from typing import Any, Dict, Optional
import json
from copy import deepcopy

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
        except Exception as e:
            logger.error("Error setting state at path %s: %s", path, str(e))

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

    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to a serializable format.
        
        Returns:
            Dictionary with saved state
        """
        try:
            # Convert state to serializable format
            serializable_state = {}
            for category, values in self._state.items():
                serializable_state[category] = {}
                for key, value in values.items():
                    # Convert numpy types to Python types
                    if hasattr(value, 'dtype'):
                        value = value.item()
                    # Convert dictionaries recursively
                    if isinstance(value, dict):
                        serializable_value = {}
                        for k, v in value.items():
                            if hasattr(v, 'dtype'):
                                v = v.item()
                            serializable_value[str(k)] = v
                        value = serializable_value
                    serializable_state[category][key] = value
            
            return {
                'version': self.VERSION,
                'state': serializable_state,
                'timestamp': json.dumps(self._history[-1] if self._history else {})
            }
        except Exception as e:
            logger.error("Error saving state: %s", str(e))
            return {}

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
                    # Update while preserving structure
                    for key, value in values.items():
                        if key in self._state[category]:
                            if isinstance(self._state[category][key], dict):
                                if isinstance(value, dict):
                                    self._state[category][key].update(value)
                            else:
                                self._state[category][key] = value
            
            logger.info("State loaded successfully")
        except Exception as e:
            logger.error("Error loading state: %s", str(e))

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
            'errors': self.get_state('errors')
        }