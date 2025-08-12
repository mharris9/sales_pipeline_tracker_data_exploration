"""
StateManager class for centralized state management.
"""
import logging
from typing import Any, Dict, Optional
import json

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages centralized state with a simplified path structure.
    """
    VERSION = 1

    def __init__(self):
        """Initialize the state manager with basic state categories."""
        self._state = {
            'feature_configs': {},  # Feature configurations
            'feature_active': {},   # Active state of features
            'feature_results': {},  # Feature calculation results
            'feature_summary': {},  # Summary information about features
        }
        self._history = []
        logger.info("StateManager initialized with categories: %s", list(self._state.keys()))

    def get_state(self, path: str, default: Any = None) -> Any:
        """
        Get state using path like 'feature_configs/category_counts'.
        
        Args:
            path: Path in format 'category/key'
            default: Default value if path doesn't exist
            
        Returns:
            State value or default
        """
        try:
            if '/' in path:
                category, key = path.split('/')
                return self._state.get(category, {}).get(key, default)
            else:
                return self._state.get(path, default)
        except Exception as e:
            logger.error("Error getting state at path %s: %s", path, str(e))
            return default

    def set_state(self, path: str, value: Any) -> None:
        """
        Set state using path like 'feature_configs/category_counts'.
        
        Args:
            path: Path in format 'category/key'
            value: Value to store
        """
        try:
            if '/' in path:
                category, key = path.split('/')
                if category not in self._state:
                    self._state[category] = {}
                self._state[category][key] = value
            else:
                self._state[path] = value
            
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
            path: Optional path to clear (format: 'category' or 'category/key')
        """
        try:
            if path is None:
                self._state = {category: {} for category in self._state}
            elif '/' in path:
                category, key = path.split('/')
                if category in self._state and key in self._state[category]:
                    del self._state[category][key]
            elif path in self._state:
                self._state[path] = {}
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
            
            # Update state categories
            for category in self._state:
                if category in state_data:
                    self._state[category] = state_data[category]
                else:
                    self._state[category] = {}
            
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
            'last_update': self._history[-1] if self._history else None
        }