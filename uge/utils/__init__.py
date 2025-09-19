"""
Utils Package

This package contains utility functions and helper classes for the UGE application.

Utilities provide common functionality used across the application:
- helpers: General utility functions (ID generation, fitness functions, etc.)
- constants: Application constants and configuration values
- logger: Custom logging utilities for Streamlit

Utils are responsible for:
- Providing reusable helper functions
- Managing application constants
- Handling common operations
- Supporting other components with shared functionality
"""

from .helpers import create_setup_id, create_run_id, mae, accuracy, fitness_eval
from .logger import StreamlitLogger

# Import constants with error handling to avoid circular imports
try:
    from .constants import DEFAULT_CONFIG, FILE_PATHS, UI_CONSTANTS
except ImportError:
    # Fallback constants if import fails
    DEFAULT_CONFIG = {}
    FILE_PATHS = {}
    UI_CONSTANTS = {}

__all__ = ['create_setup_id', 'create_run_id', 'mae', 'accuracy', 'fitness_eval',
           'DEFAULT_CONFIG', 'FILE_PATHS', 'UI_CONSTANTS', 'StreamlitLogger']