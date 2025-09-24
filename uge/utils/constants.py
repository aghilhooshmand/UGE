"""
Constants and Configuration for UGE Application

This module contains application-wide constants, default configurations,
and file paths used throughout the UGE application.

Constants:
- FILE_PATHS: Common file and directory paths
- DEFAULT_CONFIG: Default setup configuration values (loaded from settings)
- UI_CONSTANTS: User interface related constants (loaded from settings)

Author: UGE Team
"""

import json
from pathlib import Path
from uge.services.settings_service import SettingsService

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Initialize settings service
_settings_service = SettingsService()

# Load help texts from unified configuration
HELP = {}
help_path = PROJECT_ROOT / "uge" / "utils" / "tooltip_config.json"
if help_path.exists():
    try:
        config = json.loads(help_path.read_text())
        # Extract setup parameters for backward compatibility
        HELP = config.get('setup_parameters', {})
    except Exception:
        HELP = {}

# File and Directory Paths
FILE_PATHS = {
    'project_root': PROJECT_ROOT,
    'results_dir': PROJECT_ROOT / "results",
    'setups_dir': PROJECT_ROOT / "results" / "setups",
    'datasets_dir': PROJECT_ROOT / "datasets",
    'grammars_dir': PROJECT_ROOT / "grammars",
    'tooltip_config': PROJECT_ROOT / "uge" / "utils" / "tooltip_config.json",
}

# Default Setup Configuration (loaded from settings)
DEFAULT_CONFIG = {
    # Load from settings service
    **_settings_service.get_ge_parameters(),
    **_settings_service.get_defaults(),
    'default_report_items': _settings_service.get_report_items(),
    'track_generation_configs': _settings_service.get_setting('generation_config.track_generation_configs', True),
    'generation_config_params': _settings_service.get_setting('generation_config.generation_config_params', []),
    'evolution_types': _settings_service.get_evolution_types(),
    'default_evolution_type': _settings_service.get_setting('default_evolution_type', 'fixed')
}

# User Interface Constants (loaded from settings)
UI_CONSTANTS = {
    # Load from settings service
    **_settings_service.get_app_info(),
    **_settings_service.get_ui_constants(),
    **_settings_service.get_defaults(),
    
    # Navigation Options
    'navigation_pages': [
        "🏃 Run Setup",
        "📊 Dataset Manager", 
        "📝 Grammar Editor",
        "🧪 Setup Manager",
        "📈 Analysis",
        "⚖️ Comparison",
        "⚙️ Settings"
    ],
}