"""
Settings Service for UGE Application

This module provides a service for loading and saving application settings
from a JSON configuration file.

Classes:
- SettingsService: Service for managing application settings

Author: UGE Team
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from uge.utils.logger import StreamlitLogger


class SettingsService:
    """
    Service for managing application settings.
    
    This service handles loading and saving settings from/to a JSON file,
    providing a centralized way to manage application configuration.
    """
    
    def __init__(self, settings_file: str = "settings.json"):
        """
        Initialize settings service.
        
        Args:
            settings_file (str): Path to settings JSON file
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.settings_file = self.project_root / settings_file
        self._settings = {}
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load settings from JSON file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self._settings = json.load(f)
            else:
                # Create default settings file if it doesn't exist
                self._create_default_settings()
        except Exception as e:
            print(f"Error loading settings: {e}")
            self._create_default_settings()
    
    def _create_default_settings(self) -> None:
        """Create default settings file."""
        default_settings = {
            "app": {
                "name": "UGE - Grammatical Evolution Platform",
                "version": "1.0.0",
                "page_title": "Grammatical Evolution",
                "app_title": "🧬 Grammatical Evolution",
                "app_subtitle": "** Learning GE with comprehensive analysis and comparison**"
            },
            "ui": {
                "chart_height": 500,
                "chart_template": "plotly_white",
                "max_log_lines": 200,
                "max_upload_size_mb": 200,
                "allowed_dataset_extensions": [".csv", ".data", ".txt"]
            },
            "defaults": {
                "dataset": "processed.cleveland.data",
                "grammar": "heartDisease.bnf",
                "fitness_metric": "mae",
                "label_column": "class"
            },
            "ge_parameters": {
                "population": 50,
                "generations": 20,
                "n_runs": 3,
                "p_crossover": 0.8,
                "p_mutation": 0.01,
                "elite_size": 1,
                "tournsize": 7,
                "halloffame_size": 1,
                "max_tree_depth": 35,
                "min_init_tree_depth": 4,
                "max_init_tree_depth": 13,
                "min_init_genome_length": 95,
                "max_init_genome_length": 115,
                "codon_size": 255,
                "codon_consumption": "lazy",
                "genome_representation": "list",
                "initialisation": "sensible",
                "random_seed": 42,
                "test_size": 0.3
            },
            "report_items": [
                "gen", "invalid", "avg", "std", "min", "max", "fitness_test",
                "best_ind_length", "avg_length", "best_ind_nodes", "avg_nodes",
                "best_ind_depth", "avg_depth", "avg_used_codons", "best_ind_used_codons",
                "invalid_count_min", "invalid_count_avg", "invalid_count_max", "invalid_count_std",
                "nodes_length_min", "nodes_length_avg", "nodes_length_max", "nodes_length_std",
                "structural_diversity", "fitness_diversity", "selection_time", "generation_time"
            ],
            "generation_config": {
                "track_generation_configs": True,
                "generation_config_params": [
                    "population", "p_crossover", "p_mutation", "elite_size", "tournsize", 
                    "halloffame_size", "max_tree_depth", "codon_size", "codon_consumption", 
                    "genome_representation"
                ]
            },
            "evolution_types": {
                "fixed": {
                    "name": "Fixed Evolution",
                    "description": "Use the same configuration for all generations (recommended)",
                    "track_configs": True
                }
            },
            "default_evolution_type": "fixed",
            "parameter_limits": {
                "population": {"min": 10, "max": 5000},
                "generations": {"min": 1, "max": 2000},
                "n_runs": {"min": 1, "max": 100},
                "p_crossover": {"min": 0.0, "max": 1.0},
                "p_mutation": {"min": 0.0, "max": 1.0},
                "elite_size": {"min": 0, "max": 50},
                "tournsize": {"min": 2, "max": 50},
                "halloffame_size": {"min": 1, "max": 100},
                "max_tree_depth": {"min": 5, "max": 100},
                "min_init_tree_depth": {"min": 1, "max": 20},
                "max_init_tree_depth": {"min": 1, "max": 200},
                "min_init_genome_length": {"min": 10, "max": 200},
                "max_init_genome_length": {"min": 10, "max": 500},
                "codon_size": {"min": 100, "max": 512},
                "test_size": {"min": 0.1, "max": 0.5},
                "random_seed": {"min": 0, "max": 10000}
            },
            "parameter_options": {
                "codon_consumption": ["lazy", "eager"],
                "genome_representation": ["list"],
                "initialisation": ["sensible", "random"],
                "fitness_metric": ["mae", "accuracy"]
            }
        }
        self._settings = default_settings
        self.save_settings()
    
    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to setting (e.g., 'ge_parameters.population')
            default (Any): Default value if setting not found
            
        Returns:
            Any: Setting value or default
        """
        keys = key_path.split('.')
        value = self._settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, key_path: str, value: Any) -> None:
        """
        Set a setting value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to setting
            value (Any): Value to set
        """
        keys = key_path.split('.')
        current = self._settings
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings.copy()
    
    def save_settings(self) -> bool:
        """
        Save settings to JSON file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self._create_default_settings()
    
    def get_ge_parameters(self) -> Dict[str, Any]:
        """Get GE parameters section."""
        return self.get_setting('ge_parameters', {})
    
    def get_ui_constants(self) -> Dict[str, Any]:
        """Get UI constants section."""
        return self.get_setting('ui', {})
    
    def get_parameter_limits(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter limits section."""
        return self.get_setting('parameter_limits', {})
    
    def get_parameter_options(self) -> Dict[str, list]:
        """Get parameter options section."""
        return self.get_setting('parameter_options', {})
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get defaults section."""
        return self.get_setting('defaults', {})
    
    def get_report_items(self) -> list:
        """Get report items list."""
        return self.get_setting('report_items', [])
    
    def get_evolution_types(self) -> Dict[str, Any]:
        """Get evolution types section."""
        return self.get_setting('evolution_types', {})
    
    def get_app_info(self) -> Dict[str, Any]:
        """Get app info section."""
        return self.get_setting('app', {})
