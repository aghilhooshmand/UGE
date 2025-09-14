"""
Tooltip Manager for UGE Application

This module provides a centralized way to manage all tooltip text
through a configuration file, making the application more customizable.

Classes:
- TooltipManager: Manages tooltip text loading and retrieval

Author: UGE Team
"""

import json
import os
from typing import Dict, Any, Optional


class TooltipManager:
    """
    Manager for tooltip text configuration.
    
    This class loads tooltip text from a configuration file and provides
    methods to retrieve tooltip text for different UI elements.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize tooltip manager.
        
        Args:
            config_path (Optional[str]): Path to tooltip config file
        """
        if config_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'tooltip_config.json')
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load tooltip configuration from file.
        
        Returns:
            Dict[str, Any]: Tooltip configuration
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Tooltip config file not found at {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in tooltip config file: {e}")
            return {}
    
    def get_tooltip(self, category: str, key: str, subkey: Optional[str] = None) -> str:
        """
        Get tooltip text for a specific UI element.
        
        Args:
            category (str): Main category (e.g., 'metrics', 'performance_insights')
            key (str): Key within the category
            subkey (Optional[str]): Sub-key for nested values
            
        Returns:
            str: Tooltip text or empty string if not found
        """
        try:
            if category not in self._config:
                return ""
            
            category_config = self._config[category]
            if key not in category_config:
                return ""
            
            if subkey is not None:
                if isinstance(category_config[key], dict) and subkey in category_config[key]:
                    return category_config[key][subkey]
                else:
                    return ""
            else:
                value = category_config[key]
                if isinstance(value, str):
                    return value
                else:
                    return ""
        
        except (KeyError, TypeError):
            return ""
    
    def get_metric_tooltip(self, metric_name: str, context: Optional[str] = None) -> str:
        """
        Get tooltip for a metric with optional context.
        
        Args:
            metric_name (str): Name of the metric
            context (Optional[str]): Context-specific tooltip (e.g., 'maximize', 'minimize')
            
        Returns:
            str: Tooltip text
        """
        if context:
            return self.get_tooltip('metrics', metric_name, context)
        else:
            return self.get_tooltip('metrics', metric_name)
    
    def get_insight_tooltip(self, insight_type: str, level: str) -> str:
        """
        Get tooltip for performance insights.
        
        Args:
            insight_type (str): Type of insight (e.g., 'success_rate', 'consistency')
            level (str): Level of the insight (e.g., 'high', 'low', '0_percent')
            
        Returns:
            str: Tooltip text
        """
        return self.get_tooltip('performance_insights', insight_type, level)
    
    def get_threshold_tooltip(self, threshold_name: str) -> str:
        """
        Get tooltip for analysis thresholds.
        
        Args:
            threshold_name (str): Name of the threshold
            
        Returns:
            str: Tooltip text
        """
        return self.get_tooltip('analysis_thresholds', threshold_name)
    
    def get_chart_tooltip(self, chart_type: str) -> str:
        """
        Get tooltip for chart types.
        
        Args:
            chart_type (str): Type of chart
            
        Returns:
            str: Tooltip text
        """
        return self.get_tooltip('charts', chart_type)
    
    def get_export_tooltip(self, export_type: str) -> str:
        """
        Get tooltip for export options.
        
        Args:
            export_type (str): Type of export
            
        Returns:
            str: Tooltip text
        """
        return self.get_tooltip('export', export_type)
    
    def get_parameter_tooltip(self, parameter_name: str) -> str:
        """
        Get tooltip for experiment parameters.
        
        Args:
            parameter_name (str): Name of the parameter
            
        Returns:
            str: Tooltip text
        """
        return self.get_tooltip('experiment_parameters', parameter_name)
    
    def reload_config(self) -> None:
        """Reload tooltip configuration from file."""
        self._config = self._load_config()


# Global instance for easy access
tooltip_manager = TooltipManager()
