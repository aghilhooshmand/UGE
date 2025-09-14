"""
Base Controller for UGE Application

This module provides the base controller class that all other controllers inherit from.
It contains common functionality and utilities for controllers.

Classes:
- BaseController: Base class for all controllers

Author: UGE Team
"""

from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import streamlit as st

from uge.services.storage_service import StorageService
from uge.services.dataset_service import DatasetService
from uge.services.ge_service import GEService
from uge.utils.constants import FILE_PATHS


class BaseController(ABC):
    """
    Base class for all controllers in the UGE application.
    
    This class provides common functionality and utilities that all controllers
    can use, including service management, error handling, and common operations.
    
    Attributes:
        storage_service (StorageService): Service for file operations
        dataset_service (DatasetService): Service for dataset operations
        ge_service (GEService): Service for GE operations
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize base controller.
        
        Args:
            project_root (Optional[str]): Project root directory
        """
        self.project_root = project_root or str(FILE_PATHS['project_root'])
        
        # Initialize services
        self.storage_service = StorageService()
        self.dataset_service = DatasetService()
        self.ge_service = GEService()
    
    def handle_error(self, error: Exception, context: str = "", 
                    show_to_user: bool = True) -> None:
        """
        Handle and process errors.
        
        Args:
            error (Exception): Error to handle
            context (str): Additional context about where the error occurred
            show_to_user (bool): Whether to show error to user
        """
        error_msg = f"Error {context}: {str(error)}" if context else str(error)
        
        # Log error (could be extended to use proper logging)
        print(f"Controller Error: {error_msg}")
        
        if show_to_user:
            st.error(f"❌ {error_msg}")
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            key (str): Session state key
            default (Any): Default value if key doesn't exist
            
        Returns:
            Any: Value from session state
        """
        return st.session_state.get(key, default)
    
    def set_session_state(self, key: str, value: Any) -> None:
        """
        Set value in session state.
        
        Args:
            key (str): Session state key
            value (Any): Value to set
        """
        st.session_state[key] = value
    
    def clear_session_state(self, key: str) -> None:
        """
        Clear value from session state.
        
        Args:
            key (str): Session state key to clear
        """
        if key in st.session_state:
            del st.session_state[key]
    
    def validate_input(self, data: Dict[str, Any], 
                      required_fields: List[str]) -> List[str]:
        """
        Validate input data.
        
        Args:
            data (Dict[str, Any]): Data to validate
            required_fields (List[str]): List of required field names
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                errors.append(f"Field '{field}' is required")
        
        return errors
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func (Callable): Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result or None if error occurred
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, f"executing {func.__name__}")
            return None
    
    def get_available_datasets(self) -> List[str]:
        """
        Get list of available datasets.
        
        Returns:
            List[str]: List of dataset names
        """
        return self.safe_execute(self.dataset_service.list_datasets) or []
    
    def get_available_grammars(self) -> List[str]:
        """
        Get list of available grammars.
        
        Returns:
            List[str]: List of grammar names
        """
        # This would need to be implemented in a grammar service
        # For now, return empty list
        return []
    
    def get_available_experiments(self) -> List[str]:
        """
        Get list of available experiments.
        
        Returns:
            List[str]: List of experiment IDs
        """
        experiments = self.safe_execute(self.storage_service.list_experiments) or []
        return [exp.name for exp in experiments]
    
    @abstractmethod
    def handle_request(self, request_type: str, **kwargs) -> Any:
        """
        Handle a request.
        
        This method must be implemented by all subclasses.
        
        Args:
            request_type (str): Type of request
            **kwargs: Request parameters
            
        Returns:
            Any: Request result
        """
        pass