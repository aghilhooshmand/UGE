"""
Base View Component for UGE Application

This module provides the base view class that all other views inherit from.
It contains common functionality and utilities for Streamlit views.

Classes:
- BaseView: Base class for all views

Author: UGE Team
"""

import streamlit as st
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseView(ABC):
    """
    Base class for all views in the UGE application.
    
    This class provides common functionality and utilities that all views
    can use, including session state management, error handling, and
    common UI patterns.
    
    Attributes:
        title (str): Title of the view
        description (str): Description of the view
    """
    
    def __init__(self, title: str, description: str = ""):
        """
        Initialize base view.
        
        Args:
            title (str): Title of the view
            description (str): Description of the view
        """
        self.title = title
        self.description = description
    
    def render_header(self):
        """Render the view header with title and description."""
        st.header(self.title)
        if self.description:
            st.markdown(self.description)
    
    def render_sidebar_stats(self, stats: Dict[str, Any]):
        """
        Render sidebar statistics.
        
        Args:
            stats (Dict[str, Any]): Statistics to display
        """
        st.markdown("### 📊 Quick Stats")
        
        for key, value in stats.items():
            if isinstance(value, int):
                st.metric(key, value)
            else:
                st.write(f"**{key}:** {value}")
    
    def show_error(self, message: str):
        """
        Display an error message.
        
        Args:
            message (str): Error message to display
        """
        st.error(f"❌ {message}")
    
    def show_success(self, message: str):
        """
        Display a success message.
        
        Args:
            message (str): Success message to display
        """
        st.success(f"✅ {message}")
    
    def show_info(self, message: str):
        """
        Display an info message.
        
        Args:
            message (str): Info message to display
        """
        st.info(f"ℹ️ {message}")
    
    def show_warning(self, message: str):
        """
        Display a warning message.
        
        Args:
            message (str): Warning message to display
        """
        st.warning(f"⚠️ {message}")
    
    def create_columns(self, num_columns: int = 2):
        """
        Create Streamlit columns.
        
        Args:
            num_columns (int): Number of columns to create
            
        Returns:
            List: List of Streamlit column objects
        """
        return st.columns(num_columns)
    
    def create_expander(self, title: str, expanded: bool = False):
        """
        Create a Streamlit expander.
        
        Args:
            title (str): Title of the expander
            expanded (bool): Whether to expand by default
            
        Returns:
            Streamlit expander object
        """
        return st.expander(title, expanded=expanded)
    
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
    
    def set_session_state(self, key: str, value: Any):
        """
        Set value in session state.
        
        Args:
            key (str): Session state key
            value (Any): Value to set
        """
        st.session_state[key] = value
    
    def clear_session_state(self, key: str):
        """
        Clear value from session state.
        
        Args:
            key (str): Session state key to clear
        """
        if key in st.session_state:
            del st.session_state[key]
    
    def handle_error(self, error: Exception, context: str = ""):
        """
        Handle and display errors.
        
        Args:
            error (Exception): Error to handle
            context (str): Additional context about where the error occurred
        """
        error_msg = f"Error {context}: {str(error)}" if context else str(error)
        self.show_error(error_msg)
        
        # Log error details in expander for debugging
        with self.create_expander("🔍 Error Details", expanded=False):
            st.code(str(error))
    
    @abstractmethod
    def render(self):
        """
        Render the view.
        
        This method must be implemented by all subclasses.
        """
        pass