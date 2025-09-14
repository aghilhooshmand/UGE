"""
Logging Utilities for UGE Application

This module contains custom logging utilities specifically designed for Streamlit applications.
The StreamlitLogger class provides real-time logging display in Streamlit interfaces.

Classes:
- StreamlitLogger: Custom logger that displays logs in Streamlit placeholders

Author: UGE Team
"""

from io import StringIO


class StreamlitLogger(StringIO):
    """
    Custom logger that displays log output in Streamlit placeholders.
    
    This class extends StringIO to capture log output and display it in real-time
    within Streamlit applications. It's particularly useful for showing progress
    during long-running operations like GE experiments.
    
    Attributes:
        placeholder: Streamlit placeholder object for displaying logs
        buffered (str): Internal buffer for log messages
        
    Example:
        >>> import streamlit as st
        >>> placeholder = st.empty()
        >>> logger = StreamlitLogger(placeholder)
        >>> logger.write("Starting experiment...")
        >>> logger.write("Generation 1 complete")
    """
    
    def __init__(self, placeholder):
        """
        Initialize the StreamlitLogger.
        
        Args:
            placeholder: Streamlit placeholder object where logs will be displayed
        """
        super().__init__()
        self.placeholder = placeholder
        self.buffered = ""
    
    def write(self, s):
        """
        Write a log message to the buffer and display it in Streamlit.
        
        This method captures log output and displays it in the Streamlit placeholder.
        For performance reasons, it only shows the last 200 lines of output.
        
        Args:
            s (str): Log message to write
            
        Returns:
            int: Number of characters written (for StringIO compatibility)
        """
        self.buffered += s
        
        # Trim long logs for performance - only show last 200 lines
        show = '\n'.join(self.buffered.splitlines()[-200:])
        self.placeholder.code(show)
        
        
        return len(s)
    
    def flush(self):
        """
        Flush the buffer (no-op for this implementation).
        
        This method is required for StringIO compatibility but doesn't
        perform any actual flushing since we display logs in real-time.
        """
        pass