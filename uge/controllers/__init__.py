"""
Controllers Package

This package contains the control logic and orchestration for the UGE application.

Controllers coordinate between views and services, handling the application flow:
- SetupController: Orchestrates setup execution and management
- AnalysisController: Coordinates data analysis and visualization processes

Controllers are responsible for:
- Processing user requests from views
- Coordinating business logic execution
- Managing application state and flow
- Handling errors and validation
- Returning results to views
"""

from .setup_controller import SetupController
from .analysis_controller import AnalysisController

__all__ = ['SetupController', 'AnalysisController']