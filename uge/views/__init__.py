"""
Views Package

This package contains the presentation layer components for the UGE application.

Views handle all user interface logic and Streamlit components:
- ExperimentView: UI for running and managing experiments
- AnalysisView: UI for analyzing results and creating visualizations
- DatasetView: UI for dataset management and preview
- components/: Reusable UI components like charts and forms

Views are responsible for:
- Rendering user interfaces
- Handling user input
- Displaying data to users
- Managing UI state and interactions
"""

from .experiment_view import ExperimentView
from .analysis_view import AnalysisView
from .dataset_view import DatasetView

__all__ = ['ExperimentView', 'AnalysisView', 'DatasetView']