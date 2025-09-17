"""
Views Package

This package contains the presentation layer components for the UGE application.

Views handle all user interface logic and Streamlit components:
- SetupView: UI for running and managing setups
- AnalysisView: UI for analyzing results and creating visualizations
- DatasetView: UI for dataset management and preview
- GrammarView: UI for grammar editing and management
- SetupManagerView: UI for setup management and monitoring
- ComparisonView: UI for setup comparison and analysis
- components/: Reusable UI components like charts and forms

Views are responsible for:
- Rendering user interfaces
- Handling user input
- Displaying data to users
- Managing UI state and interactions
"""

from .home_view import HomeView
from .setup_view import SetupView
from .analysis_view import AnalysisView
from .dataset_view import DatasetView
from .grammar_view import GrammarView
from .setup_manager_view import SetupManagerView
from .comparison_view import ComparisonView

__all__ = ['HomeView', 'SetupView', 'AnalysisView', 'DatasetView', 'GrammarView', 'SetupManagerView', 'ComparisonView']