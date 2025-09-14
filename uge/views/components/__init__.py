"""
UI Components Package

This package contains reusable UI components for the UGE application.

Components:
- BaseView: Base class for all views
- Charts: Chart and visualization components
- Forms: Form components and input widgets

Author: UGE Team
"""

from .base_view import BaseView
from .charts import Charts
from .forms import Forms

__all__ = ['BaseView', 'Charts', 'Forms']