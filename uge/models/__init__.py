"""
Models Package

This package contains data models and business entities for the UGE application.

Models represent the core data structures and business logic of the application:
- Setup: Represents a GE setup with its configuration and results
- Dataset: Handles dataset loading, preprocessing, and management
- Grammar: Manages BNF grammar files and parsing

Each model encapsulates data and the operations that can be performed on that data.
"""

from .setup import Setup
from .dataset import Dataset
from .grammar import Grammar
from .operator import CustomOperator, OperatorRegistry

__all__ = ['Setup', 'Dataset', 'Grammar', 'CustomOperator', 'OperatorRegistry']