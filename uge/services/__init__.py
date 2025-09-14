"""
Services Package

This package contains the business logic and external integrations for the UGE application.

Services handle core business operations and external system integrations:
- GEService: Wraps the Grammatical Evolution algorithm and fitness evaluation
- StorageService: Handles file operations and data persistence
- DatasetService: Manages dataset loading, preprocessing, and validation

Services are responsible for:
- Implementing core business logic
- Integrating with external systems (grape, algorithms, functions)
- Data processing and transformation
- File I/O operations
- Algorithm execution
"""

from .ge_service import GEService
from .storage_service import StorageService
from .dataset_service import DatasetService

__all__ = ['GEService', 'StorageService', 'DatasetService']