"""
Dataset Service for UGE Application

This module provides dataset management services for the UGE application.
It handles loading, preprocessing, and managing datasets for GE experiments.

Classes:
- DatasetService: Main service for dataset operations

Author: UGE Team
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from uge.models.dataset import Dataset, DatasetInfo
from uge.utils.constants import FILE_PATHS


class DatasetService:
    """
    Service for dataset management operations.
    
    This service handles all dataset-related operations including
    loading, preprocessing, validation, and management of datasets
    used in GE experiments.
    
    Attributes:
        datasets_dir (Path): Directory containing datasets
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize dataset service.
        
        Args:
            project_root (Optional[Path]): Project root directory
        """
        self.project_root = project_root or FILE_PATHS['project_root']
        self.datasets_dir = self.project_root / "datasets"
        
        # Ensure datasets directory exists
        self.datasets_dir.mkdir(exist_ok=True)
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List[str]: List of dataset filenames
        """
        if not self.datasets_dir.exists():
            return []
        
        return [p.name for p in self.datasets_dir.glob('*') 
                if p.is_file() and p.suffix in {'.data', '.csv', ''}]
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found
        """
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            return None
        
        try:
            dataset = Dataset.from_file(dataset_path)
            return dataset.info
        except Exception as e:
            raise ValueError(f"Error getting dataset info: {e}")
    
    def load_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """
        Load a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Optional[Dataset]: Loaded dataset or None if not found
        """
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            return None
        
        try:
            return Dataset.from_file(dataset_path)
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def preprocess_dataset(self, dataset_name: str, label_column: Optional[str] = None,
                          test_size: float = 0.3, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess dataset for GE experiments.
        
        Args:
            dataset_name (str): Name of the dataset
            label_column (Optional[str]): Label column for CSV datasets
            test_size (float): Test set size ratio
            random_seed (int): Random seed for reproducibility
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, Y_train, X_test, Y_test
            
        Raises:
            ValueError: If dataset cannot be loaded or preprocessed
        """
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        try:
            if dataset_name == 'processed.cleveland.data':
                return dataset.preprocess_cleveland_data(random_seed)
            else:
                if not label_column:
                    raise ValueError('label_column not provided for CSV dataset')
                return dataset.preprocess_csv_data(label_column, test_size, random_seed)
        except Exception as e:
            raise ValueError(f"Error preprocessing dataset: {e}")
    
    def validate_dataset(self, dataset_name: str) -> List[str]:
        """
        Validate a dataset for GE experiments.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            List[str]: List of validation warnings/errors
        """
        warnings = []
        
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            warnings.append(f"Dataset not found: {dataset_name}")
            return warnings
        
        try:
            # Load dataset to check for issues
            data = dataset.load()
            
            # Check for missing values
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                warnings.append(f"Dataset has {missing_count} missing values")
            
            # Check for empty dataset
            if len(data) == 0:
                warnings.append("Dataset is empty")
            
            # Check for too few samples
            if len(data) < 10:
                warnings.append("Dataset has very few samples (less than 10)")
            
            # Check for too many features
            if len(data.columns) > 1000:
                warnings.append("Dataset has many features (more than 1000)")
            
            # Check for constant columns
            constant_cols = data.columns[data.nunique() <= 1].tolist()
            if constant_cols:
                warnings.append(f"Dataset has constant columns: {constant_cols}")
            
            # Special validation for Cleveland dataset
            if dataset_name == 'processed.cleveland.data':
                required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                               'ca', 'thal', 'class']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    warnings.append(f"Cleveland dataset missing required columns: {missing_cols}")
            
        except Exception as e:
            warnings.append(f"Error validating dataset: {e}")
        
        return warnings
    
    def get_dataset_preview(self, dataset_name: str, n_rows: int = 10) -> Optional[pd.DataFrame]:
        """
        Get a preview of a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            n_rows (int): Number of rows to preview
            
        Returns:
            Optional[pd.DataFrame]: Dataset preview or None if not found
        """
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            return None
        
        try:
            return dataset.get_preview(n_rows)
        except Exception as e:
            raise ValueError(f"Error getting dataset preview: {e}")
    
    def get_dataset_statistics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Optional[Dict[str, Any]]: Dataset statistics or None if not found
        """
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            return None
        
        try:
            return dataset.get_statistics()
        except Exception as e:
            raise ValueError(f"Error getting dataset statistics: {e}")
    
    def save_dataset(self, dataset_name: str, data: pd.DataFrame) -> Path:
        """
        Save a dataset to file.
        
        Args:
            dataset_name (str): Name for the dataset
            data (pd.DataFrame): Data to save
            
        Returns:
            Path: Path to saved dataset file
        """
        # Ensure proper file extension
        if not dataset_name.endswith(('.csv', '.data')):
            dataset_name += '.csv'
        
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            if dataset_name.endswith('.csv'):
                data.to_csv(dataset_path, index=False)
            else:
                data.to_csv(dataset_path, sep=' ', index=False)
            
            return dataset_path
        except Exception as e:
            raise ValueError(f"Error saving dataset: {e}")
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_name (str): Name of the dataset to delete
            
        Returns:
            bool: True if deleted successfully, False if not found
        """
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            return False
        
        try:
            dataset_path.unlink()
            return True
        except Exception as e:
            raise RuntimeError(f"Error deleting dataset: {e}")
    
    def get_available_datasets_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available datasets.
        
        Returns:
            List[Dict[str, Any]]: List of dataset information dictionaries
        """
        datasets = self.list_datasets()
        datasets_info = []
        
        for dataset_name in datasets:
            try:
                info = self.get_dataset_info(dataset_name)
                if info:
                    datasets_info.append(info.to_dict())
            except Exception:
                # Skip datasets that can't be loaded
                continue
        
        return datasets_info
    
    def check_dataset_compatibility(self, dataset_name: str, label_column: Optional[str] = None) -> List[str]:
        """
        Check if a dataset is compatible with GE experiments.
        
        Args:
            dataset_name (str): Name of the dataset
            label_column (Optional[str]): Label column for CSV datasets
            
        Returns:
            List[str]: List of compatibility issues
        """
        issues = []
        
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            issues.append("Dataset not found")
            return issues
        
        try:
            data = dataset.load()
            
            # Check for label column in CSV datasets
            if dataset_name != 'processed.cleveland.data':
                if not label_column:
                    issues.append("Label column not specified for CSV dataset")
                elif label_column not in data.columns:
                    issues.append(f"Label column '{label_column}' not found in dataset")
                else:
                    # Check if label column has appropriate values
                    unique_labels = data[label_column].nunique()
                    if unique_labels < 2:
                        issues.append("Label column has less than 2 unique values")
                    elif unique_labels > 10:
                        issues.append("Label column has many unique values (consider if this is classification)")
            
            # Check for numerical features
            if dataset_name != 'processed.cleveland.data':
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) == 0:
                    issues.append("Dataset has no numerical features")
                elif len(numerical_cols) < 2:
                    issues.append("Dataset has very few numerical features")
            
        except Exception as e:
            issues.append(f"Error checking compatibility: {e}")
        
        return issues