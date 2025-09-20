"""
Dataset Model for UGE Application

This module defines the Dataset data model, which represents datasets
used in Grammatical Evolution experiments.

Classes:
- Dataset: Main dataset data model
- DatasetInfo: Dataset metadata and information

Author: UGE Team
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetInfo:
    """
    Dataset metadata and information.
    
    This class contains metadata about a dataset, including its name,
    path, type, and basic statistics.
    
    Attributes:
        name (str): Dataset name
        path (Path): Path to the dataset file
        file_type (str): File extension/type
        size_bytes (int): File size in bytes
        columns (List[str]): Column names
        rows (int): Number of rows
        features (int): Number of feature columns
        has_labels (bool): Whether dataset has label column
        label_column (Optional[str]): Name of label column if exists
    """
    
    name: str
    path: Path
    file_type: str
    size_bytes: int
    columns: list
    rows: int
    features: int
    has_labels: bool = False
    label_column: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset info to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'file_type': self.file_type,
            'size_bytes': self.size_bytes,
            'columns': self.columns,
            'rows': self.rows,
            'features': self.features,
            'has_labels': self.has_labels,
            'label_column': self.label_column
        }


class Dataset:
    """
    Dataset data model for UGE experiments.
    
    This class represents a dataset used in Grammatical Evolution experiments.
    It handles loading, preprocessing, and splitting of datasets.
    
    Attributes:
        info (DatasetInfo): Dataset metadata
        data (Optional[pd.DataFrame]): Loaded dataset data
        X_train (Optional[np.ndarray]): Training features
        Y_train (Optional[np.ndarray]): Training labels
        X_test (Optional[np.ndarray]): Test features
        Y_test (Optional[np.ndarray]): Test labels
    """
    
    def __init__(self, info: DatasetInfo):
        """
        Initialize dataset with metadata.
        
        Args:
            info (DatasetInfo): Dataset metadata
        """
        self.info = info
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.Y_test: Optional[np.ndarray] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is not supported
        """
        if not self.info.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.info.path}")
        
        try:
            if self.info.file_type == '.csv':
                self.data = pd.read_csv(self.info.path)
            elif self.info.file_type == '.data':
                # Try different separators for .data files
                try:
                    self.data = pd.read_csv(self.info.path, sep=',')
                except:
                    self.data = pd.read_csv(self.info.path, sep=r'\s+')
            else:
                raise ValueError(f"Unsupported file type: {self.info.file_type}")
            
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def preprocess_cleveland_data(self, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess Cleveland heart disease dataset.
        
        This method handles the specific preprocessing required for the
        processed.cleveland.data dataset, including handling missing values,
        normalization, and one-hot encoding.
        
        Args:
            random_seed (int): Random seed for reproducibility
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, Y_train, X_test, Y_test
        """
        if self.data is None:
            self.load()
        
        # Handle missing values
        data = self.data.copy()
        data = data[data.ca != '?']
        data = data[data.thal != '?']
        
        # Convert labels to binary
        Y = data['class'].to_numpy()
        for i in range(len(Y)):
            Y[i] = 1 if Y[i] > 0 else 0
        
        # Remove class column
        data = data.drop(['class'], axis=1)
        
        # Normalize numerical columns
        cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data.loc[:, cols] = (data.loc[:, cols] - data.loc[:, cols].mean()) / data.loc[:, cols].std()
        
        # One-hot encode categorical columns
        data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
        
        # Convert to numpy array
        X = data.to_numpy()
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_seed
        )
        
        # Transpose for GE format (features as columns)
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        return X_train, Y_train, X_test, Y_test
    
    def preprocess_csv_data(self, label_column: str, test_size: float = 0.3, 
                          random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess CSV dataset with specified label column.
        
        Args:
            label_column (str): Name of the label column
            test_size (float): Test set size ratio
            random_seed (int): Random seed for reproducibility
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, Y_train, X_test, Y_test
        """
        if self.data is None:
            self.load()
        
        # Remove rows with missing values
        df = self.data.dropna()
        
        # Extract labels and features
        Y = df[label_column].astype(int).to_numpy()
        X = df.drop(columns=[label_column]).select_dtypes(include=[np.number]).to_numpy()
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_seed
        )
        
        # Transpose for GE format (features as columns)
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        return X_train, Y_train, X_test, Y_test
    
    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """
        Get a preview of the dataset.
        
        Args:
            n_rows (int): Number of rows to preview
            
        Returns:
            pd.DataFrame: Dataset preview
        """
        if self.data is None:
            self.load()
        
        return self.data.head(n_rows)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dict[str, Any]: Dataset statistics
        """
        if self.data is None:
            self.load()
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Dataset':
        """
        Create dataset from file path.
        
        Args:
            file_path (Path): Path to dataset file
            
        Returns:
            Dataset: Dataset instance
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        file_type = file_path.suffix
        size_bytes = file_path.stat().st_size
        
        # Load a small sample to get basic info
        try:
            if file_type == '.csv':
                sample = pd.read_csv(file_path, nrows=5)
            elif file_type == '.data':
                try:
                    sample = pd.read_csv(file_path, sep=',', nrows=5)
                except:
                    sample = pd.read_csv(file_path, sep=r'\s+', nrows=5)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Get full row count
            with open(file_path, 'r') as f:
                rows = sum(1 for _ in f)
            
            info = DatasetInfo(
                name=file_path.name,
                path=file_path,
                file_type=file_type,
                size_bytes=size_bytes,
                columns=list(sample.columns),
                rows=rows,
                features=len(sample.columns),
                has_labels=False  # Will be determined later
            )
            
            return cls(info)
            
        except Exception as e:
            raise ValueError(f"Error reading dataset file: {e}")