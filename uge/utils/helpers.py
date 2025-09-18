"""
Helper Functions for UGE Application

This module contains utility functions used throughout the UGE application.
These are general-purpose functions that don't belong to any specific domain.

Functions:
- create_setup_id(): Generate unique experiment identifiers
- create_run_id(): Generate unique run identifiers  
- mae(): Calculate Mean Absolute Error for fitness evaluation
- accuracy(): Calculate accuracy for fitness evaluation
- fitness_eval(): Core fitness evaluation function for GE individuals

Author: UGE Team
"""

import uuid
import datetime as dt
import numpy as np

# Import operator service for dynamic function mapping
from uge.services.operator_service import OperatorService
from uge.services.storage_service import StorageService


# Global operator service instance for dynamic function mapping
_operator_service = None


def get_operator_service():
    """
    Get the global operator service instance.
    
    Returns:
        OperatorService: The global operator service instance
    """
    global _operator_service
    if _operator_service is None:
        storage_service = StorageService()
        _operator_service = OperatorService(storage_service)
    return _operator_service


def create_setup_id():
    """
    Create a unique experiment ID with timestamp and UUID.
    
    Returns:
        str: Unique experiment ID in format 'exp_YYYYMMDD_HHMMSS_XXXXXXXX'
        
    Example:
        >>> create_setup_id()
        'exp_20240101_143022_a1b2c3d4'
    """
    return f"exp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def create_run_id():
    """
    Create a unique run ID with timestamp and UUID.
    
    Returns:
        str: Unique run ID in format 'run_YYYYMMDD_HHMMSS_XXXXXXXX'
        
    Example:
        >>> create_run_id()
        'run_20240101_143022_e5f6g7h8'
    """
    return f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def mae(y, yhat):
    """
    Calculate Mean Absolute Error between true and predicted values.
    
    This function is used as a fitness metric for Grammatical Evolution.
    Lower MAE values indicate better performance.
    
    Args:
        y (array-like): True values
        yhat (array-like): Predicted values
        
    Returns:
        float: Mean Absolute Error
        
    Example:
        >>> mae([1, 2, 3], [1.1, 1.9, 3.1])
        0.06666666666666667
    """
    return np.mean(np.abs(np.array(y) - np.array(yhat)))


def accuracy(y, yhat):
    """
    Calculate accuracy between true and predicted values.
    
    This function is used as a fitness metric for Grammatical Evolution.
    Higher accuracy values indicate better performance.
    
    Args:
        y (array-like): True values
        yhat (array-like): Predicted values
        
    Returns:
        float: Accuracy (0.0 to 1.0)
        
    Example:
        >>> accuracy([1, 0, 1], [1, 0, 0])
        0.6666666666666666
    """
    compare = np.equal(y, yhat)
    return np.mean(compare)


def fitness_eval(individual, points, metric='mae'):
    """
    Evaluate the fitness of a Grammatical Evolution individual.
    
    This is the core fitness evaluation function that:
    1. Checks if the individual is valid
    2. Evaluates the phenotype (generated code)
    3. Converts predictions to binary classifications
    4. Calculates fitness using the specified metric
    
    Args:
        individual: GE individual with phenotype and genome
        points (tuple): Training data (x, y) where x is features, y is labels
        metric (str): Fitness metric to use ('mae' or 'accuracy')
        
    Returns:
        tuple: Fitness value (single-element tuple for DEAP compatibility)
        
    Raises:
        Various exceptions during phenotype evaluation are caught and return np.nan
        
    Example:
        >>> individual = SomeGEIndividual()
        >>> points = (X_train, Y_train)
        >>> fitness = fitness_eval(individual, points, 'accuracy')
        >>> print(fitness)
        (0.85,)
    """
    x = points[0]
    Y = points[1]
    
    # Check if individual is invalid
    if individual.invalid:
        return np.nan,
    
    try:
        # Get operator service and create dynamic function mapping
        operator_service = get_operator_service()
        
        # Get all available operators for dynamic mapping
        all_operators = operator_service.get_all_operators()
        function_mapping = operator_service.create_operator_function_mapping(list(all_operators.keys()))
        
        # Create evaluation context with dynamic function mapping
        eval_context = {
            'x': x,  # Make the data available as 'x'
            'np': np,  # Make numpy available
        }
        
        # Add all dynamically mapped functions to evaluation context
        eval_context.update(function_mapping)
        
        # Evaluate the generated phenotype (Python code)
        pred = eval(individual.phenotype, eval_context)
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, NameError, SyntaxError, TypeError, ValueError):
        return np.nan,
    
    # Check if prediction is real-valued
    if not np.isrealobj(pred):
        return np.nan,
    
    try:
        # Convert predictions to binary classifications
        Y_class = [1 if pred[i] > 0 else 0 for i in range(len(Y))]
    except (IndexError, TypeError):
        return np.nan,
    
    # Calculate fitness based on metric
    if metric == 'mae':
        fitness_val = mae(Y, Y_class)
    elif metric == 'accuracy':
        fitness_val = accuracy(Y, Y_class)
    else:
        fitness_val = mae(Y, Y_class)  # Default to MAE
    
    return fitness_val,