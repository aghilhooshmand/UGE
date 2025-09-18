"""
Operator Service Module

Provides services for managing custom operators in the UGE application.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from uge.models.operator import CustomOperator, OperatorRegistry
from uge.services.storage_service import StorageService


class OperatorService:
    """
    Service for managing custom operators.
    
    This service provides functionality for:
    - Loading and saving custom operators
    - Validating operator definitions
    - Creating operator function mappings
    - Managing operator registry
    """
    
    def __init__(self, storage_service: StorageService):
        """
        Initialize the operator service.
        
        Args:
            storage_service (StorageService): Service for file operations
        """
        self.storage_service = storage_service
        self.operators_file = Path("operators") / "custom_operators.json"
        self.registry = OperatorRegistry()
        self._load_operators()
    
    def _load_operators(self):
        """Load operators from storage."""
        try:
            if self.storage_service.exists(str(self.operators_file)):
                data = self.storage_service.load_json(str(self.operators_file))
                self.registry = OperatorRegistry.from_dict(data)
            else:
                # Initialize with built-in operators
                self._initialize_builtin_operators()
                self._save_operators()
        except Exception as e:
            print(f"Error loading operators: {e}")
            self._initialize_builtin_operators()
    
    def _save_operators(self):
        """Save operators to storage."""
        try:
            data = self.registry.to_dict()
            self.storage_service.save_json(str(self.operators_file), data)
        except Exception as e:
            print(f"Error saving operators: {e}")
    
    def reload_operators(self):
        """Reload operators from storage."""
        self._load_operators()
    
    def _initialize_builtin_operators(self):
        """Initialize registry with all operators from functions.py."""
        builtin_operators = [
            # Arithmetic operators
            {
                'name': 'add',
                'display_name': 'Addition',
                'description': 'Add two numeric values',
                'function_code': 'import numpy as np\ndef add(a, b):\n    return np.add(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'arithmetic',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'sub',
                'display_name': 'Subtraction',
                'description': 'Subtract second value from first',
                'function_code': 'import numpy as np\ndef sub(a, b):\n    return np.subtract(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'arithmetic',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'mul',
                'display_name': 'Multiplication',
                'description': 'Multiply two numeric values',
                'function_code': 'import numpy as np\ndef mul(a, b):\n    return np.multiply(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'arithmetic',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'pdiv',
                'display_name': 'Protected Division',
                'description': 'Divide first value by second with protection against division by zero',
                'function_code': 'import numpy as np\ndef pdiv(a, b):\n    try:\n        with np.errstate(divide=\'ignore\', invalid=\'ignore\'):\n            return np.where(b == 0, np.ones_like(a), a / b)\n    except ZeroDivisionError:\n        return 1.0',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'arithmetic',
                'created_by': 'system',
                'is_builtin': True
            },
            
            # Mathematical functions
            {
                'name': 'sigmoid',
                'display_name': 'Sigmoid',
                'description': 'Calculate the sigmoid of input values',
                'function_code': 'import numpy as np\ndef sigmoid(arr):\n    if np.isscalar(arr):\n        arr = np.array([arr])\n    return 1 / (1 + np.exp(-arr))',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'psqrt',
                'display_name': 'Protected Square Root',
                'description': 'Calculate square root with absolute value protection',
                'function_code': 'import numpy as np\ndef psqrt(a):\n    return np.sqrt(np.abs(a))',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'psin',
                'display_name': 'Sine',
                'description': 'Calculate sine of input values',
                'function_code': 'import numpy as np\ndef psin(n):\n    return np.sin(n)',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'pcos',
                'display_name': 'Cosine',
                'description': 'Calculate cosine of input values',
                'function_code': 'import numpy as np\ndef pcos(n):\n    return np.cos(n)',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'plog',
                'display_name': 'Protected Logarithm',
                'description': 'Calculate logarithm with protection',
                'function_code': 'import numpy as np\ndef plog(a):\n    return np.log(1.0 + np.abs(a))',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'exp',
                'display_name': 'Exponential',
                'description': 'Calculate exponential of input values',
                'function_code': 'import numpy as np\ndef exp(a):\n    return np.exp(a)',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'abs_',
                'display_name': 'Absolute Value',
                'description': 'Calculate absolute value',
                'function_code': 'import numpy as np\ndef abs_(a):\n    return np.abs(a)',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            
            # Min/Max functions
            {
                'name': 'minimum',
                'display_name': 'Minimum',
                'description': 'Return minimum of two values',
                'function_code': 'import numpy as np\ndef minimum(a, b):\n    return np.minimum(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'maximum',
                'display_name': 'Maximum',
                'description': 'Return maximum of two values',
                'function_code': 'import numpy as np\ndef maximum(a, b):\n    return np.maximum(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'min_',
                'display_name': 'Min',
                'description': 'Return minimum of two values',
                'function_code': 'import numpy as np\ndef min_(a, b):\n    return np.minimum(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'max_',
                'display_name': 'Max',
                'description': 'Return maximum of two values',
                'function_code': 'import numpy as np\ndef max_(a, b):\n    return np.maximum(a, b)',
                'parameter_count': 2,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'neg',
                'display_name': 'Negation',
                'description': 'Negate the input value',
                'function_code': 'def neg(a):\n    return -a',
                'parameter_count': 1,
                'return_type': 'float',
                'category': 'mathematical',
                'created_by': 'system',
                'is_builtin': True
            },
            
            # Logical operators
            {
                'name': 'and_',
                'display_name': 'Logical AND',
                'description': 'Logical AND operation',
                'function_code': 'import numpy as np\ndef and_(a, b):\n    return np.logical_and(a, b)',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'logical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'or_',
                'display_name': 'Logical OR',
                'description': 'Logical OR operation',
                'function_code': 'import numpy as np\ndef or_(a, b):\n    return np.logical_or(a, b)',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'logical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'not_',
                'display_name': 'Logical NOT',
                'description': 'Logical NOT operation',
                'function_code': 'import numpy as np\ndef not_(a):\n    return np.logical_not(a)',
                'parameter_count': 1,
                'return_type': 'bool',
                'category': 'logical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'nand_',
                'display_name': 'Logical NAND',
                'description': 'Logical NAND operation',
                'function_code': 'import numpy as np\ndef nand_(a, b):\n    return np.logical_not(np.logical_and(a, b))',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'logical',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'nor_',
                'display_name': 'Logical NOR',
                'description': 'Logical NOR operation',
                'function_code': 'import numpy as np\ndef nor_(a, b):\n    return np.logical_not(np.logical_or(a, b))',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'logical',
                'created_by': 'system',
                'is_builtin': True
            },
            
            # Comparison operators
            {
                'name': 'greater_than_or_equal',
                'display_name': 'Greater Than or Equal',
                'description': 'Check if first value is greater than or equal to second',
                'function_code': 'def greater_than_or_equal(a, b):\n    return a >= b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'less_than_or_equal',
                'display_name': 'Less Than or Equal',
                'description': 'Check if first value is less than or equal to second',
                'function_code': 'def less_than_or_equal(a, b):\n    return a <= b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'greater_than',
                'display_name': 'Greater Than',
                'description': 'Check if first value is greater than second',
                'function_code': 'def greater_than(a, b):\n    return a > b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'less_than',
                'display_name': 'Less Than',
                'description': 'Check if first value is less than second',
                'function_code': 'def less_than(a, b):\n    return a < b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'equal',
                'display_name': 'Equal',
                'description': 'Check if two values are equal',
                'function_code': 'def equal(a, b):\n    return a == b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            {
                'name': 'not_equal',
                'display_name': 'Not Equal',
                'description': 'Check if two values are not equal',
                'function_code': 'def not_equal(a, b):\n    return a != b',
                'parameter_count': 2,
                'return_type': 'bool',
                'category': 'comparison',
                'created_by': 'system',
                'is_builtin': True
            },
            
            # Conditional operator
            {
                'name': 'if_',
                'display_name': 'Conditional',
                'description': 'If condition is true return first value, else second value',
                'function_code': 'import numpy as np\ndef if_(i, o0, o1):\n    return np.where(i, o0, o1)',
                'parameter_count': 3,
                'return_type': 'any',
                'category': 'conditional',
                'created_by': 'system',
                'is_builtin': True
            }
        ]
        
        for op_data in builtin_operators:
            operator = CustomOperator.from_dict(op_data)
            operator.created_at = datetime.now().isoformat()
            self.registry.add_operator(operator)
    
    def add_custom_operator(self, name: str, display_name: str, description: str, 
                          function_code: str, parameter_count: int, return_type: str, 
                          category: str, created_by: str = "user") -> Tuple[bool, str]:
        """
        Add a custom operator.
        
        Args:
            name (str): Operator name
            display_name (str): Display name
            description (str): Description
            function_code (str): Python function code
            parameter_count (int): Number of parameters
            return_type (str): Return type
            category (str): Category
            created_by (str): Creator identifier
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Check if operator name already exists
            if self.registry.get_operator(name):
                return False, f"Operator name '{name}' already exists. Please choose a different name."
            
            # Check if function name in code conflicts with existing operators
            function_name = self._extract_function_name(function_code)
            if function_name and function_name != name:
                return False, f"Function name '{function_name}' in code must match operator name '{name}'"
            
            # Check if function name conflicts with existing operators
            if function_name and self.registry.get_operator(function_name):
                return False, f"Function name '{function_name}' conflicts with existing operator. Please use a different function name."
            
            # Create operator
            operator = CustomOperator(
                name=name,
                display_name=display_name,
                description=description,
                function_code=function_code,
                parameter_count=parameter_count,
                return_type=return_type,
                category=category,
                created_by=created_by,
                is_builtin=False,
                created_at=datetime.now().isoformat()
            )
            
            # Validate function code
            is_valid, error_msg = operator.validate_function_code()
            if not is_valid:
                return False, f"Invalid function code: {error_msg}"
            
            # Add to registry
            if self.registry.add_operator(operator):
                self._save_operators()
                return True, f"Operator '{name}' added successfully and saved to custom operators file"
            else:
                return False, f"Failed to add operator '{name}'"
                
        except Exception as e:
            return False, f"Error adding operator: {str(e)}"
    
    def _extract_function_name(self, function_code: str) -> Optional[str]:
        """
        Extract function name from function code.
        
        Args:
            function_code (str): Python function code
            
        Returns:
            Optional[str]: Function name if found, None otherwise
        """
        try:
            import re
            # Look for function definition pattern
            match = re.search(r'def\s+(\w+)\s*\(', function_code)
            if match:
                return match.group(1)
            return None
        except Exception:
            return None
    
    def remove_custom_operator(self, name: str) -> Tuple[bool, str]:
        """
        Remove a custom operator.
        
        Args:
            name (str): Name of operator to remove
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            operator = self.registry.get_operator(name)
            if not operator:
                return False, f"Operator '{name}' not found"
            
            if operator.is_builtin:
                return False, f"Cannot remove built-in operator '{name}'"
            
            if self.registry.remove_operator(name):
                self._save_operators()
                return True, f"Operator '{name}' removed successfully"
            else:
                return False, f"Failed to remove operator '{name}'"
                
        except Exception as e:
            return False, f"Error removing operator: {str(e)}"
    
    def get_operator(self, name: str) -> Optional[CustomOperator]:
        """Get an operator by name."""
        return self.registry.get_operator(name)
    
    def get_all_operators(self) -> Dict[str, CustomOperator]:
        """Get all operators."""
        return self.registry.get_all_operators()
    
    def get_operators_by_category(self, category: str) -> Dict[str, CustomOperator]:
        """Get operators by category."""
        return self.registry.get_operators_by_category(category)
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        categories = set()
        for operator in self.registry.get_all_operators().values():
            categories.add(operator.category)
        return sorted(list(categories))
    
    def create_operator_function_mapping(self, operator_names: List[str]) -> Dict[str, callable]:
        """
        Create a mapping of operator names to function objects.
        
        Args:
            operator_names (List[str]): List of operator names to include
            
        Returns:
            Dict[str, callable]: Mapping of operator names to functions
        """
        mapping = {}
        
        for name in operator_names:
            operator = self.registry.get_operator(name)
            if operator:
                try:
                    func = operator.get_function_object()
                    mapping[name] = func
                except Exception as e:
                    print(f"Warning: Could not create function for operator '{name}': {e}")
        
        return mapping
    
    def validate_grammar_operators(self, grammar_content: str) -> Tuple[bool, List[str]]:
        """
        Validate that all operators used in grammar are available.
        
        Args:
            grammar_content (str): BNF grammar content
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_missing_operators)
        """
        # Extract operator names from grammar (simple regex approach)
        import re
        
        # Find function calls like operator_name(param1, param2)
        operator_pattern = r'(\w+)\s*\([^)]*\)'
        matches = re.findall(operator_pattern, grammar_content)
        
        # Filter out BNF keywords and common terms
        bnf_keywords = {'log_op', 'num_op', 'boolean_feature', 'nonboolean_feature', 'conditional_branches'}
        used_operators = set(matches) - bnf_keywords
        
        missing_operators = []
        for op_name in used_operators:
            if not self.registry.get_operator(op_name):
                missing_operators.append(op_name)
        
        is_valid = len(missing_operators) == 0
        return is_valid, missing_operators
