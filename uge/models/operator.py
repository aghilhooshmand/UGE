"""
Operator Model Module

Defines data models for custom operators in the UGE application.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class CustomOperator:
    """
    Represents a custom operator defined by the user.
    
    This class encapsulates all information needed to define and use
    a custom operator in grammatical evolution.
    
    Attributes:
        name (str): The name of the operator (e.g., 'greater_than_equal')
        display_name (str): Human-readable display name
        description (str): Description of what the operator does
        function_code (str): Python function code as string
        parameter_count (int): Number of parameters the operator takes
        return_type (str): Expected return type ('bool', 'float', 'int', 'any')
        category (str): Category of operator ('comparison', 'arithmetic', 'logical', 'custom')
        created_by (str): User who created the operator
        is_builtin (bool): Whether this is a built-in operator
    """
    
    name: str
    display_name: str
    description: str
    function_code: str
    parameter_count: int
    return_type: str
    category: str
    created_by: str = "user"
    is_builtin: bool = False
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operator to dictionary for serialization."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'function_code': self.function_code,
            'parameter_count': self.parameter_count,
            'return_type': self.return_type,
            'category': self.category,
            'created_by': self.created_by,
            'is_builtin': self.is_builtin,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomOperator':
        """Create operator from dictionary."""
        return cls(
            name=data.get('name', ''),
            display_name=data.get('display_name', ''),
            description=data.get('description', ''),
            function_code=data.get('function_code', ''),
            parameter_count=data.get('parameter_count', 2),
            return_type=data.get('return_type', 'any'),
            category=data.get('category', 'custom'),
            created_by=data.get('created_by', 'user'),
            is_builtin=data.get('is_builtin', False),
            created_at=data.get('created_at', '')
        )
    
    def validate_function_code(self) -> tuple[bool, str]:
        """
        Validate the function code.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Try to compile the function code
            compile(self.function_code, '<string>', 'exec')
            
            # Check if it's a function definition
            if 'def ' not in self.function_code:
                return False, "Function code must contain a function definition"
            
            # Check if function name matches operator name
            if f"def {self.name}(" not in self.function_code:
                return False, f"Function name must match operator name '{self.name}'"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error in function code: {str(e)}"
        except Exception as e:
            return False, f"Error validating function code: {str(e)}"
    
    def get_function_object(self):
        """
        Get the actual function object from the code.
        
        Returns:
            function: The compiled function object
        """
        try:
            # Create a namespace for the function
            namespace = {}
            
            # Execute the function code in the namespace
            exec(self.function_code, namespace)
            
            # Return the function object
            if self.name in namespace:
                return namespace[self.name]
            else:
                raise ValueError(f"Function '{self.name}' not found in compiled code")
                
        except Exception as e:
            raise ValueError(f"Error compiling function: {str(e)}")


@dataclass
class OperatorRegistry:
    """
    Registry for managing custom operators.
    
    This class manages a collection of custom operators and provides
    methods for adding, removing, and retrieving operators.
    """
    
    operators: Dict[str, CustomOperator] = field(default_factory=dict)
    
    def add_operator(self, operator: CustomOperator) -> bool:
        """
        Add an operator to the registry.
        
        Args:
            operator (CustomOperator): The operator to add
            
        Returns:
            bool: True if added successfully, False if operator already exists
        """
        if operator.name in self.operators:
            return False
        
        self.operators[operator.name] = operator
        return True
    
    def remove_operator(self, name: str) -> bool:
        """
        Remove an operator from the registry.
        
        Args:
            name (str): Name of the operator to remove
            
        Returns:
            bool: True if removed successfully, False if not found
        """
        if name in self.operators:
            del self.operators[name]
            return True
        return False
    
    def get_operator(self, name: str) -> Optional[CustomOperator]:
        """
        Get an operator by name.
        
        Args:
            name (str): Name of the operator
            
        Returns:
            Optional[CustomOperator]: The operator if found, None otherwise
        """
        return self.operators.get(name)
    
    def get_operators_by_category(self, category: str) -> Dict[str, CustomOperator]:
        """
        Get all operators in a specific category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            Dict[str, CustomOperator]: Operators in the category
        """
        return {name: op for name, op in self.operators.items() 
                if op.category == category}
    
    def get_all_operators(self) -> Dict[str, CustomOperator]:
        """Get all operators in the registry."""
        return self.operators.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for serialization."""
        return {
            'operators': {name: op.to_dict() for name, op in self.operators.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperatorRegistry':
        """Create registry from dictionary."""
        registry = cls()
        operators_data = data.get('operators', {})
        
        for name, op_data in operators_data.items():
            registry.operators[name] = CustomOperator.from_dict(op_data)
        
        return registry
