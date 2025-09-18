#!/usr/bin/env python3
"""
Demo Script for Custom Operator System

This script demonstrates how the custom operator system works:
1. Creating custom operators
2. Storing them in the registry
3. Creating function mappings
4. Using them in grammatical evolution

Author: Aghil
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from uge.services.storage_service import StorageService
from uge.services.operator_service import OperatorService
from uge.models.operator import CustomOperator


def demo_custom_operators():
    """Demonstrate the custom operator system."""
    print("🧬 UGE Custom Operator System Demo")
    print("=" * 50)
    
    # Initialize services
    storage_service = StorageService()
    operator_service = OperatorService(storage_service)
    
    print("\n1️⃣ Initial Built-in Operators:")
    all_operators = operator_service.get_all_operators()
    for name, op in all_operators.items():
        print(f"   • {op.display_name} ({name}) - {op.category}")
    
    print(f"\n   Total operators: {len(all_operators)}")
    
    # Add a custom operator
    print("\n2️⃣ Adding Custom Operators:")
    
    # Example 1: Custom comparison operator
    success1, msg1 = operator_service.add_custom_operator(
        name="my_distance",
        display_name="My Distance",
        description="Calculate Euclidean distance between two points",
        function_code="""def my_distance(a, b):
    \"\"\"
    Calculate Euclidean distance between two values.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        Distance between a and b
    \"\"\"
    import math
    return math.sqrt((a - b) ** 2)""",
        parameter_count=2,
        return_type="float",
        category="mathematical",
        created_by="demo_user"
    )
    
    print(f"   • my_distance: {msg1}")
    
    # Example 2: Custom logical operator
    success2, msg2 = operator_service.add_custom_operator(
        name="my_xor",
        display_name="My XOR",
        description="Exclusive OR operation",
        function_code="""def my_xor(a, b):
    \"\"\"
    Exclusive OR operation.
    
    Args:
        a: First boolean value
        b: Second boolean value
    
    Returns:
        XOR result
    \"\"\"
    return bool(a) != bool(b)""",
        parameter_count=2,
        return_type="bool",
        category="logical",
        created_by="demo_user"
    )
    
    print(f"   • my_xor: {msg2}")
    
    # Example 3: Custom arithmetic operator
    success3, msg3 = operator_service.add_custom_operator(
        name="my_power",
        display_name="My Power",
        description="Raise first value to power of second",
        function_code="""def my_power(a, b):
    \"\"\"
    Raise a to the power of b.
    
    Args:
        a: Base value
        b: Exponent
    
    Returns:
        a raised to the power of b
    \"\"\"
    try:
        return a ** b
    except:
        return 1.0""",
        parameter_count=2,
        return_type="float",
        category="arithmetic",
        created_by="demo_user"
    )
    
    print(f"   • my_power: {msg3}")
    
    # Show updated operators
    print("\n3️⃣ Updated Operator Registry:")
    all_operators = operator_service.get_all_operators()
    for name, op in all_operators.items():
        status = "🔧 Built-in" if op.is_builtin else "👤 Custom"
        print(f"   {status} {op.display_name} ({name}) - {op.category}")
    
    print(f"\n   Total operators: {len(all_operators)}")
    
    # Demonstrate function mapping
    print("\n4️⃣ Creating Function Mapping:")
    custom_operators = ["my_distance", "my_xor", "my_power", "add", "equal"]
    
    function_mapping = operator_service.create_operator_function_mapping(custom_operators)
    
    print(f"   Created mapping for {len(function_mapping)} operators:")
    for name, func in function_mapping.items():
        print(f"   • {name}: {func}")
    
    # Test the functions
    print("\n5️⃣ Testing Custom Functions:")
    
    # Test my_distance
    distance_func = function_mapping.get("my_distance")
    if distance_func:
        result = distance_func(3, 7)
        print(f"   • my_distance(3, 7) = {result}")
    
    # Test my_xor
    xor_func = function_mapping.get("my_xor")
    if xor_func:
        result1 = xor_func(True, False)
        result2 = xor_func(True, True)
        print(f"   • my_xor(True, False) = {result1}")
        print(f"   • my_xor(True, True) = {result2}")
    
    # Test my_power
    power_func = function_mapping.get("my_power")
    if power_func:
        result = power_func(2, 3)
        print(f"   • my_power(2, 3) = {result}")
    
    # Demonstrate grammar validation
    print("\n6️⃣ Grammar Validation:")
    
    # Test grammar with custom operators
    test_grammar = """
    <log_op> ::= my_xor(<log_op>,<log_op>) | and_(<log_op>,<log_op>)
    <num_op> ::= my_distance(<num_op>,<num_op>) | my_power(<num_op>,<num_op>) | add(<num_op>,<num_op>)
    """
    
    is_valid, missing = operator_service.validate_grammar_operators(test_grammar)
    
    if is_valid:
        print("   ✅ Grammar is valid - all operators are available")
    else:
        print(f"   ❌ Grammar is invalid - missing operators: {missing}")
    
    # Test grammar with missing operators
    invalid_grammar = """
    <log_op> ::= unknown_operator(<log_op>,<log_op>)
    <num_op> ::= another_unknown(<num_op>,<num_op>)
    """
    
    is_valid, missing = operator_service.validate_grammar_operators(invalid_grammar)
    
    if is_valid:
        print("   ✅ Invalid grammar test passed")
    else:
        print(f"   ❌ Invalid grammar test passed - missing operators: {missing}")
    
    print("\n7️⃣ Operator Categories:")
    categories = operator_service.get_available_categories()
    for category in categories:
        ops = operator_service.get_operators_by_category(category)
        print(f"   • {category}: {len(ops)} operators")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run the UGE application")
    print("2. Go to '🔧 Operator Manager' page")
    print("3. View, add, edit, or delete custom operators")
    print("4. Use custom operators in your grammars")
    print("5. Run setups with custom operators")


if __name__ == "__main__":
    demo_custom_operators()
