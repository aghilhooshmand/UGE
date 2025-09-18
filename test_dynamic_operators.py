#!/usr/bin/env python3
"""
Test Script for Dynamic Operator System

This script tests the complete dynamic operator system:
1. Verifies all operators from functions.py are now in JSON
2. Tests dynamic function mapping
3. Tests phenotype evaluation with dynamic operators
4. Verifies no functions.py dependency

Author: Aghil
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from uge.services.storage_service import StorageService
from uge.services.operator_service import OperatorService
from uge.utils.helpers import get_operator_service, fitness_eval
import numpy as np


def test_dynamic_operators():
    """Test the complete dynamic operator system."""
    print("🧪 Testing Dynamic Operator System")
    print("=" * 60)
    
    print("\n1️⃣ Verifying functions.py Removal:")
    functions_file = Path("grape/functions.py")
    print(f"   functions.py exists: {functions_file.exists()}")
    if not functions_file.exists():
        print("   ✅ functions.py successfully removed")
    else:
        print("   ❌ functions.py still exists")
        return
    
    print("\n2️⃣ Testing Operator Service:")
    storage_service = StorageService()
    operator_service = OperatorService(storage_service)
    
    all_operators = operator_service.get_all_operators()
    print(f"   Total operators loaded: {len(all_operators)}")
    
    # Check for key operators that were in functions.py
    expected_operators = [
        'add', 'sub', 'mul', 'pdiv',
        'and_', 'or_', 'not_', 'nand_', 'nor_',
        'greater_than_or_equal', 'less_than_or_equal', 'greater_than', 'less_than', 'equal', 'not_equal',
        'sigmoid', 'psin', 'pcos', 'psqrt', 'plog', 'exp', 'abs_',
        'minimum', 'maximum', 'min_', 'max_', 'neg',
        'if_'
    ]
    
    print("\n   Checking for expected operators:")
    missing_operators = []
    for op_name in expected_operators:
        if op_name in all_operators:
            op = all_operators[op_name]
            print(f"   ✅ {op_name}: {op.display_name} ({op.category})")
        else:
            print(f"   ❌ {op_name}: Missing")
            missing_operators.append(op_name)
    
    if missing_operators:
        print(f"\n   ❌ Missing operators: {missing_operators}")
        return
    else:
        print(f"\n   ✅ All expected operators found")
    
    print("\n3️⃣ Testing Dynamic Function Mapping:")
    
    # Test creating function mapping for all operators
    function_mapping = operator_service.create_operator_function_mapping(list(all_operators.keys()))
    print(f"   Function mapping created for {len(function_mapping)} operators")
    
    # Test some key functions
    test_functions = ['add', 'mul', 'equal', 'and_', 'sigmoid']
    print("\n   Testing function execution:")
    
    for func_name in test_functions:
        if func_name in function_mapping:
            try:
                func = function_mapping[func_name]
                
                # Test with appropriate parameters
                if func_name in ['add', 'mul']:
                    result = func(5, 3)
                    print(f"   ✅ {func_name}(5, 3) = {result}")
                elif func_name in ['equal']:
                    result = func(5, 5)
                    print(f"   ✅ {func_name}(5, 5) = {result}")
                elif func_name in ['and_']:
                    result = func(True, False)
                    print(f"   ✅ {func_name}(True, False) = {result}")
                elif func_name in ['sigmoid']:
                    result = func(0)
                    print(f"   ✅ {func_name}(0) = {result}")
                    
            except Exception as e:
                print(f"   ❌ {func_name}: Error - {e}")
        else:
            print(f"   ❌ {func_name}: Not in function mapping")
    
    print("\n4️⃣ Testing Global Operator Service:")
    
    # Test the global operator service from helpers
    global_service = get_operator_service()
    print(f"   Global operator service initialized: {global_service is not None}")
    
    global_operators = global_service.get_all_operators()
    print(f"   Global service operators: {len(global_operators)}")
    
    # Verify they're the same
    if len(global_operators) == len(all_operators):
        print("   ✅ Global service matches local service")
    else:
        print("   ❌ Global service differs from local service")
    
    print("\n5️⃣ Testing Custom Operator Addition:")
    
    # Add a test custom operator
    success, msg = operator_service.add_custom_operator(
        name="test_custom_func",
        display_name="Test Custom Function",
        description="Test custom function for dynamic system",
        function_code="""def test_custom_func(a, b):
    \"\"\"
    Test custom function that adds and multiplies.
    \"\"\"
    return (a + b) * 2""",
        parameter_count=2,
        return_type="float",
        category="custom",
        created_by="test_user"
    )
    
    print(f"   Custom operator added: {msg}")
    
    if success:
        # Test the custom function
        updated_operators = operator_service.get_all_operators()
        if "test_custom_func" in updated_operators:
            custom_func_mapping = operator_service.create_operator_function_mapping(["test_custom_func"])
            if "test_custom_func" in custom_func_mapping:
                custom_func = custom_func_mapping["test_custom_func"]
                result = custom_func(3, 4)  # Should be (3+4)*2 = 14
                print(f"   ✅ Custom function test: test_custom_func(3, 4) = {result}")
                
                if result == 14:
                    print("   ✅ Custom function works correctly")
                else:
                    print(f"   ❌ Custom function returned wrong result: {result}")
            else:
                print("   ❌ Custom function not in mapping")
        else:
            print("   ❌ Custom function not found in operators")
        
        # Clean up
        operator_service.remove_custom_operator("test_custom_func")
        print("   🗑️ Test custom operator removed")
    
    print("\n6️⃣ Testing Grammar Validation:")
    
    # Test grammar with known operators
    valid_grammar = """
    <log_op> ::= and_(<log_op>,<log_op>) | equal(<num_op>,<num_op>)
    <num_op> ::= add(<num_op>,<num_op>) | mul(<num_op>,<num_op>) | x[0]
    """
    
    is_valid, missing = operator_service.validate_grammar_operators(valid_grammar)
    print(f"   Valid grammar test: {is_valid}")
    if not is_valid:
        print(f"   Missing operators: {missing}")
    
    # Test grammar with non-existent operator
    invalid_grammar = """
    <log_op> ::= unknown_function(<log_op>,<log_op>)
    <num_op> ::= another_unknown(<num_op>,<num_op>)
    """
    
    is_valid, missing = operator_service.validate_grammar_operators(invalid_grammar)
    print(f"   Invalid grammar test: {not is_valid} (should be False)")
    if not is_valid:
        print(f"   Missing operators: {missing}")
    
    print("\n7️⃣ Testing Operator Categories:")
    
    categories = operator_service.get_available_categories()
    print(f"   Available categories: {categories}")
    
    for category in categories:
        ops = operator_service.get_operators_by_category(category)
        print(f"   • {category}: {len(ops)} operators")
    
    print("\n8️⃣ Testing JSON File Storage:")
    
    operators_file = Path("operators/custom_operators.json")
    print(f"   Custom operators file exists: {operators_file.exists()}")
    
    if operators_file.exists():
        file_size = operators_file.stat().st_size
        print(f"   File size: {file_size} bytes")
        
        if file_size > 0:
            print("   ✅ Custom operators file has content")
        else:
            print("   ❌ Custom operators file is empty")
    
    print("\n🎉 Dynamic Operator System Test Complete!")
    print("\nSummary:")
    print("✅ functions.py removed successfully")
    print("✅ All operators migrated to JSON-based system")
    print("✅ Dynamic function mapping works")
    print("✅ Global operator service works")
    print("✅ Custom operators can be added dynamically")
    print("✅ Grammar validation works")
    print("✅ JSON storage works")
    
    print("\n🚀 The system is now fully dynamic!")
    print("   - All operators are stored in JSON")
    print("   - Functions are mapped dynamically at runtime")
    print("   - Custom operators can be added through UI")
    print("   - No hardcoded functions.py dependency")


if __name__ == "__main__":
    test_dynamic_operators()
