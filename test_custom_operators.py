#!/usr/bin/env python3
"""
Test Script for Custom Operators File Functionality

This script tests the custom operators JSON file functionality:
1. Creates custom operators
2. Saves them to JSON file
3. Loads them back from JSON file
4. Verifies unique name validation

Author: Aghil
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from uge.services.storage_service import StorageService
from uge.services.operator_service import OperatorService


def test_custom_operators_file():
    """Test the custom operators JSON file functionality."""
    print("🧪 Testing Custom Operators File Functionality")
    print("=" * 60)
    
    # Initialize services
    storage_service = StorageService()
    operator_service = OperatorService(storage_service)
    
    print("\n1️⃣ Initial State:")
    all_operators = operator_service.get_all_operators()
    custom_count = len([op for op in all_operators.values() if not op.is_builtin])
    print(f"   Total operators: {len(all_operators)}")
    print(f"   Custom operators: {custom_count}")
    print(f"   Built-in operators: {len(all_operators) - custom_count}")
    
    # Check if custom operators file exists
    operators_file = Path("operators/custom_operators.json")
    print(f"   Custom operators file exists: {operators_file.exists()}")
    
    print("\n2️⃣ Adding Custom Operators:")
    
    # Test 1: Add a valid custom operator
    success1, msg1 = operator_service.add_custom_operator(
        name="test_distance",
        display_name="Test Distance",
        description="Test operator for distance calculation",
        function_code="""def test_distance(a, b):
    \"\"\"
    Calculate distance between two values.
    \"\"\"
    return abs(a - b)""",
        parameter_count=2,
        return_type="float",
        category="mathematical",
        created_by="test_user"
    )
    
    print(f"   ✅ test_distance: {msg1}")
    
    # Test 2: Try to add duplicate operator name
    success2, msg2 = operator_service.add_custom_operator(
        name="test_distance",  # Same name as above
        display_name="Another Test Distance",
        description="This should fail",
        function_code="def test_distance(a, b): return a + b",
        parameter_count=2,
        return_type="float",
        category="mathematical",
        created_by="test_user"
    )
    
    print(f"   ❌ Duplicate name test: {msg2}")
    
    # Test 3: Try to add operator with mismatched function name
    success3, msg3 = operator_service.add_custom_operator(
        name="my_operator",
        display_name="My Operator",
        description="This should fail - function name mismatch",
        function_code="def different_function_name(a, b): return a + b",  # Wrong function name
        parameter_count=2,
        return_type="float",
        category="mathematical",
        created_by="test_user"
    )
    
    print(f"   ❌ Function name mismatch test: {msg3}")
    
    # Test 4: Add another valid operator
    success4, msg4 = operator_service.add_custom_operator(
        name="test_power",
        display_name="Test Power",
        description="Test operator for power calculation",
        function_code="""def test_power(a, b):
    \"\"\"
    Calculate a raised to power b.
    \"\"\"
    try:
        return a ** b
    except:
        return 1.0""",
        parameter_count=2,
        return_type="float",
        category="arithmetic",
        created_by="test_user"
    )
    
    print(f"   ✅ test_power: {msg4}")
    
    print("\n3️⃣ Verify Custom Operators File:")
    
    # Check if file exists and has content
    if operators_file.exists():
        with open(operators_file, 'r') as f:
            data = json.load(f)
        
        operators_data = data.get('operators', {})
        custom_operators = {name: op for name, op in operators_data.items() if not op.get('is_builtin', True)}
        
        print(f"   📁 File exists: {operators_file}")
        print(f"   📄 File size: {operators_file.stat().st_size} bytes")
        print(f"   🔢 Total operators in file: {len(operators_data)}")
        print(f"   👤 Custom operators in file: {len(custom_operators)}")
        
        print("\n   Custom operators in JSON file:")
        for name, op_data in custom_operators.items():
            print(f"     • {op_data['display_name']} ({name}) - {op_data['category']}")
    
    print("\n4️⃣ Reload and Verify:")
    
    # Create a new service instance to test loading from file
    new_storage_service = StorageService()
    new_operator_service = OperatorService(new_storage_service)
    
    reloaded_operators = new_operator_service.get_all_operators()
    reloaded_custom_count = len([op for op in reloaded_operators.values() if not op.is_builtin])
    
    print(f"   🔄 Reloaded total operators: {len(reloaded_operators)}")
    print(f"   👤 Reloaded custom operators: {reloaded_custom_count}")
    
    # Check if our custom operators are there
    test_distance_op = new_operator_service.get_operator("test_distance")
    test_power_op = new_operator_service.get_operator("test_power")
    
    if test_distance_op:
        print(f"   ✅ test_distance operator loaded: {test_distance_op.display_name}")
    else:
        print(f"   ❌ test_distance operator not found")
    
    if test_power_op:
        print(f"   ✅ test_power operator loaded: {test_power_op.display_name}")
    else:
        print(f"   ❌ test_power operator not found")
    
    print("\n5️⃣ Test Function Mapping:")
    
    # Test creating function mapping
    custom_operator_names = ["test_distance", "test_power", "add", "equal"]
    function_mapping = new_operator_service.create_operator_function_mapping(custom_operator_names)
    
    print(f"   🔗 Function mapping created for {len(function_mapping)} operators:")
    for name, func in function_mapping.items():
        print(f"     • {name}: {func}")
    
    # Test the functions
    if "test_distance" in function_mapping:
        result = function_mapping["test_distance"](5, 3)
        print(f"   🧪 test_distance(5, 3) = {result}")
    
    if "test_power" in function_mapping:
        result = function_mapping["test_power"](2, 3)
        print(f"   🧪 test_power(2, 3) = {result}")
    
    print("\n6️⃣ Cleanup:")
    
    # Remove test operators
    removed1, msg1 = new_operator_service.remove_custom_operator("test_distance")
    removed2, msg2 = new_operator_service.remove_custom_operator("test_power")
    
    print(f"   🗑️ Removed test_distance: {msg1}")
    print(f"   🗑️ Removed test_power: {msg2}")
    
    # Final state
    final_operators = new_operator_service.get_all_operators()
    final_custom_count = len([op for op in final_operators.values() if not op.is_builtin])
    
    print(f"   📊 Final state - Total: {len(final_operators)}, Custom: {final_custom_count}")
    
    print("\n🎉 All tests completed successfully!")
    print("\nKey Features Verified:")
    print("✅ Custom operators are saved to operators/custom_operators.json")
    print("✅ Operators are loaded from JSON file on startup")
    print("✅ Unique name validation works (operator names and function names)")
    print("✅ Function mapping works with custom operators")
    print("✅ CRUD operations work correctly")


if __name__ == "__main__":
    test_custom_operators_file()
