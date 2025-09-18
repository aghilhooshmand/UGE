# 🔧 Custom Operator System

## Overview

The UGE Custom Operator System allows users to define their own operators for grammatical evolution through a user-friendly interface. Instead of being limited to predefined operators in `functions.py`, users can now create custom operators that are dynamically mapped to Python functions at runtime.

## 🎯 Key Features

- **📝 Visual Operator Definition**: Create operators through an intuitive UI
- **🔧 Function Code Validation**: Real-time validation of Python function code
- **🗂️ Category Organization**: Organize operators by type (arithmetic, comparison, logical, etc.)
- **💾 Persistent Storage**: Operators are saved and loaded automatically
- **🔄 Dynamic Mapping**: Operators are mapped to functions at runtime
- **📋 Grammar Validation**: Check if all operators in a grammar are available
- **🔧 Built-in + Custom**: Mix built-in and custom operators seamlessly

## 🏗️ Architecture

### Components

1. **`CustomOperator` Model**: Represents a single operator with metadata
2. **`OperatorRegistry`**: Manages collection of operators
3. **`OperatorService`**: Business logic for operator management
4. **`OperatorView`**: UI for operator management
5. **Dynamic Function Mapping**: Runtime mapping of operators to Python functions

### Data Flow

```
User Input → OperatorView → OperatorService → OperatorRegistry → Storage
     ↑                                                              ↓
UI Display ← OperatorView ← OperatorService ← OperatorRegistry ← Storage
```

## 📋 Usage Guide

### 1. Accessing the Operator Manager

1. **Start the UGE application**
2. **Navigate to "🔧 Operator Manager"** in the sidebar
3. **Use the tabs** to manage operators:
   - 📋 **View Operators**: Browse all available operators
   - ➕ **Add Operator**: Create new custom operators
   - ✏️ **Edit Operator**: Modify existing custom operators
   - 🗑️ **Delete Operator**: Remove custom operators

### 2. Creating a Custom Operator

#### Required Information:
- **Operator Name**: Internal name used in grammar (e.g., `my_custom_op`)
- **Display Name**: Human-readable name for display
- **Description**: What the operator does
- **Category**: Type of operator (arithmetic, comparison, logical, mathematical, custom)
- **Parameter Count**: Number of parameters (1-10)
- **Return Type**: Expected return type (bool, float, int, any)
- **Function Code**: Complete Python function definition

#### Example Custom Operator:

```python
def my_distance(a, b):
    """
    Calculate Euclidean distance between two values.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        Distance between a and b
    """
    import math
    return math.sqrt((a - b) ** 2)
```

#### Function Code Requirements:
- Must be a complete function definition
- Function name must match operator name
- Should handle both scalar and array inputs
- Use `import numpy as np` for array operations
- Include proper error handling

### 3. Using Custom Operators in Grammars

#### Grammar Syntax:
```bnf
<log_op> ::= <custom_comparisons> | and_(<log_op>,<log_op>) | or_(<log_op>,<log_op>)
<custom_comparisons> ::= my_custom_comparison(<num_op>,<num_op>) | equal(<num_op>,<num_op>)
<num_op> ::= my_custom_math(<num_op>,<num_op>) | add(<num_op>,<num_op>)
```

#### Example Grammar File:
```bnf
# Custom Operator Example Grammar
<log_op> ::= my_xor(<log_op>,<log_op>) | and_(<log_op>,<log_op>)
<num_op> ::= my_distance(<num_op>,<num_op>) | my_power(<num_op>,<num_op>) | add(<num_op>,<num_op>)
```

### 4. Grammar Validation

The system automatically validates that all operators used in a grammar are available:

- **✅ Valid**: All operators exist in the registry
- **❌ Invalid**: Missing operators are reported with names

## 🔧 Built-in Operators

The system comes with pre-configured built-in operators:

### Arithmetic Operators
- `add`: Addition
- `sub`: Subtraction  
- `mul`: Multiplication
- `pdiv`: Protected division (handles division by zero)

### Logical Operators
- `and_`: Logical AND
- `or_`: Logical OR
- `not_`: Logical NOT

### Comparison Operators
- `greater_than_or_equal`: Greater than or equal comparison
- `less_than_or_equal`: Less than or equal comparison
- `equal`: Equality comparison

## 📁 File Structure

```
UGE/
├── uge/
│   ├── models/
│   │   └── operator.py              # Operator data models
│   ├── services/
│   │   └── operator_service.py      # Operator business logic
│   └── views/
│       └── operator_view.py         # Operator UI
├── grammars/
│   └── custom_operator_example.bnf  # Example grammar
├── operators/
│   └── custom_operators.json       # Operator storage (auto-created)
└── demo_custom_operators.py        # Demonstration script
```

## 🧪 Testing and Demo

### Run the Demo Script:
```bash
python3 demo_custom_operators.py
```

This script demonstrates:
- Creating custom operators
- Function mapping
- Grammar validation
- Category organization

### Example Output:
```
🧬 UGE Custom Operator System Demo
==================================================

1️⃣ Initial Built-in Operators:
   • Addition (add) - arithmetic
   • Subtraction (sub) - arithmetic
   • Multiplication (mul) - arithmetic
   • Protected Division (pdiv) - arithmetic
   • Logical AND (and_) - logical
   • Logical OR (or_) - logical
   • Logical NOT (not_) - logical
   • Greater Than or Equal (greater_than_or_equal) - comparison
   • Less Than or Equal (less_than_or_equal) - comparison
   • Equal (equal) - comparison

   Total operators: 10

2️⃣ Adding Custom Operators:
   • my_distance: Operator 'my_distance' added successfully
   • my_xor: Operator 'my_xor' added successfully
   • my_power: Operator 'my_power' added successfully

3️⃣ Updated Operator Registry:
   🔧 Built-in Addition (add) - arithmetic
   🔧 Built-in Subtraction (sub) - arithmetic
   ...
   👤 Custom My Distance (my_distance) - mathematical
   👤 Custom My XOR (my_xor) - logical
   👤 Custom My Power (my_power) - arithmetic

   Total operators: 13

4️⃣ Creating Function Mapping:
   Created mapping for 5 operators:
   • my_distance: <function my_distance at 0x...>
   • my_xor: <function my_xor at 0x...>
   • my_power: <function my_power at 0x...>
   • add: <function add at 0x...>
   • equal: <function equal at 0x...>

5️⃣ Testing Custom Functions:
   • my_distance(3, 7) = 4.0
   • my_xor(True, False) = True
   • my_xor(True, True) = False
   • my_power(2, 3) = 8

6️⃣ Grammar Validation:
   ✅ Grammar is valid - all operators are available
   ❌ Invalid grammar test passed - missing operators: ['unknown_operator', 'another_unknown']

7️⃣ Operator Categories:
   • arithmetic: 4 operators
   • comparison: 3 operators
   • logical: 4 operators
   • mathematical: 1 operators

🎉 Demo completed successfully!
```

## 🔄 Integration with GE System

### Dynamic Function Mapping

When running a setup, the system:

1. **Parses the grammar** to extract operator names
2. **Validates operators** are available in the registry
3. **Creates function mapping** for available operators
4. **Injects functions** into the GE execution environment
5. **Executes the setup** with custom operators

### Runtime Process:
```python
# 1. Get operators from grammar
grammar_operators = extract_operators_from_grammar(grammar_content)

# 2. Validate operators
is_valid, missing = operator_service.validate_grammar_operators(grammar_content)

# 3. Create function mapping
function_mapping = operator_service.create_operator_function_mapping(grammar_operators)

# 4. Inject into GE environment
for name, func in function_mapping.items():
    globals()[name] = func

# 5. Run GE with custom operators
result = run_grammatical_evolution(grammar, function_mapping)
```

## 🛠️ Advanced Usage

### Custom Operator Categories

Create operators in different categories:

- **arithmetic**: Mathematical operations (add, sub, mul, div, power, etc.)
- **comparison**: Comparison operations (greater_than, less_than, equal, etc.)
- **logical**: Logical operations (and, or, not, xor, etc.)
- **mathematical**: Advanced math functions (sin, cos, sqrt, log, etc.)
- **custom**: User-defined operations

### Complex Operators

Create operators that work with arrays:

```python
def my_array_sum(a, b):
    """
    Sum two arrays element-wise.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Element-wise sum
    """
    import numpy as np
    return np.array(a) + np.array(b)
```

### Error Handling

Include proper error handling in custom operators:

```python
def my_safe_divide(a, b):
    """
    Safe division with error handling.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        Division result or default value
    """
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return 1.0
```

## 🚀 Benefits

1. **🎯 Flexibility**: Create operators tailored to your specific problem domain
2. **🔧 No Code Changes**: Add operators without modifying the core system
3. **📊 Better Performance**: Optimize operators for your specific use case
4. **🧪 Experimentation**: Test different operator combinations easily
5. **👥 Collaboration**: Share operator definitions with team members
6. **📈 Scalability**: Add as many operators as needed
7. **🔄 Reusability**: Use operators across multiple grammars and setups

## 🔍 Troubleshooting

### Common Issues

#### 1. Function Name Mismatch
**Error**: "Function name must match operator name"
**Solution**: Ensure the function name in the code matches the operator name

#### 2. Syntax Errors
**Error**: "Syntax error in function code"
**Solution**: Check Python syntax, ensure proper indentation and parentheses

#### 3. Missing Operators
**Error**: "Missing operators: ['unknown_op']"
**Solution**: Define the missing operators in the Operator Manager

#### 4. Import Errors
**Error**: "NameError: name 'np' is not defined"
**Solution**: Add `import numpy as np` at the beginning of your function code

### Validation Tips

1. **Test functions independently** before adding to the system
2. **Use simple test cases** to verify operator behavior
3. **Check return types** match the specified return type
4. **Handle edge cases** (zero division, empty arrays, etc.)
5. **Use proper imports** for required libraries

## 📚 Examples

See the following files for complete examples:

- `grammars/custom_operator_example.bnf`: Example grammar using custom operators
- `demo_custom_operators.py`: Complete demonstration script
- `uge/views/operator_view.py`: Full UI implementation

## 🎉 Conclusion

The Custom Operator System makes UGE highly extensible and user-friendly. Users can now create domain-specific operators without modifying the core system, enabling more sophisticated grammatical evolution experiments.

**Happy evolving with custom operators! 🧬✨**
