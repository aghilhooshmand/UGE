"""
Operator View Module

Handles the UI for custom operator management.
"""

import streamlit as st
from typing import Dict, List, Optional
from uge.views.components.base_view import BaseView
from uge.services.operator_service import OperatorService
from uge.models.operator import CustomOperator


class OperatorView(BaseView):
    """
    Operator View class for handling custom operator management UI.
    
    This class provides methods to render the operator management interface,
    including creating, editing, and deleting custom operators.
    """
    
    def __init__(self, operator_service: OperatorService):
        """
        Initialize the OperatorView.
        
        Args:
            operator_service (OperatorService): Service for managing operators
        """
        super().__init__(
            title="🔧 Custom Operator Manager",
            description="Define and manage custom operators for grammatical evolution"
        )
        self.operator_service = operator_service
    
    def render(self) -> None:
        """Render the operator management interface."""
        self.render_header()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 View Operators", "➕ Add Operator", "✏️ Edit Operator", "🗑️ Delete Operator"])
        
        with tab1:
            self._render_view_operators()
        
        with tab2:
            self._render_add_operator()
        
        with tab3:
            self._render_edit_operator()
        
        with tab4:
            self._render_delete_operator()
    
    def _render_view_operators(self):
        """Render the view operators tab."""
        st.subheader("📋 Available Operators")
        
        # Show information about custom operators file
        st.info("""
        **📁 Custom Operators Storage:**
        - Custom operators are saved in `operators/custom_operators.json`
        - Built-in operators are part of the system and cannot be modified
        - When mapping operators in grammars, the system searches both built-in and custom operators
        - Operator names and function names must be unique across the entire system
        """)
        
        # Get all operators
        all_operators = self.operator_service.get_all_operators()
        
        if not all_operators:
            st.info("No operators available.")
            return
        
        # Category filter
        categories = self.operator_service.get_available_categories()
        selected_category = st.selectbox(
            "Filter by Category:",
            ["All"] + categories,
            key="operator_category_filter"
        )
        
        # Filter operators
        if selected_category == "All":
            filtered_operators = all_operators
        else:
            filtered_operators = self.operator_service.get_operators_by_category(selected_category)
        
        # Display operators
        for name, operator in filtered_operators.items():
            operator_type = "🔧 Built-in" if operator.is_builtin else "👤 Custom"
            storage_info = "Stored in custom_operators.json" if not operator.is_builtin else "Built into system"
            
            with st.expander(f"{operator_type} {operator.display_name} ({name})", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {operator.description}")
                    st.write(f"**Category:** {operator.category}")
                    st.write(f"**Parameters:** {operator.parameter_count}")
                    st.write(f"**Return Type:** {operator.return_type}")
                    st.write(f"**Created By:** {operator.created_by}")
                    st.write(f"**Storage:** {storage_info}")
                    if not operator.is_builtin:
                        st.write(f"**Created At:** {operator.created_at}")
                
                with col2:
                    st.code(operator.function_code, language="python")
                
                # Show validation status
                is_valid, error_msg = operator.validate_function_code()
                if is_valid:
                    st.success("✅ Function code is valid")
                else:
                    st.error(f"❌ Function code error: {error_msg}")
                
                # Show mapping info
                if not operator.is_builtin:
                    st.info(f"💾 This custom operator is saved in `operators/custom_operators.json` and will be available for mapping in grammars.")
    
    def _render_add_operator(self):
        """Render the add operator tab."""
        st.subheader("➕ Add Custom Operator")
        
        with st.form("add_operator_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "Operator Name:",
                    help="Internal name used in grammar (e.g., 'my_custom_op')",
                    placeholder="my_custom_op"
                )
                
                display_name = st.text_input(
                    "Display Name:",
                    help="Human-readable name for display",
                    placeholder="My Custom Operator"
                )
                
                description = st.text_area(
                    "Description:",
                    help="Description of what the operator does",
                    placeholder="This operator performs a custom operation..."
                )
            
            with col2:
                category = st.selectbox(
                    "Category:",
                    ["arithmetic", "comparison", "logical", "mathematical", "custom"],
                    help="Category for organizing operators"
                )
                
                parameter_count = st.number_input(
                    "Parameter Count:",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of parameters the operator takes"
                )
                
                return_type = st.selectbox(
                    "Return Type:",
                    ["bool", "float", "int", "any"],
                    help="Expected return type of the operator"
                )
            
            # Function code input
            st.subheader("Python Function Code")
            st.markdown("""
            **Instructions:**
            - Write a complete Python function
            - Function name must match the operator name
            - Use `import numpy as np` if needed
            - Function should handle both scalar and array inputs
            """)
            
            default_code = f"""def {name if name else 'my_operator'}(a, b):
    \"\"\"
    Custom operator function.
    
    Args:
        a: First parameter
        b: Second parameter
    
    Returns:
        Result of the operation
    \"\"\"
    # Add your custom logic here
    return a + b  # Example: addition"""
            
            function_code = st.text_area(
                "Function Code:",
                value=default_code if name else "",
                height=200,
                help="Complete Python function definition"
            )
            
            # Submit button - simple design
            st.markdown("---")
            submitted = st.form_submit_button(
                "💾 Save Operator",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Debug: Show that form was submitted
                st.info("🔄 Form submitted! Starting validation...")
                
                # Validate when save button is clicked
                validation_errors = []
                
                # Check required fields
                if not name:
                    validation_errors.append("❌ Operator name is required")
                if not display_name:
                    validation_errors.append("❌ Display name is required")
                if not function_code:
                    validation_errors.append("❌ Function code is required")
                
                # Check if operator name already exists
                if name and self.operator_service.get_operator(name):
                    validation_errors.append(f"❌ Operator name '{name}' already exists")
                
                # Check function name uniqueness
                if function_code:
                    function_name = self._extract_function_name(function_code)
                    if function_name:
                        if function_name != name:
                            validation_errors.append(f"❌ Function name '{function_name}' must match operator name '{name}'")
                        elif self.operator_service.get_operator(function_name):
                            validation_errors.append(f"❌ Function name '{function_name}' conflicts with existing operator")
                    
                    # Validate function syntax
                    try:
                        compile(function_code, '<string>', 'exec')
                    except SyntaxError as e:
                        validation_errors.append(f"❌ Syntax error: {str(e)}")
                
                # Show validation errors or save
                if validation_errors:
                    st.error("**Validation Errors:**")
                    for error in validation_errors:
                        st.error(error)
                    st.info(f"🔍 Debug: Found {len(validation_errors)} validation errors")
                else:
                    st.success("✅ Validation passed! Saving operator...")
                    # Save the operator
                    success, message = self.operator_service.add_custom_operator(
                        name=name,
                        display_name=display_name,
                        description=description,
                        function_code=function_code,
                        parameter_count=parameter_count,
                        return_type=return_type,
                        category=category,
                        created_by="user"
                    )
                    
                    if success:
                        # Reload operators to ensure the new operator appears in lists
                        self.operator_service.reload_operators()
                        st.success(f"✅ SUCCESS: {message}")
                        st.balloons()
                        st.info("🔄 Reloading page to show updated operator list...")
                        st.rerun()
                    else:
                        st.error(f"❌ FAILED: {message}")
                        st.info("🔍 Debug: Save operation failed")
            
    
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
    
    def _render_edit_operator(self):
        """Render the edit operator tab."""
        st.subheader("✏️ Edit Custom Operator")
        
        # Get custom operators (exclude built-in)
        all_operators = self.operator_service.get_all_operators()
        custom_operators = {name: op for name, op in all_operators.items() if not op.is_builtin}
        
        # Debug information
        st.info(f"📊 Found {len(all_operators)} total operators, {len(custom_operators)} custom operators")
        
        if not custom_operators:
            st.info("No custom operators available for editing.")
            st.info("💡 Tip: Add a custom operator first using the 'Add Operator' tab.")
            return
        
        # Operator selection
        operator_names = list(custom_operators.keys())
        selected_operator_name = st.selectbox(
            "Select Operator to Edit:",
            operator_names,
            key="edit_operator_select"
        )
        
        if selected_operator_name:
            operator = custom_operators[selected_operator_name]
            
            with st.form("edit_operator_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input("Operator Name:", value=operator.name, disabled=True)
                    
                    display_name = st.text_input(
                        "Display Name:",
                        value=operator.display_name
                    )
                    
                    description = st.text_area(
                        "Description:",
                        value=operator.description
                    )
                
                with col2:
                    category = st.selectbox(
                        "Category:",
                        ["arithmetic", "comparison", "logical", "mathematical", "custom"],
                        index=["arithmetic", "comparison", "logical", "mathematical", "custom"].index(operator.category)
                    )
                    
                    parameter_count = st.number_input(
                        "Parameter Count:",
                        min_value=1,
                        max_value=10,
                        value=operator.parameter_count
                    )
                    
                    return_type = st.selectbox(
                        "Return Type:",
                        ["bool", "float", "int", "any"],
                        index=["bool", "float", "int", "any"].index(operator.return_type)
                    )
                
                # Function code
                function_code = st.text_area(
                    "Function Code:",
                    value=operator.function_code,
                    height=200
                )
                
                # Submit button
                submitted = st.form_submit_button("✏️ Update Operator", use_container_width=True)
                
                if submitted:
                    # Debug information
                    st.info("🔄 Updating operator...")
                    
                    # Remove old operator first
                    remove_success, remove_message = self.operator_service.remove_custom_operator(operator.name)
                    if not remove_success:
                        st.error(f"Failed to remove old operator: {remove_message}")
                        return
                    
                    # Add updated operator
                    success, message = self.operator_service.add_custom_operator(
                        name=operator.name,
                        display_name=display_name,
                        description=description,
                        function_code=function_code,
                        parameter_count=parameter_count,
                        return_type=return_type,
                        category=category,
                        created_by=operator.created_by
                    )
                    
                    if success:
                        # Reload operators to ensure the updated operator appears correctly
                        self.operator_service.reload_operators()
                        st.success(f"✅ {message}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to update operator: {message}")
    
    def _render_delete_operator(self):
        """Render the delete operator tab."""
        st.subheader("🗑️ Delete Custom Operator")
        
        # Get custom operators (exclude built-in)
        all_operators = self.operator_service.get_all_operators()
        custom_operators = {name: op for name, op in all_operators.items() if not op.is_builtin}
        
        if not custom_operators:
            st.info("No custom operators available for deletion.")
            return
        
        # Operator selection
        operator_names = list(custom_operators.keys())
        selected_operator_name = st.selectbox(
            "Select Operator to Delete:",
            operator_names,
            key="delete_operator_select"
        )
        
        if selected_operator_name:
            operator = custom_operators[selected_operator_name]
            
            # Show operator details
            st.write(f"**Operator:** {operator.display_name} ({operator.name})")
            st.write(f"**Description:** {operator.description}")
            st.write(f"**Category:** {operator.category}")
            
            # Confirmation
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🗑️ Delete Operator", type="primary", use_container_width=True):
                    success, message = self.operator_service.remove_custom_operator(operator.name)
                    
                    if success:
                        # Reload operators to ensure the deleted operator is removed from lists
                        self.operator_service.reload_operators()
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Warning
            st.warning("⚠️ Deleting an operator will affect any grammars that use it.")
    
    def render_operator_selector(self, key: str, label: str = "Select Operators:", 
                               categories: List[str] = None, multiple: bool = True) -> List[str]:
        """
        Render a multi-select widget for operators.
        
        Args:
            key (str): Unique key for the widget
            label (str): Label for the widget
            categories (List[str]): Categories to include (None for all)
            multiple (bool): Allow multiple selection
            
        Returns:
            List[str]: Selected operator names
        """
        all_operators = self.operator_service.get_all_operators()
        
        if categories:
            filtered_operators = {}
            for cat in categories:
                filtered_operators.update(self.operator_service.get_operators_by_category(cat))
            operators = filtered_operators
        else:
            operators = all_operators
        
        if not operators:
            st.info("No operators available.")
            return []
        
        # Create options with display names
        operator_options = {}
        for name, operator in operators.items():
            display_text = f"{operator.display_name} ({name}) - {operator.category}"
            operator_options[display_text] = name
        
        if multiple:
            selected_display = st.multiselect(
                label,
                list(operator_options.keys()),
                key=key
            )
            return [operator_options[display] for display in selected_display]
        else:
            selected_display = st.selectbox(
                label,
                ["None"] + list(operator_options.keys()),
                key=key
            )
            if selected_display == "None":
                return []
            return [operator_options[selected_display]]
