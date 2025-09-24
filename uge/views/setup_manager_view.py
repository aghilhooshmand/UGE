"""
Setup Manager View Module

Handles the UI for setup management and monitoring.
"""

import streamlit as st
from typing import List, Dict, Any
from uge.services.storage_service import StorageService


class SetupManagerView:
    """
    Setup Manager View class for handling setup management UI.
    
    This class provides methods to render the setup manager page,
    allowing users to view, manage, and monitor their setups.
    """
    
    def __init__(self, storage_service: StorageService):
        """
        Initialize the SetupManagerView.
        
        Args:
            storage_service (StorageService): Storage service instance
        """
        self.storage_service = storage_service
    
    def render_setup_manager(self) -> None:
        """Render the setup manager page."""
        st.header("🧪 Setup Manager")
        st.markdown("Manage and monitor your setups")
        
        try:
            setup_paths = self.storage_service.list_setups()
            
            if setup_paths:
                # Display setups in a table
                st.subheader("Available Setups")
                
                # Create setup data for display
                setup_data = []
                for setup_path in setup_paths:
                    setup_id = setup_path.name
                    try:
                        setup = self.storage_service.load_setup(setup_id)
                        if setup and setup.config:
                            setup_data.append({
                                "ID": setup_id,
                                "Name": setup.config.setup_name,
                                "Dataset": setup.config.dataset,
                                "Grammar": setup.config.grammar,
                                "Runs": len(setup.results) if setup.results else 0,
                                "Status": "Completed" if setup.results else "No runs"
                            })
                        else:
                            setup_data.append({
                                "ID": setup_id,
                                "Name": "Unknown",
                                "Dataset": "Unknown",
                                "Grammar": "Unknown",
                                "Runs": 0,
                                "Status": "Error"
                            })
                    except Exception as e:
                        setup_data.append({
                            "ID": setup_id,
                            "Name": "Error",
                            "Dataset": "Error",
                            "Grammar": "Error",
                            "Runs": 0,
                            "Status": f"Error: {str(e)}"
                        })
                
                # Display as table
                if setup_data:
                    st.dataframe(setup_data, use_container_width=True)
                    
                    st.divider()
                    
                    # Individual Setup Actions
                    st.subheader("Individual Setup Actions")
                    
                    # Setup selection dropdown
                    setup_ids = [setup["ID"] for setup in setup_data]
                    selected_setup = st.selectbox(
                        "Select a setup to manage:",
                        options=setup_ids,
                        help="Choose a setup to view details or delete"
                    )
                    
                    if selected_setup:
                        # Load the full setup to get detailed configuration
                        try:
                            setup = self.storage_service.load_setup(selected_setup)
                            if setup and setup.config:
                                # Display selected setup details
                                selected_setup_data = next(setup for setup in setup_data if setup["ID"] == selected_setup)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Setup Details:**")
                                    st.write(f"**Name:** {selected_setup_data['Name']}")
                                    st.write(f"**Dataset:** {selected_setup_data['Dataset']}")
                                    st.write(f"**Grammar:** {selected_setup_data['Grammar']}")
                                    st.write(f"**Runs:** {selected_setup_data['Runs']}")
                                    st.write(f"**Status:** {selected_setup_data['Status']}")
                                
                                with col2:
                                    st.markdown("**Actions:**")
                                    
                                    # Individual delete button
                                    delete_key = f"delete_{selected_setup}"
                                    if st.button(f"🗑️ Delete Setup", key=delete_key, type="secondary"):
                                        if st.session_state.get(f'confirm_delete_{selected_setup}', False):
                                            try:
                                                self.storage_service.delete_setup(selected_setup)
                                                st.success(f"Setup '{selected_setup}' deleted successfully!")
                                                st.session_state[f'confirm_delete_{selected_setup}'] = False
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error deleting setup: {str(e)}")
                                        else:
                                            st.session_state[f'confirm_delete_{selected_setup}'] = True
                                            st.warning(f"Click again to confirm deletion of setup '{selected_setup}'!")
                                    
                                    # Cancel delete button
                                    if st.session_state.get(f'confirm_delete_{selected_setup}', False):
                                        if st.button(f"❌ Cancel Delete", key=f"cancel_{selected_setup}"):
                                            st.session_state[f'confirm_delete_{selected_setup}'] = False
                                            st.rerun()
                                
                                # Display detailed configuration with parameter types
                                st.divider()
                                st.subheader("📋 Configuration Details")
                                
                                # Get parameter information with types
                                param_info = setup.config.get_parameter_info()
                                
                                # Check if parameter_configs is available
                                if not hasattr(setup.config, 'parameter_configs') or setup.config.parameter_configs is None:
                                    st.warning("⚠️ This setup was created before the parameter configuration system was implemented. All parameters are shown as 'Fixed' by default.")
                                
                                # Group parameters by category
                                categories = {}
                                for param_name, info in param_info.items():
                                    category = info['category']
                                    if category not in categories:
                                        categories[category] = []
                                    categories[category].append((param_name, info))
                                
                                # Display parameters by category
                                for category, params in categories.items():
                                    with st.expander(f"📁 {category} Parameters", expanded=True):
                                        for param_name, info in params:
                                            param_type = info['type']
                                            param_value = info['value']
                                            
                                            # Create type indicator
                                            if param_type == 'fixed':
                                                type_indicator = "🔒 Fixed"
                                                type_color = "green"
                                            elif param_type == 'random':
                                                type_indicator = "🎲 Random"
                                                type_color = "orange"
                                            elif param_type == 'custom':
                                                type_indicator = "⚙️ Custom Expression"
                                                type_color = "blue"
                                            else:
                                                type_indicator = "❓ Unknown"
                                                type_color = "gray"
                                            
                                            # Format parameter name
                                            display_name = param_name.replace('_', ' ').title()
                                            
                                            # Format parameter value and additional info
                                            if param_type == 'random':
                                                # Show random parameter details (dynamic mode)
                                                min_val = info.get('min_value', param_value)
                                                max_val = info.get('max_value', param_value)
                                                display_value = f"Value: {param_value} | Random between {min_val} and {max_val}"
                                            elif param_type == 'custom':
                                                # Show custom configuration details
                                                config = info.get('config', {})
                                                mode = config.get('mode', 'custom')
                                                base_value = config.get('value', param_value)
                                                change_every = config.get('change_every', 5)
                                                change_amount = config.get('change_amount', 0.01)
                                                change_operation = config.get('change_operation', 'add')
                                                min_value = config.get('min_value', base_value)
                                                max_value = config.get('max_value', base_value)
                                                
                                                # Create descriptive text
                                                if change_operation == 'add':
                                                    operation_text = f"add {change_amount}"
                                                elif change_operation == 'subtract':
                                                    operation_text = f"subtract {change_amount}"
                                                elif change_operation == 'multiply':
                                                    operation_text = f"multiply by {change_amount}"
                                                elif change_operation == 'divide':
                                                    operation_text = f"divide by {change_amount}"
                                                else:
                                                    operation_text = f"apply {change_operation} with {change_amount}"
                                                
                                                display_value = f"Starts at {base_value}, every {change_every} generations {operation_text} (range: {min_value} to {max_value})"
                                            elif isinstance(param_value, list):
                                                if len(param_value) > 5:
                                                    display_value = f"[{param_value[0]}, ..., {param_value[-1]}] ({len(param_value)} items)"
                                                else:
                                                    display_value = str(param_value)
                                            elif isinstance(param_value, str) and len(param_value) > 50:
                                                display_value = f"{param_value[:50]}..."
                                            else:
                                                display_value = str(param_value)
                                            
                                            # Display parameter with type indicator
                                            col1, col2, col3 = st.columns([3, 1, 2])
                                            with col1:
                                                st.write(f"**{display_name}:**")
                                            with col2:
                                                st.markdown(f":{type_color}[{type_indicator}]")
                                            with col3:
                                                st.code(display_value, language="text")
                            else:
                                st.error("Could not load setup configuration")
                        except Exception as e:
                            st.error(f"Error loading setup details: {str(e)}")
                    
                    st.divider()
                    
                    # Bulk Actions
                    st.subheader("Bulk Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Delete All"):
                            if st.session_state.get('confirm_delete_all', False):
                                try:
                                    for setup_path in setup_paths:
                                        self.storage_service.delete_setup(setup_path.name)
                                    st.success("All setups deleted successfully!")
                                    st.session_state.confirm_delete_all = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting setups: {str(e)}")
                            else:
                                st.session_state.confirm_delete_all = True
                                st.warning("Click again to confirm deletion of ALL setups!")
                    
                    with col3:
                        if st.session_state.get('confirm_delete_all', False):
                            if st.button("❌ Cancel"):
                                st.session_state.confirm_delete_all = False
                                st.rerun()
                else:
                    st.info("No setup data available to display.")
            else:
                st.info("No setups found. Create a setup using the 'Run Setup' page.")
                
        except Exception as e:
            st.error(f"Error loading setups: {str(e)}")
