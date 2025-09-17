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
