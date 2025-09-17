"""
Grammar View Module

Handles the UI for grammar editing and management with full CRUD functionality.
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional
from uge.utils.constants import FILE_PATHS


class GrammarView:
    """
    Grammar View class for handling grammar editing UI with full CRUD operations.
    
    This class provides methods to render the grammar editor page,
    allowing users to create, read, update, and delete BNF grammar files.
    """
    
    def __init__(self):
        """Initialize the GrammarView."""
        pass
    
    def render_grammar_editor(self, grammars: List[str]) -> None:
        """
        Render the grammar editor page with full CRUD functionality.
        
        Args:
            grammars (List[str]): List of available grammar files
        """
        st.header("📝 Grammar Editor")
        st.markdown("Create, edit, and manage BNF grammar files")
        
        # Initialize session state for grammar management
        if 'grammar_action' not in st.session_state:
            st.session_state.grammar_action = 'view'
        if 'editing_grammar' not in st.session_state:
            st.session_state.editing_grammar = None
        if 'new_grammar_name' not in st.session_state:
            st.session_state.new_grammar_name = ""
        if 'new_grammar_content' not in st.session_state:
            st.session_state.new_grammar_content = ""
        
        # Action selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👁️ View Grammar", type="primary" if st.session_state.grammar_action == 'view' else "secondary"):
                st.session_state.grammar_action = 'view'
                st.rerun()
        
        with col2:
            if st.button("➕ Add New Grammar", type="primary" if st.session_state.grammar_action == 'add' else "secondary"):
                st.session_state.grammar_action = 'add'
                st.rerun()
        
        with col3:
            if st.button("✏️ Edit Grammar", type="primary" if st.session_state.grammar_action == 'edit' else "secondary"):
                st.session_state.grammar_action = 'edit'
                st.rerun()
        
        with col4:
            if st.button("🗑️ Delete Grammar", type="primary" if st.session_state.grammar_action == 'delete' else "secondary"):
                st.session_state.grammar_action = 'delete'
                st.rerun()
        
        st.markdown("---")
        
        # Render based on selected action
        if st.session_state.grammar_action == 'view':
            self._render_view_grammar(grammars)
        elif st.session_state.grammar_action == 'add':
            self._render_add_grammar()
        elif st.session_state.grammar_action == 'edit':
            self._render_edit_grammar(grammars)
        elif st.session_state.grammar_action == 'delete':
            self._render_delete_grammar(grammars)
    
    def _render_view_grammar(self, grammars: List[str]) -> None:
        """Render view grammar interface."""
        st.subheader("👁️ View Grammar")
        
        if not grammars:
            st.info("No grammar files found in the grammars directory.")
            return
        
        selected_grammar = st.selectbox("Select Grammar to View", grammars)
        
        if selected_grammar:
            grammar_path = FILE_PATHS['grammars_dir'] / selected_grammar
            
            try:
                with open(grammar_path, 'r', encoding='utf-8') as f:
                    grammar_content = f.read()
                
                st.subheader(f"📄 {selected_grammar}")
                
                # Display grammar content in a read-only text area
                st.text_area(
                    "Grammar Content",
                    value=grammar_content,
                    height=400,
                    disabled=True,
                    help="This is a read-only view of the grammar content"
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Grammar",
                    data=grammar_content,
                    file_name=selected_grammar,
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error reading grammar: {str(e)}")
    
    def _render_add_grammar(self) -> None:
        """Render add new grammar interface."""
        st.subheader("➕ Add New Grammar")
        
        # Grammar name input
        grammar_name = st.text_input(
            "Grammar Name",
            value=st.session_state.new_grammar_name,
            placeholder="my_grammar.bnf",
            help="Enter the name for the new grammar file (include .bnf extension)"
        )
        
        # Validate grammar name
        if grammar_name:
            if not grammar_name.endswith('.bnf'):
                st.warning("⚠️ Grammar name should end with '.bnf' extension")
            elif grammar_name in [g for g in self._get_available_grammars()]:
                st.error(f"❌ Grammar '{grammar_name}' already exists!")
            else:
                st.success(f"✅ Grammar name '{grammar_name}' is valid")
        
        # Grammar content input
        grammar_content = st.text_area(
            "Grammar Content (BNF Format)",
            value=st.session_state.new_grammar_content,
            height=400,
            placeholder="# Enter your BNF grammar here\n<start> ::= <expr>\n<expr> ::= <term> | <expr> <op> <term>\n...",
            help="Enter the BNF grammar content"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save New Grammar", type="primary"):
                if self._validate_grammar_input(grammar_name, grammar_content):
                    if self._save_new_grammar(grammar_name, grammar_content):
                        st.success(f"✅ Grammar '{grammar_name}' created successfully!")
                        st.session_state.new_grammar_name = ""
                        st.session_state.new_grammar_content = ""
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reset Form"):
                st.session_state.new_grammar_name = ""
                st.session_state.new_grammar_content = ""
                st.rerun()
        
        with col3:
            if st.button("📋 Load Template"):
                st.session_state.new_grammar_content = self._get_grammar_template()
                st.rerun()
    
    def _render_edit_grammar(self, grammars: List[str]) -> None:
        """Render edit grammar interface."""
        st.subheader("✏️ Edit Grammar")
        
        if not grammars:
            st.info("No grammar files found to edit.")
            return
        
        selected_grammar = st.selectbox("Select Grammar to Edit", grammars)
        
        if selected_grammar:
            grammar_path = FILE_PATHS['grammars_dir'] / selected_grammar
            
            try:
                with open(grammar_path, 'r', encoding='utf-8') as f:
                    grammar_content = f.read()
                
                st.subheader(f"✏️ Editing: {selected_grammar}")
                
                # Text area for editing
                edited_content = st.text_area(
                    "Grammar Content",
                    value=grammar_content,
                    height=400,
                    help="Edit the BNF grammar content"
                )
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("💾 Save Changes", type="primary"):
                        if self._save_grammar(selected_grammar, edited_content):
                            st.success(f"✅ Grammar '{selected_grammar}' updated successfully!")
                            st.rerun()
                
                with col2:
                    if st.button("🔄 Reset to Original"):
                        st.rerun()
                
                with col3:
                    st.download_button(
                        label="📥 Download",
                        data=edited_content,
                        file_name=selected_grammar,
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Error reading grammar: {str(e)}")
    
    def _render_delete_grammar(self, grammars: List[str]) -> None:
        """Render delete grammar interface."""
        st.subheader("🗑️ Delete Grammar")
        
        if not grammars:
            st.info("No grammar files found to delete.")
            return
        
        selected_grammar = st.selectbox("Select Grammar to Delete", grammars)
        
        if selected_grammar:
            grammar_path = FILE_PATHS['grammars_dir'] / selected_grammar
            
            # Show grammar preview before deletion
            try:
                with open(grammar_path, 'r', encoding='utf-8') as f:
                    grammar_content = f.read()
                
                st.warning(f"⚠️ You are about to delete: **{selected_grammar}**")
                
                # Show preview of grammar content
                with st.expander("Preview Grammar Content"):
                    st.code(grammar_content, language="bnf")
                
                # Confirmation
                st.error("🚨 **This action cannot be undone!**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🗑️ Confirm Delete", type="primary"):
                        if self._delete_grammar(selected_grammar):
                            st.success(f"✅ Grammar '{selected_grammar}' deleted successfully!")
                            st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", type="secondary"):
                        st.info("Deletion cancelled.")
                
                with col3:
                    st.download_button(
                        label="📥 Backup Download",
                        data=grammar_content,
                        file_name=f"{selected_grammar}.backup",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Error reading grammar: {str(e)}")
    
    def _validate_grammar_input(self, name: str, content: str) -> bool:
        """Validate grammar input."""
        if not name or not name.strip():
            st.error("❌ Grammar name cannot be empty!")
            return False
        
        if not name.endswith('.bnf'):
            st.error("❌ Grammar name must end with '.bnf' extension!")
            return False
        
        if not content or not content.strip():
            st.error("❌ Grammar content cannot be empty!")
            return False
        
        # Check if grammar already exists
        existing_grammars = self._get_available_grammars()
        if name in existing_grammars:
            st.error(f"❌ Grammar '{name}' already exists!")
            return False
        
        return True
    
    def _save_new_grammar(self, name: str, content: str) -> bool:
        """Save a new grammar file."""
        try:
            grammar_path = FILE_PATHS['grammars_dir'] / name
            with open(grammar_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            st.error(f"Error saving grammar: {str(e)}")
            return False
    
    def _save_grammar(self, name: str, content: str) -> bool:
        """Save an existing grammar file."""
        try:
            grammar_path = FILE_PATHS['grammars_dir'] / name
            with open(grammar_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            st.error(f"Error saving grammar: {str(e)}")
            return False
    
    def _delete_grammar(self, name: str) -> bool:
        """Delete a grammar file."""
        try:
            grammar_path = FILE_PATHS['grammars_dir'] / name
            grammar_path.unlink()  # Delete the file
            return True
        except Exception as e:
            st.error(f"Error deleting grammar: {str(e)}")
            return False
    
    def _get_available_grammars(self) -> List[str]:
        """Get list of available grammar files."""
        try:
            grammars_dir = FILE_PATHS['grammars_dir']
            if grammars_dir.exists():
                return [f.name for f in grammars_dir.glob("*.bnf")]
            return []
        except Exception:
            return []
    
    def _get_grammar_template(self) -> str:
        """Get a basic grammar template."""
        return """# BNF Grammar Template
# Example grammar for classification

<start> ::= <expr>

<expr> ::= <log_op>
         | <num_op>
         | <comparison>

<log_op> ::= <boolean_feature>
           | not_(<log_op>)
           | and_(<log_op>,<log_op>)
           | or_(<log_op>,<log_op>)

<comparison> ::= less_than(<num_op>,<num_op>)
               | greater_than(<num_op>,<num_op>)
               | equal(<num_op>,<num_op>)

<num_op> ::= <numeric_feature>
           | <constant>
           | add(<num_op>,<num_op>)
           | sub(<num_op>,<num_op>)
           | mul(<num_op>,<num_op>)

<numeric_feature> ::= x[<int>]
<boolean_feature> ::= b[<int>]

<constant> ::= <int>.<int> | <int>
<int> ::= <digit> | <digit><int>
<digit> ::= 0|1|2|3|4|5|6|7|8|9"""
