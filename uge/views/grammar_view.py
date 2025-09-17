"""
Grammar View Module

Handles the UI for grammar editing and management.
"""

import streamlit as st
from pathlib import Path
from typing import List
from uge.utils.constants import FILE_PATHS


class GrammarView:
    """
    Grammar View class for handling grammar editing UI.
    
    This class provides methods to render the grammar editor page,
    allowing users to view, edit, and manage BNF grammar files.
    """
    
    def __init__(self):
        """Initialize the GrammarView."""
        pass
    
    def render_grammar_editor(self, grammars: List[str]) -> None:
        """
        Render the grammar editor page.
        
        Args:
            grammars (List[str]): List of available grammar files
        """
        st.header("📝 Grammar Editor")
        st.markdown("Edit and manage BNF grammar files")
        
        if grammars:
            selected_grammar = st.selectbox("Select Grammar", grammars)
            
            if selected_grammar:
                grammar_path = FILE_PATHS['grammars_dir'] / selected_grammar
                
                # Display grammar content
                st.subheader(f"Grammar: {selected_grammar}")
                
                try:
                    with open(grammar_path, 'r') as f:
                        grammar_content = f.read()
                    
                    # Text area for editing
                    edited_content = st.text_area(
                        "Grammar Content",
                        value=grammar_content,
                        height=400,
                        help="Edit the BNF grammar content"
                    )
                    
                    # Save button
                    if st.button("💾 Save Grammar"):
                        try:
                            with open(grammar_path, 'w') as f:
                                f.write(edited_content)
                            st.success(f"Grammar '{selected_grammar}' saved successfully!")
                        except Exception as e:
                            st.error(f"Error saving grammar: {str(e)}")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Grammar",
                        data=grammar_content,
                        file_name=selected_grammar,
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error reading grammar: {str(e)}")
        else:
            st.info("No grammar files found in the grammars directory.")
