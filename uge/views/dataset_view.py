"""
Dataset View for UGE Application

This module provides the dataset view for managing datasets
used in Grammatical Evolution experiments.

Classes:
- DatasetView: Main view for dataset operations

Author: UGE Team
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from uge.views.components.base_view import BaseView
from uge.views.components.forms import Forms


class DatasetView(BaseView):
    """
    View for dataset operations.
    
    This view handles the user interface for managing datasets
    used in GE experiments.
    
    Attributes:
        on_dataset_upload (Optional[Callable]): Callback for dataset upload
        on_dataset_delete (Optional[Callable]): Callback for dataset deletion
        on_dataset_preview (Optional[Callable]): Callback for dataset preview
    """
    
    def __init__(self, on_dataset_upload: Optional[Callable] = None,
                 on_dataset_delete: Optional[Callable] = None,
                 on_dataset_preview: Optional[Callable] = None):
        """
        Initialize dataset view.
        
        Args:
            on_dataset_upload (Optional[Callable]): Callback for dataset upload
            on_dataset_delete (Optional[Callable]): Callback for dataset deletion
            on_dataset_preview (Optional[Callable]): Callback for dataset preview
        """
        super().__init__(
            title="📊 Dataset Manager",
            description="Upload, manage, and preview your datasets"
        )
        self.on_dataset_upload = on_dataset_upload
        self.on_dataset_delete = on_dataset_delete
        self.on_dataset_preview = on_dataset_preview
    
    def render(self, datasets: List[str] = None):
        """
        Render the dataset view.
        
        Args:
            datasets (List[str]): Available datasets
        """
        if datasets is None:
            datasets = []
        
        self.render_header()
        
        # Dataset action selection
        dataset_action = st.radio(
            "Dataset Action:",
            ["➕ Add Dataset", "✏️ Edit Dataset", "👁️ Preview Dataset"],
            key="dataset_action"
        )
        
        if dataset_action == "➕ Add Dataset":
            self._render_upload_form()
        elif dataset_action == "✏️ Edit Dataset":
            self._render_management_form(datasets)
        elif dataset_action == "👁️ Preview Dataset":
            self._render_preview_form(datasets)
    
    def _render_upload_form(self):
        """Render dataset upload form."""
        form_submitted, uploaded_file = Forms.create_dataset_upload_form()
        
        if form_submitted and uploaded_file:
            self._handle_dataset_upload(uploaded_file)
    
    def _render_management_form(self, datasets: List[str]):
        """Render dataset management form."""
        action, selected_dataset = Forms.create_dataset_management_form(datasets)
        
        if action == "delete":
            self._handle_dataset_deletion(selected_dataset)
        elif action == "preview":
            self._handle_dataset_preview(selected_dataset)
    
    def _render_preview_form(self, datasets: List[str]):
        """Render dataset preview form."""
        if not datasets:
            st.info("No datasets available")
            return
        
        preview_dataset = st.selectbox("Select Dataset to Preview", datasets)
        
        if st.button("👁️ Preview Dataset"):
            self._handle_dataset_preview(preview_dataset)
    
    def _handle_dataset_upload(self, uploaded_file):
        """
        Handle dataset upload.
        
        Args:
            uploaded_file: Uploaded file object
        """
        try:
            if self.on_dataset_upload:
                self.on_dataset_upload(uploaded_file)
            else:
                self.show_success(f"Dataset uploaded: {uploaded_file.name}")
        except Exception as e:
            self.handle_error(e, "uploading dataset")
    
    def _handle_dataset_deletion(self, dataset_name: str):
        """
        Handle dataset deletion.
        
        Args:
            dataset_name (str): Name of dataset to delete
        """
        try:
            if self.on_dataset_delete:
                self.on_dataset_delete(dataset_name)
            else:
                self.show_success(f"Dataset deleted: {dataset_name}")
        except Exception as e:
            self.handle_error(e, "deleting dataset")
    
    def _handle_dataset_preview(self, dataset_name: str):
        """
        Handle dataset preview.
        
        Args:
            dataset_name (str): Name of dataset to preview
        """
        try:
            if self.on_dataset_preview:
                preview_data = self.on_dataset_preview(dataset_name)
                self._render_dataset_preview(dataset_name, preview_data)
            else:
                st.info(f"Preview for dataset: {dataset_name}")
        except Exception as e:
            self.handle_error(e, "previewing dataset")
    
    def _render_dataset_preview(self, dataset_name: str, preview_data: Dict[str, Any]):
        """
        Render dataset preview.
        
        Args:
            dataset_name (str): Name of the dataset
            preview_data (Dict[str, Any]): Preview data
        """
        st.subheader(f"Dataset Preview: {dataset_name}")
        
        if 'dataframe' in preview_data:
            df = preview_data['dataframe']
            st.dataframe(df, width='stretch', hide_index=True)
            st.caption(f"Showing first 10 rows. Shape: {df.shape}")
            
            # Dataset information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Columns:** {len(df.columns)}")
                st.write(f"**Preview rows:** {len(df)}")
            
            with col2:
                st.write("**Data types:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
        
        if 'statistics' in preview_data:
            stats = preview_data['statistics']
            with st.expander("📊 Dataset Statistics", expanded=False):
                st.json(stats)
        
        if 'validation' in preview_data:
            validation = preview_data['validation']
            if validation:
                st.subheader("⚠️ Validation Warnings")
                for warning in validation:
                    st.warning(warning)
    
    def render_dataset_list(self, datasets: List[Dict[str, Any]]):
        """
        Render list of datasets with information.
        
        Args:
            datasets (List[Dict[str, Any]]): List of dataset information
        """
        st.subheader("Available Datasets")
        
        if not datasets:
            st.info("No datasets available")
            return
        
        for dataset in datasets:
            with st.expander(f"📁 {dataset['name']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**File Type:** {dataset['file_type']}")
                    st.write(f"**Size:** {dataset['size_bytes']} bytes")
                    st.write(f"**Rows:** {dataset['rows']}")
                
                with col2:
                    st.write(f"**Columns:** {dataset['features']}")
                    st.write(f"**Has Labels:** {dataset['has_labels']}")
                    if dataset['label_column']:
                        st.write(f"**Label Column:** {dataset['label_column']}")
    
    def render_dataset_upload_success(self, filename: str):
        """
        Render dataset upload success message.
        
        Args:
            filename (str): Name of uploaded file
        """
        self.show_success(f"✅ Dataset saved as: {filename}")
        st.rerun()
    
    def render_dataset_deletion_success(self, dataset_name: str):
        """
        Render dataset deletion success message.
        
        Args:
            dataset_name (str): Name of deleted dataset
        """
        self.show_success(f"✅ Deleted: {dataset_name}")
        st.rerun()
    
    def render_dataset_upload_error(self, error: str):
        """
        Render dataset upload error message.
        
        Args:
            error (str): Error message
        """
        self.show_error(f"❌ Error saving dataset: {error}")
    
    def render_dataset_deletion_error(self, error: str):
        """
        Render dataset deletion error message.
        
        Args:
            error (str): Error message
        """
        self.show_error(f"❌ Error deleting dataset: {error}")
    
    def render_dataset_preview_error(self, error: str):
        """
        Render dataset preview error message.
        
        Args:
            error (str): Error message
        """
        self.show_error(f"❌ Error previewing dataset: {error}")
    
    def render_dataset_validation_warnings(self, warnings: List[str]):
        """
        Render dataset validation warnings.
        
        Args:
            warnings (List[str]): List of validation warnings
        """
        if warnings:
            st.subheader("⚠️ Dataset Validation Warnings")
            for warning in warnings:
                st.warning(warning)
    
    def render_dataset_compatibility_check(self, dataset_name: str, 
                                         compatibility_issues: List[str]):
        """
        Render dataset compatibility check results.
        
        Args:
            dataset_name (str): Name of the dataset
            compatibility_issues (List[str]): List of compatibility issues
        """
        st.subheader(f"Compatibility Check: {dataset_name}")
        
        if not compatibility_issues:
            self.show_success("✅ Dataset is compatible with GE experiments")
        else:
            st.subheader("⚠️ Compatibility Issues")
            for issue in compatibility_issues:
                st.warning(issue)
    
    def render_dataset_statistics(self, dataset_name: str, statistics: Dict[str, Any]):
        """
        Render dataset statistics.
        
        Args:
            dataset_name (str): Name of the dataset
            statistics (Dict[str, Any]): Dataset statistics
        """
        st.subheader(f"Dataset Statistics: {dataset_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Info:**")
            st.write(f"- Shape: {statistics['shape']}")
            st.write(f"- Columns: {len(statistics['columns'])}")
            st.write(f"- Memory Usage: {statistics['memory_usage']} bytes")
        
        with col2:
            st.write("**Data Types:**")
            for col, dtype in statistics['dtypes'].items():
                st.write(f"- {col}: {dtype}")
        
        if statistics['missing_values']:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(list(statistics['missing_values'].items()), 
                                    columns=['Column', 'Missing Count'])
            st.dataframe(missing_df, hide_index=True)