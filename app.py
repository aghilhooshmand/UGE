"""
UGE - Grammatical Evolution Platform

Main Streamlit application using MVC architecture.
This is the entry point for the UGE platform.

Author: FORGE Team
"""

import streamlit as st
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Ensure imports resolve from UGE folder
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import MVC components
from uge.controllers.setup_controller import SetupController
from uge.controllers.analysis_controller import AnalysisController
from uge.views.home_view import HomeView
from uge.views.setup_view import SetupView
from uge.views.dataset_view import DatasetView
from uge.views.analysis_view import AnalysisView
from uge.views.grammar_view import GrammarView
from uge.views.setup_manager_view import SetupManagerView
from uge.views.comparison_view import ComparisonView
from uge.services.storage_service import StorageService
from uge.services.dataset_service import DatasetService
from uge.utils.constants import FILE_PATHS, HELP
from uge import __version__, BUILD_INFO


class UGEApp:
    """
    Main UGE Application class.
    
    This class orchestrates the entire application using the MVC pattern.
    It initializes controllers, views, and services, and manages the main
    application flow.
    """
    
    def __init__(self):
        """Initialize the UGE application."""
        self.setup_controller = SetupController(
            on_setup_start=self._on_setup_start,
            on_setup_progress=self._on_setup_progress,
            on_setup_complete=self._on_setup_complete,
            on_setup_error=self._on_setup_error
        )
        
        self.analysis_controller = AnalysisController(
            on_analysis_start=self._on_analysis_start,
            on_analysis_complete=self._on_analysis_complete,
            on_analysis_error=self._on_analysis_error
        )
        
        self.storage_service = StorageService()
        self.dataset_service = DatasetService()
        
        # Initialize views
        self.home_view = HomeView()
        self.setup_view = SetupView(self.setup_controller)
        # Initialize dataset view with preview callback
        self.dataset_view = DatasetView(
            on_dataset_preview=self._on_dataset_preview
        )
        self.analysis_view = AnalysisView(
            on_setup_select=self._on_setup_select,
            on_analysis_options_change=self._on_analysis_options_change,
            on_export_data=self._on_export_data
        )
        self.grammar_view = GrammarView()
        self.setup_manager_view = SetupManagerView(self.storage_service)
        self.comparison_view = ComparisonView(self.storage_service)
    
    def _on_setup_start(self, setup):
        """Callback when setup starts."""
        st.success(f"🚀 Setup '{setup.config.setup_name}' started!")
        st.session_state['current_setup'] = setup
        # Initialize persistent placeholders for live progress
        if 'uge_progress_bar' not in st.session_state:
            st.session_state['uge_progress_bar'] = st.progress(0)
        else:
            try:
                st.session_state['uge_progress_bar'].progress(0)
            except Exception:
                st.session_state['uge_progress_bar'] = st.progress(0)
        st.session_state['uge_progress_text'] = st.empty()
    
    def _on_setup_progress(self, setup, run_number, total_runs, progress):
        """Callback for setup progress updates."""
        # Update persistent progress bar and text
        try:
            if 'uge_progress_bar' in st.session_state and st.session_state['uge_progress_bar']:
                st.session_state['uge_progress_bar'].progress(min(max(progress, 0.0), 1.0))
            else:
                st.session_state['uge_progress_bar'] = st.progress(min(max(progress, 0.0), 1.0))
        except Exception:
            st.session_state['uge_progress_bar'] = st.progress(min(max(progress, 0.0), 1.0))
        if 'uge_progress_text' in st.session_state and st.session_state['uge_progress_text']:
            st.session_state['uge_progress_text'].info(
                f"🔄 Run {run_number}/{total_runs} in '{setup.config.setup_name}'"
            )
        else:
            st.info(f"🔄 Run {run_number}/{total_runs} in '{setup.config.setup_name}'")
    
    def _on_setup_complete(self, setup):
        """Callback when setup completes."""
        st.success(f"✅ Setup '{setup.config.setup_name}' completed successfully!")
        st.session_state['current_setup'] = None
        # Clear progress placeholders
        st.session_state['uge_progress_text'] = None
        st.session_state['uge_progress_bar'] = None
    
    def _on_setup_error(self, error):
        """Callback when setup errors."""
        st.error(f"❌ Setup failed: {str(error)}")
        st.session_state['current_setup'] = None
        # Clear progress placeholders
        st.session_state['uge_progress_text'] = None
        st.session_state['uge_progress_bar'] = None
    
    def _on_analysis_start(self, setup_id):
        """Callback when analysis starts."""
        # Try to get setup name
        try:
            setup = self.storage_service.load_setup(setup_id)
            exp_name = setup.config.setup_name if setup and setup.config else setup_id
        except:
            exp_name = setup_id
        st.info(f"🔍 Starting analysis for setup '{exp_name}'...")
    
    def _on_analysis_complete(self, setup_id, results):
        """Callback when analysis completes."""
        # Try to get setup name
        try:
            setup = self.storage_service.load_setup(setup_id)
            exp_name = setup.config.setup_name if setup and setup.config else setup_id
        except:
            exp_name = setup_id
        st.success(f"✅ Analysis completed for setup '{exp_name}'")
    
    def _on_analysis_error(self, error):
        """Callback when analysis errors."""
        st.error(f"❌ Analysis failed: {str(error)}")
    
    def _on_setup_select(self, setup_id: str):
        """Callback when setup is selected for analysis."""
        try:
            setup = self.storage_service.load_setup(setup_id)
            return setup
        except Exception as e:
            st.error(f"Error loading setup: {str(e)}")
            return None
    
    def _on_analysis_options_change(self, options):
        """Callback when analysis options change."""
        # Store analysis options in session state if needed
        st.session_state['analysis_options'] = options
    
    def _on_export_data(self, setup, export_type):
        """Callback for data export."""
        try:
            return self.analysis_controller.export_setup_data(setup.id, export_type)
        except Exception as e:
            st.error(f"Export error: {str(e)}")
            return None

    def _on_dataset_preview(self, dataset_name: str):
        """Callback to build dataset preview payload for the view."""
        try:
            df = self.dataset_service.get_dataset_preview(dataset_name)
            stats = self.dataset_service.get_dataset_statistics(dataset_name)
            validation = self.dataset_service.validate_dataset(dataset_name)
            payload = {}
            if df is not None:
                payload['dataframe'] = df
            if stats is not None:
                payload['statistics'] = stats
            if validation is not None:
                payload['validation'] = validation
            return payload
        except Exception as e:
            st.error(f"Error preparing dataset preview: {str(e)}")
            return {}
    
    def render_sidebar(self):
        """Render the application sidebar."""
        with st.sidebar:
            # Make GE title clickable to go to Home
            if st.button("🧬 GE", key="home_button", help="Click to go to Home page"):
                st.session_state.current_page = "🏠 Home"
                st.rerun()
            st.markdown("---")
            
            # Main navigation
            st.markdown("### 📋 Navigation")
            
            # Get current page from session state for default value
            current_page = st.session_state.get('current_page', "🏠 Home")
            
            page = st.selectbox(
                "Select Page:",
                ["🏠 Home", "🏃 Run Setup", "📊 Dataset Manager", "📝 Grammar Editor", 
                 "🧪 Setup Manager", "📈 Analysis", "⚖️ Comparison"],
                index=["🏠 Home", "🏃 Run Setup", "📊 Dataset Manager", "📝 Grammar Editor", 
                       "🧪 Setup Manager", "📈 Analysis", "⚖️ Comparison"].index(current_page) if current_page in ["🏠 Home", "🏃 Run Setup", "📊 Dataset Manager", "📝 Grammar Editor", "🧪 Setup Manager", "📈 Analysis", "⚖️ Comparison"] else 0,
                key="main_navigation",
                label_visibility="collapsed"
            )
            
            # Check if page has changed and trigger rerun
            if page != current_page:
                st.session_state.current_page = page
                st.rerun()
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            try:
                setups = self.storage_service.list_setups()
                datasets = self.dataset_service.list_datasets()
                
                # Count total runs
                total_runs = 0
                for exp in setups:
                    runs = self.storage_service.list_setup_runs(exp.name)
                    total_runs += len(runs)
                
                st.metric("🧪 Setups", len(setups))
                st.metric("📊 Datasets", len(datasets))
                st.metric("📝 Grammars", len(self._get_available_grammars()))
                st.metric("🏃 Total Runs", total_runs)
                
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
            
            st.markdown("---")
            
            # Additional info
            st.markdown("### ℹ️ About")
            st.markdown("**UGE Platform** - Grammatical Evolution for Machine Learning")
            st.markdown("Create, run, and analyze GE setups with ease.")
            
            st.markdown("---")
            st.caption("v1.0.0 | Built with Streamlit")
            
            return page
    
    def _get_available_grammars(self):
        """Get list of available grammars."""
        try:
            grammars_dir = FILE_PATHS['grammars_dir']
            if grammars_dir.exists():
                return [f.name for f in grammars_dir.glob("*.bnf")]
            return []
        except Exception:
            return []
    
    def render_page(self, page):
        """Render the selected page."""
        if page == "🏠 Home":
            self.home_view.render_home()
        elif page == "🏃 Run Setup":
            # Get required data for setup view
            from uge.utils.constants import HELP
            datasets = self.dataset_service.list_datasets()
            grammars = self._get_available_grammars()
            self.setup_view.render(
                help_texts=HELP,
                datasets=datasets,
                grammars=grammars
            )
        elif page == "📊 Dataset Manager":
            datasets = self.dataset_service.list_datasets()
            self.dataset_view.render(datasets=datasets)
        elif page == "📝 Grammar Editor":
            grammars = self._get_available_grammars()
            self.grammar_view.render_grammar_editor(grammars)
        elif page == "🧪 Setup Manager":
            self.setup_manager_view.render_setup_manager()
        elif page == "📈 Analysis":
            # Get available setups for analysis
            try:
                setup_paths = self.storage_service.list_setups()
                # Convert paths to setup info with better names
                setups = []
                for exp_path in setup_paths:
                    exp_id = exp_path.name
                    try:
                        # Load config to get setup name
                        config = self.storage_service.load_setup_config(exp_id)
                        if config:
                            setups.append({
                                'id': exp_id,
                                'name': config.setup_name,
                                'path': str(exp_path)
                            })
                        else:
                            # Fallback to exp_id if config can't be loaded
                            setups.append({
                                'id': exp_id,
                                'name': exp_id,
                                'path': str(exp_path)
                            })
                    except Exception:
                        # Fallback to exp_id if there's any error
                        setups.append({
                            'id': exp_id,
                            'name': exp_id,
                            'path': str(exp_path)
                        })
                
                self.analysis_view.render(setups)
            except Exception as e:
                st.error(f"Error loading setups: {str(e)}")
                self.analysis_view.render([])
        elif page == "⚖️ Comparison":
            self.comparison_view.render_comparison(self.analysis_controller)
        else:
            st.error(f"Unknown page: {page}")
    
    def run(self):
        """Run the UGE application."""
        # Page configuration
        st.set_page_config(
            page_title="UGE - Grammatical Evolution Platform",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render sidebar (handles navigation internally)
        self.render_sidebar()
        
        # Get current page from session state
        current_page = st.session_state.get('current_page', "🏃 Run Setup")
        
        # Render the selected page
        self.render_page(current_page)

def main():
    """Main entry point for the UGE application."""
    try:
        app = UGEApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()