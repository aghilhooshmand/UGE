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
from uge.views.setup_view import SetupView
from uge.views.dataset_view import DatasetView
from uge.views.analysis_view import AnalysisView
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
            st.title("🧬 GE")
            st.markdown("---")
            
            # Main navigation
            st.markdown("### 📋 Navigation")
            page = st.selectbox(
                "Select Page:",
                ["🏃 Run Setup", "📊 Dataset Manager", "📝 Grammar Editor", 
                 "🧪 Setup Manager", "📈 Analysis", "⚖️ Comparison"],
                key="main_navigation",
                label_visibility="collapsed"
            )
            
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
        if page == "🏃 Run Setup":
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
            self._render_grammar_editor()
        elif page == "🧪 Setup Manager":
            self._render_setup_manager()
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
            self._render_comparison()
        else:
            st.error(f"Unknown page: {page}")
    
    def _render_grammar_editor(self):
        """Render the grammar editor page."""
        st.header("📝 Grammar Editor")
        st.markdown("Edit and manage BNF grammar files")
        
        # Get available grammars
        grammars = self._get_available_grammars()
        
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
    
    def _render_setup_manager(self):
        """Render the setup manager page."""
        st.header("🧪 Setup Manager")
        st.markdown("Manage and monitor your setups")
        
        try:
            setups = self.storage_service.list_setups()
            
            if setups:
                # Setup selection
                exp_names = [exp.name for exp in setups]
                selected_exp = st.selectbox("Select Setup", exp_names)
                
                if selected_exp:
                    # Load setup details
                    setup = self.setup_controller.get_setup(selected_exp)
                    
                    if setup:
                        # Display setup info
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Setup Details")
                            st.write(f"**Name:** {setup.config.setup_name}")
                            st.write(f"**Status:** {setup.status}")
                            st.write(f"**Created:** {setup.created_at}")
                            st.write(f"**Total Runs:** {setup.config.n_runs}")
                            st.write(f"**Completed Runs:** {len(setup.results)}")
                        
                        with col2:
                            st.subheader("⚙️ Configuration")
                            st.write(f"**Dataset:** {setup.config.dataset}")
                            st.write(f"**Grammar:** {setup.config.grammar}")
                            st.write(f"**Population:** {setup.config.population}")
                            st.write(f"**Generations:** {setup.config.generations}")
                            st.write(f"**Fitness Metric:** {setup.config.fitness_metric}")
                        
                        # Progress bar
                        if setup.config.n_runs > 0:
                            progress = len(setup.results) / setup.config.n_runs
                            st.progress(progress)
                            st.write(f"Progress: {len(setup.results)}/{setup.config.n_runs} runs completed")
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("📊 View Results"):
                                st.session_state['selected_setup'] = selected_exp
                                st.rerun()
                        
                        with col2:
                            if st.button("📥 Export Data"):
                                export_data = self.analysis_controller.export_setup_data(
                                    selected_exp, 'all'
                                )
                                if export_data:
                                    st.download_button(
                                        label="📥 Download Setup Data",
                                        data=export_data,
                                        file_name=f"{selected_exp}_data.json",
                                        mime="application/json"
                                    )
                        
                        with col3:
                            if st.button("🗑️ Delete Setup", type="secondary"):
                                if self.setup_controller.delete_setup(selected_exp):
                                    st.success(f"Setup '{selected_exp}' deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete setup")
            else:
                st.info("No setups found. Create your first setup using the 'Run Setup' page!")
                
        except Exception as e:
            st.error(f"Error loading setups: {str(e)}")
    
    def _render_comparison(self):
        """Render the comparison page."""
        st.header("⚖️ Setup Comparison")
        st.markdown("Compare multiple setups")
        
        try:
            setup_paths = self.storage_service.list_setups()
            
            if len(setup_paths) >= 2:
                # Load setup objects and create options
                exp_options = {}
                for exp_path in setup_paths:
                    exp_id = exp_path.name
                    try:
                        setup = self.storage_service.load_setup(exp_id)
                        if setup and setup.config:
                            exp_name = setup.config.setup_name
                        else:
                            exp_name = exp_id
                    except:
                        exp_name = exp_id
                    exp_options[exp_name] = exp_id
                
                selected_setup_names = st.multiselect(
                    "Select Setups to Compare",
                    list(exp_options.keys()),
                    default=list(exp_options.keys())[:2] if len(exp_options) >= 2 else list(exp_options.keys())
                )
                
                # Convert selected names back to IDs for processing
                selected_setups = [exp_options[name] for name in selected_setup_names]
                
                if len(selected_setups) >= 2:
                    if st.button("🔍 Compare Setups"):
                        comparison_results = self.analysis_controller.compare_setups(selected_setups)
                        
                        if comparison_results:
                            # Store results in session state
                            st.session_state.comparison_results = comparison_results
                            st.session_state.selected_setups = selected_setups
                            st.session_state.selected_setup_names = selected_setup_names
                    
                    # Check if we have stored comparison results
                    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
                        comparison_results = st.session_state.comparison_results
                        selected_setups = st.session_state.selected_setups
                        
                        if comparison_results:
                            st.subheader("📊 Comparison Results")
                            
                            # Add clear button
                            if st.button("🗑️ Clear Comparison Results"):
                                if 'comparison_results' in st.session_state:
                                    del st.session_state.comparison_results
                                if 'selected_setups' in st.session_state:
                                    del st.session_state.selected_setups
                                if 'selected_setup_names' in st.session_state:
                                    del st.session_state.selected_setup_names
                                st.rerun()
                            
                            # Display comparison metrics
                            if 'comparison_metrics' in comparison_results:
                                metrics = comparison_results['comparison_metrics']
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Best Overall Fitness", 
                                             f"{metrics.get('best_overall_fitness', 'N/A'):.4f}" 
                                             if metrics.get('best_overall_fitness') else "N/A")
                                
                                with col2:
                                    # Get setup name for best setup
                                    best_exp_id = metrics.get('best_setup', 'N/A')
                                    if best_exp_id != 'N/A':
                                        exp_configs = comparison_results.get('setup_configs', {})
                                        best_exp_config = exp_configs.get(best_exp_id, {})
                                        best_exp_name = best_exp_config.get('setup_name', best_exp_id)
                                    else:
                                        best_exp_name = 'N/A'
                                    
                                    st.metric("Best Setup", best_exp_name)
                                
                                with col3:
                                    st.metric("Setups Compared", 
                                             len(selected_setups))
                            
                            # Display rankings
                            if 'rankings' in comparison_results:
                                st.subheader("🏆 Rankings")
                                
                                # Create mapping from setup IDs to names
                                exp_configs = comparison_results.get('setup_configs', {})
                                id_to_name = {}
                                for exp_id, config in exp_configs.items():
                                    id_to_name[exp_id] = config.get('setup_name', exp_id)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**By Best Fitness:**")
                                    for i, exp_id in enumerate(comparison_results['rankings'].get('by_best_fitness', []), 1):
                                        exp_name = id_to_name.get(exp_id, exp_id)
                                        st.write(f"{i}. {exp_name}")
                                
                                with col2:
                                    st.write("**By Average Fitness:**")
                                    for i, exp_id in enumerate(comparison_results['rankings'].get('by_average_fitness', []), 1):
                                        exp_name = id_to_name.get(exp_id, exp_id)
                                        st.write(f"{i}. {exp_name}")
                            
                            # Display aggregate comparison charts
                            if 'aggregate_data' in comparison_results and comparison_results['aggregate_data']:
                                st.subheader("📈 Aggregate Performance Charts")
                                
                                # Chart type selection
                                chart_type = st.selectbox(
                                    "Select Chart Type:",
                                    ["Best Fitness (Max)", "Average Fitness", "Test Fitness", "All Metrics"],
                                    help="Choose which fitness metric to display in the comparison chart"
                                )
                                
                                # Create comparison chart
                                self._render_comparison_chart(comparison_results, chart_type)
                            
                            # Export comparison data
                            st.subheader("📥 Export Options")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                comparison_json = json.dumps(comparison_results, indent=2, default=str)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=comparison_json,
                                    file_name="setup_comparison.json",
                                    mime="application/json"
                                )
                            
                            with col2:
                                csv_data = self._export_comparison_csv(comparison_results)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name="setup_comparison.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.error("Failed to compare setups")
                else:
                    st.warning("Please select at least 2 setups to compare")
            else:
                st.info("You need at least 2 setups to perform a comparison. Create more setups first!")
                
        except Exception as e:
            st.error(f"Error comparing setups: {str(e)}")
    
    def _render_comparison_chart(self, comparison_results: Dict[str, Any], chart_type: str):
        """
        Render interactive comparison chart based on aggregate setup data.
        
        This method creates Plotly charts that compare multiple setups across
        different fitness metrics. It handles different chart types and includes
        error bars to show variance across runs.
        
        Args:
            comparison_results (Dict[str, Any]): Results from setup comparison
            chart_type (str): Type of chart to render ('Best Fitness', 'Average Fitness', etc.)
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Extract aggregate data from comparison results
            # This contains average and standard deviation data for each setup
            aggregate_data = comparison_results.get('aggregate_data', {})
            if not aggregate_data:
                st.warning("No aggregate data available for charting")
                return
            
            # Initialize Plotly figure and color palette
            fig = go.Figure()
            colors = px.colors.qualitative.Set1  # Use consistent color scheme
            traces_added = 0  # Track if any data was actually plotted
            
            for i, (exp_id, data) in enumerate(aggregate_data.items()):
                if not data or 'generations' not in data:
                    st.warning(f"No data available for setup {exp_id}")
                    continue
                
                color = colors[i % len(colors)]
                # Get setup name from configs if available
                exp_configs = comparison_results.get('setup_configs', {})
                exp_config = exp_configs.get(exp_id, {})
                exp_name = exp_config.get('setup_name', exp_id)
                
                # Handle different chart types based on user selection
                if chart_type == "Best Fitness (Max)":
                    # Plot best fitness evolution with error bars showing standard deviation
                    if 'avg_max' in data and data['avg_max']:
                        fig.add_trace(go.Scatter(
                            x=data['generations'],
                            y=data['avg_max'],
                            mode='lines+markers',
                            name=f'{exp_name} - Best',
                            line=dict(color=color, width=3),
                            marker=dict(size=4),
                            error_y=dict(
                                type='data',
                                array=data.get('std_max', []),
                                visible=True,
                                color=color,
                                thickness=1
                            )
                        ))
                        traces_added += 1
                elif chart_type == "Average Fitness":
                    # Plot average fitness evolution across all individuals in population
                    if 'avg_avg' in data and data['avg_avg']:
                        fig.add_trace(go.Scatter(
                            x=data['generations'],
                            y=data['avg_avg'],
                            mode='lines+markers',
                            name=f'{exp_name} - Average',
                            line=dict(color=color, width=3),
                            marker=dict(size=4),
                            error_y=dict(
                                type='data',
                                array=data.get('std_avg', []),
                                visible=True,
                                color=color,
                                thickness=1
                            )
                        ))
                        traces_added += 1
                elif chart_type == "Test Fitness":
                    # Plot test fitness evolution (generalization performance)
                    if 'avg_test' in data and data['avg_test']:
                        # Find first generation with valid test data
                        test_values = data['avg_test']
                        first_valid_gen = None
                        for i, val in enumerate(test_values):
                            if val is not None and val != 0 and not np.isnan(val):
                                first_valid_gen = i
                                break
                        
                        if first_valid_gen is not None:
                            # Only plot test data from first valid generation onwards
                            valid_test_values = test_values[first_valid_gen:]
                            valid_test_gens = data['generations'][first_valid_gen:]
                            valid_std_test = data.get('std_test', [])[first_valid_gen:] if data.get('std_test') else []
                            
                            fig.add_trace(go.Scatter(
                                x=valid_test_gens,
                                y=valid_test_values,
                                mode='lines+markers',
                                name=f'{exp_name} - Test',
                                line=dict(color=color, width=3),
                                marker=dict(size=4),
                                error_y=dict(
                                    type='data',
                                    array=valid_std_test,
                                    visible=True,
                                    color=color,
                                    thickness=1
                                )
                            ))
                            traces_added += 1
                elif chart_type == "All Metrics":
                    # Plot all three fitness metrics (best, average, test) for comprehensive comparison
                    if 'avg_max' in data and data['avg_max']:
                        fig.add_trace(go.Scatter(
                            x=data['generations'],
                            y=data['avg_max'],
                            mode='lines+markers',
                            name=f'{exp_name} - Best',
                            line=dict(color=color, width=2),
                            marker=dict(size=3)
                        ))
                        traces_added += 1
                    if 'avg_avg' in data and data['avg_avg']:
                        fig.add_trace(go.Scatter(
                            x=data['generations'],
                            y=data['avg_avg'],
                            mode='lines+markers',
                            name=f'{exp_name} - Average',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=3)
                        ))
                        traces_added += 1
                    if 'avg_test' in data and data['avg_test']:
                        # Find first generation with valid test data
                        test_values = data['avg_test']
                        first_valid_gen = None
                        for i, val in enumerate(test_values):
                            if val is not None and val != 0 and not np.isnan(val):
                                first_valid_gen = i
                                break
                        
                        if first_valid_gen is not None:
                            # Only plot test data from first valid generation onwards
                            valid_test_values = test_values[first_valid_gen:]
                            valid_test_gens = data['generations'][first_valid_gen:]
                            
                            fig.add_trace(go.Scatter(
                                x=valid_test_gens,
                                y=valid_test_values,
                                mode='lines+markers',
                                name=f'{exp_name} - Test',
                                line=dict(color=color, width=2, dash='dot'),
                                marker=dict(size=3)
                            ))
                            traces_added += 1
            
            # Check if any traces were added
            if traces_added == 0:
                st.warning(f"No data available for chart type: {chart_type}")
                return
            
            fig.update_layout(
                title=f"Setup Comparison - {chart_type}",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                hovermode='x unified',
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering comparison chart: {str(e)}")
            import traceback
            st.write("Full error details:")
            st.code(traceback.format_exc())
    
    def _export_comparison_csv(self, comparison_results: Dict[str, Any]) -> str:
        """
        Export comparison results as CSV format for spreadsheet analysis.
        
        This method converts the comparison results into a CSV format that can be
        easily imported into Excel or other spreadsheet applications for further
        analysis and reporting.
        
        Args:
            comparison_results (Dict[str, Any]): Results from setup comparison
            
        Returns:
            str: CSV data as string, or empty string if error occurs
        """
        try:
            import pandas as pd
            from io import StringIO
            
            # Create summary data for CSV export
            summary_data = []
            
            # Create mapping from setup IDs to human-readable names
            exp_configs = comparison_results.get('setup_configs', {})
            id_to_name = {}
            for exp_id, config in exp_configs.items():
                id_to_name[exp_id] = config.get('setup_name', exp_id)
            
            for exp_data in comparison_results.get('setups', []):
                exp_id = exp_data.get('setup_id', '')
                exp_name = id_to_name.get(exp_id, exp_id)
                
                summary_data.append({
                    'Setup Name': exp_name,
                    'Best Fitness': exp_data.get('best_fitness', 0),
                    'Average Fitness': exp_data.get('average_fitness', 0),
                    'Total Runs': exp_data.get('total_runs', 0),
                    'Completed Runs': exp_data.get('completed_runs', 0),
                    'Best Depth': exp_data.get('best_depth', 0),
                    'Best Genome Length': exp_data.get('best_genome_length', 0),
                    'Used Codons': exp_data.get('used_codons', 0)
                })
            
            df = pd.DataFrame(summary_data)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error exporting CSV: {str(e)}")
            return ""
    
    def run(self):
        """Run the UGE application."""
        # Page configuration
        st.set_page_config(
            page_title="Grammatical Evolution for Classification", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title with version
        st.title("🧬 Grammatical Evolution for Classification")
        st.markdown(f"**v{__version__}** - Learning GE for classification tasks with comprehensive analysis and comparison")
        
        # Version info in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**UGE v{__version__}**")
            st.markdown(f"*Released: {BUILD_INFO['release_date']}*")
            st.markdown("---")
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render the selected page
        self.render_page(page)


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