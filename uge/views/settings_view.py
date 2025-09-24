"""
Settings View for UGE Application

This module provides the settings view for managing application configuration.

Classes:
- SettingsView: View for application settings management

Author: UGE Team
"""

import streamlit as st
from typing import Dict, Any, Optional
from uge.views.components.base_view import BaseView
from uge.services.settings_service import SettingsService


class SettingsView(BaseView):
    """
    View for application settings management.
    
    This view provides a user interface for viewing and modifying
    application settings stored in the settings.json file.
    """
    
    def __init__(self, settings_service: SettingsService):
        """
        Initialize settings view.
        
        Args:
            settings_service (SettingsService): Settings service instance
        """
        super().__init__(
            title="⚙️ Settings",
            description="Configure application settings and default values"
        )
        self.settings_service = settings_service
    
    def render(self):
        """Render the settings view."""
        self.render_header()
        
        # Settings sections
        settings_sections = [
            ("🎯 Default Values", self._render_defaults),
            ("🧬 GE Parameters", self._render_ge_parameters),
            ("🎨 UI Settings", self._render_ui_settings),
            ("📊 Report Items", self._render_report_items),
            ("🔧 Parameter Limits", self._render_parameter_limits),
            ("📝 Parameter Options", self._render_parameter_options)
        ]
        
        # Create tabs for different settings sections
        tab_names = [section[0] for section in settings_sections]
        tabs = st.tabs(tab_names)
        
        for i, (section_name, render_func) in enumerate(settings_sections):
            with tabs[i]:
                render_func()
        
        # Save/Reset buttons at the bottom
        self._render_action_buttons()
    
    def _render_defaults(self):
        """Render default values section."""
        st.subheader("🎯 Default Values")
        st.markdown("Configure default values used throughout the application.")
        
        defaults = self.settings_service.get_defaults()
        
        # Get available datasets and grammars
        import os
        from pathlib import Path
        
        project_root = Path(__file__).resolve().parent.parent.parent
        datasets_dir = project_root / "datasets"
        grammars_dir = project_root / "grammars"
        
        # Get available datasets
        available_datasets = []
        if datasets_dir.exists():
            for file in datasets_dir.iterdir():
                if file.is_file() and file.suffix in ['.csv', '.data', '.txt']:
                    available_datasets.append(file.name)
        
        # Get available grammars
        available_grammars = []
        if grammars_dir.exists():
            for file in grammars_dir.iterdir():
                if file.is_file() and file.suffix == '.bnf':
                    available_grammars.append(file.name)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset selection
            if available_datasets:
                current_dataset = defaults.get('dataset', 'processed.cleveland.data')
                dataset_index = available_datasets.index(current_dataset) if current_dataset in available_datasets else 0
                new_dataset = st.selectbox(
                    "Default Dataset",
                    options=available_datasets,
                    index=dataset_index,
                    help="Default dataset file to use"
                )
            else:
                new_dataset = st.text_input(
                    "Default Dataset",
                    value=defaults.get('dataset', 'processed.cleveland.data'),
                    help="Default dataset file to use"
                )
            
            # Grammar selection
            if available_grammars:
                current_grammar = defaults.get('grammar', 'heartDisease.bnf')
                grammar_index = available_grammars.index(current_grammar) if current_grammar in available_grammars else 0
                new_grammar = st.selectbox(
                    "Default Grammar",
                    options=available_grammars,
                    index=grammar_index,
                    help="Default BNF grammar file to use"
                )
            else:
                new_grammar = st.text_input(
                    "Default Grammar",
                    value=defaults.get('grammar', 'heartDisease.bnf'),
                    help="Default BNF grammar file to use"
                )
        
        with col2:
            # Fitness metric selection
            fitness_options = self.settings_service.get_parameter_options().get('fitness_metric', ['mae', 'accuracy'])
            current_fitness = defaults.get('fitness_metric', 'mae')
            fitness_index = fitness_options.index(current_fitness) if current_fitness in fitness_options else 0
            new_fitness_metric = st.selectbox(
                "Default Fitness Metric",
                options=fitness_options,
                index=fitness_index,
                help="Default fitness metric for optimization"
            )
            
            # Label column - keep as text input since it varies by dataset
            new_label_column = st.text_input(
                "Default Label Column",
                value=defaults.get('label_column', 'class'),
                help="Default label column name in datasets"
            )
        
        # Store changes in session state
        st.session_state['settings_defaults'] = {
            'dataset': new_dataset,
            'grammar': new_grammar,
            'fitness_metric': new_fitness_metric,
            'label_column': new_label_column
        }
    
    def _render_ge_parameters(self):
        """Render GE parameters section."""
        st.subheader("🧬 GE Parameters")
        st.markdown("Configure default Grammatical Evolution parameters.")
        
        ge_params = self.settings_service.get_ge_parameters()
        limits = self.settings_service.get_parameter_limits()
        
        # Create form for GE parameters
        with st.form("ge_parameters_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Basic Parameters**")
                new_population = st.number_input(
                    "Population Size",
                    min_value=limits.get('population', {}).get('min', 10),
                    max_value=limits.get('population', {}).get('max', 5000),
                    value=ge_params.get('population', 50),
                    step=10
                )
                
                new_generations = st.number_input(
                    "Generations",
                    min_value=limits.get('generations', {}).get('min', 1),
                    max_value=limits.get('generations', {}).get('max', 2000),
                    value=ge_params.get('generations', 20),
                    step=1
                )
                
                new_n_runs = st.number_input(
                    "Number of Runs",
                    min_value=limits.get('n_runs', {}).get('min', 1),
                    max_value=limits.get('n_runs', {}).get('max', 100),
                    value=ge_params.get('n_runs', 3),
                    step=1
                )
            
            with col2:
                st.markdown("**GA Parameters**")
                new_p_crossover = st.number_input(
                    "Crossover Probability",
                    min_value=limits.get('p_crossover', {}).get('min', 0.0),
                    max_value=limits.get('p_crossover', {}).get('max', 1.0),
                    value=ge_params.get('p_crossover', 0.8),
                    step=0.01,
                    format="%.2f"
                )
                
                new_p_mutation = st.number_input(
                    "Mutation Probability",
                    min_value=limits.get('p_mutation', {}).get('min', 0.0),
                    max_value=limits.get('p_mutation', {}).get('max', 1.0),
                    value=ge_params.get('p_mutation', 0.01),
                    step=0.001,
                    format="%.3f"
                )
                
                new_elite_size = st.number_input(
                    "Elite Size",
                    min_value=limits.get('elite_size', {}).get('min', 0),
                    max_value=limits.get('elite_size', {}).get('max', 50),
                    value=ge_params.get('elite_size', 1),
                    step=1
                )
            
            with col3:
                st.markdown("**Tree Parameters**")
                new_max_tree_depth = st.number_input(
                    "Max Tree Depth",
                    min_value=limits.get('max_tree_depth', {}).get('min', 5),
                    max_value=limits.get('max_tree_depth', {}).get('max', 100),
                    value=ge_params.get('max_tree_depth', 35),
                    step=1
                )
                
                new_codon_size = st.number_input(
                    "Codon Size",
                    min_value=limits.get('codon_size', {}).get('min', 100),
                    max_value=limits.get('codon_size', {}).get('max', 512),
                    value=ge_params.get('codon_size', 255),
                    step=1
                )
                
                new_test_size = st.number_input(
                    "Test Size",
                    min_value=limits.get('test_size', {}).get('min', 0.1),
                    max_value=limits.get('test_size', {}).get('max', 0.5),
                    value=ge_params.get('test_size', 0.3),
                    step=0.05,
                    format="%.2f"
                )
            
            # Submit button for the form
            submitted = st.form_submit_button("💾 Update GE Parameters", type="primary")
            
            if submitted:
                # Store changes in session state
                st.session_state['settings_ge_parameters'] = {
                    'population': new_population,
                    'generations': new_generations,
                    'n_runs': new_n_runs,
                    'p_crossover': new_p_crossover,
                    'p_mutation': new_p_mutation,
                    'elite_size': new_elite_size,
                    'max_tree_depth': new_max_tree_depth,
                    'codon_size': new_codon_size,
                    'test_size': new_test_size,
                    'tournsize': ge_params.get('tournsize', 7),
                    'halloffame_size': ge_params.get('halloffame_size', 1),
                    'min_init_tree_depth': ge_params.get('min_init_tree_depth', 4),
                    'max_init_tree_depth': ge_params.get('max_init_tree_depth', 13),
                    'min_init_genome_length': ge_params.get('min_init_genome_length', 95),
                    'max_init_genome_length': ge_params.get('max_init_genome_length', 115),
                    'codon_consumption': ge_params.get('codon_consumption', 'lazy'),
                    'genome_representation': ge_params.get('genome_representation', 'list'),
                    'initialisation': ge_params.get('initialisation', 'sensible'),
                    'random_seed': ge_params.get('random_seed', 42)
                }
                st.success("✅ GE Parameters updated! Click 'Save Settings' to persist changes.")
    
    def _render_ui_settings(self):
        """Render UI settings section."""
        st.subheader("🎨 UI Settings")
        st.markdown("Configure user interface settings.")
        
        ui_settings = self.settings_service.get_ui_constants()
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_chart_height = st.number_input(
                "Chart Height",
                min_value=200,
                max_value=1000,
                value=ui_settings.get('chart_height', 500),
                step=50,
                help="Default height for charts in pixels"
            )
            
            new_max_log_lines = st.number_input(
                "Max Log Lines",
                min_value=50,
                max_value=1000,
                value=ui_settings.get('max_log_lines', 200),
                step=25,
                help="Maximum number of log lines to display"
            )
        
        with col2:
            new_chart_template = st.selectbox(
                "Chart Template",
                options=['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'],
                index=0,
                help="Default chart template/theme"
            )
            
            new_max_upload_size = st.number_input(
                "Max Upload Size (MB)",
                min_value=10,
                max_value=1000,
                value=ui_settings.get('max_upload_size_mb', 200),
                step=10,
                help="Maximum file upload size in megabytes"
            )
        
        # Store changes in session state
        st.session_state['settings_ui'] = {
            'chart_height': new_chart_height,
            'chart_template': new_chart_template,
            'max_log_lines': new_max_log_lines,
            'max_upload_size_mb': new_max_upload_size,
            'allowed_dataset_extensions': ui_settings.get('allowed_dataset_extensions', ['.csv', '.data', '.txt'])
        }
    
    def _render_report_items(self):
        """Render report items section."""
        st.subheader("📊 Report Items")
        st.markdown("Configure which metrics to track during evolution.")
        
        current_items = self.settings_service.get_report_items()
        
        # Available report items
        available_items = [
            'gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
            'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes',
            'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons',
            'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std',
            'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std',
            'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time'
        ]
        
        new_report_items = st.multiselect(
            "Select Report Items",
            options=available_items,
            default=current_items,
            help="Choose which metrics to track during evolution"
        )
        
        # Store changes in session state
        st.session_state['settings_report_items'] = new_report_items
    
    def _render_parameter_limits(self):
        """Render parameter limits section."""
        st.subheader("🔧 Parameter Limits")
        st.markdown("Configure minimum and maximum values for parameters.")
        
        limits = self.settings_service.get_parameter_limits()
        
        st.info("Parameter limits are used to validate user input and set slider ranges in forms.")
        
        # Display current limits in a table format
        limit_data = []
        for param, limit_dict in limits.items():
            limit_data.append({
                'Parameter': param,
                'Min': limit_dict.get('min', 'N/A'),
                'Max': limit_dict.get('max', 'N/A')
            })
        
        if limit_data:
            st.dataframe(limit_data, use_container_width=True)
        
        st.warning("Parameter limits can only be modified by editing the settings.json file directly.")
    
    def _render_parameter_options(self):
        """Render parameter options section."""
        st.subheader("📝 Parameter Options")
        st.markdown("Configure available options for categorical parameters.")
        
        options = self.settings_service.get_parameter_options()
        
        st.info("Parameter options define the available choices for categorical parameters.")
        
        # Display current options
        for param, option_list in options.items():
            st.markdown(f"**{param}:** {', '.join(option_list)}")
        
        st.warning("Parameter options can only be modified by editing the settings.json file directly.")
    
    def _render_action_buttons(self):
        """Render save and reset buttons."""
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                self._save_settings()
        
        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                self._reset_settings()
        
        with col3:
            if st.button("📥 Export Settings", use_container_width=True):
                self._export_settings()
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            # Update settings with session state values
            if 'settings_defaults' in st.session_state:
                for key, value in st.session_state['settings_defaults'].items():
                    self.settings_service.set_setting(f'defaults.{key}', value)
            
            if 'settings_ge_parameters' in st.session_state:
                for key, value in st.session_state['settings_ge_parameters'].items():
                    self.settings_service.set_setting(f'ge_parameters.{key}', value)
            
            if 'settings_ui' in st.session_state:
                for key, value in st.session_state['settings_ui'].items():
                    self.settings_service.set_setting(f'ui.{key}', value)
            
            if 'settings_report_items' in st.session_state:
                self.settings_service.set_setting('report_items', st.session_state['settings_report_items'])
            
            # Save to file
            if self.settings_service.save_settings():
                st.success("✅ Settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings!")
                
        except Exception as e:
            st.error(f"❌ Error saving settings: {str(e)}")
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        if st.button("⚠️ Confirm Reset", type="secondary"):
            try:
                self.settings_service.reset_to_defaults()
                st.success("✅ Settings reset to defaults!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error resetting settings: {str(e)}")
    
    def _export_settings(self):
        """Export current settings as JSON."""
        try:
            settings_json = self.settings_service.get_all_settings()
            import json
            json_str = json.dumps(settings_json, indent=2)
            
            st.download_button(
                label="📥 Download Settings",
                data=json_str,
                file_name="uge_settings.json",
                mime="application/json",
                help="Download current settings as JSON file"
            )
        except Exception as e:
            st.error(f"❌ Error exporting settings: {str(e)}")
