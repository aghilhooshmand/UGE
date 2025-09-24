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
            
            # Second row of parameters
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.markdown("**Selection Parameters**")
                new_tournsize = st.number_input(
                    "Tournament Size",
                    min_value=limits.get('tournsize', {}).get('min', 2),
                    max_value=limits.get('tournsize', {}).get('max', 50),
                    value=ge_params.get('tournsize', 7),
                    step=1
                )
                
                new_halloffame_size = st.number_input(
                    "Hall of Fame Size",
                    min_value=limits.get('halloffame_size', {}).get('min', 1),
                    max_value=limits.get('halloffame_size', {}).get('max', 100),
                    value=ge_params.get('halloffame_size', 1),
                    step=1
                )
                
                new_random_seed = st.number_input(
                    "Random Seed",
                    min_value=limits.get('random_seed', {}).get('min', 0),
                    max_value=limits.get('random_seed', {}).get('max', 10000),
                    value=ge_params.get('random_seed', 42),
                    step=1
                )
            
            with col5:
                st.markdown("**Initialization Parameters**")
                new_min_init_tree_depth = st.number_input(
                    "Min Init Tree Depth",
                    min_value=limits.get('min_init_tree_depth', {}).get('min', 1),
                    max_value=limits.get('min_init_tree_depth', {}).get('max', 20),
                    value=ge_params.get('min_init_tree_depth', 4),
                    step=1
                )
                
                new_max_init_tree_depth = st.number_input(
                    "Max Init Tree Depth",
                    min_value=limits.get('max_init_tree_depth', {}).get('min', 1),
                    max_value=limits.get('max_init_tree_depth', {}).get('max', 30),
                    value=ge_params.get('max_init_tree_depth', 13),
                    step=1
                )
                
                new_min_init_genome_length = st.number_input(
                    "Min Init Genome Length",
                    min_value=limits.get('min_init_genome_length', {}).get('min', 10),
                    max_value=limits.get('min_init_genome_length', {}).get('max', 200),
                    value=ge_params.get('min_init_genome_length', 95),
                    step=1
                )
            
            with col6:
                st.markdown("**Genome Parameters**")
                new_max_init_genome_length = st.number_input(
                    "Max Init Genome Length",
                    min_value=limits.get('max_init_genome_length', {}).get('min', 10),
                    max_value=limits.get('max_init_genome_length', {}).get('max', 500),
                    value=ge_params.get('max_init_genome_length', 115),
                    step=1
                )
                
                # Categorical parameters
                codon_consumption_options = self.settings_service.get_parameter_options().get('codon_consumption', ['lazy', 'eager'])
                current_codon_consumption = ge_params.get('codon_consumption', 'lazy')
                codon_consumption_index = codon_consumption_options.index(current_codon_consumption) if current_codon_consumption in codon_consumption_options else 0
                new_codon_consumption = st.selectbox(
                    "Codon Consumption",
                    options=codon_consumption_options,
                    index=codon_consumption_index
                )
                
                genome_representation_options = self.settings_service.get_parameter_options().get('genome_representation', ['list'])
                current_genome_representation = ge_params.get('genome_representation', 'list')
                genome_representation_index = genome_representation_options.index(current_genome_representation) if current_genome_representation in genome_representation_options else 0
                new_genome_representation = st.selectbox(
                    "Genome Representation",
                    options=genome_representation_options,
                    index=genome_representation_index
                )
                
                initialisation_options = self.settings_service.get_parameter_options().get('initialisation', ['sensible', 'random'])
                current_initialisation = ge_params.get('initialisation', 'sensible')
                initialisation_index = initialisation_options.index(current_initialisation) if current_initialisation in initialisation_options else 0
                new_initialisation = st.selectbox(
                    "Initialisation",
                    options=initialisation_options,
                    index=initialisation_index
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
                    'tournsize': new_tournsize,
                    'halloffame_size': new_halloffame_size,
                    'min_init_tree_depth': new_min_init_tree_depth,
                    'max_init_tree_depth': new_max_init_tree_depth,
                    'min_init_genome_length': new_min_init_genome_length,
                    'max_init_genome_length': new_max_init_genome_length,
                    'codon_consumption': new_codon_consumption,
                    'genome_representation': new_genome_representation,
                    'initialisation': new_initialisation,
                    'random_seed': new_random_seed
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
        
        # Create editable form for parameter limits
        with st.form("parameter_limits_form"):
            st.markdown("**Edit Parameter Limits**")
            
            # Create columns for better layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("**Parameter**")
            with col2:
                st.markdown("**Min Value**")
            with col3:
                st.markdown("**Max Value**")
            
            # Store edited limits in session state
            if 'settings_parameter_limits' not in st.session_state:
                st.session_state['settings_parameter_limits'] = limits.copy()
            
            edited_limits = {}
            
            # Create input fields for each parameter
            for param, limit_dict in limits.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(param)
                
                with col2:
                    current_min = st.session_state['settings_parameter_limits'].get(param, {}).get('min', limit_dict.get('min', 0))
                    new_min = st.number_input(
                        f"min_{param}",
                        value=current_min,
                        step=0.1 if isinstance(current_min, float) else 1,
                        key=f"min_{param}",
                        label_visibility="collapsed"
                    )
                
                with col3:
                    current_max = st.session_state['settings_parameter_limits'].get(param, {}).get('max', limit_dict.get('max', 100))
                    new_max = st.number_input(
                        f"max_{param}",
                        value=current_max,
                        step=0.1 if isinstance(current_max, float) else 1,
                        key=f"max_{param}",
                        label_visibility="collapsed"
                    )
                
                edited_limits[param] = {'min': new_min, 'max': new_max}
            
            # Submit button for the form
            submitted = st.form_submit_button("💾 Update Parameter Limits", type="primary")
            
            if submitted:
                # Store changes in session state
                st.session_state['settings_parameter_limits'] = edited_limits
                st.success("✅ Parameter limits updated! Click 'Save Settings' to persist changes.")
        
        # Display current limits in a table format for reference
        st.markdown("**Current Parameter Limits**")
        limit_data = []
        current_limits = st.session_state.get('settings_parameter_limits', limits)
        for param, limit_dict in current_limits.items():
            limit_data.append({
                'Parameter': param,
                'Min': limit_dict.get('min', 'N/A'),
                'Max': limit_dict.get('max', 'N/A')
            })
        
        if limit_data:
            st.dataframe(limit_data, use_container_width=True)
    
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
            
            if 'settings_parameter_limits' in st.session_state:
                self.settings_service.set_setting('parameter_limits', st.session_state['settings_parameter_limits'])
            
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
