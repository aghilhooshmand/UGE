"""
Forms Component for UGE Application

This module provides form components and input widgets for the UGE application.

Classes:
- Forms: Form components and input utilities

Author: UGE Team
"""

import streamlit as st
import datetime as dt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import with error handling to avoid circular imports
try:
    from uge.models.setup import SetupConfig
    from uge.utils.constants import DEFAULT_CONFIG, UI_CONSTANTS
except ImportError:
    # Fallback constants if import fails due to circular imports
    DEFAULT_CONFIG = {
        'population': 500,
        'generations': 200,
        'n_runs': 3,
        'p_crossover': 0.8,
        'p_mutation': 0.01,
        'elite_size': 1,
        'tournsize': 7,
        'halloffame_size': 1,
        'max_tree_depth': 35,
        'min_init_tree_depth': 4,
        'max_init_tree_depth': 13,
        'min_init_genome_length': 95,
        'max_init_genome_length': 115,
        'codon_size': 255,
        'codon_consumption': 'lazy',
        'genome_representation': 'list',
        'initialisation': 'sensible',
        'random_seed': 42,
        'test_size': 0.3,
        'fitness_metric': 'mae',
        'label_column': 'class',
        'default_report_items': [
            'gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
            'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes',
            'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons',
            'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std',
            'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std',
            'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time'
        ],
        'track_generation_configs': True,
        'generation_config_params': [
            'population', 'p_crossover', 'p_mutation', 'elite_size', 'tournsize', 
            'halloffame_size', 'max_tree_depth', 'codon_size', 'codon_consumption', 
            'genome_representation'
        ],
        'evolution_types': {
            'fixed': {
                'name': 'Fixed Evolution',
                'description': 'Use the same configuration for all generations (recommended for beginners)',
                'track_configs': True
            },
            'dynamic': {
                'name': 'Dynamic Evolution', 
                'description': 'Allow configuration to change per generation (advanced feature)',
                'track_configs': True
            }
        },
        'default_evolution_type': 'fixed'
    }
    UI_CONSTANTS = {
        'default_dataset': 'processed.cleveland.data',
        'default_grammar': 'heartDisease.bnf'
    }
    SetupConfig = None  # Will be handled gracefully in the code


class Forms:
    """
    Form components and input utilities for the UGE application.
    
    This class provides methods for creating various types of forms
    and input widgets used throughout the application.
    """
    
    @staticmethod
    def create_setup_form(help_texts: Dict[str, str] = None, 
                              datasets: List[str] = None, 
                              grammars: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Create setup configuration form.
        
        Args:
            help_texts (Dict[str, str]): Help texts for form fields
            datasets (List[str]): Available datasets
            grammars (List[str]): Available grammars
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (form_submitted, form_data)
        """
        if help_texts is None:
            help_texts = {}
        if datasets is None:
            datasets = []
        if grammars is None:
            grammars = []
        
        # Section 1: Fixed Setup Parameters (Outside Form)
        st.subheader("🔧 1. Setup Configuration")
        st.markdown("*These parameters are fixed throughout the entire setup and all runs.*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Setup Info**")
            setup_name = st.text_input(
                "Setup Name", 
                value=f"Setup_{dt.datetime.now().strftime('%Y%m%d_%H%M')}", 
                help=help_texts.get('setup_name', "Unique name for this setup")
            )
        
        with col2:
            st.markdown("**Dataset Selection**")
            dataset = st.selectbox(
                "Dataset", 
                options=datasets, 
                index=datasets.index(UI_CONSTANTS['default_dataset']) if UI_CONSTANTS['default_dataset'] in datasets else (0 if datasets else None), 
                help=help_texts.get('dataset', "Select dataset for the setup")
            )
        
        with col3:
            st.markdown("**Grammar Selection**")
            grammar = st.selectbox(
                "Grammar", 
                options=grammars, 
                index=grammars.index(UI_CONSTANTS['default_grammar']) if UI_CONSTANTS['default_grammar'] in grammars else (0 if grammars else None), 
                key="grammar_selectbox",  # Add key for reactivity
                help=help_texts.get('grammar', "Select BNF grammar for the setup")
            )
        
        # Second row of fixed parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Evolution Parameters**")
            population = st.number_input(
                "Population Size", 
                min_value=10, max_value=5000, 
                value=DEFAULT_CONFIG['population'], 
                step=10, 
                help=help_texts.get('population', "Number of individuals in population")
            )
        
        with col2:
            generations = st.number_input(
                "Generations", 
                min_value=1, max_value=2000, 
                value=DEFAULT_CONFIG['generations'], 
                step=1, 
                help=help_texts.get('generations', "Number of generations to evolve")
            )
        
        with col3:
            n_runs = st.number_input(
                "Number of Runs", 
                min_value=1, max_value=100, 
                value=DEFAULT_CONFIG['n_runs'], 
                help=help_texts.get('n_runs', "Number of independent runs for this setup")
            )
        
        # Third row of fixed parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Fitness Configuration**")
            fitness_metric = st.selectbox(
                "Fitness Metric", 
                options=["mae", "accuracy"], 
                index=0, 
                help=help_texts.get('fitness_metric', "Choose which fitness to optimize. MAE: lower is better (minimization). Accuracy: higher is better (maximization).")
            )
            st.session_state['fitness_metric'] = fitness_metric
            
            # Show fitness direction based on selected metric
            if fitness_metric == "mae":
                fitness_direction = -1   # minimize (lower is better)
                direction_text = "Minimize (Lower is Better)"
                st.info("🎯 **Fitness Direction:** Minimize (Lower is Better)")
            else:  # accuracy
                fitness_direction = 1    # maximize (higher is better)
                direction_text = "Maximize (Higher is Better)"
                st.info("🎯 **Fitness Direction:** Maximize (Higher is Better)")
            
            st.session_state['fitness_direction'] = fitness_direction
        
        with col2:
            st.markdown("**Randomization**")
            random_seed = st.number_input(
                "Random Seed", 
                min_value=0, max_value=10_000, 
                value=DEFAULT_CONFIG['random_seed'], 
                step=1, 
                help=help_texts.get('random_seed', "Random seed for reproducibility")
            )
        
        
        # Show grammar content for selected grammar
        if 'grammar_selectbox' in st.session_state:
            selected_grammar = st.session_state['grammar_selectbox']
            if selected_grammar:
                st.markdown("**📄 Grammar Content Preview**")
                
                # Show info message based on selected grammar
                if selected_grammar == "UGE_Classification.bnf":
                    st.info("🔧 **Dynamic Grammar**: This grammar will be automatically adapted to your dataset's column types.")
                elif selected_grammar == "heartDisease.bnf":
                    st.info("❤️ **Heart Disease Grammar**: Specific grammar for heart disease classification.")
                else:
                    st.info(f"📄 **Static Grammar**: Content from grammars/{selected_grammar}")

                # Load grammar content from file (without info message)
                grammar_content = Forms._load_grammar_content_raw(selected_grammar)

                # Display grammar in text box
                st.text_area(
                    "Grammar Content (BNF Format)",
                    value=grammar_content,
                    height=200,
                    disabled=True,  # Make it read-only
                    key=f"grammar_content_display_{selected_grammar}",  # Add key for reactivity
                    help="This shows the grammar content that will be used for the setup"
                )

                # Add download button for grammar
                st.download_button(
                    label="📥 Download Grammar",
                    data=grammar_content,
                    file_name=selected_grammar,
                    mime="text/plain",
                    help="Download the selected grammar file"
                )
        
        st.divider()
        
        # Section 1: Parameter Configuration System and Current Parameter Values (Two Column Layout)
        st.subheader("🎛️ Parameter Configuration System & Current Parameter Values")
        
        # Create two columns for better appearance
        config_col, values_col = st.columns([2, 1])
        
        with config_col:
            st.markdown("**⚙️ Configure Parameters**")
            st.markdown("*Configure each parameter to be Fixed (same value throughout evolution) or Dynamic (varies each generation).*")
            
            # Initialize session state for parameter configurations
            if 'parameter_configs' not in st.session_state:
                st.session_state.parameter_configs = {}
            
            # Define configurable parameters
            config_params = {
                # Genetic Algorithm Parameters
            'elite_size': {
                'name': 'Elite Size',
                'default': DEFAULT_CONFIG['elite_size'],
                'min': 0,
                'max': 50,
                'help': 'Number of elite individuals to preserve'
            },
            'p_crossover': {
                'name': 'Crossover Probability',
                'default': DEFAULT_CONFIG['p_crossover'],
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'help': 'Probability of crossover operation'
            },
            'p_mutation': {
                'name': 'Mutation Probability', 
                'default': DEFAULT_CONFIG['p_mutation'],
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'help': 'Probability of mutation operation'
            },
            'tournsize': {
                'name': 'Tournament Size',
                'default': DEFAULT_CONFIG['tournsize'],
                'min': 2,
                'max': 50,
                'help': 'Size of tournament for selection'
            },
            'halloffame_size': {
                'name': 'Hall of Fame Size',
                'default': DEFAULT_CONFIG['halloffame_size'],
                'min': 1,
                'max': 100,
                'help': 'Size of hall of fame'
            },
            
            # Tree Parameters
            'max_tree_depth': {
                'name': 'Max Tree Depth',
                'default': DEFAULT_CONFIG['max_tree_depth'],
                'min': 5,
                'max': 100,
                'help': 'Maximum depth of the evolved trees'
            },
            'min_init_tree_depth': {
                'name': 'Min Init Tree Depth',
                'default': DEFAULT_CONFIG['min_init_tree_depth'],
                'min': 1,
                'max': 20,
                'help': 'Minimum depth for initial trees'
            },
            'max_init_tree_depth': {
                'name': 'Max Init Tree Depth',
                'default': DEFAULT_CONFIG['max_init_tree_depth'],
                'min': 1,
                'max': 30,
                'help': 'Maximum depth for initial trees'
            },
            
            # Genome Parameters
            'min_init_genome_length': {
                'name': 'Min Init Genome Length',
                'default': DEFAULT_CONFIG['min_init_genome_length'],
                'min': 10,
                'max': 200,
                'help': 'Minimum length for initial genomes'
            },
            'max_init_genome_length': {
                'name': 'Max Init Genome Length',
                'default': DEFAULT_CONFIG['max_init_genome_length'],
                'min': 10,
                'max': 500,
                'help': 'Maximum length for initial genomes'
            },
            'codon_size': {
                'name': 'Codon Size',
                'default': DEFAULT_CONFIG['codon_size'],
                'min': 100,
                'max': 512,
                'help': 'Size of codons in the genome'
            },
            
            # Categorical Parameters
            'codon_consumption': {
                'name': 'Codon Consumption',
                'default': DEFAULT_CONFIG['codon_consumption'],
                'options': ['lazy', 'eager'],
                'help': 'How codons are consumed during tree generation'
            },
            'genome_representation': {
                'name': 'Genome Representation',
                'default': DEFAULT_CONFIG['genome_representation'],
                'options': ['list'],
                'help': 'How genomes are represented in memory'
            },
            'initialisation': {
                'name': 'Initialisation',
                'default': DEFAULT_CONFIG['initialisation'],
                'options': ['sensible', 'random'],
                'help': 'Method for generating initial trees'
            }
            }
            
            # Create configuration forms for each parameter
            parameter_configs = {}
            
            for param_key, param_info in config_params.items():
                with st.expander(f"⚙️ {param_info['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Mode selection
                        mode_key = f"{param_key}_mode"
                    if mode_key not in st.session_state:
                        st.session_state[mode_key] = "Fixed"
                    
                    mode = st.selectbox(
                        "Mode",
                        ["Fixed", "Dynamic"],
                        index=0 if st.session_state[mode_key] == "Fixed" else 1,
                        key=f"{param_key}_mode_select"
                    )
                    
                    # Update session state
                    if mode != st.session_state[mode_key]:
                        st.session_state[mode_key] = mode
                        st.rerun()
                
                with col2:
                    if mode == "Fixed":
                        # Fixed value input
                        if 'options' in param_info:
                            # Categorical parameter - use selectbox
                            default_index = param_info['options'].index(param_info['default']) if param_info['default'] in param_info['options'] else 0
                            value = st.selectbox(
                                f"{param_info['name']} (Fixed)",
                                options=param_info['options'],
                                index=default_index,
                                help=param_info['help'],
                                key=f"{param_key}_fixed"
                            )
                        elif 'step' in param_info:
                            value = st.number_input(
                                f"{param_info['name']} (Fixed)",
                                min_value=param_info['min'],
                                max_value=param_info['max'],
                                value=param_info['default'],
                                step=param_info['step'],
                                help=param_info['help'],
                                key=f"{param_key}_fixed"
                            )
                        else:
                            value = st.number_input(
                                f"{param_info['name']} (Fixed)",
                                min_value=param_info['min'],
                                max_value=param_info['max'],
                                value=param_info['default'],
                                help=param_info['help'],
                                key=f"{param_key}_fixed"
                            )
                        
                        parameter_configs[param_key] = {
                            'mode': 'fixed',
                            'value': value,
                            'options': param_info.get('options', None)
                        }
                        st.info(f"ℹ️ {param_info['name']} will remain constant at {value}")
                        
                    else:  # Dynamic
                        if 'options' in param_info:
                            # Categorical parameter - show available options
                            st.markdown(f"**Available Options:**")
                            for option in param_info['options']:
                                st.markdown(f"- {option}")
                            
                            parameter_configs[param_key] = {
                                'mode': 'dynamic',
                                'value': param_info['options'][0],  # Use first option as default
                                'options': param_info['options']
                            }
                            st.success(f"✅ {param_info['name']} will be randomly selected from: {', '.join(param_info['options'])}")
                        else:
                            # Numerical parameter - Dynamic range inputs
                            col_low, col_high = st.columns(2)
                            
                            with col_low:
                                if 'step' in param_info:
                                    low_value = st.number_input(
                                        f"{param_info['name']} (Low)",
                                        min_value=param_info['min'],
                                        max_value=param_info['max'],
                                        value=max(param_info['min'], param_info['default'] - param_info['step'] * 5),
                                        step=param_info['step'],
                                        help=f"Minimum {param_info['name'].lower()} for random variation",
                                        key=f"{param_key}_low"
                                    )
                                else:
                                    low_value = st.number_input(
                                        f"{param_info['name']} (Low)",
                                        min_value=param_info['min'],
                                        max_value=param_info['max'],
                                        value=max(param_info['min'], param_info['default'] - 1),
                                        help=f"Minimum {param_info['name'].lower()} for random variation",
                                        key=f"{param_key}_low"
                                    )
                            
                            with col_high:
                                if 'step' in param_info:
                                    high_value = st.number_input(
                                        f"{param_info['name']} (High)",
                                        min_value=param_info['min'],
                                        max_value=param_info['max'],
                                        value=min(param_info['max'], param_info['default'] + param_info['step'] * 5),
                                        step=param_info['step'],
                                        help=f"Maximum {param_info['name'].lower()} for random variation",
                                        key=f"{param_key}_high"
                                    )
                                else:
                                    high_value = st.number_input(
                                        f"{param_info['name']} (High)",
                                        min_value=param_info['min'],
                                        max_value=param_info['max'],
                                        value=min(param_info['max'], param_info['default'] + 2),
                                        help=f"Maximum {param_info['name'].lower()} for random variation",
                                        key=f"{param_key}_high"
                                    )
                            
                            # Ensure low <= high
                            if low_value > high_value:
                                st.warning(f"⚠️ Low must be ≤ High. Adjusting values.")
                                low_value = high_value
                            
                            parameter_configs[param_key] = {
                                'mode': 'dynamic',
                                'value': low_value,  # Use low as default
                                'low': low_value,
                                'high': high_value,
                                'options': None
                            }
                            st.success(f"✅ {param_info['name']} will vary randomly between {low_value} and {high_value}")
        
        with values_col:
            st.markdown("**📊 Current Parameter Values**")
            st.markdown("*These are the current parameter values based on your configuration. Dynamic parameters will vary during evolution.*")
            
            # Get values and modes from parameter configurations
            def get_param_display(param_key, param_name, default_value):
                param_config = parameter_configs.get(param_key, {})
                value = param_config.get('value', default_value)
                mode = param_config.get('mode', 'fixed')
                
                if mode == 'dynamic':
                    if 'options' in param_config:
                        # Categorical dynamic
                        return f"**{param_name}:** {value} 🔄 *Dynamic* (from {param_config['options']})"
                    else:
                        # Numerical dynamic
                        return f"**{param_name}:** {value} 🔄 *Dynamic* ({param_config.get('low', 'N/A')}-{param_config.get('high', 'N/A')})"
                else:
                    return f"**{param_name}:** {value} 🔒 *Fixed*"
            
            # GA Parameters
            st.markdown("**🧬 Genetic Algorithm Parameters:**")
            st.markdown(get_param_display('elite_size', 'Elite Size', DEFAULT_CONFIG['elite_size']))
            st.markdown(get_param_display('p_crossover', 'Crossover Probability', DEFAULT_CONFIG['p_crossover']))
            st.markdown(get_param_display('p_mutation', 'Mutation Probability', DEFAULT_CONFIG['p_mutation']))
            st.markdown(get_param_display('tournsize', 'Tournament Size', DEFAULT_CONFIG['tournsize']))
            st.markdown(get_param_display('halloffame_size', 'Hall of Fame Size', DEFAULT_CONFIG['halloffame_size']))
            
            # Tree Parameters
            st.markdown("**🌳 Tree Parameters:**")
            st.markdown(get_param_display('max_tree_depth', 'Max Tree Depth', DEFAULT_CONFIG['max_tree_depth']))
            st.markdown(get_param_display('min_init_tree_depth', 'Min Init Tree Depth', DEFAULT_CONFIG['min_init_tree_depth']))
            st.markdown(get_param_display('max_init_tree_depth', 'Max Init Tree Depth', DEFAULT_CONFIG['max_init_tree_depth']))
            
            # Genome Parameters
            st.markdown("**🧬 Genome Parameters:**")
            st.markdown(get_param_display('min_init_genome_length', 'Min Init Genome Length', DEFAULT_CONFIG['min_init_genome_length']))
            st.markdown(get_param_display('max_init_genome_length', 'Max Init Genome Length', DEFAULT_CONFIG['max_init_genome_length']))
            st.markdown(get_param_display('codon_size', 'Codon Size', DEFAULT_CONFIG['codon_size']))
            
            # Categorical Parameters
            st.markdown("**📝 Categorical Parameters:**")
            st.markdown(get_param_display('codon_consumption', 'Codon Consumption', DEFAULT_CONFIG['codon_consumption']))
            st.markdown(get_param_display('genome_representation', 'Genome Representation', DEFAULT_CONFIG['genome_representation']))
            st.markdown(get_param_display('initialisation', 'Initialisation', DEFAULT_CONFIG['initialisation']))
            
            # Dynamic Parameters Summary
            st.markdown("**📊 Dynamic Parameters Summary**")
            
            # Dynamic parameters indicator
            dynamic_count = sum(1 for config in parameter_configs.values() if config.get('mode') == 'dynamic')
            if dynamic_count > 0:
                st.success(f"🔄 **{dynamic_count} parameters** are set to dynamic mode!")
                
                # Show which parameters are dynamic
                st.markdown("**Dynamic Parameters:**")
                for param_key, param_config in parameter_configs.items():
                    if param_config.get('mode') == 'dynamic':
                        param_name = param_config.get('name', param_key)
                        if 'options' in param_config:
                            st.markdown(f"- {param_name}: {param_config['options']}")
                        else:
                            st.markdown(f"- {param_name}: {param_config.get('low', 'N/A')}-{param_config.get('high', 'N/A')}")
            else:
                st.info("🔒 **All parameters** are set to fixed mode.")
            
            # Store parameter configurations for form submission
            parameter_configs_data = parameter_configs
        
        st.divider()
        
        # Section 2: Dataset Configuration
        st.subheader("📊 2. Dataset Configuration")
        with st.form("dataset_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Label column selection
                label_column = st.text_input(
                    "Label Column", 
                    value=DEFAULT_CONFIG['label_column'], 
                    help=help_texts.get('label_column', "Name of the target column in the dataset"),
                    key="label_column"
                )
            
            with col2:
                # Test size
                test_size = st.slider(
                    "Test Size", 
                    min_value=0.1, max_value=0.5, 
                    value=DEFAULT_CONFIG['test_size'], 
                    step=0.05, 
                    help=help_texts.get('test_size', "Proportion of data for testing"),
                    key="test_size"
                )
            
            # Submit button for dataset configuration
            st.form_submit_button("✅ Confirm Dataset Configuration", type="secondary")
        
        st.divider()
        
        # Section 3: Report Items Configuration
        st.subheader("📝 3. Report Items Configuration")
        with st.form("report_config_form"):
            # Report items
            report_items = st.multiselect(
                "Select report items", 
                options=DEFAULT_CONFIG['default_report_items'], 
                default=DEFAULT_CONFIG['default_report_items'], 
                help=help_texts.get('report_items', "Items to include in setup reports"),
                key="report_items"
            )
            
            # Submit button for report configuration
            st.form_submit_button("✅ Confirm Report Items", type="secondary")
        
        st.divider()
        
        # Section 5: Execute Setup
        st.subheader("🚀 5. Execute Setup")
        
        # Run Setup button outside of any form
        run_setup = st.button("🚀 Run Setup", type="primary", use_container_width=True)
        
        # Handle form submission when button is clicked
        if run_setup:
            # Get grammar from session state since it's outside the form
            selected_grammar = st.session_state.get('grammar_selectbox', UI_CONSTANTS['default_grammar'])
            
            # Get dataset configuration from session state
            label_column = st.session_state.get('label_column', DEFAULT_CONFIG['label_column'])
            test_size = st.session_state.get('test_size', DEFAULT_CONFIG['test_size'])
            report_items = st.session_state.get('report_items', DEFAULT_CONFIG['default_report_items'])
            
            # Get current parameter values from configurations
            elite_size = parameter_configs_data.get('elite_size', {}).get('value', DEFAULT_CONFIG['elite_size'])
            p_crossover = parameter_configs_data.get('p_crossover', {}).get('value', DEFAULT_CONFIG['p_crossover'])
            p_mutation = parameter_configs_data.get('p_mutation', {}).get('value', DEFAULT_CONFIG['p_mutation'])
            tournsize = parameter_configs_data.get('tournsize', {}).get('value', DEFAULT_CONFIG['tournsize'])
            halloffame_size = parameter_configs_data.get('halloffame_size', {}).get('value', DEFAULT_CONFIG['halloffame_size'])
            max_tree_depth = parameter_configs_data.get('max_tree_depth', {}).get('value', DEFAULT_CONFIG['max_tree_depth'])
            min_init_tree_depth = parameter_configs_data.get('min_init_tree_depth', {}).get('value', DEFAULT_CONFIG['min_init_tree_depth'])
            max_init_tree_depth = parameter_configs_data.get('max_init_tree_depth', {}).get('value', DEFAULT_CONFIG['max_init_tree_depth'])
            min_init_genome_length = parameter_configs_data.get('min_init_genome_length', {}).get('value', DEFAULT_CONFIG['min_init_genome_length'])
            max_init_genome_length = parameter_configs_data.get('max_init_genome_length', {}).get('value', DEFAULT_CONFIG['max_init_genome_length'])
            codon_size = parameter_configs_data.get('codon_size', {}).get('value', DEFAULT_CONFIG['codon_size'])
            codon_consumption = parameter_configs_data.get('codon_consumption', {}).get('value', DEFAULT_CONFIG['codon_consumption'])
            genome_representation = parameter_configs_data.get('genome_representation', {}).get('value', DEFAULT_CONFIG['genome_representation'])
            initialisation = parameter_configs_data.get('initialisation', {}).get('value', DEFAULT_CONFIG['initialisation'])
            
            form_data = {
                'setup_name': setup_name,
                'dataset': dataset,
                'grammar': selected_grammar,
                'fitness_metric': fitness_metric,
                'fitness_direction': fitness_direction,
                'n_runs': int(n_runs),
                'generations': int(generations),
                'population': int(population),
                'p_crossover': float(p_crossover),
                'p_mutation': float(p_mutation),
                'elite_size': int(elite_size),
                'tournsize': int(tournsize),
                'halloffame_size': int(halloffame_size),
                'parameter_configs': parameter_configs_data,  # Parameter configuration system
                'max_tree_depth': int(max_tree_depth),
                'min_init_tree_depth': int(min_init_tree_depth),
                'max_init_tree_depth': int(max_init_tree_depth),
                'min_init_genome_length': int(min_init_genome_length),
                'max_init_genome_length': int(max_init_genome_length),
                'codon_size': int(codon_size),
                'codon_consumption': codon_consumption,
                'genome_representation': genome_representation,
                'initialisation': initialisation,
                'random_seed': int(random_seed),
                'label_column': label_column,
                'test_size': float(test_size),
                'report_items': report_items
            }
            return True, form_data
        
        return False, {}
    
    @staticmethod
    def create_dataset_upload_form() -> Tuple[bool, Optional[Any]]:
        """
        Create dataset upload form.
        
        Returns:
            Tuple[bool, Optional[Any]]: (form_submitted, uploaded_file)
        """
        st.subheader("Upload New Dataset")
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'data', 'txt'],
            help="Upload CSV, .data, or .txt files"
        )
        
        if uploaded_file is not None:
            filename = uploaded_file.name
            if not filename.endswith(('.csv', '.data', '.txt')):
                filename += '.csv'
            
            st.info(f"📁 File: {filename} ({uploaded_file.size} bytes)")
            
            if st.button("💾 Save Dataset"):
                return True, uploaded_file
        
        return False, None
    
    @staticmethod
    def create_dataset_management_form(datasets: List[str]) -> Tuple[str, str]:
        """
        Create dataset management form.
        
        Args:
            datasets (List[str]): Available datasets
            
        Returns:
            Tuple[str, str]: (action, selected_dataset)
        """
        st.subheader("Manage Existing Datasets")
        
        if not datasets:
            st.info("No datasets available")
            return "none", ""
        
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Delete Dataset"):
                return "delete", selected_dataset
        
        with col2:
            if st.button("👁️ Preview Dataset"):
                return "preview", selected_dataset
        
        return "none", selected_dataset
    
    @staticmethod
    def _load_grammar_content_raw(grammar_name: str) -> str:
        """
        Load grammar content from file without showing info messages.
        
        Args:
            grammar_name (str): Name of the grammar file
            
        Returns:
            str: Grammar content or error message
        """
        try:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            grammar_path = project_root / "grammars" / grammar_name
            
            if grammar_path.exists():
                content = grammar_path.read_text(encoding='utf-8')
                return content
            else:
                error_msg = f"# Error: Grammar file not found\n# Path: {grammar_path}\n# Please check if the file exists in the grammars directory."
                return error_msg
                
        except Exception as e:
            error_msg = f"# Error loading grammar file\n# Grammar: {grammar_name}\n# Error: {str(e)}"
            return error_msg
    
    @staticmethod
    def create_setup_selection_form(setups: List[Dict[str, str]]) -> Optional[str]:
        """
        Create setup selection form.
        
        Args:
            setups (List[Dict[str, str]]): Available setups with id, name, path
            
        Returns:
            Optional[str]: Selected setup ID
        """
        if not setups:
            st.info("No setups available")
            return None
        
        # Create options for selectbox (show name, return id)
        setup_options = {exp['name']: exp['id'] for exp in setups}
        
        selected_setup_name = st.selectbox(
            "Select Setup",
            list(setup_options.keys()),
            help="Choose an setup to analyze"
        )
        
        if selected_setup_name:
            return setup_options[selected_setup_name]
        
        return None
    
    @staticmethod
    def create_analysis_options_form() -> Dict[str, Any]:
        """
        Create analysis options form.
        
        Returns:
            Dict[str, Any]: Analysis options
        """
        st.subheader("Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_charts = st.checkbox("Show Charts", value=True)
            show_statistics = st.checkbox("Show Statistics", value=True)
            show_config_charts = st.checkbox("Show Configuration Charts", value=True, 
                                           help="Show charts for configuration parameter evolution across generations")
            show_best_individual = st.checkbox("Show Best Individual", value=True)
        
        with col2:
            export_data = st.checkbox("Export Data", value=False)
        
        return {
            'show_charts': show_charts,
            'show_statistics': show_statistics,
            'show_config_charts': show_config_charts,
            'show_best_individual': show_best_individual,
            'export_data': export_data
        }