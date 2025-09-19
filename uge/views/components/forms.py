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
            st.markdown("**Evolution Type**")
            evolution_type_options = list(DEFAULT_CONFIG['evolution_types'].keys())
            evolution_type_index = evolution_type_options.index(DEFAULT_CONFIG['default_evolution_type'])
            
            evolution_type = st.selectbox(
                "Evolution Type", 
                options=evolution_type_options, 
                index=evolution_type_index,
                format_func=lambda x: DEFAULT_CONFIG['evolution_types'][x]['name'],
                help=help_texts.get('evolution_type', "Choose evolution type: Fixed (same config all generations) or Dynamic (config can change per generation)")
            )
            
            # Show evolution type description
            evolution_info = DEFAULT_CONFIG['evolution_types'][evolution_type]
            st.info(f"**{evolution_info['name']}**: {evolution_info['description']}")
        
        with col3:
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
        
        # Section 2: Advanced Parameters (Inside Form)
        with st.form("setup_form"):
            st.subheader("⚙️ 2. Advanced Parameters")
            st.markdown("*These parameters control the detailed behavior of the evolutionary algorithm.*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # GE Parameters
                st.markdown("**Genetic Algorithm Parameters**")
                p_crossover = st.slider(
                    "Crossover Probability", 
                    min_value=0.0, max_value=1.0, 
                    value=DEFAULT_CONFIG['p_crossover'], 
                    step=0.01, 
                    help=help_texts.get('p_crossover', "Probability of crossover operation")
                )
                p_mutation = st.slider(
                    "Mutation Probability", 
                    min_value=0.0, max_value=1.0, 
                    value=DEFAULT_CONFIG['p_mutation'], 
                    step=0.01, 
                    help=help_texts.get('p_mutation', "Probability of mutation operation")
                )
                elite_size = st.number_input(
                    "Elite Size", 
                    min_value=0, max_value=50, 
                    value=DEFAULT_CONFIG['elite_size'], 
                    help=help_texts.get('elite_size', "Number of elite individuals to preserve")
                )
                tournsize = st.number_input(
                    "Tournament Size", 
                    min_value=2, max_value=50, 
                    value=DEFAULT_CONFIG['tournsize'], 
                    help=help_texts.get('tournsize', "Size of tournament for selection")
                )
                halloffame_size = st.number_input(
                    "Hall of Fame Size", 
                    min_value=1, max_value=100, 
                    value=max(1, int(elite_size)), 
                    help=help_texts.get('halloffame_size', "Size of hall of fame")
                )
            
            with col2:
                # Tree Parameters
                st.markdown("**Tree Parameters**")
                max_tree_depth = st.number_input(
                    "Max Tree Depth", 
                    min_value=1, max_value=100, 
                    value=DEFAULT_CONFIG['max_tree_depth'], 
                    help=help_texts.get('max_tree_depth', "Maximum tree depth")
                )
                min_init_tree_depth = st.number_input(
                    "Min Init Tree Depth", 
                    min_value=1, max_value=50, 
                    value=DEFAULT_CONFIG['min_init_tree_depth'], 
                    help=help_texts.get('min_init_tree_depth', "Minimum initial tree depth")
                )
                max_init_tree_depth = st.number_input(
                    "Max Init Tree Depth", 
                    min_value=1, max_value=50, 
                    value=DEFAULT_CONFIG['max_init_tree_depth'], 
                    help=help_texts.get('max_init_tree_depth', "Maximum initial tree depth")
                )
                
                # Genome Parameters
                st.markdown("**Genome Parameters**")
                min_init_genome_length = st.number_input(
                    "Min Init Genome Length", 
                    min_value=1, max_value=5000, 
                    value=DEFAULT_CONFIG['min_init_genome_length'], 
                    help=help_texts.get('min_init_genome_length', "Minimum initial genome length")
                )
                max_init_genome_length = st.number_input(
                    "Max Init Genome Length", 
                    min_value=1, max_value=5000, 
                    value=DEFAULT_CONFIG['max_init_genome_length'], 
                    help=help_texts.get('max_init_genome_length', "Maximum initial genome length")
                )
                codon_size = st.number_input(
                    "Codon Size", 
                    min_value=2, max_value=65535, 
                    value=DEFAULT_CONFIG['codon_size'], 
                    help=help_texts.get('codon_size', "Codon size for genome representation")
                )
                codon_consumption = st.selectbox(
                    "Codon Consumption", 
                    options=["lazy", "eager"], 
                    index=0, 
                    help=help_texts.get('codon_consumption', "Codon consumption strategy")
                )
                genome_representation = st.selectbox(
                    "Genome Representation", 
                    options=["list"], 
                    index=0, 
                    help=help_texts.get('genome_representation', "Genome representation type")
                )
                initialisation = st.selectbox(
                    "Initialisation", 
                    options=["sensible", "random"], 
                    index=0, 
                    help=help_texts.get('initialisation', "Population initialization strategy")
                )
            
            # Additional Parameters Row
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**Report Items**")
                report_items = st.multiselect(
                    "Select report items", 
                    options=DEFAULT_CONFIG['default_report_items'], 
                    default=DEFAULT_CONFIG['default_report_items'], 
                    help=help_texts.get('report_items', "Items to include in setup reports")
                )
            
            with col4:
                st.markdown("**Dataset Options**")
                
                # Always show label column input (required field)
                label_column = st.text_input(
                    "Label Column", 
                    value=DEFAULT_CONFIG['label_column'], 
                    help=help_texts.get('label_column', "Name of the label column")
                )
                
                test_size = st.slider(
                    "Test Size", 
                    0.1, 0.5, 
                    DEFAULT_CONFIG['test_size'], 
                    0.05, 
                    help=help_texts.get('test_size', "Test set size ratio")
                )
            
            # Submit button inside form
            run_setup = st.form_submit_button("🚀 Run Setup", type="primary", use_container_width=True)
        
        st.divider()
        
        # Section 3: Run Setup Status
        st.subheader("🚀 3. Execute Setup")
        
        # Handle form submission when button is clicked
        if run_setup:
            # Get grammar from session state since it's outside the form
            selected_grammar = st.session_state.get('grammar_selectbox', UI_CONSTANTS['default_grammar'])
            form_data = {
                'setup_name': setup_name,
                'dataset': dataset,
                'grammar': selected_grammar,
                'fitness_metric': fitness_metric,
                'fitness_direction': fitness_direction,
                'evolution_type': evolution_type,
                'n_runs': int(n_runs),
                'generations': int(generations),
                'population': int(population),
                'p_crossover': float(p_crossover),
                'p_mutation': float(p_mutation),
                'elite_size': int(elite_size),
                'tournsize': int(tournsize),
                'halloffame_size': int(halloffame_size),
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