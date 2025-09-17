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
from uge.models.setup import SetupConfig
from uge.utils.constants import DEFAULT_CONFIG, UI_CONSTANTS


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
        
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Setup Info")
                setup_name = st.text_input(
                    "Setup Name", 
                    value=f"Setup_{dt.datetime.now().strftime('%Y%m%d_%H%M')}", 
                    help=help_texts.get('setup_name', "Unique name for this setup")
                )
                
                st.subheader("Dataset & Grammar")
                dataset = st.selectbox(
                    "Dataset", 
                    options=datasets, 
                    index=datasets.index(UI_CONSTANTS['default_dataset']) if UI_CONSTANTS['default_dataset'] in datasets else (0 if datasets else None), 
                    help=help_texts.get('dataset', "Select dataset for the setup")
                )
                grammar = st.selectbox(
                    "Grammar", 
                    options=grammars, 
                    index=grammars.index(UI_CONSTANTS['default_grammar']) if UI_CONSTANTS['default_grammar'] in grammars else (0 if grammars else None), 
                    help=help_texts.get('grammar', "Select BNF grammar for the setup")
                )
                
                st.subheader("GA Parameters")
                population = st.number_input(
                    "Population Size", 
                    min_value=10, max_value=5000, 
                    value=DEFAULT_CONFIG['population'], 
                    step=10, 
                    help=help_texts.get('population', "Number of individuals in population")
                )
                generations = st.number_input(
                    "Generations", 
                    min_value=1, max_value=2000, 
                    value=DEFAULT_CONFIG['generations'], 
                    step=1, 
                    help=help_texts.get('generations', "Number of generations to evolve")
                )
                n_runs = st.number_input(
                    "Number of Runs", 
                    min_value=1, max_value=100, 
                    value=DEFAULT_CONFIG['n_runs'], 
                    help=help_texts.get('n_runs', "Number of independent runs for this setup")
                )
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
            
            with col2:
                st.subheader("GA Parameters (continued)")
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
                
                st.subheader("GE/GRAPE Parameters")
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
            
            st.subheader("Additional Parameters")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Report Items")
                report_items = st.multiselect(
                    "Select report items", 
                    options=DEFAULT_CONFIG['default_report_items'], 
                    default=DEFAULT_CONFIG['default_report_items'], 
                    help=help_texts.get('report_items', "Items to include in setup reports")
                )
            
            with col4:
                st.subheader("Dataset Options")
                
                # Always show label column input (required field)
                label_column = st.text_input(
                    "Label Column", 
                    value=DEFAULT_CONFIG['label_column'], 
                    help=help_texts.get('label_column', "Name of the label column")
                )
                
                test_size = DEFAULT_CONFIG['test_size']
                
                test_size = st.slider(
                    "Test Size", 
                    0.1, 0.5, 
                    DEFAULT_CONFIG['test_size'], 
                    0.05, 
                    help=help_texts.get('test_size', "Test set size ratio")
                )
                
                random_seed = st.number_input(
                    "Random Seed", 
                    min_value=0, max_value=10_000, 
                    value=DEFAULT_CONFIG['random_seed'], 
                    step=1, 
                    help=help_texts.get('random_seed', "Random seed for reproducibility")
                )
                fitness_metric = st.selectbox(
                    "Fitness Metric", 
                    options=["mae", "accuracy"], 
                    index=0, 
                    help=help_texts.get('fitness_metric', "Choose which fitness to optimize. Both MAE and Accuracy: higher is better (maximization).")
                )
                st.session_state['fitness_metric'] = fitness_metric
                
                # Fitness direction selection
                if fitness_metric == "mae":
                    fitness_direction = 1   # maximize
                    direction_text = "Maximize (Higher is Better)"
                else:  # accuracy
                    fitness_direction = 1   # maximize
                    direction_text = "Maximize (Higher is Better)"
                
                st.info(f"**Fitness Direction:** {direction_text}")
                st.session_state['fitness_direction'] = fitness_direction
            
            run_setup = st.form_submit_button("🚀 Run Setup", type="primary")
            
            if run_setup:
                form_data = {
                    'setup_name': setup_name,
                    'dataset': dataset,
                    'grammar': grammar,
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
    def create_grammar_editor_form(grammars: List[str] = None) -> Tuple[str, str, str]:
        """
        Create grammar editor form.
        
        Args:
            grammars (List[str]): Available grammars
            
        Returns:
            Tuple[str, str, str]: (action, grammar_name, grammar_content)
        """
        if grammars is None:
            grammars = []
        
        action = st.radio(
            "Grammar Action:",
            ["➕ Add Grammar", "✏️ Edit Grammar", "👁️ Preview Grammar"],
            key="grammar_action"
        )
        
        if action == "➕ Add Grammar":
            st.subheader("Create New Grammar")
            new_grammar_name = st.text_input("Grammar Name", placeholder="my_grammar.bnf")
            new_grammar_content = st.text_area(
                "Grammar Content (BNF format)", 
                height=300, 
                placeholder="<start> ::= <expr>\n<expr> ::= ..."
            )
            
            if st.button("💾 Save Grammar"):
                return "add", new_grammar_name, new_grammar_content
        
        elif action == "✏️ Edit Grammar":
            st.subheader("Edit Existing Grammar")
            if grammars:
                selected_grammar = st.selectbox("Select Grammar", grammars)
                # This would need to be handled by the controller to load content
                edited_content = st.text_area("Edit Grammar Content", height=300)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Update Grammar"):
                        return "edit", selected_grammar, edited_content
                with col2:
                    if st.button("🗑️ Delete Grammar"):
                        return "delete", selected_grammar, ""
            else:
                st.info("No grammars available")
        
        elif action == "👁️ Preview Grammar":
            st.subheader("Grammar Preview")
            if grammars:
                preview_grammar = st.selectbox("Select Grammar to Preview", grammars)
                return "preview", preview_grammar, ""
            else:
                st.info("No grammars available")
        
        return "none", "", ""
    
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
            show_best_individual = st.checkbox("Show Best Individual", value=True)
        
        with col2:
            export_data = st.checkbox("Export Data", value=False)
        
        return {
            'show_charts': show_charts,
            'show_statistics': show_statistics,
            'show_best_individual': show_best_individual,
            'export_data': export_data
        }