"""
Constants and Configuration for UGE Application

This module contains application-wide constants, default configurations,
and file paths used throughout the UGE application.

Constants:
- FILE_PATHS: Common file and directory paths
- DEFAULT_CONFIG: Default experiment configuration values
- UI_CONSTANTS: User interface related constants

Author: UGE Team
"""

import json
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load help texts from unified configuration
HELP = {}
help_path = PROJECT_ROOT / "uge" / "utils" / "tooltip_config.json"
if help_path.exists():
    try:
        config = json.loads(help_path.read_text())
        # Extract experiment parameters for backward compatibility
        HELP = config.get('experiment_parameters', {})
    except Exception:
        HELP = {}

# File and Directory Paths
FILE_PATHS = {
    'project_root': PROJECT_ROOT,
    'results_dir': PROJECT_ROOT / "results",
    'experiments_dir': PROJECT_ROOT / "results" / "experiments",
    'datasets_dir': PROJECT_ROOT / "datasets",
    'grammars_dir': PROJECT_ROOT / "grammars",
    'tooltip_config': PROJECT_ROOT / "uge" / "utils" / "tooltip_config.json",
}

# Default Experiment Configuration
DEFAULT_CONFIG = {
    # GA Parameters
    'population': 50,
    'generations': 20,
    'n_runs': 3,
    'p_crossover': 0.8,
    'p_mutation': 0.01,
    'elite_size': 1,
    'tournsize': 7,
    'halloffame_size': 1,
    
    # GE/GRAPE Parameters
    'max_tree_depth': 35,
    'min_init_tree_depth': 4,
    'max_init_tree_depth': 13,
    'min_init_genome_length': 95,
    'max_init_genome_length': 115,
    'codon_size': 255,
    'codon_consumption': 'lazy',
    'genome_representation': 'list',
    'initialisation': 'sensible',
    
    # Dataset Parameters
    'test_size': 0.3,
    'random_seed': 42,
    'fitness_metric': 'mae',
    'label_column': 'class',
    # Report Items
    'default_report_items': [
        'gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
        'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes',
        'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons',
        'invalid_count_min', 'invalid_count_avg', 'invalid_count_max',
        'nodes_length_min', 'nodes_length_avg', 'nodes_length_max',
        'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time'
    ]
}

# User Interface Constants
UI_CONSTANTS = {
    'page_title': "Grammatical Evolution",
    'app_title': "🧬 Grammatical Evolution",
    'app_subtitle': "** Learning GE with comprehensive analysis and comparison**",
    
    # Navigation Options
    'navigation_pages': [
        "🏃 Run Experiment",
        "📊 Dataset Manager", 
        "📝 Grammar Editor",
        "🧪 Experiment Manager",
        "📈 Analysis",
        "⚖️ Comparison"
    ],
    
    # Default Dataset and Grammar
    'default_dataset': 'processed.cleveland.data',
    'default_grammar': 'heartDisease.bnf',
    
    # Chart Configuration
    'chart_height': 500,
    'chart_template': 'plotly_white',
    'max_log_lines': 200,
    
    # File Upload
    'allowed_dataset_extensions': ['.csv', '.data', '.txt'],
    'max_upload_size_mb': 200,
}