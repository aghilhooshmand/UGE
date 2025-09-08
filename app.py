import os
import json
import time
import uuid
import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random

import sys
from pathlib import Path
from io import StringIO
import contextlib

# Ensure imports resolve from UGE folder
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Load help texts
HELP = {}
help_path = CURRENT_DIR / 'config_help.json'
if help_path.exists():
    try:
        HELP = json.loads(help_path.read_text())
    except Exception:
        HELP = {}

import grape  # existing framework
import algorithms  # existing framework
from deap import creator, base, tools
from functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal

from sklearn.model_selection import train_test_split

RESULTS_DIR = CURRENT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Experiment Management ----------

def create_experiment_id():
    """Create unique experiment ID"""
    return f"exp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def create_run_id():
    """Create unique run ID"""
    return f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def save_experiment_config(exp_id: str, config: dict):
    """Save experiment configuration"""
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(exist_ok=True)
    
    config_file = exp_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return exp_dir

def save_run_result(exp_id: str, run_id: str, result: dict):
    """Save run result within experiment"""
    exp_dir = EXPERIMENTS_DIR / exp_id
    run_dir = exp_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save result data
    result_file = run_dir / "result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save CSV log
    if 'logbook' in result:
        csv_file = run_dir / "logbook.csv"
        log_df = pd.DataFrame(result['logbook'])
        log_df.to_csv(csv_file, index=False)
    
    return run_dir

def list_experiments():
    """List all experiments"""
    if not EXPERIMENTS_DIR.exists():
        return []
    return [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir() and p.name.startswith('exp_')]

def load_experiment_config(exp_id: str):
    """Load experiment configuration"""
    config_file = EXPERIMENTS_DIR / exp_id / "experiment_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def list_experiment_runs(exp_id: str):
    """List all runs for an experiment"""
    runs_dir = EXPERIMENTS_DIR / exp_id / "runs"
    if not runs_dir.exists():
        return []
    return [p for p in runs_dir.iterdir() if p.is_dir()]

def load_run_result(exp_id: str, run_id: str):
    """Load run result"""
    result_file = EXPERIMENTS_DIR / exp_id / "runs" / run_id / "result.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

# ---------- Helpers ----------

def list_datasets():
    ds_dir = CURRENT_DIR / "datasets"
    return [p.name for p in ds_dir.glob('*') if p.is_file() and p.suffix in {'.data', '.csv', ''}]

def list_grammars():
    g_dir = CURRENT_DIR / "grammars"
    return [p.name for p in g_dir.glob('*.bnf')]

# Data loaders aligned with example_classification.py expectations

def load_dataset(name: str, random_seed: int):
    np.random.seed(random_seed)
    ds_dir = CURRENT_DIR / "datasets"

    if name == "processed.cleveland.data":
        data = pd.read_csv(ds_dir / name, sep=",")
        data = data[data.ca != '?']
        data = data[data.thal != '?']
        Y = data['class'].to_numpy()
        for i in range(len(Y)):
            Y[i] = 1 if Y[i] > 0 else 0
        data = data.drop(['class'], axis=1)
        cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data.loc[:, cols] = (data.loc[:, cols] - data.loc[:, cols].mean())/data.loc[:, cols].std()
        data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_seed)
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        return X_train, Y_train, X_test, Y_test

    # Generic CSV path
    df = pd.read_csv(ds_dir / name)
    return df, None, None, None


def mae(y, yhat):
    # Calculate actual Mean Absolute Error
    return np.mean(np.abs(np.array(y) - np.array(yhat)))

# Add near other fitness helpers

def accuracy(y, yhat):
    compare = np.equal(y, yhat)
    return np.mean(compare)

# Modify fitness_eval to switch by metric

def fitness_eval(individual, points, metric='mae'):
    x = points[0]
    Y = points[1]
    if individual.invalid:
        return np.nan,
    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError):
        return np.nan,
    if not np.isrealobj(pred):
        return np.nan,
    try:
        Y_class = [1 if pred[i] > 0 else 0 for i in range(len(Y))]
    except (IndexError, TypeError):
        return np.nan,
    
    if metric == 'mae':
        fitness_val = mae(Y, Y_class)
    elif metric == 'accuracy':
        fitness_val = accuracy(Y, Y_class)
    else:
        fitness_val = mae(Y, Y_class)
    
    return fitness_val,


class StreamlitLogger(StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.buffered = ""
    def write(self, s):
        self.buffered += s
        # Trim long logs for performance
        show = '\n'.join(self.buffered.splitlines()[-200:])
        self.placeholder.code(show)
        return len(s)
    def flush(self):
        pass


def run_ge_classification(config: dict, report_items, live_placeholder=None):
    random_seed = config.get('random_seed', 42)
    np.random.seed(random_seed)
    import random as pyrand
    pyrand.seed(random_seed)

    # Data
    dataset_name = config['dataset']
    grammar_name = config['grammar']

    if dataset_name == 'processed.cleveland.data':
        X_train, Y_train, X_test, Y_test = load_dataset(dataset_name, random_seed)
    else:
        df: pd.DataFrame = load_dataset(dataset_name, random_seed)[0]
        label_col = config.get('label_column')
        if not label_col:
            raise ValueError('label_column not provided for CSV dataset')
        df = df.dropna()
        Y = df[label_col].astype(int).to_numpy()
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=config.get('test_size', 0.3), random_state=random_seed)
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)

    # Grammar
    BNF_GRAMMAR = grape.Grammar(str(CURRENT_DIR / "grammars" / grammar_name))

    # Toolbox
    toolbox = base.Toolbox()
    # Avoid multiple re-creation errors
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    # Choose fitness type based on metric
    current_metric = st.session_state.get('fitness_metric', 'mae')
    fitness_class = creator.FitnessMax if current_metric == 'accuracy' else creator.FitnessMin
    
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', grape.Individual, fitness=fitness_class)

    if config.get('initialisation', 'sensible') == 'random':
        toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
    else:
        toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    # Create a fitness function wrapper that captures the current metric
    def fitness_wrapper(individual, points):
        # Get the current metric from session state at evaluation time
        current_metric = st.session_state.get('fitness_metric', 'mae')
        return fitness_eval(individual, points, current_metric)
    
    toolbox.register("evaluate", fitness_wrapper)
    toolbox.register("select", tools.selTournament, tournsize=config.get('tournsize', 7))
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    # Params
    POPULATION_SIZE = config.get('population', 300)
    ELITE_SIZE = config.get('elite_size', 1)
    HALLOFFAME_SIZE = config.get('halloffame_size', max(1, ELITE_SIZE))
    MAX_INIT_TREE_DEPTH = config.get('max_init_tree_depth', 13)
    MIN_INIT_TREE_DEPTH = config.get('min_init_tree_depth', 4)
    MAX_GENERATIONS = config.get('generations', 100)
    P_CROSSOVER = config.get('p_crossover', 0.8)
    P_MUTATION = config.get('p_mutation', 0.01)
    MIN_INIT_GENOME_LENGTH = config.get('min_init_genome_length', 95)
    MAX_INIT_GENOME_LENGTH = config.get('max_init_genome_length', 115)
    CODON_CONSUMPTION = config.get('codon_consumption', 'lazy')
    GENOME_REPRESENTATION = config.get('genome_representation', 'list')
    MAX_GENOME_LENGTH = config.get('max_genome_length', None)
    MAX_TREE_DEPTH = config.get('max_tree_depth', 35)
    CODON_SIZE = config.get('codon_size', 255)
    


    # Population
    if config.get('initialisation', 'sensible') == 'random':
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                               bnf_grammar=BNF_GRAMMAR,
                                               min_init_genome_length=MIN_INIT_GENOME_LENGTH,
                                               max_init_genome_length=MAX_INIT_GENOME_LENGTH,
                                               max_init_depth=MAX_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION)
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                               bnf_grammar=BNF_GRAMMAR,
                                               min_init_depth=MIN_INIT_TREE_DEPTH,
                                               max_init_depth=MAX_INIT_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION)

    # HOF and stats
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # Live log capture
    logger = StreamlitLogger(live_placeholder) if live_placeholder is not None else None
    
    with (contextlib.redirect_stdout(logger) if logger else contextlib.nullcontext()):
        population, logbook = algorithms.ge_eaSimpleWithElitism(
            population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
            ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
            bnf_grammar=BNF_GRAMMAR,
            codon_size=CODON_SIZE,
            max_tree_depth=MAX_TREE_DEPTH,
            max_genome_length=MAX_GENOME_LENGTH,
            points_train=[X_train, Y_train],
            points_test=[X_test, Y_test],
            codon_consumption=CODON_CONSUMPTION,
            report_items=report_items,
            genome_representation=GENOME_REPRESENTATION,
            stats=stats, halloffame=hof, verbose=False)

    # Prepare outputs conditionally
    available = set(logbook.header)
    series = {}
    for key in ['max','avg','min','std','fitness_test']:
        if key in available:
            series[key] = logbook.select(key)
        else:
            series[key] = []

    result = {
        'config': config,
        'report_items': list(report_items),
        'max': list(map(float, series.get('max', []))) if series.get('max') else [],
        'avg': list(map(float, series.get('avg', []))) if series.get('avg') else [],
        'min': list(map(float, series.get('min', []))) if series.get('min') else [],
        'std': list(map(float, series.get('std', []))) if series.get('std') else [],
        'fitness_test': [float(x) if x == x else None for x in series.get('fitness_test', [])] if series.get('fitness_test') else [],
        'best_phenotype': hof.items[0].phenotype if hof.items else None,
        'best_training_fitness': float(hof.items[0].fitness.values[0]) if hof.items else None,
        'best_depth': int(hof.items[0].depth) if hof.items else None,
        'best_genome_length': len(hof.items[0].genome) if hof.items else None,
        'best_used_codons': float(hof.items[0].used_codons)/len(hof.items[0].genome) if hof.items else None,
        'timestamp': dt.datetime.now(dt.timezone.utc).isoformat()
    }

    return result


def save_run(result: dict):
    run_id = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    rows = zip(range(len(result.get('max', []))), result.get('avg', []), result.get('std', []), result.get('min', []), result.get('max', []), result.get('fitness_test', []))
    with open(out_dir / 'log.csv', 'w') as f:
        f.write('\t'.join(['gen','avg','std','min','max','fitness_test']) + '\n')
        for i, a, s, mn, mx, ft in rows:
            f.write('\t'.join(map(lambda v: '' if v is None else str(v), [i, a, s, mn, mx, ft])) + '\n')
    # Register under experiment if provided
    exp_name = (result.get('config') or {}).get('experiment')
    if exp_name:
        _register_run_to_experiment(exp_name, run_id, result.get('config'))
    return str(out_dir)


def plot_result(result: dict, title="Fitness Evolution"):
    gens = list(range(len(result.get('max', []))))
    
    # Get fitness metric for proper labeling
    fitness_metric = st.session_state.get('fitness_metric', 'mae')
    ylabel = 'Fitness (higher is better)' if fitness_metric == 'accuracy' else 'Fitness (lower is better)'
    
    # Create Plotly figure
    fig = go.Figure()
    
    if result.get('max'):
        fig.add_trace(go.Scatter(
            x=gens, y=result['max'],
            mode='lines',
            name='Best (train)',
            line=dict(color='red', width=2)
        ))
    
    if result.get('avg'):
        fig.add_trace(go.Scatter(
            x=gens, y=result['avg'],
            mode='lines',
            name='Average (train)',
            line=dict(color='blue', width=2)
        ))
    
    if result.get('min'):
        fig.add_trace(go.Scatter(
            x=gens, y=result['min'],
            mode='lines',
            name='Min (train)',
            line=dict(color='green', width=2)
        ))
    
    if result.get('fitness_test') and any(v is not None for v in result['fitness_test']):
        test_values = [v if v is not None else np.nan for v in result['fitness_test']]
        fig.add_trace(go.Scatter(
            x=gens, y=test_values,
            mode='lines',
            name='Test',
            line=dict(color='magenta', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Generation',
        yaxis_title=ylabel,
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Grammatical Evolution", layout="wide")
st.title("🧬 Grammatical Evolution ")
st.markdown("** Learning GE with comprehensive analysis and comparison**")

# Sidebar navigation (like real software)
with st.sidebar:
    st.title("🧬 GE")
    st.markdown("---")
    
    # Main navigation with better styling
    st.markdown("### 📋 Navigation")
    page = st.selectbox(
        "Select Page:",
        ["🏃 Run Experiment", "📊 Dataset Manager", "📝 Grammar Editor", "🧪 Experiment Manager", "📈 Analysis", "⚖️ Comparison"],
        key="main_navigation",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats in sidebar
    st.markdown("### 📊 Quick Stats")
    experiments = list_experiments()
    datasets = list_datasets()
    grammars = list_grammars()
    
    # Count total runs
    total_runs = 0
    for exp in experiments:
        runs = list_experiment_runs(exp.name)
        total_runs += len(runs)
    
    # Display stats in a cleaner way
    st.metric("🧪 Experiments", len(experiments))
    st.metric("📊 Datasets", len(datasets))
    st.metric("📝 Grammars", len(grammars))
    st.metric("🏃 Total Runs", total_runs)
    
    st.markdown("---")
    
    # Additional sidebar info
    st.markdown("### ℹ️ About")
    st.markdown("**UGE Platform** - Grammatical Evolution for Machine Learning")
    st.markdown("Create, run, and analyze GE experiments with ease.")
    
    # Version info
    st.markdown("---")
    st.caption("v1.0.0 | Built with Streamlit")

# Main content area (like real software pages)
if page == "🏃 Run Experiment":
    st.header("🚀 Run Experiment")
    st.markdown("Create and execute new Grammatical Evolution experiments")
    
    with st.form("experiment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Experiment Info")
            experiment_name = st.text_input("Experiment Name", value=f"Experiment_{dt.datetime.now().strftime('%Y%m%d_%H%M')}", help="Unique name for this experiment")
            
            st.subheader("Dataset & Grammar")
            datasets = list_datasets()
            grammars = list_grammars()
            dataset = st.selectbox("Dataset", options=datasets, index=datasets.index('processed.cleveland.data') if 'processed.cleveland.data' in datasets else (0 if datasets else None), help=HELP.get('dataset'))
            grammar = st.selectbox("Grammar", options=grammars, index=grammars.index('UGE_classification.bnf') if 'UGE_classification.bnf' in grammars else (grammars.index('heartDisease.bnf') if 'heartDisease.bnf' in grammars else (0 if grammars else None)), help=HELP.get('grammar'))
            
            st.subheader("GA Parameters")
            population = st.number_input("Population Size", min_value=10, max_value=5000, value=500, step=10, help=HELP.get('population'))
            generations = st.number_input("Generations", min_value=1, max_value=2000, value=200, step=1, help=HELP.get('generations'))
            n_runs = st.number_input("Number of Runs", min_value=1, max_value=20, value=3, help="Number of independent runs for this experiment")
            p_crossover = st.slider("Crossover Probability", min_value=0.0, max_value=1.0, value=0.8, step=0.01, help=HELP.get('p_crossover'))
            p_mutation = st.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=0.01, step=0.01, help=HELP.get('p_mutation'))
            
        with col2:
            st.subheader("GA Parameters (continued)")
            elite_size = st.number_input("Elite Size", min_value=0, max_value=50, value=1, help=HELP.get('elite_size'))
            tournsize = st.number_input("Tournament Size", min_value=2, max_value=50, value=7, help=HELP.get('tournsize'))
            halloffame_size = st.number_input("Hall of Fame Size", min_value=1, max_value=100, value=max(1, int(elite_size)), help=HELP.get('halloffame_size'))
        
            st.subheader("GE/GRAPE Parameters")
            max_tree_depth = st.number_input("Max Tree Depth", min_value=1, max_value=100, value=35, help=HELP.get('max_tree_depth'))
            min_init_tree_depth = st.number_input("Min Init Tree Depth", min_value=1, max_value=50, value=4, help=HELP.get('min_init_tree_depth'))
            max_init_tree_depth = st.number_input("Max Init Tree Depth", min_value=1, max_value=50, value=13, help=HELP.get('max_init_tree_depth'))
            min_init_genome_length = st.number_input("Min Init Genome Length", min_value=1, max_value=5000, value=95, help=HELP.get('min_init_genome_length'))
            max_init_genome_length = st.number_input("Max Init Genome Length", min_value=1, max_value=5000, value=115, help=HELP.get('max_init_genome_length'))
            codon_size = st.number_input("Codon Size", min_value=2, max_value=65535, value=255, help=HELP.get('codon_size'))
            codon_consumption = st.selectbox("Codon Consumption", options=["lazy","eager"], index=0, help=HELP.get('codon_consumption'))
            genome_representation = st.selectbox("Genome Representation", options=["list"], index=0, help=HELP.get('genome_representation'))
            initialisation = st.selectbox("Initialisation", options=["sensible","random"], index=0, help=HELP.get('initialisation'))
        
        st.subheader("Additional Parameters")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Report Items")
            default_report = ['gen','invalid','avg','std','min','max','fitness_test','best_ind_length','avg_length','best_ind_nodes','avg_nodes','best_ind_depth','avg_depth','avg_used_codons','best_ind_used_codons','structural_diversity','fitness_diversity','selection_time','generation_time']
            report_items = st.multiselect("Select report items", options=default_report, default=default_report, help=HELP.get('report_items'))
            
        with col4:
            st.subheader("Dataset Options")
            label_column = None
            test_size = 0.3
            if dataset.endswith('.csv') and dataset != 'processed.cleveland.data':
                df_preview = pd.read_csv(CURRENT_DIR / 'datasets' / dataset, nrows=200)
                cols = list(df_preview.columns)
                label_column = st.selectbox("Label Column", options=cols, index=(cols.index('target') if 'target' in cols else 0), help=HELP.get('label_column'))
                test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05, help=HELP.get('test_size'))
            
            random_seed = st.number_input("Random Seed", min_value=0, max_value=10_000, value=42, step=1, help=HELP.get('random_seed'))
            fitness_metric = st.selectbox("Fitness Metric", options=["mae","accuracy"], index=0, help="Choose which fitness to optimize. MAE: lower is better (minimization). Accuracy: higher is better (maximization). Test fitness will be calculated per generation. For accuracy, consider increasing mutation rate and population size for better exploration.")
            st.session_state['fitness_metric'] = fitness_metric
        
        run_experiment = st.form_submit_button("🚀 Run Experiment", type="primary")

    # Execute experiment when button is clicked
    if run_experiment:
        # Create experiment
        exp_id = create_experiment_id()
        
        # Prepare configuration
        config = {
            'experiment_name': experiment_name,
            'dataset': dataset,
            'grammar': grammar,
            'fitness_metric': fitness_metric,
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
            'report_items': report_items,
            'created_at': dt.datetime.now().isoformat()
        }
        
        # Save experiment config
        exp_dir = save_experiment_config(exp_id, config)
        
        st.success(f"Created experiment: {experiment_name} (ID: {exp_id})")
        
        # Create a collapsible section for running details
        with st.expander("🔍 Show Running Details", expanded=False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a container for all run details
            all_runs_container = st.container()
            
            # Run all runs for this experiment
            all_run_results = {}
            
            for run_idx in range(int(n_runs)):
                status_text.text(f"Running {experiment_name} - Run {run_idx+1}/{n_runs}")
                
                # Create a separate placeholder for each run
                with all_runs_container:
                    st.subheader(f"🏃 Run {run_idx+1} Details")
                    run_placeholder = st.empty()
                
                # Update config for this run
                run_config = config.copy()
                run_config['random_seed'] = int(random_seed) + run_idx
                
                # Run GE
                run_id = create_run_id()
                result = run_ge_classification(run_config, report_items, run_placeholder)
                
                # Save run result
                save_run_result(exp_id, run_id, result)
                all_run_results[run_id] = result
                
                # Update progress
                progress_bar.progress((run_idx + 1) / int(n_runs))
            
            status_text.text("✅ Experiment completed!")
        
        # Store results in session state to persist across reruns
        st.session_state['latest_experiment'] = {
            'config': config,
            'results': all_run_results,
            'exp_id': exp_id,
            'experiment_name': experiment_name
        }
        
        # Show completion message
        st.success(f"✅ Experiment '{experiment_name}' completed successfully!")
        st.info("📊 Go to the 'Analysis' page to view detailed results, charts, and export data.")

elif page == "📊 Dataset Manager":
    st.header("📊 Dataset Manager")
    st.markdown("Upload, manage, and preview your datasets")
    
    # Dataset submenu
    dataset_action = st.radio(
        "Dataset Action:",
        ["➕ Add Dataset", "✏️ Edit Dataset", "👁️ Preview Dataset"],
        key="dataset_action"
    )
    
    if dataset_action == "➕ Add Dataset":
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
                try:
                    datasets_dir = CURRENT_DIR / 'datasets'
                    datasets_dir.mkdir(exist_ok=True)
                    file_path = datasets_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Dataset saved as: {filename}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving dataset: {e}")
    
    elif dataset_action == "✏️ Edit Dataset":
        st.subheader("Manage Existing Datasets")
        datasets = list_datasets()
        if datasets:
            selected_dataset = st.selectbox("Select Dataset to Delete", datasets)
            if st.button("🗑️ Delete Dataset"):
                try:
                    dataset_path = CURRENT_DIR / 'datasets' / selected_dataset
                    if dataset_path.exists():
                        dataset_path.unlink()
                        st.success(f"✅ Deleted: {selected_dataset}")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.info("No datasets available")
    
    elif dataset_action == "👁️ Preview Dataset":
        st.subheader("Dataset Preview")
        datasets = list_datasets()
        if datasets:
            preview_dataset = st.selectbox("Select Dataset to Preview", datasets)
            try:
                dataset_path = CURRENT_DIR / 'datasets' / preview_dataset
                if dataset_path.exists():
                    if preview_dataset.endswith('.csv'):
                        df_preview = pd.read_csv(dataset_path, nrows=10)
                    elif preview_dataset.endswith('.data'):
                        try:
                            df_preview = pd.read_csv(dataset_path, nrows=10)
                        except:
                            df_preview = pd.read_csv(dataset_path, sep=r'\s+', nrows=10)
                    else:
                        df_preview = pd.read_csv(dataset_path, nrows=10)
                    
                    st.dataframe(df_preview, width='stretch', hide_index=True)
                    st.caption(f"Showing first 10 rows. Shape: {df_preview.shape}")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**Columns:** {len(df_preview.columns)}")
                        st.write(f"**Preview rows:** {len(df_preview)}")
                    with col_info2:
                        st.write(f"**Data types:**")
                        for col, dtype in df_preview.dtypes.items():
                            st.write(f"- {col}: {dtype}")
                else:
                    st.error("File not found")
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("No datasets available")

elif page == "📝 Grammar Editor":
    st.header("📝 Grammar Editor")
    st.markdown("Create, edit, and manage BNF grammars for Grammatical Evolution")
    
    # Grammar submenu
    grammar_action = st.radio(
        "Grammar Action:",
        ["➕ Add Grammar", "✏️ Edit Grammar", "👁️ Preview Grammar"],
        key="grammar_action"
    )
    
    if grammar_action == "➕ Add Grammar":
        st.subheader("Create New Grammar")
        new_grammar_name = st.text_input("Grammar Name", placeholder="my_grammar.bnf")
        new_grammar_content = st.text_area("Grammar Content (BNF format)", height=300, placeholder="<start> ::= <expr>\n<expr> ::= ...")
        
        if st.button("💾 Save Grammar"):
            if new_grammar_name and new_grammar_content:
                if not new_grammar_name.endswith('.bnf'):
                    new_grammar_name += '.bnf'
                try:
                    grammar_path = CURRENT_DIR / 'grammars' / new_grammar_name
                    with open(grammar_path, 'w') as f:
                        f.write(new_grammar_content)
                    st.success(f"✅ Grammar saved as: {new_grammar_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("Enter name and content")
    
    elif grammar_action == "✏️ Edit Grammar":
        st.subheader("Edit Existing Grammar")
        grammars = list_grammars()
        if grammars:
            selected_grammar = st.selectbox("Select Grammar", grammars)
            
            # Load grammar content
            try:
                grammar_path = CURRENT_DIR / 'grammars' / selected_grammar
                if grammar_path.exists():
                    with open(grammar_path, 'r') as f:
                        current_content = f.read()
                else:
                    current_content = ""
            except:
                current_content = ""
            
            edited_content = st.text_area("Edit Grammar Content", value=current_content, height=300)
            
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                if st.button("💾 Update Grammar"):
                    try:
                        with open(grammar_path, 'w') as f:
                            f.write(edited_content)
                        st.success(f"✅ Updated: {selected_grammar}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            with col_edit2:
                if st.button("🗑️ Delete Grammar"):
                    try:
                        if grammar_path.exists():
                            grammar_path.unlink()
                            st.success(f"✅ Deleted: {selected_grammar}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        else:
            st.info("No grammars available")
    
    elif grammar_action == "👁️ Preview Grammar":
        st.subheader("Grammar Preview")
        grammars = list_grammars()
        if grammars:
            preview_grammar = st.selectbox("Select Grammar to Preview", grammars)
            try:
                grammar_path = CURRENT_DIR / 'grammars' / preview_grammar
                if grammar_path.exists():
                    with open(grammar_path, 'r') as f:
                        grammar_content = f.read()
                    st.code(grammar_content, language='bnf')
                else:
                    st.error("File not found")
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("No grammars available")

elif page == "🧪 Experiment Manager":
    st.header("🧪 Experiment Manager")
    st.markdown("View, manage, and delete your experiments")
    
    # Experiment submenu
    exp_action = st.radio(
        "Experiment Action:",
        ["📋 List Experiments", "📊 View Results", "🗑️ Delete Experiment"],
        key="exp_action"
    )
    
    if exp_action == "📋 List Experiments":
        st.subheader("Available Experiments")
        experiments = list_experiments()
        if experiments:
            for exp in experiments:
                exp_config = load_experiment_config(exp.name)
                if exp_config:
                    exp_name = exp_config.get('experiment_name', exp.name)
                    runs = list_experiment_runs(exp.name)
                    
                    with st.expander(f"🧪 {exp_name}"):
                        st.write(f"**ID:** {exp.name}")
                        st.write(f"**Dataset:** {exp_config.get('dataset', 'N/A')}")
                        st.write(f"**Grammar:** {exp_config.get('grammar', 'N/A')}")
                        st.write(f"**Runs:** {len(runs)}")
                        st.write(f"**Created:** {exp_config.get('created_at', 'N/A')}")
        else:
            st.info("No experiments found")
    
    elif exp_action == "📊 View Results":
        st.subheader("Experiment Results")
        experiments = list_experiments()
        if experiments:
            exp_options = {}
            for exp in experiments:
                exp_config = load_experiment_config(exp.name)
                if exp_config:
                    exp_name = exp_config.get('experiment_name', exp.name)
                    exp_options[exp_name] = exp.name
            
            selected_exp_name = st.selectbox("Select Experiment", list(exp_options.keys()))
            if selected_exp_name:
                exp_id = exp_options[selected_exp_name]
                runs = list_experiment_runs(exp_id)
                
                if runs:
                    st.write(f"**Runs for {selected_exp_name}:**")
                    for run in runs:
                        run_result = load_run_result(exp_id, run.name)
                        if run_result:
                            st.write(f"• Run {run.name}: Best fitness = {run_result.get('best_training_fitness', 'N/A')}")
                else:
                    st.info("No runs found for this experiment")
        else:
            st.info("No experiments available")
    
    elif exp_action == "🗑️ Delete Experiment":
        st.subheader("Delete Experiment")
        experiments = list_experiments()
        if experiments:
            exp_options = {}
            for exp in experiments:
                exp_config = load_experiment_config(exp.name)
                if exp_config:
                    exp_name = exp_config.get('experiment_name', exp.name)
                    exp_options[exp_name] = exp.name
            
            selected_exp_name = st.selectbox("Select Experiment to Delete", list(exp_options.keys()))
            if st.button("🗑️ Delete Experiment"):
                try:
                    exp_id = exp_options[selected_exp_name]
                    # Delete experiment directory
                    exp_dir = CURRENT_DIR / 'results' / 'experiments' / exp_id
                    if exp_dir.exists():
                        import shutil
                        shutil.rmtree(exp_dir)
                        st.success(f"✅ Deleted experiment: {selected_exp_name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.info("No experiments available")


elif page == "⚖️ Comparison":
    st.header("⚖️ Multi-Experiment Comparison")
    st.markdown("Compare multiple experiments based on their average performance across all runs")
    
    experiments = list_experiments()
    if len(experiments) >= 2:
        # Get experiment options with user-friendly names
        exp_options = {}
        for exp in experiments:
            exp_config = load_experiment_config(exp.name)
            if exp_config:
                exp_name = exp_config.get('experiment_name', exp.name)
                exp_options[exp_name] = exp.name
            else:
                exp_options[exp.name] = exp.name
        
        st.subheader("🔬 Select Experiments to Compare")
        st.markdown("**Choose multiple experiments to compare their average performance:**")
        
        # Multi-select for experiments
        selected_experiments = st.multiselect(
            "Select Experiments:",
            options=list(exp_options.keys()),
            default=list(exp_options.keys())[:2] if len(exp_options) >= 2 else [],
            help="Select 2 or more experiments to compare. You can choose as many as you want!"
        )
        
        if len(selected_experiments) >= 2:
            st.success(f"✅ **Selected {len(selected_experiments)} experiments** for comparison")
            
            # Load experiment data and calculate averages
            experiment_stats = {}
            experiment_configs = {}
            
            for exp_name in selected_experiments:
                exp_id = exp_options[exp_name]
                exp_config = load_experiment_config(exp_id)
                experiment_configs[exp_name] = exp_config
                
                # Load all runs for this experiment
                runs = list_experiment_runs(exp_id)
                
                if runs:
                    # Calculate average statistics across all runs
                    all_training_fitness = []
                    all_test_fitness = []
                    all_best_phenotypes = []
                    
                    for run in runs:
                        run_result = load_run_result(exp_id, run.name)
                        if run_result:
                            if run_result.get('max') and len(run_result['max']) > 0:
                                all_training_fitness.append(run_result['max'][-1])
                            if run_result.get('fitness_test') and len(run_result['fitness_test']) > 0:
                                all_test_fitness.append(run_result['fitness_test'][-1])
                            if run_result.get('best_phenotype'):
                                all_best_phenotypes.append(run_result['best_phenotype'])
                    
                    # Store average statistics
                    experiment_stats[exp_name] = {
                        'avg_training_fitness': np.mean(all_training_fitness) if all_training_fitness else None,
                        'std_training_fitness': np.std(all_training_fitness) if all_training_fitness else None,
                        'best_training_fitness': np.max(all_training_fitness) if all_training_fitness else None,
                        'min_training_fitness': np.min(all_training_fitness) if all_training_fitness else None,
                        'avg_test_fitness': np.mean(all_test_fitness) if all_test_fitness else None,
                        'std_test_fitness': np.std(all_test_fitness) if all_test_fitness else None,
                        'best_test_fitness': np.max(all_test_fitness) if all_test_fitness else None,
                        'min_test_fitness': np.min(all_test_fitness) if all_test_fitness else None,
                        'num_runs': len(runs),
                        'best_phenotypes': all_best_phenotypes
                    }
            
            if experiment_stats:
                st.markdown("---")
                
                # Experiment overview table
                st.subheader("📊 Experiment Overview")
                
                overview_data = []
                for exp_name in selected_experiments:
                    exp_config = experiment_configs[exp_name]
                    stats = experiment_stats[exp_name]
                    
                    overview_data.append({
                        'Experiment': exp_name,
                        'Dataset': exp_config.get('dataset', 'N/A'),
                        'Grammar': exp_config.get('grammar', 'N/A'),
                        'Fitness Metric': exp_config.get('fitness_metric', 'N/A').upper(),
                        'Population': exp_config.get('population', 'N/A'),
                        'Generations': exp_config.get('generations', 'N/A'),
                        'Runs': stats['num_runs'],
                        'Avg Training': f"{stats['avg_training_fitness']:.4f}" if stats['avg_training_fitness'] is not None else 'N/A',
                        'Best Training': f"{stats['best_training_fitness']:.4f}" if stats['best_training_fitness'] is not None else 'N/A',
                        'Avg Test': f"{stats['avg_test_fitness']:.4f}" if stats['avg_test_fitness'] is not None else 'N/A',
                        'Best Test': f"{stats['best_test_fitness']:.4f}" if stats['best_test_fitness'] is not None else 'N/A'
                    })
                
                overview_df = pd.DataFrame(overview_data)
                st.dataframe(overview_df, width='stretch', hide_index=True)
                
                # Performance comparison metrics
                st.markdown("---")
                st.subheader("📈 Performance Comparison (Average-based)")
                
                # Create comparison metrics
                fitness_metric = experiment_configs[selected_experiments[0]].get('fitness_metric', 'mae')
                metric_label = 'Accuracy' if fitness_metric == 'accuracy' else 'MAE'
                better_direction = 'higher' if fitness_metric == 'accuracy' else 'lower'
                
                st.markdown(f"**Fitness Metric:** `{metric_label}` (Direction: {better_direction} is better)")
                
                # Training performance comparison
                st.markdown("#### 🎯 Training Performance Comparison")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Average Training Fitness**")
                    for exp_name in selected_experiments:
                        stats = experiment_stats[exp_name]
                        if stats['avg_training_fitness'] is not None:
                            st.metric(
                                exp_name,
                                f"{stats['avg_training_fitness']:.4f}",
                                f"±{stats['std_training_fitness']:.4f}" if stats['std_training_fitness'] is not None else None
                            )
                
                with col2:
                    st.markdown("**Best Training Fitness**")
                    for exp_name in selected_experiments:
                        stats = experiment_stats[exp_name]
                        if stats['best_training_fitness'] is not None:
                            st.metric(exp_name, f"{stats['best_training_fitness']:.4f}")
                
                with col3:
                    st.markdown("**Average Test Fitness**")
                    for exp_name in selected_experiments:
                        stats = experiment_stats[exp_name]
                        if stats['avg_test_fitness'] is not None:
                            st.metric(
                                exp_name,
                                f"{stats['avg_test_fitness']:.4f}",
                                f"±{stats['std_test_fitness']:.4f}" if stats['std_test_fitness'] is not None else None
                            )
                
                with col4:
                    st.markdown("**Best Test Fitness**")
                    for exp_name in selected_experiments:
                        stats = experiment_stats[exp_name]
                        if stats['best_test_fitness'] is not None:
                            st.metric(exp_name, f"{stats['best_test_fitness']:.4f}")
                
                # Aggregate charts across runs and generations
                st.markdown("---")
                st.subheader("📊 Aggregate Performance Across Runs & Generations")
                st.markdown("Shows how fitness evolves across generations, averaged across all runs for each experiment")
                
                # Calculate aggregate data across runs and generations
                aggregate_data = {}
                colors = px.colors.qualitative.Set1
                
                for i, exp_name in enumerate(selected_experiments):
                    exp_id = exp_options[exp_name]
                    runs = list_experiment_runs(exp_id)
                    
                    if runs:
                        # Collect all generation data across runs
                        all_generations_data = {
                            'max': [],  # Best fitness per generation
                            'avg': [],  # Average fitness per generation
                            'min': [],  # Min fitness per generation
                            'fitness_test': []  # Test fitness per generation
                        }
                        
                        max_generations = 0
                        for run in runs:
                            run_result = load_run_result(exp_id, run.name)
                            if run_result:
                                # Get the maximum number of generations across all runs
                                max_generations = max(max_generations, len(run_result.get('max', [])))
                        
                        # Initialize arrays for each generation
                        for gen in range(max_generations):
                            all_generations_data['max'].append([])
                            all_generations_data['avg'].append([])
                            all_generations_data['min'].append([])
                            all_generations_data['fitness_test'].append([])
                        
                        # Collect data from all runs
                        for run in runs:
                            run_result = load_run_result(exp_id, run.name)
                            if run_result:
                                for gen in range(max_generations):
                                    if gen < len(run_result.get('max', [])):
                                        all_generations_data['max'][gen].append(run_result['max'][gen])
                                    if gen < len(run_result.get('avg', [])):
                                        all_generations_data['avg'][gen].append(run_result['avg'][gen])
                                    if gen < len(run_result.get('min', [])):
                                        all_generations_data['min'][gen].append(run_result['min'][gen])
                                    if gen < len(run_result.get('fitness_test', [])) and run_result['fitness_test'][gen] is not None:
                                        all_generations_data['fitness_test'][gen].append(run_result['fitness_test'][gen])
                        
                        # Calculate aggregate statistics per generation
                        aggregate_stats = {
                            'generations': list(range(max_generations)),
                            'avg_max': [],
                            'std_max': [],
                            'avg_avg': [],
                            'std_avg': [],
                            'avg_min': [],
                            'std_min': [],
                            'avg_test': [],
                            'std_test': []
                        }
                        
                        for gen in range(max_generations):
                            # Best fitness (max)
                            if all_generations_data['max'][gen]:
                                aggregate_stats['avg_max'].append(np.mean(all_generations_data['max'][gen]))
                                aggregate_stats['std_max'].append(np.std(all_generations_data['max'][gen]))
                            else:
                                aggregate_stats['avg_max'].append(np.nan)
                                aggregate_stats['std_max'].append(np.nan)
                            
                            # Average fitness
                            if all_generations_data['avg'][gen]:
                                aggregate_stats['avg_avg'].append(np.mean(all_generations_data['avg'][gen]))
                                aggregate_stats['std_avg'].append(np.std(all_generations_data['avg'][gen]))
                            else:
                                aggregate_stats['avg_avg'].append(np.nan)
                                aggregate_stats['std_avg'].append(np.nan)
                            
                            # Min fitness
                            if all_generations_data['min'][gen]:
                                aggregate_stats['avg_min'].append(np.mean(all_generations_data['min'][gen]))
                                aggregate_stats['std_min'].append(np.std(all_generations_data['min'][gen]))
                            else:
                                aggregate_stats['avg_min'].append(np.nan)
                                aggregate_stats['std_min'].append(np.nan)
                            
                            # Test fitness
                            if all_generations_data['fitness_test'][gen]:
                                aggregate_stats['avg_test'].append(np.mean(all_generations_data['fitness_test'][gen]))
                                aggregate_stats['std_test'].append(np.std(all_generations_data['fitness_test'][gen]))
                            else:
                                aggregate_stats['avg_test'].append(np.nan)
                                aggregate_stats['std_test'].append(np.nan)
                        
                        aggregate_data[exp_name] = aggregate_stats
                
                # Create aggregate comparison charts
                if aggregate_data:
                    # Chart selection
                    chart_type = st.selectbox(
                        "Select Chart Type:",
                        ["Best Fitness (Max)", "Average Fitness", "Test Fitness", "All Metrics"],
                        help="Choose which fitness metric to display in the aggregate chart"
                    )
                    
                    fig = go.Figure()
                    
                    for i, (exp_name, stats) in enumerate(aggregate_data.items()):
                        color = colors[i % len(colors)]
                        
                        if chart_type == "Best Fitness (Max)":
                            # Add main line
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_max'],
                                mode='lines+markers',
                                name=f'{exp_name} - Best',
                                line=dict(color=color, width=3),
                                marker=dict(size=4)
                            ))
                            
                            # Add confidence interval (std)
                            if any(not np.isnan(std) for std in stats['std_max']):
                                upper_bound = [avg + std if not np.isnan(avg) and not np.isnan(std) else np.nan 
                                             for avg, std in zip(stats['avg_max'], stats['std_max'])]
                                lower_bound = [avg - std if not np.isnan(avg) and not np.isnan(std) else np.nan 
                                             for avg, std in zip(stats['avg_max'], stats['std_max'])]
                                
                                fig.add_trace(go.Scatter(
                                    x=stats['generations'] + stats['generations'][::-1],
                                    y=upper_bound + lower_bound[::-1],
                                    fill='tonexty' if i > 0 else 'tozeroy',
                                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    hoverinfo="skip"
                                ))
                        
                        elif chart_type == "Average Fitness":
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_avg'],
                                mode='lines+markers',
                                name=f'{exp_name} - Average',
                                line=dict(color=color, width=3),
                                marker=dict(size=4)
                            ))
                        
                        elif chart_type == "Test Fitness":
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_test'],
                                mode='lines+markers',
                                name=f'{exp_name} - Test',
                                line=dict(color=color, width=3),
                                marker=dict(size=4)
                            ))
                        
                        elif chart_type == "All Metrics":
                            # Best fitness
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_max'],
                                mode='lines+markers',
                                name=f'{exp_name} - Best',
                                line=dict(color=color, width=2, dash='solid'),
                                marker=dict(size=3)
                            ))
                            
                            # Average fitness
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_avg'],
                                mode='lines+markers',
                                name=f'{exp_name} - Average',
                                line=dict(color=color, width=2, dash='dash'),
                                marker=dict(size=3)
                            ))
                            
                            # Test fitness
                            fig.add_trace(go.Scatter(
                                x=stats['generations'],
                                y=stats['avg_test'],
                                mode='lines+markers',
                                name=f'{exp_name} - Test',
                                line=dict(color=color, width=2, dash='dot'),
                                marker=dict(size=3)
                            ))
                    
                    # Update layout
                    fitness_metric = experiment_configs[selected_experiments[0]].get('fitness_metric', 'mae')
                    ylabel = 'Fitness (higher is better)' if fitness_metric == 'accuracy' else 'Fitness (lower is better)'
                    
                    fig.update_layout(
                        title=f'📊 Aggregate Performance: {chart_type} Across Generations',
                        xaxis_title='Generation',
                        yaxis_title=ylabel,
                        hovermode='x unified',
                        showlegend=True,
                        template='plotly_white',
                        height=600,
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generation-by-generation statistics table
                    st.markdown("#### 📋 Generation-by-Generation Statistics")
                    
                    # Create detailed statistics table
                    gen_stats_data = []
                    for exp_name, stats in aggregate_data.items():
                        for gen in range(len(stats['generations'])):
                            gen_stats_data.append({
                                'Experiment': exp_name,
                                'Generation': gen,
                                'Avg Best Fitness': f"{stats['avg_max'][gen]:.4f}" if not np.isnan(stats['avg_max'][gen]) else 'N/A',
                                'Std Best Fitness': f"±{stats['std_max'][gen]:.4f}" if not np.isnan(stats['std_max'][gen]) else 'N/A',
                                'Avg Average Fitness': f"{stats['avg_avg'][gen]:.4f}" if not np.isnan(stats['avg_avg'][gen]) else 'N/A',
                                'Std Average Fitness': f"±{stats['std_avg'][gen]:.4f}" if not np.isnan(stats['std_avg'][gen]) else 'N/A',
                                'Avg Test Fitness': f"{stats['avg_test'][gen]:.4f}" if not np.isnan(stats['avg_test'][gen]) else 'N/A',
                                'Std Test Fitness': f"±{stats['std_test'][gen]:.4f}" if not np.isnan(stats['std_test'][gen]) else 'N/A'
                            })
                    
                    if gen_stats_data:
                        gen_stats_df = pd.DataFrame(gen_stats_data)
                        
                        # Show only first 20 generations to avoid overwhelming display
                        if len(gen_stats_df) > 20:
                            st.info(f"Showing first 20 generations. Total generations: {len(gen_stats_df)}")
                            display_df = gen_stats_df.head(20)
                        else:
                            display_df = gen_stats_df
                        
                        st.dataframe(display_df, width='stretch', hide_index=True)
                        
                        # Export generation statistics
                        st.markdown("#### 📥 Export Generation Statistics")
                        csv_gen_stats = gen_stats_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Generation Statistics CSV",
                            data=csv_gen_stats,
                            file_name=f"generation_statistics_{len(selected_experiments)}_experiments.csv",
                            mime="text/csv"
                        )
                
                # Ranking table
                st.markdown("---")
                st.subheader("🏆 Performance Ranking")
                
                # Create ranking based on average training fitness
                ranking_data = []
                for exp_name in selected_experiments:
                    stats = experiment_stats[exp_name]
                    if stats['avg_training_fitness'] is not None:
                        ranking_data.append({
                            'Experiment': exp_name,
                            'Avg Training Fitness': stats['avg_training_fitness'],
                            'Best Training Fitness': stats['best_training_fitness'],
                            'Avg Test Fitness': stats['avg_test_fitness'],
                            'Best Test Fitness': stats['best_test_fitness'],
                            'Number of Runs': stats['num_runs']
                        })
                
                if ranking_data:
                    ranking_df = pd.DataFrame(ranking_data)
                    
                    # Sort by average training fitness (higher is better for accuracy, lower is better for MAE)
                    if fitness_metric == 'accuracy':
                        ranking_df = ranking_df.sort_values('Avg Training Fitness', ascending=False)
                    else:
                        ranking_df = ranking_df.sort_values('Avg Training Fitness', ascending=True)
                    
                    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                    ranking_df = ranking_df[['Rank', 'Experiment', 'Avg Training Fitness', 'Best Training Fitness', 'Avg Test Fitness', 'Best Test Fitness', 'Number of Runs']]
                    
                    st.dataframe(ranking_df, width='stretch', hide_index=True)
                    
                    # Show best experiment
                    best_exp = ranking_df.iloc[0]
                    st.success(f"🏅 **Best Performing Experiment:** {best_exp['Experiment']} with average training fitness of {best_exp['Avg Training Fitness']:.4f}")
                
                # Export comparison data
                st.markdown("---")
                st.subheader("📥 Export Comparison Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export overview data
                    csv_overview = overview_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Overview CSV",
                        data=csv_overview,
                        file_name=f"experiment_overview_{len(selected_experiments)}_experiments.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export ranking data
                    if ranking_data:
                        csv_ranking = ranking_df.to_csv(index=False)
                        st.download_button(
                            label="🏆 Download Ranking CSV",
                            data=csv_ranking,
                            file_name=f"experiment_ranking_{len(selected_experiments)}_experiments.csv",
                            mime="text/csv"
                        )
        else:
            st.warning("⚠️ Please select at least 2 experiments to compare")
    else:
        st.info("Need at least 2 experiments to compare. Create more experiments using the 'Run Experiment' page!")

elif page == "📈 Analysis":
    st.header("📊 Experiment Analysis")
    st.markdown("Detailed analysis of individual experiments with comprehensive statistics and visualizations")
    
    experiments = list_experiments()
    if experiments:
        # Experiment selection - show user-friendly names
        exp_options = {}
        for exp in experiments:
            exp_config = load_experiment_config(exp.name)
            if exp_config:
                exp_name = exp_config.get('experiment_name', exp.name)
                exp_options[exp_name] = exp.name
            else:
                exp_options[exp.name] = exp.name
        
        selected_exp = st.selectbox("Select Experiment", list(exp_options.keys()))
        exp_id = exp_options[selected_exp]
        
        # Load experiment config
        exp_config = load_experiment_config(exp_id)
        if exp_config:
            st.subheader(f"🔬 {exp_config.get('experiment_name', exp_id)}")
            
            # Show experiment info with better formatting
            st.markdown("### 📋 Experiment Configuration")
            
            # Create styled info boxes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🗂️ **Dataset & Grammar**")
                st.info(f"**Dataset:** `{exp_config.get('dataset', 'N/A')}`")
                st.info(f"**Grammar:** `{exp_config.get('grammar', 'N/A')}`")
                st.info(f"**Fitness Metric:** `{exp_config.get('fitness_metric', 'N/A').upper()}`")
            
            with col2:
                st.markdown("#### ⚙️ **Algorithm Parameters**")
                st.success(f"**Runs:** `{exp_config.get('n_runs', 'N/A')}`")
                st.success(f"**Generations:** `{exp_config.get('generations', 'N/A')}`")
                st.success(f"**Population:** `{exp_config.get('population', 'N/A')}`")
            
            with col3:
                st.markdown("#### 🎯 **Genetic Operators**")
                st.warning(f"**Crossover:** `{exp_config.get('p_crossover', 'N/A')}`")
                st.warning(f"**Mutation:** `{exp_config.get('p_mutation', 'N/A')}`")
                st.warning(f"**Created:** `{exp_config.get('created_at', 'N/A')[:10]}`")
            
            # Load all runs for this experiment
            runs = list_experiment_runs(exp_id)
            if runs:
                # Calculate comprehensive statistics
                runs_data = {}
                all_training_fitness = []
                all_test_fitness = []
                all_best_phenotypes = []
                
                for run in runs:
                    run_result = load_run_result(exp_id, run.name)
                    if run_result:
                        runs_data[run.name] = run_result
                        # Use the correct keys from the run results
                        if run_result.get('max') and len(run_result['max']) > 0:
                            all_training_fitness.append(run_result['max'][-1])  # Best fitness from last generation
                        if run_result.get('fitness_test') and len(run_result['fitness_test']) > 0:
                            all_test_fitness.append(run_result['fitness_test'][-1])
                        # For now, we'll use the max fitness as the "best phenotype" indicator
                        if run_result.get('max') and len(run_result['max']) > 0:
                            all_best_phenotypes.append(f"Run {run.name} - Best Fitness: {run_result['max'][-1]:.4f}")
                
                # Display comprehensive statistics with better formatting
                st.markdown("---")
                st.markdown("### 📊 **Performance Summary Statistics**")
                
                fitness_metric = exp_config.get('fitness_metric', 'mae')
                metric_label = 'Accuracy' if fitness_metric == 'accuracy' else 'MAE'
                better_direction = 'higher' if fitness_metric == 'accuracy' else 'lower'
                
                # Create styled metrics with better visual hierarchy
                st.markdown(f"**Fitness Metric:** `{metric_label}` (Direction: {better_direction} is better)")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("#### 🎯 **Training Performance**")
                    if all_training_fitness:
                        mean_val = np.mean(all_training_fitness)
                        std_val = np.std(all_training_fitness)
                        st.metric(
                            "Average Training", 
                            f"{mean_val:.4f}",
                            f"±{std_val:.4f}",
                            help=f"Average {metric_label} across all runs"
                        )
                        st.metric(
                            "Best Training", 
                            f"{np.max(all_training_fitness):.4f}",
                            help=f"Best {metric_label} across all runs"
                        )
                    else:
                        st.metric("Average Training", "N/A")
                        st.metric("Best Training", "N/A")
                
                with col2:
                    st.markdown("#### 🧪 **Test Performance**")
                    if all_test_fitness:
                        mean_val = np.mean(all_test_fitness)
                        std_val = np.std(all_test_fitness)
                        st.metric(
                            "Average Test", 
                            f"{mean_val:.4f}",
                            f"±{std_val:.4f}",
                            help=f"Average test {metric_label} across all runs"
                        )
                        st.metric(
                            "Best Test", 
                            f"{np.max(all_test_fitness):.4f}",
                            help=f"Best test {metric_label} across all runs"
                        )
                    else:
                        st.metric("Average Test", "N/A")
                        st.metric("Best Test", "N/A")
                
                with col3:
                    st.markdown("#### 📈 **Training Range**")
                    if all_training_fitness:
                        st.metric(
                            "Minimum Training", 
                            f"{np.min(all_training_fitness):.4f}",
                            help=f"Minimum {metric_label} across all runs"
                        )
                        st.metric(
                            "Maximum Training", 
                            f"{np.max(all_training_fitness):.4f}",
                            help=f"Maximum {metric_label} across all runs"
                        )
                    else:
                        st.metric("Minimum Training", "N/A")
                        st.metric("Maximum Training", "N/A")
                
                with col4:
                    st.markdown("#### 📉 **Test Range**")
                    if all_test_fitness:
                        st.metric(
                            "Minimum Test", 
                            f"{np.min(all_test_fitness):.4f}",
                            help=f"Minimum test {metric_label} across all runs"
                        )
                        st.metric(
                            "Maximum Test", 
                            f"{np.max(all_test_fitness):.4f}",
                            help=f"Maximum test {metric_label} across all runs"
                        )
                    else:
                        st.metric("Minimum Test", "N/A")
                        st.metric("Maximum Test", "N/A")
                
                # Best run summary across all runs
                if all_best_phenotypes and all_training_fitness:
                    st.markdown("---")
                    st.markdown("### 🏆 **Best Run Summary**")
                    
                    best_overall_idx = np.argmax(all_training_fitness) if all_training_fitness else 0
                    best_overall_phenotype = all_best_phenotypes[best_overall_idx]
                    
                    # Create a highlighted success box for the best run
                    st.success(f"**🏅 Best Overall Run:** {best_overall_phenotype}")
                    
                    # Show the best fitness value prominently
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Best Training Fitness", 
                            f"{all_training_fitness[best_overall_idx]:.4f}",
                            help="Best training fitness achieved across all runs"
                        )
                    with col2:
                        if all_test_fitness and best_overall_idx < len(all_test_fitness):
                            st.metric(
                                "Corresponding Test Fitness", 
                                f"{all_test_fitness[best_overall_idx]:.4f}",
                                help="Test fitness for the best training run"
                            )
            
            # Load all runs for this experiment
            runs = list_experiment_runs(exp_id)
            if runs:
                st.markdown("---")
                st.markdown("### 📈 **Analysis Options**")
                
                # Create a nice info box showing run count
                st.info(f"📊 **Found {len(runs)} runs** available for detailed analysis")
                
                # Analysis type selection with better styling
                st.markdown("#### 🔍 **Select Analysis Type**")
                analysis_type = st.selectbox(
                    "Choose how to analyze this experiment:", 
                    ["Across Generations", "Across Runs", "Individual Run Details"],
                    help="Choose how to analyze this experiment",
                    label_visibility="collapsed"
                )
                
                # Metric selection with better styling
                st.markdown("#### 📊 **Select Metrics to Display**")
                available_metrics = ['max', 'avg', 'min', 'std', 'fitness_test']
                metric_labels = {
                    'max': 'Best Training Fitness',
                    'avg': 'Average Training Fitness', 
                    'min': 'Minimum Training Fitness',
                    'std': 'Standard Deviation',
                    'fitness_test': 'Test Fitness'
                }
                
                selected_metrics = st.multiselect(
                    "Choose metrics to include in analysis:", 
                    available_metrics,
                    default=['max', 'fitness_test'],
                    format_func=lambda x: metric_labels.get(x, x),
                    help="Select which fitness metrics to display in charts and analysis"
                )
                
                if selected_metrics:
                    # Show selected metrics in a nice format
                    selected_labels = [metric_labels.get(m, m) for m in selected_metrics]
                    st.success(f"✅ **Selected metrics:** {', '.join(selected_labels)}")
                    
                    # Load run data
                    runs_data = {}
                    for run in runs:
                        run_result = load_run_result(exp_id, run.name)
                        if run_result:
                            runs_data[run.name] = run_result
                    
                    st.info(f"📁 **Loaded {len(runs_data)} runs** with complete data")
                    
                    if analysis_type == "Across Generations":
                        st.markdown("---")
                        st.markdown("### 📊 **Individual Run Analysis**")
                        st.markdown("View detailed fitness evolution for each run individually")
                        
                        # Show individual run plots
                        for i, (run_name, run_data) in enumerate(runs_data.items(), 1):
                            with st.expander(f"🏃 **Run {i}** - {run_name}", expanded=False):
                                st.markdown(f"**Run ID:** `{run_name}`")
                                
                                # Show key metrics for this run
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Training", f"{run_data.get('best_training_fitness', 'N/A'):.4f}" if run_data.get('best_training_fitness') is not None else 'N/A')
                                with col2:
                                    st.metric("Final Test", f"{run_data.get('fitness_test', [])[-1]:.4f}" if run_data.get('fitness_test') and run_data['fitness_test'][-1] is not None else 'N/A')
                                with col3:
                                    st.metric("Generations", len(run_data.get('max', [])))
                                
                                # Show the plot
                                fig = plot_result(run_data, title=f"Fitness Evolution - Run {i}")
                    
                    elif analysis_type == "Across Runs":
                        st.markdown("---")
                        st.markdown("### 📊 **Cross-Run Comparison**")
                        st.markdown("Compare fitness evolution across all runs in a single view")
                        
                        # Show comparison across runs
                        fig = go.Figure()
                        colors = px.colors.qualitative.Set3
                        
                        for i, (run_name, run_data) in enumerate(runs_data.items()):
                            color = colors[i % len(colors)]
                            for metric in selected_metrics:
                                if metric in run_data and run_data[metric]:
                                    gens = list(range(len(run_data[metric])))
                                    values = [float(v) if v is not None else np.nan for v in run_data[metric]]
                                    valid_indices = [j for j, v in enumerate(values) if v == v]
                                    if valid_indices:
                                        valid_gens = [gens[j] for j in valid_indices]
                                        valid_values = [values[j] for j in valid_indices]
                                        metric_label = metric_labels.get(metric, metric)
                                        label = f'Run {i+1} - {metric_label}'
                                        fig.add_trace(go.Scatter(
                                            x=valid_gens, y=valid_values,
                                            mode='lines',
                                            name=label,
                                            line=dict(color=color, width=2),
                                            opacity=0.8
                                        ))
                        
                        fitness_metric = exp_config.get('fitness_metric', 'mae')
                        ylabel = 'Fitness (higher is better)' if fitness_metric == 'accuracy' else 'Fitness (lower is better)'
                        
                        fig.update_layout(
                            title=f'📊 Cross-Run Comparison - {exp_config.get("experiment_name")}',
                            xaxis_title='Generation',
                            yaxis_title=ylabel,
                            hovermode='x unified',
                            showlegend=True,
                            template='plotly_white',
                            height=600,
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics with better formatting
                        st.markdown("---")
                        st.markdown("### 📋 **Summary Statistics**")
                        st.markdown("Comprehensive performance metrics across all runs")
                        summary_data = []
                        for i, (run_name, run_data) in enumerate(runs_data.items()):
                            summary_data.append({
                                'Run': f'Run {i+1}',
                                'Run_ID': run_name,
                                'Best Training Fitness': f"{run_data.get('best_training_fitness', 'N/A'):.4f}" if run_data.get('best_training_fitness') is not None else 'N/A',
                                'Final Test Fitness': f"{run_data.get('fitness_test', [])[-1]:.4f}" if run_data.get('fitness_test') and run_data['fitness_test'][-1] is not None else 'N/A',
                                'Final Avg Training': f"{run_data.get('avg', [])[-1]:.4f}" if run_data.get('avg') and run_data['avg'][-1] is not None else 'N/A',
                                'Final Min Training': f"{run_data.get('min', [])[-1]:.4f}" if run_data.get('min') and run_data['min'][-1] is not None else 'N/A',
                                'Final Max Training': f"{run_data.get('max', [])[-1]:.4f}" if run_data.get('max') and run_data['max'][-1] is not None else 'N/A',
                                'Generations': len(run_data.get('max', [])),
                                'Best Phenotype': run_data.get('best_phenotype', 'N/A')[:100] + '...' if run_data.get('best_phenotype') and len(run_data.get('best_phenotype', '')) > 100 else run_data.get('best_phenotype', 'N/A'),
                                # Configuration data
                                'Experiment_Name': exp_config.get('experiment_name'),
                                'Dataset': exp_config.get('dataset'),
                                'Grammar': exp_config.get('grammar'),
                                'Population': exp_config.get('population'),
                                'Generations_Config': exp_config.get('generations'),
                                'Crossover_Prob': exp_config.get('p_crossover'),
                                'Mutation_Prob': exp_config.get('p_mutation'),
                                'Elite_Size': exp_config.get('elite_size'),
                                'Tournament_Size': exp_config.get('tournsize'),
                                'HOF_Size': exp_config.get('halloffame_size'),
                                'Max_Tree_Depth': exp_config.get('max_tree_depth'),
                                'Min_Init_Tree_Depth': exp_config.get('min_init_tree_depth'),
                                'Max_Init_Tree_Depth': exp_config.get('max_init_tree_depth'),
                                'Min_Init_Genome_Length': exp_config.get('min_init_genome_length'),
                                'Max_Init_Genome_Length': exp_config.get('max_init_genome_length'),
                                'Codon_Size': exp_config.get('codon_size'),
                                'Codon_Consumption': exp_config.get('codon_consumption'),
                                'Genome_Representation': exp_config.get('genome_representation'),
                                'Initialisation': exp_config.get('initialisation'),
                                'Fitness_Metric': exp_config.get('fitness_metric'),
                                'Random_Seed': exp_config.get('random_seed') + i,
                                'Test_Size': exp_config.get('test_size'),
                                'Created_At': exp_config.get('created_at')
                            })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, width='stretch', hide_index=True)
                            
                            # CSV Export for Summary Statistics
                            st.subheader("📥 Export Summary Statistics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export summary CSV
                                csv_summary = summary_df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Summary CSV",
                                    data=csv_summary,
                                    file_name=f"{exp_config.get('experiment_name', 'experiment')}_summary.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Export detailed CSV with all generation data and config
                                detailed_data = []
                                for i, (run_name, run_data) in enumerate(runs_data.items()):
                                    gens = list(range(len(run_data.get('max', []))))
                                    for gen in gens:
                                        row = {
                                            'Run': f'Run {i+1}',
                                            'Run_ID': run_name,
                                            'Generation': gen,
                                            'Best_Training_Fitness': run_data.get('max', [])[gen] if gen < len(run_data.get('max', [])) else None,
                                            'Avg_Training_Fitness': run_data.get('avg', [])[gen] if gen < len(run_data.get('avg', [])) else None,
                                            'Min_Training_Fitness': run_data.get('min', [])[gen] if gen < len(run_data.get('min', [])) else None,
                                            'Std_Training_Fitness': run_data.get('std', [])[gen] if gen < len(run_data.get('std', [])) else None,
                                            'Test_Fitness': run_data.get('fitness_test', [])[gen] if gen < len(run_data.get('fitness_test', [])) else None,
                                            'Experiment_Name': exp_config.get('experiment_name'),
                                            'Dataset': exp_config.get('dataset'),
                                            'Grammar': exp_config.get('grammar'),
                                            'Population': exp_config.get('population'),
                                            'Generations': exp_config.get('generations'),
                                            'Crossover_Prob': exp_config.get('p_crossover'),
                                            'Mutation_Prob': exp_config.get('p_mutation'),
                                            'Elite_Size': exp_config.get('elite_size'),
                                            'Tournament_Size': exp_config.get('tournsize'),
                                            'HOF_Size': exp_config.get('halloffame_size'),
                                            'Max_Tree_Depth': exp_config.get('max_tree_depth'),
                                            'Min_Init_Tree_Depth': exp_config.get('min_init_tree_depth'),
                                            'Max_Init_Tree_Depth': exp_config.get('max_init_tree_depth'),
                                            'Min_Init_Genome_Length': exp_config.get('min_init_genome_length'),
                                            'Max_Init_Genome_Length': exp_config.get('max_init_genome_length'),
                                            'Codon_Size': exp_config.get('codon_size'),
                                            'Codon_Consumption': exp_config.get('codon_consumption'),
                                            'Genome_Representation': exp_config.get('genome_representation'),
                                            'Initialisation': exp_config.get('initialisation'),
                                            'Fitness_Metric': exp_config.get('fitness_metric'),
                                            'Random_Seed': exp_config.get('random_seed') + i,
                                            'Test_Size': exp_config.get('test_size'),
                                            'Created_At': exp_config.get('created_at')
                                        }
                                        detailed_data.append(row)
                                
                                if detailed_data:
                                    detailed_df = pd.DataFrame(detailed_data)
                                    csv_detailed = detailed_df.to_csv(index=False)
                                    st.download_button(
                                        label="📈 Download Detailed CSV (All Generations + Config)",
                                        data=csv_detailed,
                                        file_name=f"{exp_config.get('experiment_name', 'experiment')}_detailed.csv",
                                        mime="text/csv"
                                    )
                    
                    elif analysis_type == "Individual Run Details":
                        # Individual run details
                        selected_run = st.selectbox("Select Run", [run.name for run in runs])
                        
                        if selected_run:
                            run_result = load_run_result(exp_id, selected_run)
                            if run_result:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Best Training Fitness", f"{run_result.get('best_training_fitness', 'N/A'):.4f}")
                                    st.metric("Final Test Fitness", f"{run_result.get('fitness_test', [])[-1]:.4f}" if run_result.get('fitness_test') and run_result['fitness_test'][-1] is not None else 'N/A')
                                
                                with col2:
                                    st.metric("Fitness Metric", run_result.get('fitness_metric', 'N/A'))
                                    st.metric("Generations", len(run_result.get('max', [])))
                                
                                # Best individual
                                st.subheader("Best Individual (Phenotype)")
                                st.code(run_result.get('best_phenotype', '<none>'))
                                
                                # CSV data
                                if 'logbook' in run_result:
                                    st.subheader("Generation Data")
                                    logbook_df = pd.DataFrame(run_result['logbook'])
                                    st.dataframe(logbook_df, width='stretch')
                                    
                                    # Individual run CSV download
                                    st.subheader("📥 Download Individual Run Data")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Download generation data
                                        csv_data = logbook_df.to_csv(index=False)
                                        st.download_button(
                                            label="📊 Download Generation Data CSV",
                                            data=csv_data,
                                            file_name=f"{exp_config.get('experiment_name', 'experiment')}_run_{selected_run}_generations.csv",
                                            mime="text/csv"
                                        )
                                    
                                    with col2:
                                        # Download complete run data with config
                                        run_summary_data = {
                                            'Run_ID': selected_run,
                                            'Experiment_Name': exp_config.get('experiment_name'),
                                            'Best_Training_Fitness': run_result.get('best_training_fitness'),
                                            'Final_Test_Fitness': run_result.get('fitness_test', [])[-1] if run_result.get('fitness_test') and run_result['fitness_test'][-1] is not None else None,
                                            'Best_Phenotype': run_result.get('best_phenotype'),
                                            'Generations': len(run_result.get('max', [])),
                                            'Dataset': exp_config.get('dataset'),
                                            'Grammar': exp_config.get('grammar'),
                                            'Population': exp_config.get('population'),
                                            'Generations_Config': exp_config.get('generations'),
                                            'Crossover_Prob': exp_config.get('p_crossover'),
                                            'Mutation_Prob': exp_config.get('p_mutation'),
                                            'Elite_Size': exp_config.get('elite_size'),
                                            'Tournament_Size': exp_config.get('tournsize'),
                                            'HOF_Size': exp_config.get('halloffame_size'),
                                            'Max_Tree_Depth': exp_config.get('max_tree_depth'),
                                            'Min_Init_Tree_Depth': exp_config.get('min_init_tree_depth'),
                                            'Max_Init_Tree_Depth': exp_config.get('max_init_tree_depth'),
                                            'Min_Init_Genome_Length': exp_config.get('min_init_genome_length'),
                                            'Max_Init_Genome_Length': exp_config.get('max_init_genome_length'),
                                            'Codon_Size': exp_config.get('codon_size'),
                                            'Codon_Consumption': exp_config.get('codon_consumption'),
                                            'Genome_Representation': exp_config.get('genome_representation'),
                                            'Initialisation': exp_config.get('initialisation'),
                                            'Fitness_Metric': exp_config.get('fitness_metric'),
                                            'Random_Seed': exp_config.get('random_seed') + list(runs).index([r for r in runs if r.name == selected_run][0]),
                                            'Test_Size': exp_config.get('test_size'),
                                            'Created_At': exp_config.get('created_at')
                                        }
                                        
                                        run_summary_df = pd.DataFrame([run_summary_data])
                                        csv_run_summary = run_summary_df.to_csv(index=False)
                                        st.download_button(
                                            label="📋 Download Run Summary CSV",
                                            data=csv_run_summary,
                                            file_name=f"{exp_config.get('experiment_name', 'experiment')}_run_{selected_run}_summary.csv",
                                            mime="text/csv"
                                        )
            
            # Delete experiment
            if st.button("🗑️ Delete Experiment", type="secondary"):
                import shutil
                shutil.rmtree(EXPERIMENTS_DIR / exp_id, ignore_errors=True)
                st.success(f"Deleted experiment {exp_id}")
                st.rerun()
    else:
        st.info("No experiments found. Create your first experiment using the 'Run Experiment' page!")