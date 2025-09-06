import os
import json
import time
import uuid
import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ---------- Helpers ----------

def list_datasets():
    ds_dir = CURRENT_DIR / "datasets"
    return [p.name for p in ds_dir.glob('*') if p.is_file() and p.suffix in {'.data', '.csv', ''}]

def list_grammars():
    g_dir = CURRENT_DIR / "grammars"
    return [p.name for p in g_dir.glob('*.bnf')]

def load_dataset(name: str, random_seed: int):
    """Load dataset and return X, y, feature_names"""
    ds_path = CURRENT_DIR / "datasets" / name
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset {name} not found")
    
    if name.endswith('.data'):
        # Cleveland heart disease format
        df = pd.read_csv(ds_path, header=None, na_values='?')
        df = df.dropna()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # Convert to binary classification (0, 1)
        y = (y > 0).astype(int)
        feature_names = [f'x[{i}]' for i in range(X.shape[1])]
    else:
        # CSV format
        df = pd.read_csv(ds_path)
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = [f'x[{i}]' for i in range(X.shape[1])]
    
    return X, y, feature_names

def load_grammar(name: str):
    """Load grammar file"""
    g_path = CURRENT_DIR / "grammars" / name
    if not g_path.exists():
        raise FileNotFoundError(f"Grammar {name} not found")
    return g_path.read_text()

def create_experiment_id():
    """Create unique experiment ID"""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def create_run_id():
    """Create unique run ID"""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

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
    return [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()]

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

# ---------- Fitness Functions ----------

def mae(individual, X_train, y_train, X_test, y_test, grammar, feature_names):
    """Mean Absolute Error fitness function"""
    try:
        phenotype = grape.mapper_lazy(individual, grammar, 200)
        if not phenotype:
            return (float('inf'),)
        
        # Evaluate on training set
        train_pred = []
        for i in range(len(X_train)):
            try:
                # Create local namespace with feature values
                local_vars = {name: X_train[i, j] for j, name in enumerate(feature_names)}
                local_vars.update(globals())
                pred = eval(phenotype, {"__builtins__": {}}, local_vars)
                train_pred.append(pred)
            except:
                train_pred.append(0.0)
        
        train_pred = np.array(train_pred)
        train_mae = np.mean(np.abs(train_pred - y_train))
        
        # Evaluate on test set
        test_pred = []
        for i in range(len(X_test)):
            try:
                local_vars = {name: X_test[i, j] for j, name in enumerate(feature_names)}
                local_vars.update(globals())
                pred = eval(phenotype, {"__builtins__": {}}, local_vars)
                test_pred.append(pred)
            except:
                test_pred.append(0.0)
        
        test_pred = np.array(test_pred)
        test_mae = np.mean(np.abs(test_pred - y_test))
        
        return (train_mae,), test_mae
        
    except Exception as e:
        return (float('inf'),), float('inf')

def accuracy(individual, X_train, y_train, X_test, y_test, grammar, feature_names):
    """Accuracy fitness function"""
    try:
        phenotype = grape.mapper_lazy(individual, grammar, 200)
        if not phenotype:
            return (0.0,)
        
        # Evaluate on training set
        train_pred = []
        for i in range(len(X_train)):
            try:
                local_vars = {name: X_train[i, j] for j, name in enumerate(feature_names)}
                local_vars.update(globals())
                pred = eval(phenotype, {"__builtins__": {}}, local_vars)
                train_pred.append(1 if pred > 0.5 else 0)
            except:
                train_pred.append(0)
        
        train_pred = np.array(train_pred)
        train_acc = np.mean(train_pred == y_train)
        
        # Evaluate on test set
        test_pred = []
        for i in range(len(X_test)):
            try:
                local_vars = {name: X_test[i, j] for j, name in enumerate(feature_names)}
                local_vars.update(globals())
                pred = eval(phenotype, {"__builtins__": {}}, local_vars)
                test_pred.append(1 if pred > 0.5 else 0)
            except:
                test_pred.append(0)
        
        test_pred = np.array(test_pred)
        test_acc = np.mean(test_pred == y_test)
        
        return (train_acc,), test_acc
        
    except Exception as e:
        return (0.0,), 0.0

# ---------- Main GE Function ----------

def run_ge_classification(config, report_items, live_placeholder=None):
    """Run GE classification with live progress updates"""
    
    # Load data
    X, y, feature_names = load_dataset(config['dataset'], config['random_seed'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_seed'], stratify=y
    )
    
    # Load grammar
    grammar_text = load_grammar(config['grammar'])
    grammar = grape.Grammar(grammar_text)
    
    # Create fitness function
    fitness_metric = config.get('fitness_metric', 'mae')
    if fitness_metric == 'accuracy':
        fitness_func = accuracy
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        fitness_class = creator.FitnessMax
    else:
        fitness_func = mae
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        fitness_class = creator.FitnessMin
    
    # Create individual class
    creator.create("Individual", list, fitness=fitness_class)
    
    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", grape.random_initialisation, creator.Individual, 
                    grammar, config['min_init_genome_length'], config['max_init_genome_length'])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_func, X_train=X_train, y_train=y_train, 
                    X_test=X_test, y_test=y_test, grammar=grammar, feature_names=feature_names)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon, 
                    indpb=config['p_mutation'])
    toolbox.register("select", tools.selTournament, tournsize=config['tournsize'])
    
    # Create population
    population = toolbox.population(n=config['population'])
    
    # Create hall of fame
    hof = tools.HallOfFame(config['halloffame_size'])
    
    # Create statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run algorithm with live updates
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Initialize
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit[0]  # Training fitness
        ind.test_fitness = fit[1]    # Test fitness
    
    hof.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    
    # Add test fitness to logbook
    if hof:
        logbook.record(gen=0, fitness_test=hof[0].test_fitness)
    
    # Evolution loop
    for gen in range(1, config['generations'] + 1):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config['p_crossover']:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < config['p_mutation']:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.test_fitness = fit[1]
        
        # Update hall of fame
        hof.update(offspring)
        
        # Replace population
        population[:] = offspring
        
        # Record statistics
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        # Add test fitness
        if hof:
            logbook.record(gen=gen, fitness_test=hof[0].test_fitness)
        
        # Live update
        if live_placeholder and gen % 10 == 0:
            with live_placeholder.container():
                st.write(f"Generation {gen}/{config['generations']} - Best: {hof[0].fitness.values[0]:.4f}")
    
    # Prepare result
    result = {
        'logbook': logbook,
        'hof': hof,
        'best_individual': hof[0] if hof else None,
        'best_phenotype': grape.mapper_lazy(hof[0], grammar, 200) if hof else None,
        'best_training_fitness': hof[0].fitness.values[0] if hof else None,
        'best_test_fitness': hof[0].test_fitness if hof else None,
        'config': config,
        'fitness_metric': fitness_metric
    }
    
    # Convert logbook to dict for JSON serialization
    logbook_dict = {}
    for field in logbook.header:
        logbook_dict[field] = [record[field] for record in logbook]
    result['logbook_dict'] = logbook_dict
    
    return result

# ---------- Plotting Functions ----------

def plot_across_generations(data, title="Fitness Across Generations", ylabel="Fitness"):
    """Plot fitness across generations for one or more runs"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if isinstance(data, dict) and 'logbook_dict' in data:
        # Single run
        logbook = data['logbook_dict']
        gens = logbook.get('gen', [])
        max_vals = logbook.get('max', [])
        avg_vals = logbook.get('avg', [])
        
        if gens and max_vals:
            ax.plot(gens, max_vals, label='Best', color='blue', linewidth=2)
        if gens and avg_vals:
            ax.plot(gens, avg_vals, label='Average', color='red', linewidth=2)
    else:
        # Multiple runs
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for i, (run_name, run_data) in enumerate(data.items()):
            if 'logbook_dict' in run_data:
                logbook = run_data['logbook_dict']
                gens = logbook.get('gen', [])
                max_vals = logbook.get('max', [])
                if gens and max_vals:
                    ax.plot(gens, max_vals, label=f'{run_name} - Best', color=colors[i], linewidth=2)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_across_runs(data, title="Fitness Across Runs", ylabel="Fitness"):
    """Plot final fitness across runs for one or more experiments"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if isinstance(data, dict) and 'runs' in data:
        # Single experiment
        runs = data['runs']
        run_names = list(runs.keys())
        final_fitness = [runs[run].get('best_training_fitness', 0) for run in run_names]
        
        ax.bar(range(len(run_names)), final_fitness, alpha=0.7)
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels([f'Run {i+1}' for i in range(len(run_names))], rotation=45)
    else:
        # Multiple experiments
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for i, (exp_name, exp_data) in enumerate(data.items()):
            if 'runs' in exp_data:
                runs = exp_data['runs']
                run_names = list(runs.keys())
                final_fitness = [runs[run].get('best_training_fitness', 0) for run in run_names]
                ax.plot(range(len(run_names)), final_fitness, 
                       label=f'{exp_name}', color=colors[i], marker='o', linewidth=2)
    
    ax.set_xlabel('Run Number')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_across_experiments(data, title="Fitness Across Experiments", ylabel="Fitness"):
    """Plot average final fitness across experiments"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    exp_names = list(data.keys())
    avg_fitness = []
    
    for exp_name, exp_data in data.items():
        if 'runs' in exp_data:
            runs = exp_data['runs']
            final_fitness = [runs[run].get('best_training_fitness', 0) for run in runs.keys()]
            avg_fitness.append(np.mean(final_fitness) if final_fitness else 0)
        else:
            avg_fitness.append(0)
    
    bars = ax.bar(range(len(exp_names)), avg_fitness, alpha=0.7)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_fitness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ---------- Streamlit App ----------

st.set_page_config(page_title="UGE - Grammatical Evolution", layout="wide")

st.title("🧬 UGE - Grammatical Evolution Platform")
st.markdown("**Experiment-based GE with flexible visualization options**")

# Sidebar for experiment creation
with st.sidebar:
    st.header("🔬 Create New Experiment")
    
    # Experiment basic info
    exp_name = st.text_input("Experiment Name", value=f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    # Dataset selection
    datasets = list_datasets()
    dataset = st.selectbox("Dataset", datasets, help=HELP.get('dataset', ''))
    
    # Grammar selection
    grammars = list_grammars()
    grammar = st.selectbox("Grammar", grammars, help=HELP.get('grammar', ''))
    
    # Fitness metric
    fitness_metric = st.selectbox("Fitness Metric", ['mae', 'accuracy'], 
                                 help=HELP.get('fitness_metric', ''))
    
    # Experiment parameters
    st.subheader("Experiment Parameters")
    n_runs = st.number_input("Number of Runs", min_value=1, max_value=20, value=3,
                            help=HELP.get('n_runs', ''))
    generations = st.number_input("Generations per Run", min_value=10, max_value=1000, value=100,
                                 help=HELP.get('generations', ''))
    
    # GA Parameters
    st.subheader("GA Parameters")
    population = st.number_input("Population Size", min_value=10, max_value=1000, value=200,
                                help=HELP.get('population', ''))
    p_crossover = st.slider("Crossover Probability", 0.0, 1.0, 0.7, 0.05,
                           help=HELP.get('p_crossover', ''))
    p_mutation = st.slider("Mutation Probability", 0.0, 1.0, 0.1, 0.01,
                          help=HELP.get('p_mutation', ''))
    elite_size = st.number_input("Elite Size", min_value=1, max_value=50, value=1,
                                help=HELP.get('elite_size', ''))
    tournsize = st.number_input("Tournament Size", min_value=2, max_value=20, value=3,
                               help=HELP.get('tournsize', ''))
    halloffame_size = st.number_input("Hall of Fame Size", min_value=1, max_value=50, value=1,
                                     help=HELP.get('halloffame_size', ''))
    
    # GE Parameters
    st.subheader("GE Parameters")
    max_tree_depth = st.number_input("Max Tree Depth", min_value=5, max_value=50, value=17,
                                    help=HELP.get('max_tree_depth', ''))
    min_init_tree_depth = st.number_input("Min Init Tree Depth", min_value=2, max_value=10, value=2,
                                         help=HELP.get('min_init_tree_depth', ''))
    max_init_tree_depth = st.number_input("Max Init Tree Depth", min_value=2, max_value=10, value=6,
                                         help=HELP.get('max_init_tree_depth', ''))
    min_init_genome_length = st.number_input("Min Init Genome Length", min_value=10, max_value=200, value=20,
                                            help=HELP.get('min_init_genome_length', ''))
    max_init_genome_length = st.number_input("Max Init Genome Length", min_value=10, max_value=200, value=100,
                                            help=HELP.get('max_init_genome_length', ''))
    codon_size = st.number_input("Codon Size", min_value=8, max_value=32, value=8,
                                help=HELP.get('codon_size', ''))
    codon_consumption = st.selectbox("Codon Consumption", ['lazy', 'eager'],
                                    help=HELP.get('codon_consumption', ''))
    genome_representation = st.selectbox("Genome Representation", ['list', 'array'],
                                        help=HELP.get('genome_representation', ''))
    initialisation = st.selectbox("Initialisation", ['random', 'sensible'],
                                 help=HELP.get('initialisation', ''))
    
    # Data Parameters
    st.subheader("Data Parameters")
    random_seed = st.number_input("Random Seed", min_value=0, max_value=10000, value=42,
                                 help=HELP.get('random_seed', ''))
    label_column = st.text_input("Label Column", value="", help=HELP.get('label_column', ''))
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, help=HELP.get('test_size', ''))
    
    # Report items
    st.subheader("Report Items")
    report_items = st.multiselect("Select metrics to track", 
                                 ['max', 'avg', 'min', 'std', 'fitness_test'],
                                 default=['max', 'avg', 'fitness_test'],
                                 help=HELP.get('report_items', ''))
    
    # Run experiment button
    run_experiment = st.button("🚀 Run Experiment", type="primary")

# Main content area
if run_experiment:
    # Create experiment
    exp_id = create_experiment_id()
    
    # Prepare configuration
    config = {
        'experiment_name': exp_name,
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
        'created_at': datetime.now().isoformat()
    }
    
    # Save experiment config
    exp_dir = save_experiment_config(exp_id, config)
    
    st.success(f"Created experiment: {exp_name} (ID: {exp_id})")
    
    # Run all runs for this experiment
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_run_results = {}
    
    for run_idx in range(int(n_runs)):
        status_text.text(f"Running {run_name} - Run {run_idx+1}/{n_runs}")
        
        # Update config for this run
        run_config = config.copy()
        run_config['random_seed'] = int(random_seed) + run_idx
        
        # Run GE
        run_id = create_run_id()
        result = run_ge_classification(run_config, report_items)
        
        # Save run result
        save_run_result(exp_id, run_id, result)
        all_run_results[run_id] = result
        
        # Update progress
        progress_bar.progress((run_idx + 1) / int(n_runs))
    
    status_text.text("✅ Experiment completed!")
    
    # Show experiment summary
    st.subheader(f"📊 Experiment Results: {exp_name}")
    
    # Final fitness across runs
    final_fitness = [result['best_training_fitness'] for result in all_run_results.values()]
    st.write(f"**Final Fitness Range:** {min(final_fitness):.4f} - {max(final_fitness):.4f}")
    st.write(f"**Average Final Fitness:** {np.mean(final_fitness):.4f} ± {np.std(final_fitness):.4f}")

# Experiment Management
st.header("📁 Experiment Management")

experiments = list_experiments()
if experiments:
    # Experiment selection
    exp_options = {f"{exp.name} ({exp.name.split('_')[-1]})": exp.name for exp in experiments}
    selected_exp = st.selectbox("Select Experiment", list(exp_options.keys()))
    exp_id = exp_options[selected_exp]
    
    # Load experiment config
    exp_config = load_experiment_config(exp_id)
    if exp_config:
        st.subheader(f"🔬 {exp_config.get('experiment_name', exp_id)}")
        
        # Show experiment info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset", exp_config.get('dataset', 'N/A'))
            st.metric("Grammar", exp_config.get('grammar', 'N/A'))
            st.metric("Fitness Metric", exp_config.get('fitness_metric', 'N/A'))
        with col2:
            st.metric("Runs", exp_config.get('n_runs', 'N/A'))
            st.metric("Generations", exp_config.get('generations', 'N/A'))
            st.metric("Population", exp_config.get('population', 'N/A'))
        with col3:
            st.metric("Crossover", exp_config.get('p_crossover', 'N/A'))
            st.metric("Mutation", exp_config.get('p_mutation', 'N/A'))
            st.metric("Created", exp_config.get('created_at', 'N/A')[:10])
        
        # Load all runs for this experiment
        runs = list_experiment_runs(exp_id)
        if runs:
            st.subheader("📈 Visualization Options")
            
            # Chart type selection
            chart_type = st.selectbox("Chart Type", 
                                    ["Across Generations", "Across Runs", "Across Experiments"],
                                    help="Choose how to visualize the results")
            
            # Metric selection
            available_metrics = ['max', 'avg', 'min', 'std', 'fitness_test']
            selected_metrics = st.multiselect("Select Metrics", available_metrics, 
                                            default=['max', 'fitness_test'])
            
            if selected_metrics:
                # Load run data
                runs_data = {}
                for run in runs:
                    run_result = load_run_result(exp_id, run.name)
                    if run_result:
                        runs_data[run.name] = run_result
                
                if chart_type == "Across Generations":
                    # Show individual run plots
                    for run_name, run_data in runs_data.items():
                        with st.expander(f"Run: {run_name}"):
                            fig = plot_across_generations(run_data, 
                                                        f"Fitness Across Generations - {run_name}",
                                                        "Fitness (lower is better)" if exp_config.get('fitness_metric') == 'mae' else "Fitness (higher is better)")
                            st.pyplot(fig)
                
                elif chart_type == "Across Runs":
                    # Show comparison across runs
                    fig = plot_across_runs({exp_id: {'runs': runs_data}}, 
                                         f"Final Fitness Across Runs - {exp_config.get('experiment_name')}",
                                         "Fitness (lower is better)" if exp_config.get('fitness_metric') == 'mae' else "Fitness (higher is better)")
                    st.pyplot(fig)
                
                elif chart_type == "Across Experiments":
                    # Show comparison across experiments
                    all_experiments_data = {}
                    for exp in experiments:
                        exp_runs = list_experiment_runs(exp.name)
                        exp_runs_data = {}
                        for run in exp_runs:
                            run_result = load_run_result(exp.name, run.name)
                            if run_result:
                                exp_runs_data[run.name] = run_result
                        if exp_runs_data:
                            all_experiments_data[exp.name] = {'runs': exp_runs_data}
                    
                    if len(all_experiments_data) > 1:
                        fig = plot_across_experiments(all_experiments_data, 
                                                    "Average Final Fitness Across Experiments",
                                                    "Fitness (lower is better)" if exp_config.get('fitness_metric') == 'mae' else "Fitness (higher is better)")
                        st.pyplot(fig)
                    else:
                        st.info("Need at least 2 experiments to compare across experiments")
            
            # Run details
            st.subheader("🔍 Run Details")
            selected_run = st.selectbox("Select Run", [run.name for run in runs])
            
            if selected_run:
                run_result = load_run_result(exp_id, selected_run)
                if run_result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Best Training Fitness", f"{run_result.get('best_training_fitness', 'N/A'):.4f}")
                        st.metric("Best Test Fitness", f"{run_result.get('best_test_fitness', 'N/A'):.4f}")
                    
                    with col2:
                        st.metric("Fitness Metric", run_result.get('fitness_metric', 'N/A'))
                        st.metric("Generations", len(run_result.get('logbook_dict', {}).get('gen', [])))
                    
                    # Best individual
                    st.subheader("Best Individual (Phenotype)")
                    st.code(run_result.get('best_phenotype', '<none>'))
                    
                    # CSV data
                    if 'logbook_dict' in run_result:
                        st.subheader("Generation Data")
                        logbook_df = pd.DataFrame(run_result['logbook_dict'])
                        st.dataframe(logbook_df, width='stretch')
        
        # Delete experiment
        if st.button("🗑️ Delete Experiment", type="secondary"):
            import shutil
            shutil.rmtree(EXPERIMENTS_DIR / exp_id, ignore_errors=True)
            st.success(f"Deleted experiment {exp_id}")
            st.experimental_rerun()

else:
    st.info("No experiments found. Create your first experiment using the sidebar!")

# Footer
st.markdown("---")
st.markdown("**UGE - Grammatical Evolution Platform** | Built with Streamlit")