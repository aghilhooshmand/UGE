"""
GE Service for UGE Application

This module provides the core Grammatical Evolution service that wraps
the third-party libraries (grape, algorithms, functions) and provides
a clean interface for running GE setups.

Classes:
- GEService: Main service for Grammatical Evolution operations

Author: UGE Team
"""

import numpy as np
import random
import contextlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path

# Import third-party libraries
from grape import grape, algorithms
from deap import creator, base, tools

# Import our models and utilities
from uge.models.setup import SetupConfig, SetupResult, GenerationConfig
from uge.models.dataset import Dataset
from uge.models.grammar import Grammar
from uge.utils.logger import StreamlitLogger
from uge.utils.constants import DEFAULT_CONFIG
from uge.utils.helpers import fitness_eval

# Import fitness evaluation functions directly to avoid circular imports
def mae(y, yhat):
    """Calculate Mean Absolute Error between true and predicted values."""
    return np.mean(np.abs(np.array(y) - np.array(yhat)))

def accuracy(y, yhat):
    """Calculate accuracy between true and predicted values."""
    compare = np.equal(y, yhat)
    return np.mean(compare)

def fitness_eval(individual, points, metric='mae'):
    """
    Evaluate the fitness of a Grammatical Evolution individual.
    
    This function is copied here to avoid circular import issues.
    """
    x = points[0]
    Y = points[1]
    
    # Check if individual is invalid
    if individual.invalid:
        return np.nan,
    
    try:
        # Simple evaluation context without operator service to avoid circular imports
        eval_context = {
            'x': x,  # Make the data available as 'x'
            'np': np,  # Make numpy available
        }
        
        # Evaluate the generated phenotype (Python code)
        pred = eval(individual.phenotype, eval_context)
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, NameError, SyntaxError, TypeError, ValueError):
        return np.nan,
    
    # Check if prediction is real-valued
    if not np.isrealobj(pred):
        return np.nan,
    
    try:
        # Convert predictions to binary classifications
        Y_class = [1 if pred[i] > 0 else 0 for i in range(len(Y))]
    except (IndexError, TypeError):
        return np.nan,
    
    # Calculate fitness based on metric
    if metric == 'mae':
        fitness_val = mae(Y, Y_class)
    elif metric == 'accuracy':
        fitness_val = accuracy(Y, Y_class)
    else:
        fitness_val = mae(Y, Y_class)  # Default to MAE
    
    return fitness_val,


class GEService:
    """
    Service for Grammatical Evolution operations.
    
    This service wraps the third-party GE libraries and provides a clean
    interface for running GE setups. It handles all the complex
    setup and execution of GE algorithms.
    
    Attributes:
        logger (Optional[Callable]): Optional logging function
    """
    
    def __init__(self, logger: Optional[Callable] = None):
        """
        Initialize GE service.
        
        Args:
            logger (Optional[Callable]): Optional logging function for output
        """
        self.logger = logger
        self._setup_deap_creators()
    
    def _setup_deap_creators(self):
        """Setup DEAP creator classes for fitness and individuals."""
        # Avoid multiple re-creation errors
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    def _create_fitness_class(self, fitness_direction: int):
        """
        Create appropriate fitness class based on fitness direction.
        
        Args:
            fitness_direction (int): Fitness direction (1 for maximize, -1 for minimize)
            
        Returns:
            DEAP fitness class
        """
        if fitness_direction == 1:  # maximize
            return creator.FitnessMax
        else:  # minimize
            return creator.FitnessMin
    
    def _create_individual_class(self, fitness_class):
        """
        Create individual class with specified fitness.
        
        Args:
            fitness_class: DEAP fitness class
            
        Returns:
            DEAP individual class
        """
        if not hasattr(creator, 'Individual'):
            creator.create('Individual', grape.Individual, fitness=fitness_class)
        return creator.Individual
    
    def _setup_toolbox(self, config: SetupConfig, fitness_class, bnf_grammar) -> base.Toolbox:
        """
        Setup DEAP toolbox with all required operators.
        
        Args:
            config (SetupConfig): Setup configuration
            fitness_class: DEAP fitness class
            bnf_grammar: BNF grammar for crossover operations
            
        Returns:
            base.Toolbox: Configured DEAP toolbox
        """
        toolbox = base.Toolbox()
        
        # Register population creator
        if config.initialisation == 'random':
            toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
        else:
            toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
        
        # Create fitness wrapper that uses the specified metric
        def fitness_wrapper(individual, *args, **kwargs):
            return fitness_eval(individual, self.current_points, config.fitness_metric)
        
        # Import partial for binding parameters
        from functools import partial
        
        # Create crossover wrapper
        def crossover_wrapper(ind1, ind2, *args, **kwargs):
            return grape.crossover_onepoint(ind1, ind2, bnf_grammar, config.max_tree_depth, config.codon_consumption)
        
        # Create mutation wrapper
        def mutation_wrapper(ind, *args, **kwargs):
            return grape.mutation_int_flip_per_codon(ind, config.p_mutation, config.codon_size, 
                                                   bnf_grammar, config.max_tree_depth, config.codon_consumption)
        
        # Register operators
        toolbox.register("evaluate", fitness_wrapper)
        toolbox.register("select", tools.selTournament, tournsize=config.tournsize)
        toolbox.register("mate", crossover_wrapper)
        toolbox.register("mutate", mutation_wrapper)
        
        return toolbox
    
    def _create_population(self, toolbox: base.Toolbox, config: SetupConfig, 
                          bnf_grammar, live_placeholder=None) -> List:
        """
        Create initial population.
        
        Args:
            toolbox (base.Toolbox): DEAP toolbox
            config (SetupConfig): Setup configuration
            bnf_grammar: grape.Grammar object
            live_placeholder: Optional placeholder for debug output
            
        Returns:
            List: Initial population
        """
        if config.initialisation == 'random':
            population = toolbox.populationCreator(
                pop_size=config.population,
                bnf_grammar=bnf_grammar,
                min_init_genome_length=config.min_init_genome_length,
                max_init_genome_length=config.max_init_genome_length,
                max_init_depth=config.max_tree_depth,
                codon_size=config.codon_size,
                codon_consumption=config.codon_consumption,
                genome_representation=config.genome_representation
            )
        else:
            population = toolbox.populationCreator(
                pop_size=config.population,
                bnf_grammar=bnf_grammar,
                min_init_depth=config.min_init_tree_depth,
                max_init_depth=config.max_init_tree_depth,
                codon_size=config.codon_size,
                codon_consumption=config.codon_consumption,
                genome_representation=config.genome_representation
            )
        
        return population
    
    def _setup_statistics(self) -> tools.Statistics:
        """
        Setup comprehensive statistics collection matching the standard algorithm.
        
        Returns:
            tools.Statistics: Configured statistics object with comprehensive metrics
        """
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        return stats
    
    def _generate_dynamic_parameter(self, param_config: dict, generation: int, random_seed: int) -> any:
        """
        Generate a dynamic parameter value based on configuration.
        
        Args:
            param_config (dict): Parameter configuration with mode, value, low, high, options
            generation (int): Current generation number
            random_seed (int): Base random seed
            
        Returns:
            any: Generated parameter value (int, float, or string)
        """
        import random
        
        if param_config['mode'] == 'fixed':
            return param_config['value']
        elif param_config['mode'] == 'dynamic':
            # Use generation-specific seed for reproducible randomness
            random.seed(random_seed + generation)
            
            # Check if this is a categorical parameter
            if param_config.get('options') is not None:
                # Categorical parameter - randomly select from options
                value = random.choice(param_config['options'])
                return value
            else:
                # Numerical parameter - generate random value within range
                if isinstance(param_config['low'], float):
                    # For float parameters (probabilities)
                    value = random.uniform(param_config['low'], param_config['high'])
                else:
                    # For integer parameters
                    value = random.randint(param_config['low'], param_config['high'])
                
                
                return value
        else:
            # Fallback to fixed value
            return param_config['value']

    def _create_generation_config(self, config: SetupConfig, generation: int, parameter_configs: dict = None) -> GenerationConfig:
        """
        Create generation-specific configuration using dynamic parameter system.
        
        Args:
            config (SetupConfig): Base setup configuration
            generation (int): Generation number
            parameter_configs (dict): Parameter configuration settings for each parameter
            
        Returns:
            GenerationConfig: Configuration for the specified generation
        """
        # If no parameter configs provided, use fixed values
        if not parameter_configs:
            dynamic_configs = {
                'elite_size': {'mode': 'fixed', 'value': config.elite_size},
                'p_crossover': {'mode': 'fixed', 'value': config.p_crossover},
                'p_mutation': {'mode': 'fixed', 'value': config.p_mutation},
                'tournsize': {'mode': 'fixed', 'value': config.tournsize},
                'halloffame_size': {'mode': 'fixed', 'value': config.halloffame_size},
                'max_tree_depth': {'mode': 'fixed', 'value': config.max_tree_depth},
                'min_init_tree_depth': {'mode': 'fixed', 'value': config.min_init_tree_depth},
                'max_init_tree_depth': {'mode': 'fixed', 'value': config.max_init_tree_depth},
                'min_init_genome_length': {'mode': 'fixed', 'value': config.min_init_genome_length},
                'max_init_genome_length': {'mode': 'fixed', 'value': config.max_init_genome_length},
                'codon_size': {'mode': 'fixed', 'value': config.codon_size},
                'codon_consumption': {'mode': 'fixed', 'value': config.codon_consumption},
                'genome_representation': {'mode': 'fixed', 'value': config.genome_representation},
                'initialisation': {'mode': 'fixed', 'value': config.initialisation}
            }
        
        # Generate dynamic parameters
        elite_size = self._generate_dynamic_parameter(
            parameter_configs.get('elite_size', {'mode': 'fixed', 'value': config.elite_size}),
            generation, config.random_seed
        )
        
        p_crossover = self._generate_dynamic_parameter(
            parameter_configs.get('p_crossover', {'mode': 'fixed', 'value': config.p_crossover}),
            generation, config.random_seed
        )
        
        p_mutation = self._generate_dynamic_parameter(
            parameter_configs.get('p_mutation', {'mode': 'fixed', 'value': config.p_mutation}),
            generation, config.random_seed
        )
        
        tournsize = self._generate_dynamic_parameter(
            parameter_configs.get('tournsize', {'mode': 'fixed', 'value': config.tournsize}),
            generation, config.random_seed
        )
        
        halloffame_size = self._generate_dynamic_parameter(
            parameter_configs.get('halloffame_size', {'mode': 'fixed', 'value': config.halloffame_size}),
            generation, config.random_seed
        )
        
        # Generate dynamic tree parameters
        max_tree_depth = self._generate_dynamic_parameter(
            parameter_configs.get('max_tree_depth', {'mode': 'fixed', 'value': config.max_tree_depth}),
            generation, config.random_seed
        )
        
        min_init_tree_depth = self._generate_dynamic_parameter(
            parameter_configs.get('min_init_tree_depth', {'mode': 'fixed', 'value': config.min_init_tree_depth}),
            generation, config.random_seed
        )
        
        max_init_tree_depth = self._generate_dynamic_parameter(
            parameter_configs.get('max_init_tree_depth', {'mode': 'fixed', 'value': config.max_init_tree_depth}),
            generation, config.random_seed
        )
        
        # Generate dynamic genome parameters
        min_init_genome_length = self._generate_dynamic_parameter(
            parameter_configs.get('min_init_genome_length', {'mode': 'fixed', 'value': config.min_init_genome_length}),
            generation, config.random_seed
        )
        
        max_init_genome_length = self._generate_dynamic_parameter(
            parameter_configs.get('max_init_genome_length', {'mode': 'fixed', 'value': config.max_init_genome_length}),
            generation, config.random_seed
        )
        
        codon_size = self._generate_dynamic_parameter(
            parameter_configs.get('codon_size', {'mode': 'fixed', 'value': config.codon_size}),
            generation, config.random_seed
        )
        
        # Generate dynamic categorical parameters
        codon_consumption = self._generate_dynamic_parameter(
            parameter_configs.get('codon_consumption', {'mode': 'fixed', 'value': config.codon_consumption}),
            generation, config.random_seed
        )
        
        genome_representation = self._generate_dynamic_parameter(
            parameter_configs.get('genome_representation', {'mode': 'fixed', 'value': config.genome_representation}),
            generation, config.random_seed
        )
        
        initialisation = self._generate_dynamic_parameter(
            parameter_configs.get('initialisation', {'mode': 'fixed', 'value': config.initialisation}),
            generation, config.random_seed
        )
        
        return GenerationConfig(
            generation=generation,
            population=config.population,
            p_crossover=p_crossover,
            p_mutation=p_mutation,
            elite_size=int(elite_size),  # Ensure integer for elite_size
            tournsize=int(tournsize),    # Ensure integer for tournsize
            halloffame_size=int(halloffame_size),  # Ensure integer for halloffame_size
            max_tree_depth=int(max_tree_depth),  # Ensure integer for max_tree_depth
            min_init_tree_depth=int(min_init_tree_depth),  # Ensure integer for min_init_tree_depth
            max_init_tree_depth=int(max_init_tree_depth),  # Ensure integer for max_init_tree_depth
            min_init_genome_length=int(min_init_genome_length),  # Ensure integer for min_init_genome_length
            max_init_genome_length=int(max_init_genome_length),  # Ensure integer for max_init_genome_length
            codon_size=int(codon_size),  # Ensure integer for codon_size
            codon_consumption=codon_consumption,
            genome_representation=genome_representation,
            initialisation=initialisation
        )
    
    
    
    def run_setup(self, config: SetupConfig, dataset: Dataset, 
                      grammar: Grammar, report_items: List[str],
                      parameter_configs: dict = None, live_placeholder=None) -> SetupResult:
        """
        Run a complete GE setup.
        
        This method orchestrates the entire GE setup process:
        1. Setup random seeds for reproducibility
        2. Prepare data and grammar
        3. Configure DEAP toolbox and operators
        4. Create initial population
        5. Run the evolutionary algorithm
        6. Collect and return results
        
        Args:
            config (SetupConfig): Setup configuration
            dataset (Dataset): Dataset to use
            grammar (Grammar): BNF grammar to use
            report_items (List[str]): Items to include in reports
            live_placeholder: Optional Streamlit placeholder for live output
            
        Returns:
            SetupResult: Complete setup results
            
        Raises:
            ValueError: If dataset or grammar cannot be loaded
            RuntimeError: If setup execution fails
        """
        try:
            # Setup random seeds for reproducibility
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
            
            # Prepare data - Cleveland dataset needs special preprocessing
            if config.dataset == 'processed.cleveland.data':
                X_train, Y_train, X_test, Y_test = dataset.preprocess_cleveland_data(config.random_seed)
            else:
                # Use 'class' as the default label column
                label_column = config.label_column or DEFAULT_CONFIG['label_column']
                X_train, Y_train, X_test, Y_test = dataset.preprocess_csv_data(
                    label_column, config.test_size, config.random_seed
                )
            
            # Load grammar
            grammar.load()
            bnf_grammar = grape.Grammar(str(grammar.info.path.resolve()))
            
            # Setup fitness and individual classes
            fitness_class = self._create_fitness_class(config.fitness_direction)
            individual_class = self._create_individual_class(fitness_class)
            
            # Set current points for fitness evaluation
            self.current_points = (X_train, Y_train)
            
            # Setup toolbox
            toolbox = self._setup_toolbox(config, fitness_class, bnf_grammar)
            
            # Create population
            population = self._create_population(toolbox, config, bnf_grammar, live_placeholder)
            
            # Setup hall of fame and statistics
            hof = tools.HallOfFame(config.halloffame_size)
            stats = self._setup_statistics()
            
            # Setup logging
            logger = StreamlitLogger(live_placeholder) if live_placeholder else None
            
            # Run the evolutionary algorithm with generation config tracking
            generation_configs = []
            
            # Check if we should track generation configurations
            track_configs = DEFAULT_CONFIG.get('track_generation_configs', True)
            
            with (contextlib.redirect_stdout(logger) if logger else contextlib.nullcontext()):
                # Check if we have any dynamic configurations
                has_dynamic = parameter_configs and any(
                    param_config.get('mode') == 'dynamic' 
                    for param_config in parameter_configs.values()
                )
                
                if has_dynamic:
                    # For dynamic parameters, we need to use initial values and track changes separately
                    # The algorithm will run with initial parameter values, but we'll track the dynamic changes
                    # This is a limitation of using the standard algorithm with dynamic parameters
                    if self.logger:
                        self.logger("Note: Dynamic parameters detected. Algorithm will use initial parameter values.")
                    
                    # Use initial parameter values for the algorithm
                    population, logbook = algorithms.ge_eaSimpleWithElitism(
                        population, toolbox, 
                        cxpb=config.p_crossover, 
                        mutpb=config.p_mutation,
                        ngen=config.generations, 
                        elite_size=config.elite_size,
                        bnf_grammar=bnf_grammar,
                        codon_size=config.codon_size,
                        max_tree_depth=config.max_tree_depth,
                        max_genome_length=config.max_init_genome_length,
                        points_train=[X_train, Y_train],
                        points_test=[X_test, Y_test],
                        codon_consumption=config.codon_consumption,
                        report_items=report_items,
                        genome_representation=config.genome_representation,
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True
                    )
                else:
                    # Use standard algorithm with fixed parameters
                    population, logbook = algorithms.ge_eaSimpleWithElitism(
                        population, toolbox, 
                        cxpb=config.p_crossover, 
                        mutpb=config.p_mutation,
                        ngen=config.generations, 
                        elite_size=config.elite_size,
                        bnf_grammar=bnf_grammar,
                        codon_size=config.codon_size,
                        max_tree_depth=config.max_tree_depth,
                        max_genome_length=config.max_init_genome_length,
                        points_train=[X_train, Y_train],
                        points_test=[X_test, Y_test],
                        codon_consumption=config.codon_consumption,
                        report_items=report_items,
                        genome_representation=config.genome_representation,
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True
                    )
            
            # Create generation configurations for all generations
            if track_configs:
                for gen in range(config.generations + 1):  # +1 to include generation 0
                    generation_configs.append(self._create_generation_config(config, gen, parameter_configs))
            
            # Process results
            result = self._process_results(config, logbook, hof, report_items, generation_configs)
            
            if self.logger:
                self.logger(f"Setup completed successfully. Best fitness: {result.best_training_fitness}")
            
            return result
            
        except Exception as e:
            error_msg = f"Setup failed: {str(e)}"
            if live_placeholder:
                live_placeholder.code(f"ERROR: {error_msg}")
            if self.logger:
                self.logger(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _process_results(self, config: SetupConfig, logbook, hof, 
                        report_items: List[str], generation_configs: List[GenerationConfig] = None) -> SetupResult:
        """
        Process setup results into SetupResult object.
        
        Args:
            config (SetupConfig): Setup configuration
            logbook: DEAP logbook with statistics
            hof: DEAP hall of fame
            report_items (List[str]): Report items
            generation_configs (List[GenerationConfig], optional): Generation configurations
            
        Returns:
            SetupResult: Processed results
        """
        # Extract series from logbook
        available = set(logbook.header)
        series = {}
        for key in ['max', 'avg', 'min', 'std', 'fitness_test', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std']:
            if key in available:
                series[key] = logbook.select(key)
            else:
                series[key] = []
        
        # Extract best individual information
        best_phenotype = None
        best_training_fitness = None
        best_depth = None
        best_genome_length = None
        best_used_codons = None
        
        if hof.items:
            best_individual = hof.items[0]
            best_phenotype = best_individual.phenotype
            best_training_fitness = float(best_individual.fitness.values[0])
            best_depth = int(best_individual.depth)
            best_genome_length = len(best_individual.genome)
            best_used_codons = float(best_individual.used_codons) / len(best_individual.genome)
        
        # Create result object
        result = SetupResult(
            config=config,
            report_items=report_items,
            max=list(map(float, series.get('max', []))) if series.get('max') else [],
            avg=list(map(float, series.get('avg', []))) if series.get('avg') else [],
            min=list(map(float, series.get('min', []))) if series.get('min') else [],
            std=list(map(float, series.get('std', []))) if series.get('std') else [],
            fitness_test=[float(x) if x == x else None for x in series.get('fitness_test', [])] 
                         if series.get('fitness_test') else [],
            best_phenotype=best_phenotype,
            best_training_fitness=best_training_fitness,
            best_depth=best_depth,
            best_genome_length=best_genome_length,
            best_used_codons=best_used_codons,
            invalid_count_min=list(map(int, series.get('invalid_count_min', []))) if series.get('invalid_count_min') else [],
            invalid_count_avg=list(map(float, series.get('invalid_count_avg', []))) if series.get('invalid_count_avg') else [],
            invalid_count_max=list(map(int, series.get('invalid_count_max', []))) if series.get('invalid_count_max') else [],
            invalid_count_std=list(map(float, series.get('invalid_count_std', []))) if series.get('invalid_count_std') else [],
            nodes_length_min=list(map(int, series.get('nodes_length_min', []))) if series.get('nodes_length_min') else [],
            nodes_length_avg=list(map(float, series.get('nodes_length_avg', []))) if series.get('nodes_length_avg') else [],
            nodes_length_max=list(map(int, series.get('nodes_length_max', []))) if series.get('nodes_length_max') else [],
            nodes_length_std=list(map(float, series.get('nodes_length_std', []))) if series.get('nodes_length_std') else [],
            generation_configs=generation_configs or []
        )
        
        return result
    
    def validate_config(self, config: SetupConfig) -> List[str]:
        """
        Validate setup configuration.
        
        Args:
            config (SetupConfig): Configuration to validate
            
        Returns:
            List[str]: List of validation warnings/errors
        """
        warnings = []
        
        # Check parameter ranges
        if config.population < 10:
            warnings.append("Population size too small (minimum 10)")
        if config.population > 5000:
            warnings.append("Population size very large (maximum 5000)")
        
        if config.generations < 1:
            warnings.append("Generations must be at least 1")
        if config.generations > 2000:
            warnings.append("Generations very large (maximum 2000)")
        
        if config.p_crossover < 0.0 or config.p_crossover > 1.0:
            warnings.append("Crossover probability must be between 0.0 and 1.0")
        
        if config.p_mutation < 0.0 or config.p_mutation > 1.0:
            warnings.append("Mutation probability must be between 0.0 and 1.0")
        
        if config.elite_size > config.population:
            warnings.append("Elite size cannot be larger than population size")
        
        if config.test_size <= 0.0 or config.test_size >= 1.0:
            warnings.append("Test size must be between 0.0 and 1.0")
        
        # Check fitness metric
        if config.fitness_metric not in ['mae', 'accuracy']:
            warnings.append("Fitness metric must be 'mae' or 'accuracy'")
        
        # Check fitness direction
        if config.fitness_direction not in [1, -1]:
            warnings.append("Fitness direction must be 1 (maximize) or -1 (minimize)")
        
        # Check evolution type
        valid_evolution_types = ['fixed', 'dynamic']
        if config.evolution_type not in valid_evolution_types:
            warnings.append(f"Evolution type must be one of: {valid_evolution_types}")
        
        # Check dynamic elite configuration
        if config.elite_dynamic_config:
            if not isinstance(config.elite_dynamic_config, dict):
                warnings.append("Elite dynamic config must be a dictionary")
            else:
                if 'low' not in config.elite_dynamic_config or 'high' not in config.elite_dynamic_config:
                    warnings.append("Elite dynamic config must contain 'low' and 'high' keys")
                else:
                    low = config.elite_dynamic_config['low']
                    high = config.elite_dynamic_config['high']
                    
                    if not isinstance(low, int) or not isinstance(high, int):
                        warnings.append("Elite dynamic config values must be integers")
                    elif low < 0:
                        warnings.append("Elite dynamic config 'low' must be >= 0")
                    elif high > config.population:
                        warnings.append("Elite dynamic config 'high' cannot be larger than population size")
                    elif low > high:
                        warnings.append("Elite dynamic config 'low' must be <= 'high'")
        
        return warnings