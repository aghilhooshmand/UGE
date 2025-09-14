"""
GE Service for UGE Application

This module provides the core Grammatical Evolution service that wraps
the third-party libraries (grape, algorithms, functions) and provides
a clean interface for running GE experiments.

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
from grape import grape, algorithms, functions
from deap import creator, base, tools

# Import our models and utilities
from uge.models.experiment import ExperimentConfig, ExperimentResult
from uge.models.dataset import Dataset
from uge.models.grammar import Grammar
from uge.utils.helpers import fitness_eval
from uge.utils.logger import StreamlitLogger


class GEService:
    """
    Service for Grammatical Evolution operations.
    
    This service wraps the third-party GE libraries and provides a clean
    interface for running GE experiments. It handles all the complex
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
    
    def _setup_toolbox(self, config: ExperimentConfig, fitness_class) -> base.Toolbox:
        """
        Setup DEAP toolbox with all required operators.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            fitness_class: DEAP fitness class
            
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
        def fitness_wrapper(individual, points):
            return fitness_eval(individual, points, config.fitness_metric)
        
        # Register operators
        toolbox.register("evaluate", fitness_wrapper)
        toolbox.register("select", tools.selTournament, tournsize=config.tournsize)
        toolbox.register("mate", grape.crossover_onepoint)
        toolbox.register("mutate", grape.mutation_int_flip_per_codon)
        
        return toolbox
    
    def _create_population(self, toolbox: base.Toolbox, config: ExperimentConfig, 
                          bnf_grammar, live_placeholder=None) -> List:
        """
        Create initial population.
        
        Args:
            toolbox (base.Toolbox): DEAP toolbox
            config (ExperimentConfig): Experiment configuration
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
        Setup statistics collection.
        
        Returns:
            tools.Statistics: Configured statistics object
        """
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        return stats
    
    def run_experiment(self, config: ExperimentConfig, dataset: Dataset, 
                      grammar: Grammar, report_items: List[str],
                      live_placeholder=None) -> ExperimentResult:
        """
        Run a complete GE experiment.
        
        This method orchestrates the entire GE experiment process:
        1. Setup random seeds for reproducibility
        2. Prepare data and grammar
        3. Configure DEAP toolbox and operators
        4. Create initial population
        5. Run the evolutionary algorithm
        6. Collect and return results
        
        Args:
            config (ExperimentConfig): Experiment configuration
            dataset (Dataset): Dataset to use
            grammar (Grammar): BNF grammar to use
            report_items (List[str]): Items to include in reports
            live_placeholder: Optional Streamlit placeholder for live output
            
        Returns:
            ExperimentResult: Complete experiment results
            
        Raises:
            ValueError: If dataset or grammar cannot be loaded
            RuntimeError: If experiment execution fails
        """
        try:
            # Setup random seeds for reproducibility
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
            
            # Prepare data
            if config.dataset == 'processed.cleveland.data':
                X_train, Y_train, X_test, Y_test = dataset.preprocess_cleveland_data(config.random_seed)
            else:
                if not config.label_column:
                    raise ValueError('label_column not provided for CSV dataset')
                X_train, Y_train, X_test, Y_test = dataset.preprocess_csv_data(
                    config.label_column, config.test_size, config.random_seed
                )
            
            # Load grammar
            grammar.load()
            bnf_grammar = grape.Grammar(str(grammar.info.path.resolve()))
            
            # Setup fitness and individual classes
            fitness_class = self._create_fitness_class(config.fitness_direction)
            individual_class = self._create_individual_class(fitness_class)
            
            # Setup toolbox
            toolbox = self._setup_toolbox(config, fitness_class)
            
            # Create population
            population = self._create_population(toolbox, config, bnf_grammar, live_placeholder)
            
            # Setup hall of fame and statistics
            hof = tools.HallOfFame(config.halloffame_size)
            stats = self._setup_statistics()
            
            # Setup logging like UGE_ref
            logger = StreamlitLogger(live_placeholder) if live_placeholder else None
            
            # Run the evolutionary algorithm
            with (contextlib.redirect_stdout(logger) if logger else contextlib.nullcontext()):
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
            
            # Process results
            result = self._process_results(config, logbook, hof, report_items)
            
            if self.logger:
                self.logger(f"Experiment completed successfully. Best fitness: {result.best_training_fitness}")
            
            return result
            
        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            if live_placeholder:
                live_placeholder.code(f"ERROR: {error_msg}")
            if self.logger:
                self.logger(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _process_results(self, config: ExperimentConfig, logbook, hof, 
                        report_items: List[str]) -> ExperimentResult:
        """
        Process experiment results into ExperimentResult object.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            logbook: DEAP logbook with statistics
            hof: DEAP hall of fame
            report_items (List[str]): Report items
            
        Returns:
            ExperimentResult: Processed results
        """
        # Extract series from logbook
        available = set(logbook.header)
        series = {}
        for key in ['max', 'avg', 'min', 'std', 'fitness_test']:
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
        result = ExperimentResult(
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
            best_used_codons=best_used_codons
        )
        
        return result
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """
        Validate experiment configuration.
        
        Args:
            config (ExperimentConfig): Configuration to validate
            
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
        
        return warnings