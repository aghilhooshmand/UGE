"""
Experiment Model for UGE Application

This module defines the Experiment data model, which represents a complete
Grammatical Evolution experiment with its configuration, results, and metadata.

Classes:
- Experiment: Main experiment data model
- ExperimentConfig: Configuration parameters for experiments
- ExperimentResult: Results and performance metrics

Author: UGE Team
"""

import datetime as dt
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """
    Configuration parameters for a Grammatical Evolution experiment.
    
    This class encapsulates all the parameters needed to run a GE experiment,
    including GA parameters, GE parameters, dataset settings, and other options.
    
    The configuration follows the MVC pattern where:
    - Models: Data structures (this class)
    - Views: User interface (forms, charts)
    - Controllers: Business logic (experiment execution)
    
    Attributes:
        experiment_name (str): Human-readable name for the experiment
        dataset (str): Name of the dataset to use (e.g., 'clinical_breast_cancer_RFC.csv')
        grammar (str): Name of the BNF grammar file (e.g., 'UGE_Classification.bnf')
        fitness_metric (str): Fitness metric to optimize ('mae' or 'accuracy')
        fitness_direction (int): Optimization direction (1 for maximize, -1 for minimize)
        n_runs (int): Number of independent runs to perform (typically 3-10)
        generations (int): Number of generations to evolve (typically 50-200)
        population (int): Population size (typically 100-500)
        p_crossover (float): Crossover probability (typically 0.7-0.9)
        p_mutation (float): Mutation probability (typically 0.1-0.3)
        elite_size (int): Number of elite individuals to preserve (typically 5-20)
        tournsize (int): Tournament size for selection (typically 3-7)
        halloffame_size (int): Size of hall of fame (typically 10-50)
        max_tree_depth (int): Maximum tree depth (prevents overfitting)
        min_init_tree_depth (int): Minimum initial tree depth
        max_init_tree_depth (int): Maximum initial tree depth
        min_init_genome_length (int): Minimum initial genome length
        max_init_genome_length (int): Maximum initial genome length
        codon_size (int): Codon size for genome representation (typically 255)
        codon_consumption (str): Codon consumption strategy ('lazy' or 'eager')
        genome_representation (str): Genome representation type ('list' or 'array')
        initialisation (str): Initialization strategy ('sensible' or 'random')
        random_seed (int): Random seed for reproducibility
        label_column (str): Name of the target/label column in dataset
        test_size (float): Proportion of data for testing (typically 0.2-0.3)
        report_items (List[str]): Metrics to track during evolution
        created_at (str): ISO timestamp of configuration creation
    """
    
    # Experiment metadata
    experiment_name: str
    dataset: str
    grammar: str
    fitness_metric: str = 'mae'
    fitness_direction: int = -1  # -1 for minimize (MAE), 1 for maximize (accuracy)
    n_runs: int = 3
    created_at: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    
    # GA Parameters
    generations: int = 200
    population: int = 500
    p_crossover: float = 0.8
    p_mutation: float = 0.01
    elite_size: int = 1
    tournsize: int = 7
    halloffame_size: int = 1
    
    # GE/GRAPE Parameters
    max_tree_depth: int = 35
    min_init_tree_depth: int = 4
    max_init_tree_depth: int = 13
    min_init_genome_length: int = 95
    max_init_genome_length: int = 115
    codon_size: int = 255
    codon_consumption: str = 'lazy'
    genome_representation: str = 'list'
    initialisation: str = 'sensible'
    
    # Dataset Parameters
    random_seed: int = 42
    label_column: Optional[str] = None
    test_size: float = 0.3
    report_items: List[str] = field(default_factory=lambda: [
        'gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
        'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes',
        'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons',
        'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'experiment_name': self.experiment_name,
            'dataset': self.dataset,
            'grammar': self.grammar,
            'fitness_metric': self.fitness_metric,
            'fitness_direction': self.fitness_direction,
            'n_runs': self.n_runs,
            'generations': self.generations,
            'population': self.population,
            'p_crossover': self.p_crossover,
            'p_mutation': self.p_mutation,
            'elite_size': self.elite_size,
            'tournsize': self.tournsize,
            'halloffame_size': self.halloffame_size,
            'max_tree_depth': self.max_tree_depth,
            'min_init_tree_depth': self.min_init_tree_depth,
            'max_init_tree_depth': self.max_init_tree_depth,
            'min_init_genome_length': self.min_init_genome_length,
            'max_init_genome_length': self.max_init_genome_length,
            'codon_size': self.codon_size,
            'codon_consumption': self.codon_consumption,
            'genome_representation': self.genome_representation,
            'initialisation': self.initialisation,
            'random_seed': self.random_seed,
            'label_column': self.label_column,
            'test_size': self.test_size,
            'report_items': self.report_items,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """
    Results from a single GE experiment run.
    
    This class encapsulates all the results and performance metrics
    from a single run of a Grammatical Evolution experiment.
    
    Attributes:
        config (ExperimentConfig): Configuration used for this run
        report_items (List[str]): Items included in the report
        max (List[float]): Maximum fitness values per generation
        avg (List[float]): Average fitness values per generation
        min (List[float]): Minimum fitness values per generation
        std (List[float]): Standard deviation of fitness per generation
        fitness_test (List[Optional[float]]): Test fitness values per generation
        best_phenotype (Optional[str]): Best individual's phenotype
        best_training_fitness (Optional[float]): Best training fitness achieved
        best_depth (Optional[int]): Depth of best individual
        best_genome_length (Optional[int]): Genome length of best individual
        best_used_codons (Optional[float]): Used codons ratio of best individual
        timestamp (str): ISO timestamp of result generation
    """
    
    config: ExperimentConfig
    report_items: List[str]
    max: List[float] = field(default_factory=list)
    avg: List[float] = field(default_factory=list)
    min: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    fitness_test: List[Optional[float]] = field(default_factory=list)
    best_phenotype: Optional[str] = None
    best_training_fitness: Optional[float] = None
    best_depth: Optional[int] = None
    best_genome_length: Optional[int] = None
    best_used_codons: Optional[float] = None
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'config': self.config.to_dict(),
            'report_items': self.report_items,
            'max': self.max,
            'avg': self.avg,
            'min': self.min,
            'std': self.std,
            'fitness_test': self.fitness_test,
            'best_phenotype': self.best_phenotype,
            'best_training_fitness': self.best_training_fitness,
            'best_depth': self.best_depth,
            'best_genome_length': self.best_genome_length,
            'best_used_codons': self.best_used_codons,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create result from dictionary."""
        config = ExperimentConfig.from_dict(data['config'])
        return cls(
            config=config,
            report_items=data['report_items'],
            max=data.get('max', []),
            avg=data.get('avg', []),
            min=data.get('min', []),
            std=data.get('std', []),
            fitness_test=data.get('fitness_test', []),
            best_phenotype=data.get('best_phenotype'),
            best_training_fitness=data.get('best_training_fitness'),
            best_depth=data.get('best_depth'),
            best_genome_length=data.get('best_genome_length'),
            best_used_codons=data.get('best_used_codons'),
            timestamp=data.get('timestamp', dt.datetime.now(dt.timezone.utc).isoformat())
        )


@dataclass
class Experiment:
    """
    Complete experiment data model.
    
    This class represents a complete Grammatical Evolution experiment,
    including its configuration, all run results, and metadata.
    
    Attributes:
        id (str): Unique experiment identifier
        config (ExperimentConfig): Experiment configuration
        results (Dict[str, ExperimentResult]): Results from each run
        status (str): Current status of the experiment
        created_at (str): ISO timestamp of creation
        completed_at (Optional[str]): ISO timestamp of completion
    """
    
    id: str
    config: ExperimentConfig
    results: Dict[str, ExperimentResult] = field(default_factory=dict)
    status: str = 'created'  # created, running, completed, failed
    created_at: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def add_result(self, run_id: str, result: ExperimentResult) -> None:
        """Add a result from a single run."""
        self.results[run_id] = result
    
    def get_best_result(self) -> Optional[ExperimentResult]:
        """Get the best result across all runs."""
        if not self.results:
            return None
        
        # Find the result with the best training fitness based on fitness direction
        best_result = None
        if self.config.fitness_direction == 1:  # maximize
            best_fitness = float('-inf')
            for result in self.results.values():
                if result.best_training_fitness is not None:
                    if result.best_training_fitness > best_fitness:
                        best_fitness = result.best_training_fitness
                        best_result = result
        else:  # minimize (fitness_direction == -1)
            best_fitness = float('inf')
            for result in self.results.values():
                if result.best_training_fitness is not None:
                    if result.best_training_fitness < best_fitness:
                        best_fitness = result.best_training_fitness
                        best_result = result
        
        return best_result
    
    def get_average_fitness(self) -> Optional[float]:
        """Get the average best fitness across all runs."""
        if not self.results:
            return None
        
        fitnesses = [r.best_training_fitness for r in self.results.values() 
                    if r.best_training_fitness is not None]
        
        if not fitnesses:
            return None
        
        return sum(fitnesses) / len(fitnesses)
    
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return len(self.results) >= self.config.n_runs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary format."""
        return {
            'id': self.id,
            'config': self.config.to_dict(),
            'results': {run_id: result.to_dict() for run_id, result in self.results.items()},
            'status': self.status,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        config = ExperimentConfig.from_dict(data['config'])
        results = {run_id: ExperimentResult.from_dict(result_data) 
                  for run_id, result_data in data.get('results', {}).items()}
        
        return cls(
            id=data['id'],
            config=config,
            results=results,
            status=data.get('status', 'created'),
            created_at=data.get('created_at', dt.datetime.now().isoformat()),
            completed_at=data.get('completed_at')
        )