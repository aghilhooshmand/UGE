"""
Setup Model for UGE Application

This module defines the Setup data model, which represents a complete
Grammatical Evolution setup with its configuration, results, and metadata.

Classes:
- Setup: Main setup data model
- SetupConfig: Configuration parameters for setups
- SetupResult: Results and performance metrics

Author: UGE Team
"""

import datetime as dt
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SetupConfig:
    """
    Configuration parameters for a Grammatical Evolution setup.
    
    This class encapsulates all the parameters needed to run a GE setup,
    including GA parameters, GE parameters, dataset settings, and other options.
    
    The configuration follows the MVC pattern where:
    - Models: Data structures (this class)
    - Views: User interface (forms, charts)
    - Controllers: Business logic (setup execution)
    
    Attributes:
        setup_name (str): Human-readable name for the setup
        dataset (str): Name of the dataset to use (e.g., 'clinical_breast_cancer_RFC.csv')
        grammar (str): Name of the BNF grammar file (e.g., 'UGE_Classification.bnf')
        fitness_metric (str): Fitness metric to optimize ('mae' or 'accuracy')
        fitness_direction (int): Optimization direction (1 for maximize, -1 for minimize)
        n_runs (int): Number of independent runs to perform (typically 3-10)
        evolution_type (str): Evolution type - 'fixed' (same config all generations) or 'dynamic' (config can change per generation)
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
    
    # Setup metadata
    setup_name: str
    dataset: str
    grammar: str
    fitness_metric: str = 'mae'
    fitness_direction: int = field(default=-1, init=False)  # Will be set in __post_init__
    n_runs: int = 3
    evolution_type: str = 'fixed'  # 'fixed' or 'dynamic'
    created_at: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    
    def __post_init__(self):
        """Set fitness_direction based on fitness_metric after initialization."""
        if self.fitness_metric == 'mae':
            self.fitness_direction = -1  # minimize (lower is better)
        elif self.fitness_metric == 'accuracy':
            self.fitness_direction = 1   # maximize (higher is better)
        else:
            self.fitness_direction = -1  # default to minimize
    
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
            'setup_name': self.setup_name,
            'dataset': self.dataset,
            'grammar': self.grammar,
            'fitness_metric': self.fitness_metric,
            'fitness_direction': self.fitness_direction,
            'n_runs': self.n_runs,
            'evolution_type': self.evolution_type,
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
    def from_dict(cls, data: Dict[str, Any]) -> 'SetupConfig':
        """Create configuration from dictionary."""
        # Remove fitness_direction since it's set automatically in __post_init__
        data_copy = data.copy()
        data_copy.pop('fitness_direction', None)
        return cls(**data_copy)


@dataclass
class GenerationConfig:
    """
    Configuration parameters for a specific generation.
    
    This class tracks configuration parameters that may vary per generation.
    Currently, all parameters are the same across generations, but this
    structure allows for future dynamic configuration changes.
    
    Attributes:
        generation (int): Generation number (0-based)
        population (int): Population size for this generation
        p_crossover (float): Crossover probability for this generation
        p_mutation (float): Mutation probability for this generation
        elite_size (int): Elite size for this generation
        tournsize (int): Tournament size for this generation
        halloffame_size (int): Hall of fame size for this generation
        max_tree_depth (int): Maximum tree depth for this generation
        codon_size (int): Codon size for this generation
        codon_consumption (str): Codon consumption strategy for this generation
        genome_representation (str): Genome representation for this generation
        timestamp (str): ISO timestamp when this generation config was recorded
    """
    
    generation: int
    population: int
    p_crossover: float
    p_mutation: float
    elite_size: int
    tournsize: int
    halloffame_size: int
    max_tree_depth: int
    codon_size: int
    codon_consumption: str
    genome_representation: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert generation config to dictionary format."""
        return {
            'generation': self.generation,
            'population': self.population,
            'p_crossover': self.p_crossover,
            'p_mutation': self.p_mutation,
            'elite_size': self.elite_size,
            'tournsize': self.tournsize,
            'halloffame_size': self.halloffame_size,
            'max_tree_depth': self.max_tree_depth,
            'codon_size': self.codon_size,
            'codon_consumption': self.codon_consumption,
            'genome_representation': self.genome_representation,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        """Create generation config from dictionary."""
        return cls(**data)


@dataclass
class SetupResult:
    """
    Results from a single GE setup run.
    
    This class encapsulates all the results and performance metrics
    from a single run of a Grammatical Evolution setup.
    
    Attributes:
        config (SetupConfig): Configuration used for this run
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
        invalid_count_min (List[int]): Minimum number of invalid individuals per generation
        invalid_count_avg (List[float]): Average number of invalid individuals per generation
        invalid_count_max (List[int]): Maximum number of invalid individuals per generation
        invalid_count_std (List[float]): Standard deviation of invalid individuals per generation
        nodes_length_min (List[int]): Minimum number of terminal symbols per generation
        nodes_length_avg (List[float]): Average number of terminal symbols per generation
        nodes_length_max (List[int]): Maximum number of terminal symbols per generation
        nodes_length_std (List[float]): Standard deviation of terminal symbols per generation
        generation_configs (List[GenerationConfig]): Configuration for each generation
        timestamp (str): ISO timestamp of result generation
    """
    
    config: SetupConfig
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
    invalid_count_min: List[int] = field(default_factory=list)
    invalid_count_avg: List[float] = field(default_factory=list)
    invalid_count_max: List[int] = field(default_factory=list)
    invalid_count_std: List[float] = field(default_factory=list)
    nodes_length_min: List[int] = field(default_factory=list)
    nodes_length_avg: List[float] = field(default_factory=list)
    nodes_length_max: List[int] = field(default_factory=list)
    nodes_length_std: List[float] = field(default_factory=list)
    generation_configs: List[GenerationConfig] = field(default_factory=list)
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
            'invalid_count_min': self.invalid_count_min,
            'invalid_count_avg': self.invalid_count_avg,
            'invalid_count_max': self.invalid_count_max,
            'invalid_count_std': self.invalid_count_std,
            'nodes_length_min': self.nodes_length_min,
            'nodes_length_avg': self.nodes_length_avg,
            'nodes_length_max': self.nodes_length_max,
            'nodes_length_std': self.nodes_length_std,
            'generation_configs': [gc.to_dict() for gc in self.generation_configs],
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SetupResult':
        """Create result from dictionary."""
        config = SetupConfig.from_dict(data['config'])
        
        # Handle generation_configs with backward compatibility
        generation_configs = []
        if 'generation_configs' in data:
            generation_configs = [GenerationConfig.from_dict(gc_data) 
                                for gc_data in data['generation_configs']]
        
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
            invalid_count_min=data.get('invalid_count_min', []),
            invalid_count_avg=data.get('invalid_count_avg', []),
            invalid_count_max=data.get('invalid_count_max', []),
            invalid_count_std=data.get('invalid_count_std', []),
            nodes_length_min=data.get('nodes_length_min', []),
            nodes_length_avg=data.get('nodes_length_avg', []),
            nodes_length_max=data.get('nodes_length_max', []),
            nodes_length_std=data.get('nodes_length_std', []),
            generation_configs=generation_configs,
            timestamp=data.get('timestamp', dt.datetime.now(dt.timezone.utc).isoformat())
        )


@dataclass
class Setup:
    """
    Complete setup data model.
    
    This class represents a complete Grammatical Evolution setup,
    including its configuration, all run results, and metadata.
    
    Attributes:
        id (str): Unique setup identifier
        config (SetupConfig): Setup configuration
        results (Dict[str, SetupResult]): Results from each run
        status (str): Current status of the setup
        created_at (str): ISO timestamp of creation
        completed_at (Optional[str]): ISO timestamp of completion
    """
    
    id: str
    config: SetupConfig
    results: Dict[str, SetupResult] = field(default_factory=dict)
    status: str = 'created'  # created, running, completed, failed
    created_at: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def add_result(self, run_id: str, result: SetupResult) -> None:
        """Add a result from a single run."""
        self.results[run_id] = result
    
    def get_best_result(self) -> Optional[SetupResult]:
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
        """Check if setup is completed."""
        return len(self.results) >= self.config.n_runs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert setup to dictionary format."""
        return {
            'id': self.id,
            'config': self.config.to_dict(),
            'results': {run_id: result.to_dict() for run_id, result in self.results.items()},
            'status': self.status,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Setup':
        """Create setup from dictionary."""
        config = SetupConfig.from_dict(data['config'])
        results = {run_id: SetupResult.from_dict(result_data) 
                  for run_id, result_data in data.get('results', {}).items()}
        
        return cls(
            id=data['id'],
            config=config,
            results=results,
            status=data.get('status', 'created'),
            created_at=data.get('created_at', dt.datetime.now().isoformat()),
            completed_at=data.get('completed_at')
        )