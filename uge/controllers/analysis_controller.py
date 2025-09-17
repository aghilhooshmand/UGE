"""
Analysis Controller for UGE Application

This module provides the analysis controller that orchestrates analysis
operations between views and services.

Classes:
- AnalysisController: Main controller for analysis operations

Author: UGE Team
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from io import StringIO

from uge.controllers.base_controller import BaseController
from uge.models.experiment import Experiment, ExperimentResult
from uge.services.storage_service import StorageService


class AnalysisController(BaseController):
    """
    Controller for analysis operations.
    
    This controller orchestrates analysis operations between views and services,
    handling experiment analysis, data export, and visualization.
    
    Attributes:
        on_analysis_start (Optional[Callable]): Callback when analysis starts
        on_analysis_complete (Optional[Callable]): Callback when analysis completes
        on_analysis_error (Optional[Callable]): Callback when analysis errors
    """
    
    def __init__(self, on_analysis_start: Optional[Callable] = None,
                 on_analysis_complete: Optional[Callable] = None,
                 on_analysis_error: Optional[Callable] = None):
        """
        Initialize analysis controller.
        
        Args:
            on_analysis_start (Optional[Callable]): Callback when analysis starts
            on_analysis_complete (Optional[Callable]): Callback when analysis completes
            on_analysis_error (Optional[Callable]): Callback when analysis errors
        """
        super().__init__()
        self.on_analysis_start = on_analysis_start
        self.on_analysis_complete = on_analysis_complete
        self.on_analysis_error = on_analysis_error
        self.storage_service = StorageService()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Optional[Experiment]: Experiment object or None if not found
        """
        try:
            return self.storage_service.load_experiment(experiment_id)
        except Exception as e:
            self.handle_error(e, f"loading experiment '{experiment_id}'")
            return None
    
    def handle_request(self, request_type: str, **kwargs) -> Any:
        """
        Handle analysis requests.
        
        Args:
            request_type (str): Type of request
            **kwargs: Request parameters
            
        Returns:
            Any: Request result
        """
        if request_type == "analyze_experiment":
            return self.analyze_experiment(kwargs.get('experiment_id'))
        elif request_type == "compare_experiments":
            return self.compare_experiments(kwargs.get('experiment_ids', []))
        elif request_type == "export_experiment_data":
            return self.export_experiment_data(kwargs.get('experiment_id'), kwargs.get('export_type'))
        elif request_type == "get_experiment_summary":
            return self.get_experiment_summary(kwargs.get('experiment_id'))
        elif request_type == "get_analysis_statistics":
            return self.get_analysis_statistics()
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def analyze_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single experiment.
        
        Args:
            experiment_id (str): Experiment ID to analyze
            
        Returns:
            Optional[Dict[str, Any]]: Analysis results or None if failed
        """
        try:
            if self.on_analysis_start:
                self.on_analysis_start(experiment_id)
            
            # Load experiment
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                self.handle_error(
                    ValueError(f"Experiment '{experiment_id}' not found"),
                    "analyzing experiment"
                )
                return None
            
            # Perform analysis
            analysis_results = self._perform_experiment_analysis(experiment)
            
            if self.on_analysis_complete:
                self.on_analysis_complete(experiment_id, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.handle_error(e, f"analyzing experiment '{experiment_id}'")
            if self.on_analysis_error:
                self.on_analysis_error(e)
            return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids (List[str]): List of experiment IDs to compare
            
        Returns:
            Optional[Dict[str, Any]]: Comparison results or None if failed
        """
        try:
            if self.on_analysis_start:
                self.on_analysis_start(experiment_ids)
            
            # Load experiments
            experiments = []
            for exp_id in experiment_ids:
                experiment = self.get_experiment(exp_id)
                if experiment:
                    experiments.append(experiment)
                else:
                    self.handle_error(
                        ValueError(f"Experiment '{exp_id}' not found"),
                        "comparing experiments"
                    )
                    return None
            
            # Perform comparison
            comparison_results = self._perform_experiment_comparison(experiments)
            
            if self.on_analysis_complete:
                self.on_analysis_complete(experiment_ids, comparison_results)
            
            return comparison_results
            
        except Exception as e:
            self.handle_error(e, "comparing experiments")
            if self.on_analysis_error:
                self.on_analysis_error(e)
            return None
    
    def export_experiment_data(self, experiment_id: str, export_type: str) -> Optional[str]:
        """
        Export experiment data.
        
        Args:
            experiment_id (str): Experiment ID to export
            export_type (str): Type of export ('results', 'config', 'all')
            
        Returns:
            Optional[str]: Exported data as string or None if failed
        """
        try:
            # Load experiment
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                self.handle_error(
                    ValueError(f"Experiment '{experiment_id}' not found"),
                    "exporting experiment data"
                )
                return None
            
            # Export based on type
            if export_type == 'results':
                return self._export_results_data(experiment)
            elif export_type == 'config':
                return self._export_config_data(experiment)
            elif export_type == 'all':
                return self._export_all_data(experiment)
            else:
                raise ValueError(f"Unknown export type: {export_type}")
                
        except Exception as e:
            self.handle_error(e, f"exporting experiment data '{experiment_id}'")
            return None
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment summary.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Optional[Dict[str, Any]]: Experiment summary or None if failed
        """
        try:
            # Load experiment
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                return None
            
            # Create summary
            summary = {
                'experiment_id': experiment_id,
                'experiment_name': experiment.config.experiment_name,
                'status': experiment.status,
                'created_at': experiment.created_at,
                'completed_at': experiment.completed_at,
                'total_runs': experiment.config.n_runs,
                'completed_runs': len(experiment.results),
                'is_completed': experiment.is_completed(),
                'best_fitness': None,
                'average_fitness': None,
                'fitness_metric': experiment.config.fitness_metric
            }
            
            # Add fitness information if available
            if experiment.results:
                best_result = experiment.get_best_result()
                if best_result:
                    summary['best_fitness'] = best_result.best_training_fitness
                
                avg_fitness = experiment.get_average_fitness()
                if avg_fitness:
                    summary['average_fitness'] = avg_fitness
            
            return summary
            
        except Exception as e:
            self.handle_error(e, f"getting experiment summary '{experiment_id}'")
            return None
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get overall analysis statistics.
        
        Returns:
            Dict[str, Any]: Analysis statistics
        """
        try:
            # Get experiment statistics
            exp_stats = self.storage_service.get_experiment_stats()
            
            # Get dataset statistics
            datasets = self.dataset_service.list_datasets()
            
            # Get grammar statistics (placeholder)
            grammars = []  # Would need grammar service
            
            return {
                'experiments': exp_stats,
                'datasets': {
                    'total': len(datasets),
                    'list': datasets
                },
                'grammars': {
                    'total': len(grammars),
                    'list': grammars
                }
            }
            
        except Exception as e:
            self.handle_error(e, "getting analysis statistics")
            return {}
    
    def _perform_experiment_analysis(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single experiment.
        
        Args:
            experiment (Experiment): Experiment to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {
            'experiment_id': experiment.id,
            'experiment_name': experiment.config.experiment_name,
            'config': experiment.config.to_dict(),
            'results': {},
            'statistics': {},
            'best_individual': None
        }
        
        # Analyze each run
        for run_id, result in experiment.results.items():
            analysis['results'][run_id] = {
                'result': result.to_dict(),
                'statistics': self._calculate_run_statistics(result)
            }
        
        # Calculate overall statistics
        analysis['statistics'] = self._calculate_experiment_statistics(experiment)
        
        # Get best individual
        best_result = experiment.get_best_result()
        if best_result:
            analysis['best_individual'] = {
                'run_id': None,  # Would need to track which run
                'fitness': best_result.best_training_fitness,
                'phenotype': best_result.best_phenotype,
                'depth': best_result.best_depth,
                'genome_length': best_result.best_genome_length,
                'used_codons': best_result.best_used_codons
            }
        
        return analysis
    
    def _perform_experiment_comparison(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """
        Perform comparison analysis of multiple experiments.
        
        Args:
            experiments (List[Experiment]): Experiments to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison = {
            'experiments': [],
            'comparison_metrics': {},
            'rankings': {},
            'aggregate_data': {},
            'experiment_configs': {}
        }
        
        # Analyze each experiment
        for experiment in experiments:
            exp_analysis = self._perform_experiment_analysis(experiment)
            comparison['experiments'].append(exp_analysis)
            
            # Store experiment config for reference
            comparison['experiment_configs'][experiment.id] = experiment.config.to_dict()
            
            # Calculate aggregate data for charts
            comparison['aggregate_data'][experiment.id] = self._calculate_aggregate_data(experiment)
        
        # Calculate comparison metrics
        comparison['comparison_metrics'] = self._calculate_comparison_metrics(experiments)
        
        # Calculate rankings
        comparison['rankings'] = self._calculate_rankings(experiments)
        
        return comparison
    
    def _calculate_aggregate_data(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Calculate aggregate data across all runs for charting.
        
        Args:
            experiment (Experiment): Experiment to analyze
            
        Returns:
            Dict[str, Any]: Aggregate data for charting
        """
        if not experiment.results:
            return {}
        
        # Find maximum generations across all runs
        max_generations = 0
        for result in experiment.results.values():
            if result.max:
                max_generations = max(max_generations, len(result.max))
        
        if max_generations == 0:
            return {}
        
        # Initialize aggregate data structure
        aggregate_data = {
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
        
        # Collect data for each generation
        for gen in range(max_generations):
            gen_max_values = []
            gen_avg_values = []
            gen_min_values = []
            gen_test_values = []
            
            for result in experiment.results.values():
                if result.max and gen < len(result.max):
                    gen_max_values.append(result.max[gen])
                if result.avg and gen < len(result.avg):
                    gen_avg_values.append(result.avg[gen])
                if result.min and gen < len(result.min):
                    gen_min_values.append(result.min[gen])
                if result.fitness_test and gen < len(result.fitness_test) and result.fitness_test[gen] is not None:
                    gen_test_values.append(result.fitness_test[gen])
            
            # Calculate statistics for this generation
            aggregate_data['avg_max'].append(sum(gen_max_values) / len(gen_max_values) if gen_max_values else 0)
            aggregate_data['std_max'].append(self._calculate_std(gen_max_values) if gen_max_values else 0)
            aggregate_data['avg_avg'].append(sum(gen_avg_values) / len(gen_avg_values) if gen_avg_values else 0)
            aggregate_data['std_avg'].append(self._calculate_std(gen_avg_values) if gen_avg_values else 0)
            aggregate_data['avg_min'].append(sum(gen_min_values) / len(gen_min_values) if gen_min_values else 0)
            aggregate_data['std_min'].append(self._calculate_std(gen_min_values) if gen_min_values else 0)
            aggregate_data['avg_test'].append(sum(gen_test_values) / len(gen_test_values) if gen_test_values else None)
            aggregate_data['std_test'].append(self._calculate_std(gen_test_values) if gen_test_values else None)
        
        return aggregate_data
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_run_statistics(self, result: ExperimentResult) -> Dict[str, Any]:
        """
        Calculate statistics for a single run.
        
        Args:
            result (ExperimentResult): Run result
            
        Returns:
            Dict[str, Any]: Run statistics
        """
        stats = {
            'best_fitness': result.best_training_fitness,
            'final_generation': len(result.max),
            'fitness_improvement': None,
            'convergence_generation': None
        }
        
        # Calculate fitness improvement
        if result.max and len(result.max) > 1:
            stats['fitness_improvement'] = result.max[-1] - result.max[0]
        
        # Find convergence generation (where fitness stops improving significantly)
        if result.max and len(result.max) > 10:
            # Simple convergence detection (could be more sophisticated)
            for i in range(len(result.max) - 10, len(result.max)):
                if abs(result.max[i] - result.max[i-1]) < 0.001:
                    stats['convergence_generation'] = i
                    break
        
        return stats
    
    def _calculate_experiment_statistics(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Calculate overall experiment statistics.
        
        Args:
            experiment (Experiment): Experiment to analyze
            
        Returns:
            Dict[str, Any]: Experiment statistics
        """
        stats = {
            'total_runs': len(experiment.results),
            'completed_runs': len(experiment.results),
            'best_fitness': None,
            'average_fitness': None,
            'fitness_std': None,
            'best_run': None,
            'worst_run': None
        }
        
        if experiment.results:
            fitnesses = [r.best_training_fitness for r in experiment.results.values() 
                        if r.best_training_fitness is not None]
            
            if fitnesses:
                stats['best_fitness'] = max(fitnesses)
                stats['average_fitness'] = sum(fitnesses) / len(fitnesses)
                stats['fitness_std'] = (sum((f - stats['average_fitness'])**2 for f in fitnesses) / len(fitnesses))**0.5
                
                # Find best and worst runs
                best_run = max(experiment.results.items(), key=lambda x: x[1].best_training_fitness or 0)
                worst_run = min(experiment.results.items(), key=lambda x: x[1].best_training_fitness or float('inf'))
                
                stats['best_run'] = best_run[0]
                stats['worst_run'] = worst_run[0]
        
        return stats
    
    def _calculate_comparison_metrics(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """
        Calculate comparison metrics between experiments.
        
        Args:
            experiments (List[Experiment]): Experiments to compare
            
        Returns:
            Dict[str, Any]: Comparison metrics
        """
        metrics = {
            'best_overall_fitness': None,
            'best_experiment': None,
            'average_fitness_by_experiment': {},
            'fitness_consistency': {}
        }
        
        if not experiments:
            return metrics
        
        # Find best overall fitness and experiment
        best_fitness = float('-inf')
        best_exp = None
        
        for experiment in experiments:
            avg_fitness = experiment.get_average_fitness()
            if avg_fitness is not None:
                metrics['average_fitness_by_experiment'][experiment.id] = avg_fitness
                
                if avg_fitness > best_fitness:
                    best_fitness = avg_fitness
                    best_exp = experiment.id
        
        metrics['best_overall_fitness'] = best_fitness
        metrics['best_experiment'] = best_exp
        
        return metrics
    
    def _calculate_rankings(self, experiments: List[Experiment]) -> Dict[str, List[str]]:
        """
        Calculate rankings of experiments.
        
        Args:
            experiments (List[Experiment]): Experiments to rank
            
        Returns:
            Dict[str, List[str]]: Rankings by different metrics
        """
        rankings = {
            'by_best_fitness': [],
            'by_average_fitness': [],
            'by_consistency': []
        }
        
        # Rank by best fitness
        best_fitness_ranking = sorted(
            experiments,
            key=lambda x: max([r.best_training_fitness for r in x.results.values() 
                              if r.best_training_fitness is not None], default=0),
            reverse=True
        )
        rankings['by_best_fitness'] = [exp.id for exp in best_fitness_ranking]
        
        # Rank by average fitness
        avg_fitness_ranking = sorted(
            experiments,
            key=lambda x: x.get_average_fitness() or 0,
            reverse=True
        )
        rankings['by_average_fitness'] = [exp.id for exp in avg_fitness_ranking]
        
        return rankings
    
    def _export_results_data(self, experiment: Experiment) -> str:
        """
        Export experiment results as CSV.
        
        Args:
            experiment (Experiment): Experiment to export
            
        Returns:
            str: CSV data
        """
        # Create DataFrame with all run results
        data = []
        for run_id, result in experiment.results.items():
            for gen in range(len(result.max)):
                data.append({
                    'run_id': run_id,
                    'generation': gen,
                    'max_fitness': result.max[gen] if gen < len(result.max) else None,
                    'avg_fitness': result.avg[gen] if gen < len(result.avg) else None,
                    'min_fitness': result.min[gen] if gen < len(result.min) else None,
                    'std_fitness': result.std[gen] if gen < len(result.std) else None,
                    'test_fitness': result.fitness_test[gen] if gen < len(result.fitness_test) else None
                })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def _export_config_data(self, experiment: Experiment) -> str:
        """
        Export experiment configuration as JSON.
        
        Args:
            experiment (Experiment): Experiment to export
            
        Returns:
            str: JSON data
        """
        return json.dumps(experiment.config.to_dict(), indent=2)
    
    def _export_all_data(self, experiment: Experiment) -> str:
        """
        Export all experiment data as JSON.
        
        Args:
            experiment (Experiment): Experiment to export
            
        Returns:
            str: JSON data
        """
        return json.dumps(experiment.to_dict(), indent=2)