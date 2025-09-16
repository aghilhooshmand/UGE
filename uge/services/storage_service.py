"""
Storage Service for UGE Application

This module provides file storage and persistence services for the UGE application.
It handles saving and loading experiments, results, and other data.

Classes:
- StorageService: Main service for file operations and data persistence

Author: UGE Team
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from uge.models.experiment import Experiment, ExperimentConfig, ExperimentResult
from uge.utils.constants import FILE_PATHS


class StorageService:
    """
    Service for file storage and data persistence operations.
    
    This service handles all file I/O operations for the UGE application,
    including saving/loading experiments, results, and managing the
    file system structure.
    
    Attributes:
        project_root (Path): Root directory of the project
        results_dir (Path): Directory for storing results
        experiments_dir (Path): Directory for storing experiments
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize storage service.
        
        Args:
            project_root (Optional[Path]): Project root directory
        """
        self.project_root = project_root or FILE_PATHS['project_root']
        self.results_dir = self.project_root / "results"
        self.experiments_dir = self.results_dir / "experiments"
        
        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_config(self, exp_id: str, config: ExperimentConfig) -> Path:
        """
        Save experiment configuration to file.
        
        Args:
            exp_id (str): Experiment ID
            config (ExperimentConfig): Configuration to save
            
        Returns:
            Path: Path to saved configuration file
        """
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        config_file = exp_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        return config_file
    
    def load_experiment_config(self, exp_id: str) -> Optional[ExperimentConfig]:
        """
        Load experiment configuration from file.
        
        Args:
            exp_id (str): Experiment ID
            
        Returns:
            Optional[ExperimentConfig]: Loaded configuration or None if not found
        """
        config_file = self.experiments_dir / exp_id / "experiment_config.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            return ExperimentConfig.from_dict(data)
        except Exception as e:
            raise ValueError(f"Error loading experiment config: {e}")
    
    def save_run_result(self, exp_id: str, run_id: str, result: ExperimentResult) -> Path:
        """
        Save run result to file.
        
        Args:
            exp_id (str): Experiment ID
            run_id (str): Run ID
            result (ExperimentResult): Result to save
            
        Returns:
            Path: Path to saved result file
        """
        exp_dir = self.experiments_dir / exp_id
        run_dir = exp_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result data
        result_file = run_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save CSV log if available
        if hasattr(result, 'logbook') and result.logbook:
            csv_file = run_dir / "logbook.csv"
            log_df = pd.DataFrame(result.logbook)
            log_df.to_csv(csv_file, index=False)
        
        return result_file
    
    def load_run_result(self, exp_id: str, run_id: str) -> Optional[ExperimentResult]:
        """
        Load run result from file.
        
        Args:
            exp_id (str): Experiment ID
            run_id (str): Run ID
            
        Returns:
            Optional[ExperimentResult]: Loaded result or None if not found
        """
        result_file = self.experiments_dir / exp_id / "runs" / run_id / "result.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            return ExperimentResult.from_dict(data)
        except Exception as e:
            raise ValueError(f"Error loading run result: {e}")
    
    def list_experiments(self) -> List[Path]:
        """
        List all experiment directories sorted by name (newest first).
        
        Returns:
            List[Path]: List of experiment directory paths sorted by name descending
        """
        if not self.experiments_dir.exists():
            return []
        
        # Get all experiment directories and sort by name descending (newest first)
        # Since experiment names include timestamp (exp_YYYYMMDD_HHMMSS_*), 
        # sorting by name descending gives us newest experiments first
        experiment_dirs = [p for p in self.experiments_dir.iterdir() 
                          if p.is_dir() and p.name.startswith('exp_')]
        
        return sorted(experiment_dirs, key=lambda x: x.name, reverse=True)
    
    def list_experiment_runs(self, exp_id: str) -> List[Path]:
        """
        List all runs for an experiment sorted by name (newest first).
        
        Args:
            exp_id (str): Experiment ID
            
        Returns:
            List[Path]: List of run directory paths sorted by name descending
        """
        runs_dir = self.experiments_dir / exp_id / "runs"
        if not runs_dir.exists():
            return []
        
        # Get all run directories and sort by name descending (newest first)
        # Since run names include timestamp (run_YYYYMMDD_HHMMSS_*), 
        # sorting by name descending gives us newest runs first
        run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        
        return sorted(run_dirs, key=lambda x: x.name, reverse=True)
    
    def load_experiment(self, exp_id: str) -> Optional[Experiment]:
        """
        Load complete experiment with all runs.
        
        Args:
            exp_id (str): Experiment ID
            
        Returns:
            Optional[Experiment]: Complete experiment or None if not found
        """
        # Load configuration
        config = self.load_experiment_config(exp_id)
        if not config:
            return None
        
        # Load all run results
        results = {}
        run_dirs = self.list_experiment_runs(exp_id)
        
        for run_dir in run_dirs:
            run_id = run_dir.name
            result = self.load_run_result(exp_id, run_id)
            if result:
                results[run_id] = result
        
        # Create experiment object
        experiment = Experiment(
            id=exp_id,
            config=config,
            results=results
        )
        
        # Update status based on results
        if experiment.is_completed():
            experiment.status = 'completed'
            experiment.completed_at = max(r.timestamp for r in results.values())
        elif results:
            experiment.status = 'running'
        
        return experiment
    
    def save_experiment(self, experiment: Experiment) -> None:
        """
        Save complete experiment with all runs.
        
        Args:
            experiment (Experiment): Experiment to save
        """
        # Save configuration
        self.save_experiment_config(experiment.id, experiment.config)
        
        # Save all run results
        for run_id, result in experiment.results.items():
            self.save_run_result(experiment.id, run_id, result)
    
    def delete_experiment(self, exp_id: str) -> bool:
        """
        Delete experiment and all its data.
        
        Args:
            exp_id (str): Experiment ID
            
        Returns:
            bool: True if deleted successfully, False if not found
        """
        exp_dir = self.experiments_dir / exp_id
        if not exp_dir.exists():
            return False
        
        try:
            import shutil
            shutil.rmtree(exp_dir)
            return True
        except Exception as e:
            raise RuntimeError(f"Error deleting experiment: {e}")
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored experiments.
        
        Returns:
            Dict[str, Any]: Experiment statistics
        """
        experiments = self.list_experiments()
        
        total_runs = 0
        completed_experiments = 0
        running_experiments = 0
        
        for exp_dir in experiments:
            exp_id = exp_dir.name
            experiment = self.load_experiment(exp_id)
            if experiment:
                total_runs += len(experiment.results)
                if experiment.is_completed():
                    completed_experiments += 1
                elif experiment.results:
                    running_experiments += 1
        
        return {
            'total_experiments': len(experiments),
            'completed_experiments': completed_experiments,
            'running_experiments': running_experiments,
            'total_runs': total_runs
        }
    
    def cleanup_old_experiments(self, days_old: int = 30) -> int:
        """
        Clean up experiments older than specified days.
        
        Args:
            days_old (int): Number of days to keep experiments
            
        Returns:
            int: Number of experiments deleted
        """
        import datetime as dt
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=days_old)
        deleted_count = 0
        
        experiments = self.list_experiments()
        for exp_dir in experiments:
            try:
                # Check creation date
                creation_time = dt.datetime.fromtimestamp(exp_dir.stat().st_ctime)
                if creation_time < cutoff_date:
                    if self.delete_experiment(exp_dir.name):
                        deleted_count += 1
            except Exception:
                # Skip if we can't determine age
                continue
        
        return deleted_count