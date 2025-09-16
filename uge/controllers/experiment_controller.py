"""
Experiment Controller for UGE Application

This module provides the experiment controller that orchestrates experiment
operations between views and services.

Classes:
- ExperimentController: Main controller for experiment operations

Author: UGE Team
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import time

from uge.controllers.base_controller import BaseController
from uge.models.experiment import Experiment, ExperimentConfig, ExperimentResult
from uge.models.dataset import Dataset
from uge.models.grammar import Grammar
from uge.utils.helpers import create_experiment_id, create_run_id


class ExperimentController(BaseController):
    """
    Controller for experiment operations.
    
    This controller orchestrates experiment operations between views and services,
    handling experiment creation, execution, and management.
    
    Attributes:
        on_experiment_start (Optional[Callable]): Callback when experiment starts
        on_experiment_progress (Optional[Callable]): Callback for experiment progress
        on_experiment_complete (Optional[Callable]): Callback when experiment completes
        on_experiment_error (Optional[Callable]): Callback when experiment errors
    """
    
    def __init__(self, on_experiment_start: Optional[Callable] = None,
                 on_experiment_progress: Optional[Callable] = None,
                 on_experiment_complete: Optional[Callable] = None,
                 on_experiment_error: Optional[Callable] = None):
        """
        Initialize experiment controller.
        
        Args:
            on_experiment_start (Optional[Callable]): Callback when experiment starts
            on_experiment_progress (Optional[Callable]): Callback for experiment progress
            on_experiment_complete (Optional[Callable]): Callback when experiment completes
            on_experiment_error (Optional[Callable]): Callback when experiment errors
        """
        super().__init__()
        self.on_experiment_start = on_experiment_start
        self.on_experiment_progress = on_experiment_progress
        self.on_experiment_complete = on_experiment_complete
        self.on_experiment_error = on_experiment_error
    
    def handle_request(self, request_type: str, **kwargs) -> Any:
        """
        Handle experiment requests.
        
        Args:
            request_type (str): Type of request
            **kwargs: Request parameters
            
        Returns:
            Any: Request result
        """
        if request_type == "create_experiment":
            return self.create_experiment(kwargs.get('config'))
        elif request_type == "run_experiment":
            return self.run_experiment(kwargs.get('config'))
        elif request_type == "get_experiment":
            return self.get_experiment(kwargs.get('experiment_id'))
        elif request_type == "list_experiments":
            return self.list_experiments()
        elif request_type == "delete_experiment":
            return self.delete_experiment(kwargs.get('experiment_id'))
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def create_experiment(self, config: ExperimentConfig) -> Optional[Experiment]:
        """
        Create a new experiment.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            
        Returns:
            Optional[Experiment]: Created experiment or None if failed
        """
        try:
            # Validate configuration
            validation_errors = self.ge_service.validate_config(config)
            if validation_errors:
                self.handle_error(
                    ValueError(f"Configuration validation failed: {validation_errors}"),
                    "creating experiment"
                )
                return None
            
            # Create experiment ID
            exp_id = create_experiment_id()
            
            # Create experiment object
            experiment = Experiment(
                id=exp_id,
                config=config,
                status='created'
            )
            
            # Save experiment configuration
            self.storage_service.save_experiment_config(exp_id, config)
            
            # Call callback if provided
            if self.on_experiment_start:
                self.on_experiment_start(experiment)
            
            return experiment
            
        except Exception as e:
            self.handle_error(e, "creating experiment")
            if self.on_experiment_error:
                self.on_experiment_error(e)
            return None
    
    def run_experiment(self, config: ExperimentConfig, live_placeholder=None, 
                      progress_bar=None, status_text=None, all_runs_container=None) -> Optional[Experiment]:
        """
        Run a complete experiment with multiple runs.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            
        Returns:
            Optional[Experiment]: Completed experiment or None if failed
        """
        try:
            # Create experiment
            experiment = self.create_experiment(config)
            if not experiment:
                return None

            # Background control flags in session_state
            cancel_key = f"cancel_{experiment.id}"
            running_key = f"running_{experiment.id}"
            st.session_state[cancel_key] = False
            st.session_state[running_key] = True
            
            # Load dataset and grammar
            dataset = self._load_dataset(config.dataset)
            grammar = self._load_grammar(config.grammar)
            
            if not dataset or not grammar:
                self.handle_error(
                    ValueError("Failed to load dataset or grammar"),
                    "running experiment"
                )
                return None
            
            def _run_all_runs():
                try:
                    all_run_results = {}
                    for run_idx in range(config.n_runs):
                        if st.session_state.get(cancel_key):
                            break
                        
                        # Update status text
                        if status_text:
                            status_text.text(f"Running {config.experiment_name} - Run {run_idx+1}/{config.n_runs}")
                        
                        # Create run section in container
                        if all_runs_container:
                            with all_runs_container:
                                st.subheader(f"🏃 Run {run_idx+1} Details")
                                run_placeholder = st.empty()
                        else:
                            run_placeholder = live_placeholder
                        
                        run_config = config
                        run_config.random_seed = config.random_seed + run_idx
                        run_id = create_run_id()
                        
                        # Simple approach - just pass the placeholder

                        result = self._run_single_experiment(
                            run_config, dataset, grammar, run_id, run_idx + 1, config.n_runs, run_placeholder
                        )
                        if result:
                            all_run_results[run_id] = result
                            self.storage_service.save_run_result(experiment.id, run_id, result)
                        
                        # Update progress bar
                        if progress_bar:
                            progress_bar.progress((run_idx + 1) / config.n_runs)
                        
                        if self.on_experiment_progress:
                            progress = (run_idx + 1) / config.n_runs
                            self.on_experiment_progress(experiment, run_idx + 1, config.n_runs, progress)
                    experiment.results = all_run_results
                    experiment.status = 'completed' if experiment.is_completed() else 'running'
                    self.storage_service.save_experiment(experiment)
                    
                    # Store results in session state for analysis page
                    st.session_state['latest_experiment'] = {
                        'config': config.to_dict(),
                        'results': {run_id: result.to_dict() for run_id, result in all_run_results.items()},
                        'exp_id': experiment.id,
                        'experiment_name': config.experiment_name
                    }
                    
                    # Update status text to completed
                    if status_text:
                        status_text.text("✅ Experiment completed!")
                    
                    if self.on_experiment_complete:
                        self.on_experiment_complete(experiment)
                except Exception as e:
                    self.handle_error(e, "running experiment")
                    if self.on_experiment_error:
                        self.on_experiment_error(e)
                finally:
                    st.session_state[running_key] = False

            _run_all_runs()
            return experiment
            
        except Exception as e:
            self.handle_error(e, "running experiment")
            if self.on_experiment_error:
                self.on_experiment_error(e)
            return None
    
    def _run_single_experiment(self, config: ExperimentConfig, dataset: Dataset, 
                              grammar: Grammar, run_id: str, run_number: int, 
                              total_runs: int, live_placeholder=None) -> Optional[ExperimentResult]:
        """
        Run a single experiment run.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            dataset (Dataset): Dataset to use
            grammar (Grammar): Grammar to use
            run_id (str): Run ID
            run_number (int): Run number
            total_runs (int): Total number of runs
            
        Returns:
            Optional[ExperimentResult]: Run result or None if failed
        """
        try:
            # Create live placeholder for output if not provided
            if live_placeholder is None and st:
                live_placeholder = st.empty()
            
            # Run GE experiment
            result = self.ge_service.run_experiment(
                config=config,
                dataset=dataset,
                grammar=grammar,
                report_items=config.report_items,
                live_placeholder=live_placeholder
            )
            
            return result
            
        except Exception as e:
            self.handle_error(e, f"running single experiment (run {run_number})")
            return None
    
    def _load_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """
        Load dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Optional[Dataset]: Loaded dataset or None if failed
        """
        try:
            return self.dataset_service.load_dataset(dataset_name)
        except Exception as e:
            self.handle_error(e, f"loading dataset '{dataset_name}'")
            return None
    
    def _load_grammar(self, grammar_name: str) -> Optional[Grammar]:
        """
        Load grammar by name.
        
        Args:
            grammar_name (str): Name of the grammar
            
        Returns:
            Optional[Grammar]: Loaded grammar or None if failed
        """
        try:
            grammar_path = Path(self.project_root) / "grammars" / grammar_name
            return Grammar.from_file(grammar_path)
        except Exception as e:
            self.handle_error(e, f"loading grammar '{grammar_name}'")
            return None
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Optional[Experiment]: Experiment or None if not found
        """
        try:
            return self.storage_service.load_experiment(experiment_id)
        except Exception as e:
            self.handle_error(e, f"loading experiment '{experiment_id}'")
            return None
    
    def list_experiments(self) -> List[Experiment]:
        """
        List all experiments sorted by creation time (newest first).
        
        Returns:
            List[Experiment]: List of experiments sorted by creation time descending
        """
        try:
            experiments = []
            exp_dirs = self.storage_service.list_experiments()
            
            for exp_dir in exp_dirs:
                exp_id = exp_dir.name
                experiment = self.get_experiment(exp_id)
                if experiment:
                    experiments.append(experiment)
            
            # Sort experiments by creation time (newest first)
            # Since experiment IDs include timestamp, we can sort by ID descending
            experiments.sort(key=lambda x: x.created_at, reverse=True)
            
            return experiments
        except Exception as e:
            self.handle_error(e, "listing experiments")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete experiment by ID.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            return self.storage_service.delete_experiment(experiment_id)
        except Exception as e:
            self.handle_error(e, f"deleting experiment '{experiment_id}'")
            return False
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        Get experiment statistics.
        
        Returns:
            Dict[str, Any]: Experiment statistics
        """
        try:
            return self.storage_service.get_experiment_stats()
        except Exception as e:
            self.handle_error(e, "getting experiment statistics")
            return {}
    
    def validate_experiment_config(self, config: ExperimentConfig) -> List[str]:
        """
        Validate experiment configuration.
        
        Args:
            config (ExperimentConfig): Configuration to validate
            
        Returns:
            List[str]: List of validation errors
        """
        try:
            return self.ge_service.validate_config(config)
        except Exception as e:
            self.handle_error(e, "validating experiment configuration")
            return [f"Validation error: {str(e)}"]
    
    def get_experiment_progress(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment progress.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Dict[str, Any]: Progress information
        """
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                return {}
            
            return {
                'experiment_id': experiment_id,
                'status': experiment.status,
                'total_runs': experiment.config.n_runs,
                'completed_runs': len(experiment.results),
                'progress': len(experiment.results) / experiment.config.n_runs,
                'is_completed': experiment.is_completed()
            }
        except Exception as e:
            self.handle_error(e, f"getting experiment progress '{experiment_id}'")
            return {}