"""
Setup Controller for UGE Application

This module provides the setup controller that orchestrates setup
operations between views and services.

Classes:
- SetupController: Main controller for setup operations

Author: UGE Team
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import time

from uge.controllers.base_controller import BaseController
from uge.models.setup import Setup, SetupConfig, SetupResult
from uge.models.dataset import Dataset
from uge.models.grammar import Grammar
from uge.utils.helpers import create_setup_id, create_run_id


class SetupController(BaseController):
    """
    Controller for setup operations.
    
    This controller orchestrates setup operations between views and services,
    handling setup creation, execution, and management.
    
    Attributes:
        on_setup_start (Optional[Callable]): Callback when setup starts
        on_setup_progress (Optional[Callable]): Callback for setup progress
        on_setup_complete (Optional[Callable]): Callback when setup completes
        on_setup_error (Optional[Callable]): Callback when setup errors
    """
    
    def __init__(self, on_setup_start: Optional[Callable] = None,
                 on_setup_progress: Optional[Callable] = None,
                 on_setup_complete: Optional[Callable] = None,
                 on_setup_error: Optional[Callable] = None):
        """
        Initialize setup controller.
        
        Args:
            on_setup_start (Optional[Callable]): Callback when setup starts
            on_setup_progress (Optional[Callable]): Callback for setup progress
            on_setup_complete (Optional[Callable]): Callback when setup completes
            on_setup_error (Optional[Callable]): Callback when setup errors
        """
        super().__init__()
        self.on_setup_start = on_setup_start
        self.on_setup_progress = on_setup_progress
        self.on_setup_complete = on_setup_complete
        self.on_setup_error = on_setup_error
    
    def handle_request(self, request_type: str, **kwargs) -> Any:
        """
        Handle setup requests.
        
        Args:
            request_type (str): Type of request
            **kwargs: Request parameters
            
        Returns:
            Any: Request result
        """
        if request_type == "create_setup":
            return self.create_setup(kwargs.get('config'))
        elif request_type == "run_setup":
            return self.run_setup(kwargs.get('config'))
        elif request_type == "get_setup":
            return self.get_setup(kwargs.get('setup_id'))
        elif request_type == "list_setups":
            return self.list_setups()
        elif request_type == "delete_setup":
            return self.delete_setup(kwargs.get('setup_id'))
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def create_setup(self, config: SetupConfig) -> Optional[Setup]:
        """
        Create a new setup.
        
        Args:
            config (SetupConfig): Setup configuration
            
        Returns:
            Optional[Setup]: Created setup or None if failed
        """
        try:
            # Validate configuration
            validation_errors = self.ge_service.validate_config(config)
            if validation_errors:
                self.handle_error(
                    ValueError(f"Configuration validation failed: {validation_errors}"),
                    "creating setup"
                )
                return None
            
            # Create setup ID
            exp_id = create_setup_id()
            
            # Create setup object
            setup = Setup(
                id=exp_id,
                config=config,
                status='created'
            )
            
            # Save setup configuration
            self.storage_service.save_setup_config(exp_id, config)
            
            # Call callback if provided
            if self.on_setup_start:
                self.on_setup_start(setup)
            
            return setup
            
        except Exception as e:
            self.handle_error(e, "creating setup")
            if self.on_setup_error:
                self.on_setup_error(e)
            return None
    
    def run_setup(self, config: SetupConfig, live_placeholder=None, 
                      progress_bar=None, status_text=None, all_runs_container=None) -> Optional[Setup]:
        """
        Run a complete setup with multiple runs.
        
        Args:
            config (SetupConfig): Setup configuration
            
        Returns:
            Optional[Setup]: Completed setup or None if failed
        """
        try:
            # Create setup
            setup = self.create_setup(config)
            if not setup:
                return None

            # Background control flags in session_state
            cancel_key = f"cancel_{setup.id}"
            running_key = f"running_{setup.id}"
            st.session_state[cancel_key] = False
            st.session_state[running_key] = True
            
            # Load dataset and grammar
            dataset = self._load_dataset(config.dataset)
            grammar = self._load_grammar(config.grammar)
            
            if not dataset or not grammar:
                self.handle_error(
                    ValueError("Failed to load dataset or grammar"),
                    "running setup"
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
                            status_text.text(f"Running {config.setup_name} - Run {run_idx+1}/{config.n_runs}")
                        
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

                        result = self._run_single_setup(
                            run_config, dataset, grammar, run_id, run_idx + 1, config.n_runs, run_placeholder
                        )
                        if result:
                            all_run_results[run_id] = result
                            self.storage_service.save_run_result(setup.id, run_id, result)
                        
                        # Update progress bar
                        if progress_bar:
                            progress_bar.progress((run_idx + 1) / config.n_runs)
                        
                        if self.on_setup_progress:
                            progress = (run_idx + 1) / config.n_runs
                            self.on_setup_progress(setup, run_idx + 1, config.n_runs, progress)
                    setup.results = all_run_results
                    setup.status = 'completed' if setup.is_completed() else 'running'
                    self.storage_service.save_setup(setup)
                    
                    # Store results in session state for analysis page
                    st.session_state['latest_setup'] = {
                        'config': config.to_dict(),
                        'results': {run_id: result.to_dict() for run_id, result in all_run_results.items()},
                        'exp_id': setup.id,
                        'setup_name': config.setup_name
                    }
                    
                    # Update status text to completed
                    if status_text:
                        status_text.text("✅ Setup completed!")
                    
                    if self.on_setup_complete:
                        self.on_setup_complete(setup)
                except Exception as e:
                    self.handle_error(e, "running setup")
                    if self.on_setup_error:
                        self.on_setup_error(e)
                finally:
                    st.session_state[running_key] = False

            _run_all_runs()
            return setup
            
        except Exception as e:
            self.handle_error(e, "running setup")
            if self.on_setup_error:
                self.on_setup_error(e)
            return None
    
    def _run_single_setup(self, config: SetupConfig, dataset: Dataset, 
                              grammar: Grammar, run_id: str, run_number: int, 
                              total_runs: int, live_placeholder=None) -> Optional[SetupResult]:
        """
        Run a single setup run.
        
        Args:
            config (SetupConfig): Setup configuration
            dataset (Dataset): Dataset to use
            grammar (Grammar): Grammar to use
            run_id (str): Run ID
            run_number (int): Run number
            total_runs (int): Total number of runs
            
        Returns:
            Optional[SetupResult]: Run result or None if failed
        """
        try:
            # Create live placeholder for output if not provided
            if live_placeholder is None and st:
                live_placeholder = st.empty()
            
            # Run GE setup
            result = self.ge_service.run_setup(
                config=config,
                dataset=dataset,
                grammar=grammar,
                report_items=config.report_items,
                live_placeholder=live_placeholder
            )
            
            return result
            
        except Exception as e:
            self.handle_error(e, f"running single setup (run {run_number})")
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
    
    def get_setup(self, setup_id: str) -> Optional[Setup]:
        """
        Get setup by ID.
        
        Args:
            setup_id (str): Setup ID
            
        Returns:
            Optional[Setup]: Setup or None if not found
        """
        try:
            return self.storage_service.load_setup(setup_id)
        except Exception as e:
            self.handle_error(e, f"loading setup '{setup_id}'")
            return None
    
    def list_setups(self) -> List[Setup]:
        """
        List all setups sorted by creation time (newest first).
        
        Returns:
            List[Setup]: List of setups sorted by creation time descending
        """
        try:
            setups = []
            exp_dirs = self.storage_service.list_setups()
            
            for exp_dir in exp_dirs:
                exp_id = exp_dir.name
                setup = self.get_setup(exp_id)
                if setup:
                    setups.append(setup)
            
            # Sort setups by creation time (newest first)
            # Since setup IDs include timestamp, we can sort by ID descending
            setups.sort(key=lambda x: x.created_at, reverse=True)
            
            return setups
        except Exception as e:
            self.handle_error(e, "listing setups")
            return []
    
    def delete_setup(self, setup_id: str) -> bool:
        """
        Delete setup by ID.
        
        Args:
            setup_id (str): Setup ID
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            return self.storage_service.delete_setup(setup_id)
        except Exception as e:
            self.handle_error(e, f"deleting setup '{setup_id}'")
            return False
    
    def get_setup_statistics(self) -> Dict[str, Any]:
        """
        Get setup statistics.
        
        Returns:
            Dict[str, Any]: Setup statistics
        """
        try:
            return self.storage_service.get_setup_stats()
        except Exception as e:
            self.handle_error(e, "getting setup statistics")
            return {}
    
    def validate_setup_config(self, config: SetupConfig) -> List[str]:
        """
        Validate setup configuration.
        
        Args:
            config (SetupConfig): Configuration to validate
            
        Returns:
            List[str]: List of validation errors
        """
        try:
            return self.ge_service.validate_config(config)
        except Exception as e:
            self.handle_error(e, "validating setup configuration")
            return [f"Validation error: {str(e)}"]
    
    def get_setup_progress(self, setup_id: str) -> Dict[str, Any]:
        """
        Get setup progress.
        
        Args:
            setup_id (str): Setup ID
            
        Returns:
            Dict[str, Any]: Progress information
        """
        try:
            setup = self.get_setup(setup_id)
            if not setup:
                return {}
            
            return {
                'setup_id': setup_id,
                'status': setup.status,
                'total_runs': setup.config.n_runs,
                'completed_runs': len(setup.results),
                'progress': len(setup.results) / setup.config.n_runs,
                'is_completed': setup.is_completed()
            }
        except Exception as e:
            self.handle_error(e, f"getting setup progress '{setup_id}'")
            return {}