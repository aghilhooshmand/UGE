"""
Storage Service for UGE Application

This module provides file storage and persistence services for the UGE application.
It handles saving and loading setups, results, and other data.

Classes:
- StorageService: Main service for file operations and data persistence

Author: UGE Team
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from uge.models.setup import Setup, SetupConfig, SetupResult
from uge.utils.constants import FILE_PATHS


class StorageService:
    """
    Service for file storage and data persistence operations.
    
    This service handles all file I/O operations for the UGE application,
    including saving/loading setups, results, and managing the
    file system structure.
    
    Attributes:
        project_root (Path): Root directory of the project
        results_dir (Path): Directory for storing results
        setups_dir (Path): Directory for storing setups
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize storage service.
        
        Args:
            project_root (Optional[Path]): Project root directory
        """
        self.project_root = project_root or FILE_PATHS['project_root']
        self.results_dir = self.project_root / "results"
        self.setups_dir = self.results_dir / "setups"
        
        # Ensure directories exist
        self.setups_dir.mkdir(parents=True, exist_ok=True)
    
    def save_setup_config(self, exp_id: str, config: SetupConfig) -> Path:
        """
        Save setup configuration to file.
        
        Args:
            exp_id (str): Setup ID
            config (SetupConfig): Configuration to save
            
        Returns:
            Path: Path to saved configuration file
        """
        exp_dir = self.setups_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        config_file = exp_dir / "setup_config.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        return config_file
    
    def load_setup_config(self, exp_id: str) -> Optional[SetupConfig]:
        """
        Load setup configuration from file.
        
        Args:
            exp_id (str): Setup ID
            
        Returns:
            Optional[SetupConfig]: Loaded configuration or None if not found
        """
        config_file = self.setups_dir / exp_id / "setup_config.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            return SetupConfig.from_dict(data)
        except Exception as e:
            raise ValueError(f"Error loading setup config: {e}")
    
    def save_run_result(self, exp_id: str, run_id: str, result: SetupResult) -> Path:
        """
        Save run result to file.
        
        Args:
            exp_id (str): Setup ID
            run_id (str): Run ID
            result (SetupResult): Result to save
            
        Returns:
            Path: Path to saved result file
        """
        exp_dir = self.setups_dir / exp_id
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
    
    def load_run_result(self, exp_id: str, run_id: str) -> Optional[SetupResult]:
        """
        Load run result from file.
        
        Args:
            exp_id (str): Setup ID
            run_id (str): Run ID
            
        Returns:
            Optional[SetupResult]: Loaded result or None if not found
        """
        result_file = self.setups_dir / exp_id / "runs" / run_id / "result.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            return SetupResult.from_dict(data)
        except Exception as e:
            raise ValueError(f"Error loading run result: {e}")
    
    def list_setups(self) -> List[Path]:
        """
        List all setup directories sorted by name (newest first).
        
        Returns:
            List[Path]: List of setup directory paths sorted by name descending
        """
        if not self.setups_dir.exists():
            return []
        
        # Get all setup directories and sort by name descending (newest first)
        # Since setup names include timestamp (exp_YYYYMMDD_HHMMSS_*), 
        # sorting by name descending gives us newest setups first
        setup_dirs = [p for p in self.setups_dir.iterdir() 
                          if p.is_dir() and p.name.startswith('exp_')]
        
        return sorted(setup_dirs, key=lambda x: x.name, reverse=True)
    
    def list_setup_runs(self, exp_id: str) -> List[Path]:
        """
        List all runs for an setup sorted by name (newest first).
        
        Args:
            exp_id (str): Setup ID
            
        Returns:
            List[Path]: List of run directory paths sorted by name descending
        """
        runs_dir = self.setups_dir / exp_id / "runs"
        if not runs_dir.exists():
            return []
        
        # Get all run directories and sort by name descending (newest first)
        # Since run names include timestamp (run_YYYYMMDD_HHMMSS_*), 
        # sorting by name descending gives us newest runs first
        run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        
        return sorted(run_dirs, key=lambda x: x.name, reverse=True)
    
    def load_setup(self, exp_id: str) -> Optional[Setup]:
        """
        Load complete setup with all runs.
        
        Args:
            exp_id (str): Setup ID
            
        Returns:
            Optional[Setup]: Complete setup or None if not found
        """
        # Load configuration
        config = self.load_setup_config(exp_id)
        if not config:
            return None
        
        # Load all run results
        results = {}
        run_dirs = self.list_setup_runs(exp_id)
        
        for run_dir in run_dirs:
            run_id = run_dir.name
            result = self.load_run_result(exp_id, run_id)
            if result:
                results[run_id] = result
        
        # Create setup object
        setup = Setup(
            id=exp_id,
            config=config,
            results=results
        )
        
        # Update status based on results
        if setup.is_completed():
            setup.status = 'completed'
            setup.completed_at = max(r.timestamp for r in results.values())
        elif results:
            setup.status = 'running'
        
        return setup
    
    def save_setup(self, setup: Setup) -> None:
        """
        Save complete setup with all runs.
        
        Args:
            setup (Setup): Setup to save
        """
        # Save configuration
        self.save_setup_config(setup.id, setup.config)
        
        # Save all run results
        for run_id, result in setup.results.items():
            self.save_run_result(setup.id, run_id, result)
    
    def delete_setup(self, exp_id: str) -> bool:
        """
        Delete setup and all its data.
        
        Args:
            exp_id (str): Setup ID
            
        Returns:
            bool: True if deleted successfully, False if not found
        """
        exp_dir = self.setups_dir / exp_id
        if not exp_dir.exists():
            return False
        
        try:
            import shutil
            shutil.rmtree(exp_dir)
            return True
        except Exception as e:
            raise RuntimeError(f"Error deleting setup: {e}")
    
    def get_setup_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored setups.
        
        Returns:
            Dict[str, Any]: Setup statistics
        """
        setups = self.list_setups()
        
        total_runs = 0
        completed_setups = 0
        running_setups = 0
        
        for exp_dir in setups:
            exp_id = exp_dir.name
            setup = self.load_setup(exp_id)
            if setup:
                total_runs += len(setup.results)
                if setup.is_completed():
                    completed_setups += 1
                elif setup.results:
                    running_setups += 1
        
        return {
            'total_setups': len(setups),
            'completed_setups': completed_setups,
            'running_setups': running_setups,
            'total_runs': total_runs
        }
    
    def cleanup_old_setups(self, days_old: int = 30) -> int:
        """
        Clean up setups older than specified days.
        
        Args:
            days_old (int): Number of days to keep setups
            
        Returns:
            int: Number of setups deleted
        """
        import datetime as dt
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=days_old)
        deleted_count = 0
        
        setups = self.list_setups()
        for exp_dir in setups:
            try:
                # Check creation date
                creation_time = dt.datetime.fromtimestamp(exp_dir.stat().st_ctime)
                if creation_time < cutoff_date:
                    if self.delete_setup(exp_dir.name):
                        deleted_count += 1
            except Exception:
                # Skip if we can't determine age
                continue
        
        return deleted_count