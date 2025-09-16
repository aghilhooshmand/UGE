"""
Experiment View for UGE Application

This module provides the experiment view for running and managing
Grammatical Evolution experiments.

Classes:
- ExperimentView: Main view for experiment operations

Author: UGE Team
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
from uge.views.components.base_view import BaseView
from uge.views.components.forms import Forms
from uge.views.components.charts import Charts
from uge.models.experiment import ExperimentConfig, ExperimentResult


class ExperimentView(BaseView):
    """
    View for experiment operations.
    
    This view handles the user interface for running and managing
    Grammatical Evolution experiments.
    
    Attributes:
        on_experiment_submit (Optional[Callable]): Callback for experiment submission
        on_experiment_cancel (Optional[Callable]): Callback for experiment cancellation
    """
    
    def __init__(self, experiment_controller, on_experiment_submit: Optional[Callable] = None,
                 on_experiment_cancel: Optional[Callable] = None):
        """
        Initialize experiment view.
        
        Args:
            experiment_controller: Controller for experiment operations
            on_experiment_submit (Optional[Callable]): Callback for experiment submission
            on_experiment_cancel (Optional[Callable]): Callback for experiment cancellation
        """
        super().__init__(
            title="🚀 Run Experiment",
            description="Create and execute new Grammatical Evolution experiments"
        )
        self.experiment_controller = experiment_controller
        self.on_experiment_submit = on_experiment_submit
        self.on_experiment_cancel = on_experiment_cancel
    
    def render(self, help_texts: Dict[str, str] = None, 
               datasets: List[str] = None, 
               grammars: List[str] = None):
        """
        Render the experiment view.
        
        Args:
            help_texts (Dict[str, str]): Help texts for form fields
            datasets (List[str]): Available datasets
            grammars (List[str]): Available grammars
        """
        self.render_header()
        
        # Create experiment form
        form_submitted, form_data = Forms.create_experiment_form(
            help_texts=help_texts,
            datasets=datasets,
            grammars=grammars
        )
        
        if form_submitted:
            self._handle_experiment_submission(form_data)
    
    def _handle_experiment_submission(self, form_data: Dict[str, Any]):
        """
        Handle experiment form submission.
        
        Args:
            form_data (Dict[str, Any]): Form data
        """
        try:
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_name=form_data['experiment_name'],
                dataset=form_data['dataset'],
                grammar=form_data['grammar'],
                fitness_metric=form_data['fitness_metric'],
                n_runs=form_data['n_runs'],
                generations=form_data['generations'],
                population=form_data['population'],
                p_crossover=form_data['p_crossover'],
                p_mutation=form_data['p_mutation'],
                elite_size=form_data['elite_size'],
                tournsize=form_data['tournsize'],
                halloffame_size=form_data['halloffame_size'],
                max_tree_depth=form_data['max_tree_depth'],
                min_init_tree_depth=form_data['min_init_tree_depth'],
                max_init_tree_depth=form_data['max_init_tree_depth'],
                min_init_genome_length=form_data['min_init_genome_length'],
                max_init_genome_length=form_data['max_init_genome_length'],
                codon_size=form_data['codon_size'],
                codon_consumption=form_data['codon_consumption'],
                genome_representation=form_data['genome_representation'],
                initialisation=form_data['initialisation'],
                random_seed=form_data['random_seed'],
                label_column=form_data['label_column'],
                test_size=form_data['test_size'],
                report_items=form_data['report_items']
            )
            
            # Validate that label column exists in dataset
            if config.dataset != "none":
                try:
                    dataset_service = self.experiment_controller.dataset_service
                    dataset_info = dataset_service.get_dataset_info(config.dataset)
                    if dataset_info and dataset_info.columns:
                        if config.label_column not in dataset_info.columns:
                            st.error(f"❌ Label column '{config.label_column}' not found in dataset '{config.dataset}'. Available columns: {', '.join(dataset_info.columns)}")
                            return
                except Exception as e:
                    st.warning(f"⚠️ Could not validate label column: {e}")
            
            # Run the experiment using the controller
            st.info(f"🚀 Starting experiment '{config.experiment_name}'...")
            
            # Create expandable section for running details like UGE_ref
            with st.expander("🔍 Show Running Details", expanded=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a container for all run details
                all_runs_container = st.container()
                
                # Dedicated live log placeholder
                live_placeholder = st.empty()
                
                # Run the experiment
                experiment = self.experiment_controller.run_experiment(
                    config, live_placeholder=live_placeholder,
                    progress_bar=progress_bar, status_text=status_text, all_runs_container=all_runs_container
                )
            
            if experiment:
                st.success(f"✅ Experiment '{config.experiment_name}' completed successfully!")
                
                # Show results
                self._show_experiment_results(experiment)
            else:
                st.error(f"❌ Experiment '{config.experiment_name}' failed!")
                
        except Exception as e:
            self.handle_error(e, "creating experiment configuration")
    
    def _show_experiment_results(self, experiment):
        """
        Show experiment results.
        
        Args:
            experiment: Completed experiment object
        """
        st.subheader("📊 Experiment Results")
        
        # Show basic experiment info
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Experiment Name", experiment.config.experiment_name)
            st.metric("Total Runs", experiment.config.n_runs)
            st.metric("Completed Runs", len(experiment.results))
        
        with col2:
            st.metric("Dataset", experiment.config.dataset)
            st.metric("Grammar", experiment.config.grammar)
            st.metric("Population", experiment.config.population)
        
        # Show best results
        if experiment.results:
            best_result = experiment.get_best_result()
            if best_result:
                st.subheader("🏆 Best Result")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best Fitness", f"{best_result.best_training_fitness:.4f}")
                with col2:
                    st.metric("Test Fitness", f"{best_result.fitness_test[-1]:.4f}" if best_result.fitness_test else "N/A")
                with col3:
                    st.metric("Generations", len(best_result.max))
        
        st.success("🎉 Experiment completed successfully!")
    
    def render_experiment_progress(self, experiment_name: str, current_run: int, 
                                 total_runs: int, progress: float):
        """
        Render experiment progress.
        
        Args:
            experiment_name (str): Name of the experiment
            current_run (int): Current run number
            total_runs (int): Total number of runs
            progress (float): Progress (0.0 to 1.0)
        """
        st.subheader(f"🏃 Running: {experiment_name}")
        st.progress(progress)
        st.write(f"Run {current_run} of {total_runs}")
    
    def render_run_details(self, run_number: int, result: ExperimentResult):
        """
        Render run details.
        
        Args:
            run_number (int): Run number
            result (ExperimentResult): Run result
        """
        with st.expander(f"🔍 Run {run_number} Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Fitness:**", f"{result.best_training_fitness:.4f}")
                st.write("**Best Depth:**", result.best_depth)
                st.write("**Genome Length:**", result.best_genome_length)
            
            with col2:
                st.write("**Used Codons:**", f"{result.best_used_codons:.2%}")
                st.write("**Generations:**", len(result.max))
                st.write("**Timestamp:**", result.timestamp)
            
            if result.best_phenotype:
                st.write("**Best Phenotype:**")
                st.code(result.best_phenotype, language='python')
    
    def render_experiment_completion(self, experiment_name: str, 
                                   total_runs: int, results: List[ExperimentResult]):
        """
        Render experiment completion message.
        
        Args:
            experiment_name (str): Name of the experiment
            total_runs (int): Total number of runs
            results (List[ExperimentResult]): All run results
        """
        self.show_success(f"✅ Experiment '{experiment_name}' completed successfully!")
        st.info("📊 Go to the 'Analysis' page to view detailed results, charts, and export data.")
        
        # Show summary statistics
        if results:
            best_fitness = max(r.best_training_fitness for r in results if r.best_training_fitness is not None)
            avg_fitness = sum(r.best_training_fitness for r in results if r.best_training_fitness is not None) / len(results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Runs", total_runs)
            with col2:
                st.metric("Best Fitness", f"{best_fitness:.4f}")
            with col3:
                st.metric("Average Fitness", f"{avg_fitness:.4f}")
    
    def render_live_output(self, placeholder, output: str):
        """
        Render live output during experiment execution.
        
        Args:
            placeholder: Streamlit placeholder for output
            output (str): Output text to display
        """
        placeholder.code(output)
    
    def render_experiment_form_validation(self, errors: List[str]):
        """
        Render form validation errors.
        
        Args:
            errors (List[str]): List of validation errors
        """
        for error in errors:
            self.show_error(error)
    
    def render_experiment_configuration_summary(self, config: ExperimentConfig):
        """
        Render experiment configuration summary.
        
        Args:
            config (ExperimentConfig): Experiment configuration
        """
        with st.expander("📋 Experiment Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Info:**")
                st.write(f"- Name: {config.experiment_name}")
                st.write(f"- Dataset: {config.dataset}")
                st.write(f"- Grammar: {config.grammar}")
                st.write(f"- Fitness Metric: {config.fitness_metric}")
                st.write(f"- Runs: {config.n_runs}")
                
                st.write("**GA Parameters:**")
                st.write(f"- Population: {config.population}")
                st.write(f"- Generations: {config.generations}")
                st.write(f"- Crossover: {config.p_crossover}")
                st.write(f"- Mutation: {config.p_mutation}")
            
            with col2:
                st.write("**GE Parameters:**")
                st.write(f"- Max Tree Depth: {config.max_tree_depth}")
                st.write(f"- Genome Length: {config.min_init_genome_length}-{config.max_init_genome_length}")
                st.write(f"- Codon Size: {config.codon_size}")
                st.write(f"- Initialization: {config.initialisation}")
                
                st.write("**Dataset Parameters:**")
                st.write(f"- Test Size: {config.test_size}")
                st.write(f"- Random Seed: {config.random_seed}")
                if config.label_column:
                    st.write(f"- Label Column: {config.label_column}")