"""
Analysis View for UGE Application

This module provides the analysis view for analyzing and visualizing
Grammatical Evolution experiment results.

Classes:
- AnalysisView: Main view for analysis operations

Author: UGE Team
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from uge.views.components.base_view import BaseView
from uge.views.components.charts import Charts
from uge.views.components.forms import Forms
from uge.models.experiment import Experiment, ExperimentResult
from uge.utils.tooltip_manager import tooltip_manager


class AnalysisView(BaseView):
    """
    View for analysis operations.
    
    This view handles the user interface for analyzing and visualizing
    GE experiment results.
    
    Attributes:
        on_experiment_select (Optional[Callable]): Callback for experiment selection
        on_analysis_options_change (Optional[Callable]): Callback for analysis options change
        on_export_data (Optional[Callable]): Callback for data export
    """
    
    def __init__(self, on_experiment_select: Optional[Callable] = None,
                 on_analysis_options_change: Optional[Callable] = None,
                 on_export_data: Optional[Callable] = None):
        """
        Initialize analysis view.
        
        Args:
            on_experiment_select (Optional[Callable]): Callback for experiment selection
            on_analysis_options_change (Optional[Callable]): Callback for analysis options change
            on_export_data (Optional[Callable]): Callback for data export
        """
        super().__init__(
            title="📈 Analysis",
            description="Analyze and visualize experiment results"
        )
        self.on_experiment_select = on_experiment_select
        self.on_analysis_options_change = on_analysis_options_change
        self.on_export_data = on_export_data
    
    def render(self, experiments: List[Dict[str, str]] = None):
        """
        Render the analysis view.
        
        Args:
            experiments (List[Dict[str, str]]): Available experiments with id, name, path
        """
        if experiments is None:
            experiments = []
        
        self.render_header()
        
        # Always require experiment selection
        if not experiments:
            self.render_no_experiments()
            return
            
        # Experiment selection
        selected_experiment = Forms.create_experiment_selection_form(experiments)
        
        if selected_experiment:
            self._handle_experiment_selection(selected_experiment)
    
    def _handle_experiment_selection(self, experiment_id: str):
        """
        Handle experiment selection.
        
        Args:
            experiment_id (str): Selected experiment ID
        """
        try:
            if self.on_experiment_select:
                experiment = self.on_experiment_select(experiment_id)
                if experiment:
                    self._render_experiment_analysis(experiment)
            else:
                st.info(f"Selected experiment: {experiment_id}")
        except Exception as e:
            self.handle_error(e, "selecting experiment")
    
    def _render_experiment_analysis(self, experiment: Experiment):
        """
        Render experiment analysis.
        
        Args:
            experiment (Experiment): Experiment to analyze
        """
        st.subheader(f"Analysis: {experiment.config.experiment_name}")
        
        # Analysis options
        analysis_options = Forms.create_analysis_options_form()
        
        if self.on_analysis_options_change:
            self.on_analysis_options_change(analysis_options)
        
        # Render analysis based on options
        if analysis_options.get('show_statistics', True):
            with st.expander("📊 Experiment Statistics", expanded=True):
                self._render_experiment_statistics(experiment)
        
        if analysis_options.get('show_charts', True):
            with st.expander("📈 Experiment Charts", expanded=True):
                self._render_experiment_charts(experiment)
        
        if analysis_options.get('show_best_individual', True):
            with st.expander("🏆 Best Individual", expanded=True):
                self._render_best_individual(experiment)
        
        if analysis_options.get('export_data', False):
            with st.expander("📤 Export Data", expanded=True):
                self._render_export_options(experiment)
    
    def _render_experiment_statistics(self, experiment: Experiment):
        """Render experiment statistics."""
        # Basic Metrics Panel
        with st.expander("📈 Basic Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Runs", 
                len(experiment.results),
                help=tooltip_manager.get_metric_tooltip('total_runs')
            )
        
        with col2:
            best_fitness = experiment.get_best_result()
            best_fitness_val = best_fitness.best_training_fitness if best_fitness else 0
            if experiment.config.fitness_direction == 1:  # maximize
                help_text = tooltip_manager.get_metric_tooltip('best_fitness', 'maximize')
            else:  # minimize
                help_text = tooltip_manager.get_metric_tooltip('best_fitness', 'minimize')
            st.metric("Best Fitness", f"{best_fitness_val:.4f}", help=help_text)
        
        with col3:
            avg_fitness = experiment.get_average_fitness()
            avg_fitness_val = avg_fitness if avg_fitness else 0
            st.metric(
                "Average Fitness", 
                f"{avg_fitness_val:.4f}",
                help=tooltip_manager.get_metric_tooltip('average_fitness')
            )
        
        with col4:
            status = "✅ Completed" if experiment.is_completed() else "🔄 Running"
            st.metric(
                "Status", 
                status,
                help=tooltip_manager.get_metric_tooltip('status')
            )
        
        # Advanced Statistics Panel
        if experiment.results:
            with st.expander("📊 Advanced Statistics", expanded=True):
                # Calculate additional statistics
                fitnesses = [r.best_training_fitness for r in experiment.results.values() if r.best_training_fitness is not None]
                depths = [r.best_depth for r in experiment.results.values() if r.best_depth is not None]
                genome_lengths = [r.best_genome_length for r in experiment.results.values() if r.best_genome_length is not None]
                used_codons = [r.best_used_codons for r in experiment.results.values() if r.best_used_codons is not None]
                generations = [len(r.max) for r in experiment.results.values()]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write("**📈 Fitness Statistics**")
                    if fitnesses:
                        fitness_std = (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5
                        st.metric(
                            "Std Dev", 
                            f"{fitness_std:.4f}",
                            help="Standard deviation of fitness values. Lower = more consistent results."
                        )
                        
                        fitness_range = max(fitnesses) - min(fitnesses)
                        st.metric(
                            "Range", 
                            f"{fitness_range:.4f}",
                            help="Difference between best and worst fitness. Smaller range = more stable algorithm."
                        )
                    else:
                        st.metric("Std Dev", "N/A")
                        st.metric("Range", "N/A")
                
                with col2:
                    st.write("**🌳 Tree Depth Statistics**")
                    if depths:
                        avg_depth = sum(depths)/len(depths)
                        st.metric(
                            "Avg Depth", 
                            f"{avg_depth:.1f}",
                            help="Average tree depth of best individuals. Indicates solution complexity."
                        )
                        
                        max_depth = max(depths)
                        st.metric(
                            "Max Depth", 
                            f"{max_depth}",
                            help="Maximum tree depth reached. Shows if depth limit is being used effectively."
                        )
                    else:
                        st.metric("Avg Depth", "N/A")
                        st.metric("Max Depth", "N/A")
                
                with col3:
                    st.write("**🧬 Genome Statistics**")
                    if genome_lengths:
                        avg_genome = sum(genome_lengths)/len(genome_lengths)
                        st.metric(
                            "Avg Length", 
                            f"{avg_genome:.1f}",
                            help="Average genome length. Longer genomes may indicate more complex solutions."
                        )
                        
                        max_genome = max(genome_lengths)
                        st.metric(
                            "Max Length", 
                            f"{max_genome}",
                            help="Maximum genome length. Shows if length limits are being utilized."
                        )
                    else:
                        st.metric("Avg Length", "N/A")
                        st.metric("Max Length", "N/A")
                
                with col4:
                    st.write("**⚡ Efficiency Statistics**")
                    if used_codons:
                        avg_used_codons = sum(used_codons)/len(used_codons)
                        st.metric(
                            "Used Codons", 
                            f"{avg_used_codons:.2%}",
                            help="Percentage of genome actually used. Higher = more efficient genome utilization."
                        )
                        
                        avg_gens = sum(generations)/len(generations)
                        st.metric(
                            "Avg Generations", 
                            f"{avg_gens:.1f}",
                            help="Average generations completed. Shows if runs are finishing early or using full time."
                        )
                    else:
                        st.metric("Used Codons", "N/A")
                        st.metric("Avg Generations", "N/A")
        
        # Performance Analysis Panel
        with st.expander("🎯 Performance Analysis", expanded=True):
            # Configurable thresholds
            st.write("**⚙️ Analysis Thresholds**")
            col1, col2 = st.columns(2)
            
            with col1:
                if experiment.config.fitness_direction == 1:  # maximize
                    default_target = 0.95
                    target_desc = "Success Target (≥)"
                    help_text = "Minimum fitness value to consider a run successful"
                else:  # minimize
                    default_target = 0.05
                    target_desc = "Success Target (≤)"
                    help_text = "Maximum fitness value to consider a run successful"
                
                success_target = st.number_input(
                    target_desc,
                    min_value=0.0,
                    max_value=1.0 if experiment.config.fitness_direction == 1 else 10.0,
                    value=default_target,
                    step=0.01,
                    help=help_text
                )
            
            with col2:
                convergence_threshold = st.number_input(
                    "Convergence Threshold",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.4f",
                    help="Minimum improvement per generation to consider still converging"
                )
            
            # Additional thresholds row
            col3, col4 = st.columns(2)
            
            with col3:
                high_consistency_threshold = st.number_input(
                    "High Consistency Threshold (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Coefficient of variation below this is considered 'High' consistency"
                )
            
            with col4:
                medium_consistency_threshold = st.number_input(
                    "Medium Consistency Threshold (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    help="Coefficient of variation below this is considered 'Medium' consistency"
                )
            
            if fitnesses:
                # Find best and worst runs based on fitness direction
                if experiment.config.fitness_direction == 1:  # maximize
                    best_run = max(experiment.results.items(), key=lambda x: x[1].best_training_fitness or 0)
                    worst_run = min(experiment.results.items(), key=lambda x: x[1].best_training_fitness or float('inf'))
                else:  # minimize
                    best_run = min(experiment.results.items(), key=lambda x: x[1].best_training_fitness or float('inf'))
                    worst_run = max(experiment.results.items(), key=lambda x: x[1].best_training_fitness or 0)
                
                # Convert run IDs to simple run numbers
                sorted_runs = sorted(experiment.results.items(), key=lambda x: x[1].timestamp)
                run_id_to_number = {run_id: idx + 1 for idx, (run_id, _) in enumerate(sorted_runs)}
                
                best_run_number = run_id_to_number.get(best_run[0], 1)
                worst_run_number = run_id_to_number.get(worst_run[0], 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Best Run", 
                        f"RUN_{best_run_number}", 
                        f"{best_run[1].best_training_fitness:.4f}",
                        help=f"The run that achieved the highest fitness value. Full run ID: {best_run[0]}"
                    )
                    
                    st.metric(
                        "Worst Run", 
                        f"RUN_{worst_run_number}", 
                        f"{worst_run[1].best_training_fitness:.4f}",
                        help=f"The run with the lowest fitness value. Full run ID: {worst_run[0]}"
                    )
                    
                    # Convergence analysis using configurable threshold
                    convergence_gens = []
                    for result in experiment.results.values():
                        if result.max and len(result.max) > 10:
                            # Find where fitness improvement becomes minimal
                            for i in range(len(result.max) - 10, len(result.max)):
                                if abs(result.max[i] - result.max[i-1]) < convergence_threshold:
                                    convergence_gens.append(i)
                                    break
                    
                    if convergence_gens:
                        avg_convergence = sum(convergence_gens) / len(convergence_gens)
                        st.metric(
                            "Avg Convergence Gen", 
                            f"{avg_convergence:.1f}",
                            help=f"Average generation where runs stop improving significantly (< {convergence_threshold:.4f} improvement for 10 generations). Lower values indicate faster convergence."
                        )
                
                with col2:
                    # Success rate analysis using configurable threshold
                    if experiment.config.fitness_direction == 1:  # maximize
                        successful_runs = sum(1 for f in fitnesses if f >= success_target)
                        target_desc = f"≥ {success_target:.2f} fitness"
                    else:  # minimize
                        successful_runs = sum(1 for f in fitnesses if f <= success_target)
                        target_desc = f"≤ {success_target:.2f} fitness"
                    
                    success_rate = (successful_runs / len(fitnesses)) * 100 if fitnesses else 0
                    st.metric(
                        "Success Rate", 
                        f"{success_rate:.1f}%",
                        help=f"Percentage of runs achieving target performance ({target_desc}). Higher is better."
                    )
                    
                    # Consistency analysis using configurable thresholds
                    cv = 0  # Initialize cv variable
                    if len(fitnesses) > 1:
                        cv = (fitness_std / (sum(fitnesses)/len(fitnesses))) * 100 if sum(fitnesses)/len(fitnesses) != 0 else 0
                        st.metric(
                            "Coefficient of Variation", 
                            f"{cv:.1f}%",
                            help="Measures consistency between runs. Lower values indicate more stable, reproducible results."
                        )
                        
                        consistency = "High" if cv < high_consistency_threshold else "Medium" if cv < medium_consistency_threshold else "Low"
                        consistency_color = "🟢" if consistency == "High" else "🟡" if consistency == "Medium" else "🔴"
                        st.metric(
                            "Consistency", 
                            f"{consistency_color} {consistency}",
                            help=f"High: < {high_consistency_threshold:.1f}% variation (very stable), Medium: {high_consistency_threshold:.1f}-{medium_consistency_threshold:.1f}% (moderately stable), Low: > {medium_consistency_threshold:.1f}% (unstable)"
                        )
                
                # Performance insights
                st.write("**💡 Performance Insights:**")
                performance_range = max(fitnesses) - min(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                
                insights = []
                if success_rate == 0:
                    insights.append(("🔴 **No runs achieved target performance** - consider increasing generations, adjusting parameters, or trying different initialization strategies", 
                                   tooltip_manager.get_insight_tooltip('success_rate', '0_percent')))
                elif success_rate < 50:
                    insights.append(("🟡 **Low success rate** - algorithm needs improvement or more runs", 
                                   tooltip_manager.get_insight_tooltip('success_rate', 'low_percent')))
                else:
                    insights.append(("🟢 **Good success rate** - algorithm is performing well", 
                                   tooltip_manager.get_insight_tooltip('success_rate', 'good_percent')))
                
                if cv < 10:
                    insights.append(("🟢 **High consistency** - algorithm is stable and reproducible", 
                                   tooltip_manager.get_insight_tooltip('consistency', 'high')))
                elif cv < 20:
                    insights.append(("🟡 **Medium consistency** - some variation between runs", 
                                   tooltip_manager.get_insight_tooltip('consistency', 'medium')))
                else:
                    insights.append(("🔴 **Low consistency** - high variation, algorithm may be unstable", 
                                   tooltip_manager.get_insight_tooltip('consistency', 'low')))
                
                if convergence_gens and avg_convergence < len(experiment.results[list(experiment.results.keys())[0]].max) * 0.5:
                    insights.append(("🟢 **Fast convergence** - algorithm finds solutions quickly", 
                                   tooltip_manager.get_insight_tooltip('convergence', 'fast')))
                elif convergence_gens:
                    insights.append(("🟡 **Slow convergence** - consider running more generations", 
                                   tooltip_manager.get_insight_tooltip('convergence', 'slow')))
                
                if performance_range < avg_fitness * 0.1:
                    insights.append(("🟢 **Small performance range** - consistent results across runs", 
                                   tooltip_manager.get_insight_tooltip('performance_range', 'small')))
                else:
                    insights.append(("🟡 **Large performance range** - significant variation in run quality", 
                                   tooltip_manager.get_insight_tooltip('performance_range', 'large')))
                
                for insight_text, help_text in insights:
                    col1, col2 = st.columns([0.95, 0.05])
                    with col1:
                        st.info(insight_text)
                    with col2:
                        st.metric("", "", help=help_text)
        
        # Runs Statistics Panel
        if experiment.results:
            with st.expander("📋 Runs Statistics", expanded=True):
                run_data = []
                # Sort runs by timestamp to get consistent ordering
                sorted_runs = sorted(experiment.results.items(), key=lambda x: x[1].timestamp)
                
                for run_idx, (run_id, result) in enumerate(sorted_runs, 1):
                    # Calculate comprehensive metrics for each run
                    final_fitness = result.max[-1] if result.max else 0
                    initial_fitness = result.max[0] if result.max else 0
                    improvement = final_fitness - initial_fitness
                    
                    # Calculate average fitness across all generations
                    avg_fitness = sum(result.avg) / len(result.avg) if result.avg else 0
                    
                    # Calculate standard deviation
                    std_fitness = result.std[-1] if result.std else 0
                    
                    # Get test fitness if available
                    test_fitness = result.fitness_test[-1] if result.fitness_test else None
                    
                    # Calculate convergence generation (where improvement becomes minimal)
                    convergence_gen = 0
                    if result.max and len(result.max) > 5:
                        total_improvement = result.max[-1] - result.max[0]
                        if total_improvement > 0:
                            for i in range(1, len(result.max)):
                                improvement_so_far = result.max[i] - result.max[0]
                                if improvement_so_far >= 0.95 * total_improvement:
                                    convergence_gen = i
                                    break
                    
                    run_data.append({
                        'Run': f"RUN_{run_idx}",
                        'Best Fitness': f"{result.best_training_fitness or 0:.4f}",
                        'Final Fitness': f"{final_fitness:.4f}",
                        'Initial Fitness': f"{initial_fitness:.4f}",
                        'Average Fitness': f"{avg_fitness:.4f}",
                        'Std Dev': f"{std_fitness:.4f}",
                        'Improvement': f"{improvement:.4f}",
                        'Test Fitness': f"{test_fitness:.4f}" if test_fitness is not None else "N/A",
                        'Best Depth': result.best_depth or 0,
                        'Genome Length': result.best_genome_length or 0,
                        'Used Codons': f"{(result.best_used_codons or 0):.2%}",
                        'Generations': len(result.max),
                        'Convergence Gen': convergence_gen,
                        'Best Phenotype': result.best_phenotype or "N/A",
                        'Timestamp': result.timestamp
                    })
                
                run_df = pd.DataFrame(run_data)
                st.dataframe(run_df, hide_index=True)
    
    def _render_experiment_charts(self, experiment: Experiment):
        """Render experiment charts."""
        
        if not experiment.results:
            st.info("No results available for charting")
            return
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Individual Run Charts", "Experiment-wide Chart"],
            help="Choose between individual run charts or experiment-wide analysis"
        )
        
        # Measurement selection
        st.subheader("📊 Measurement Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Data:**")
            show_train_avg = st.checkbox("Average", value=True)
            show_train_min = st.checkbox("Minimum", value=True)
            show_train_max = st.checkbox("Maximum", value=True)
        
        with col2:
            st.write("**Test Data:**")
            show_test_avg = st.checkbox("Test Average", value=False)
            show_test_min = st.checkbox("Test Minimum", value=False)
            show_test_max = st.checkbox("Test Maximum", value=False)
        
        # Create measurement options
        measurement_options = {
            'train_avg': show_train_avg,
            'train_min': show_train_min,
            'train_max': show_train_max,
            'test_avg': show_test_avg,
            'test_min': show_test_min,
            'test_max': show_test_max
        }
        
        if chart_type == "Individual Run Charts":
            self._render_individual_run_charts(experiment, measurement_options)
        else:
            self._render_experiment_wide_chart(experiment, measurement_options)
    
    def _render_individual_run_charts(self, experiment: Experiment, measurement_options: Dict[str, bool]):
        """Render individual run chart for selected run."""
        # Sort runs by timestamp to get consistent ordering
        sorted_runs = sorted(experiment.results.items(), key=lambda x: x[1].timestamp)
        
        # Create run selection options
        run_options = []
        for run_idx, (run_id, result) in enumerate(sorted_runs, 1):
            run_options.append(f"RUN_{run_idx}")
        
        # Add run selection dropdown
        selected_run = st.selectbox(
            "Select Run to Display:",
            options=run_options,
            help="Choose which individual run to display in the chart"
        )
        
        # Find the selected run
        selected_run_idx = run_options.index(selected_run)
        selected_run_id, selected_result = sorted_runs[selected_run_idx]
        
        # Display the selected run
        st.subheader(f"{selected_run}")
        Charts.plot_individual_run_with_bars(
            selected_result, 
            title=f"Fitness Evolution - {selected_run}",
            fitness_metric=experiment.config.fitness_metric,
            fitness_direction=experiment.config.fitness_direction,
            measurement_options=measurement_options
        )
    
    def _render_experiment_wide_chart(self, experiment: Experiment, measurement_options: Dict[str, bool]):
        """Render experiment-wide chart with min/max/avg across all runs."""
        Charts.plot_experiment_wide_with_bars(
            experiment.results, 
            title=f"Experiment-wide Analysis - {experiment.config.experiment_name}",
            fitness_metric=experiment.config.fitness_metric,
            fitness_direction=experiment.config.fitness_direction,
            measurement_options=measurement_options
        )
    
    def _render_best_individual(self, experiment: Experiment):
        """Render best individual information."""
        
        best_result = experiment.get_best_result()
        if not best_result:
            st.info("No best individual found")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fitness Information:**")
            st.write(f"- Best Training Fitness: {best_result.best_training_fitness:.4f}")
            st.write(f"- Best Depth: {best_result.best_depth}")
            st.write(f"- Genome Length: {best_result.best_genome_length}")
            st.write(f"- Used Codons: {best_result.best_used_codons:.2%}")
        
        with col2:
            st.write("**Evolution Information:**")
            st.write(f"- Total Generations: {len(best_result.max)}")
            st.write(f"- Final Generation Best: {best_result.max[-1] if best_result.max else 'N/A'}")
            st.write(f"- Final Generation Avg: {best_result.avg[-1] if best_result.avg else 'N/A'}")
            st.write(f"- Timestamp: {best_result.timestamp}")
        
        if best_result.best_phenotype:
            st.subheader("🧬 Best Phenotype")
            st.code(best_result.best_phenotype, language='python')
    
    
    def _render_export_options(self, experiment: Experiment):
        """Render export options."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Results as CSV"):
                self._handle_export_results(experiment)
        
        with col2:
            if st.button("📋 Export Configuration as JSON"):
                self._handle_export_config(experiment)
    
    def _handle_export_results(self, experiment: Experiment):
        """Handle results export."""
        try:
            if self.on_export_data:
                export_data = self.on_export_data(experiment, 'results')
                if export_data:
                    st.download_button(
                        label="💾 Download Results CSV",
                        data=export_data,
                        file_name=f"{experiment.config.experiment_name}_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            self.handle_error(e, "exporting results")
    
    def _handle_export_config(self, experiment: Experiment):
        """Handle configuration export."""
        try:
            if self.on_export_data:
                export_data = self.on_export_data(experiment, 'config')
                if export_data:
                    st.download_button(
                        label="💾 Download Configuration JSON",
                        data=export_data,
                        file_name=f"{experiment.config.experiment_name}_config.json",
                        mime="application/json"
                    )
        except Exception as e:
            self.handle_error(e, "exporting configuration")
    
    def render_experiment_not_found(self, experiment_id: str):
        """
        Render experiment not found message.
        
        Args:
            experiment_id (str): Experiment ID that was not found
        """
        self.show_error(f"Experiment '{experiment_id}' not found")
    
    def render_no_experiments(self):
        """Render no experiments available message."""
        st.info("No experiments available for analysis")
        st.markdown("""
        To analyze experiments:
        1. Go to the "Run Experiment" page
        2. Create and run an experiment
        3. Return to this page to analyze the results
        """)
    
    def render_analysis_error(self, error: str):
        """
        Render analysis error message.
        
        Args:
            error (str): Error message
        """
        self.show_error(f"Analysis error: {error}")
    
    def render_export_success(self, filename: str):
        """
        Render export success message.
        
        Args:
            filename (str): Name of exported file
        """
        self.show_success(f"✅ Data exported successfully: {filename}")
    
    def render_export_error(self, error: str):
        """
        Render export error message.
        
        Args:
            error (str): Error message
        """
        self.show_error(f"❌ Export error: {error}")