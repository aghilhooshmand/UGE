"""
Charts Component for UGE Application

This module provides chart and visualization components for displaying
experiment results and analysis data.

Classes:
- Charts: Chart and visualization utilities

Author: UGE Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any


class Charts:
    """
    Chart and visualization utilities for the UGE application.
    
    This class provides methods for creating various types of charts
    and visualizations used throughout the application.
    """
    
    @staticmethod
    def plot_fitness_evolution(result: Dict[str, Any], title: str = "Fitness Evolution", 
                              fitness_metric: str = 'mae') -> None:
        """
        Plot fitness evolution over generations.
        
        Args:
            result (Dict[str, Any]): Experiment result data
            title (str): Chart title
            fitness_metric (str): Fitness metric used ('mae' or 'accuracy')
        """
        gens = list(range(len(result.get('max', []))))
        
        # Get fitness metric for proper labeling
        ylabel = 'Fitness (higher is better)' if fitness_metric == 'accuracy' else 'Fitness (lower is better)'
        
        # Create Plotly figure
        fig = go.Figure()
        
        if result.get('max'):
            fig.add_trace(go.Scatter(
                x=gens, y=result['max'],
                mode='lines',
                name='Best (train)',
                line=dict(color='blue', width=2)
            ))
        
        if result.get('avg'):
            fig.add_trace(go.Scatter(
                x=gens, y=result['avg'],
                mode='lines',
                name='Average (train)',
                line=dict(color='orange', width=2)
            ))
        
        if result.get('min'):
            fig.add_trace(go.Scatter(
                x=gens, y=result['min'],
                mode='lines',
                name='Min (train)',
                line=dict(color='green', width=2)
            ))
        
        if result.get('fitness_test') and any(v is not None for v in result['fitness_test']):
            test_values = [v if v is not None else np.nan for v in result['fitness_test']]
            fig.add_trace(go.Scatter(
                x=gens, y=test_values,
                mode='lines',
                name='Test',
                line=dict(color='magenta', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=ylabel,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_comparison_chart(results: List[Dict[str, Any]], 
                             title: str = "Experiment Comparison") -> None:
        """
        Plot comparison chart for multiple experiments.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results
            title (str): Chart title
        """
        if not results:
            st.warning("No results to compare")
            return
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, result in enumerate(results):
            gens = list(range(len(result.get('max', []))))
            color = colors[i % len(colors)]
            
            if result.get('max'):
                fig.add_trace(go.Scatter(
                    x=gens, y=result['max'],
                    mode='lines',
                    name=f"Experiment {i+1}",
                    line=dict(color=color, width=2)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title='Fitness',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_statistics_chart(experiments: List[Dict[str, Any]], 
                             title: str = "Experiment Statistics") -> None:
        """
        Plot statistics comparison chart.
        
        Args:
            experiments (List[Dict[str, Any]]): List of experiment data
            title (str): Chart title
        """
        if not experiments:
            st.warning("No experiments to display")
            return
        
        # Extract statistics
        exp_names = [exp.get('name', f'Experiment {i+1}') for i, exp in enumerate(experiments)]
        best_fitness = [exp.get('best_fitness', 0) for exp in experiments]
        avg_fitness = [exp.get('avg_fitness', 0) for exp in experiments]
        total_runs = [exp.get('total_runs', 0) for exp in experiments]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Best Fitness', 'Average Fitness', 'Total Runs', 'Success Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Best fitness
        fig.add_trace(
            go.Bar(x=exp_names, y=best_fitness, name='Best Fitness'),
            row=1, col=1
        )
        
        # Average fitness
        fig.add_trace(
            go.Bar(x=exp_names, y=avg_fitness, name='Average Fitness'),
            row=1, col=2
        )
        
        # Total runs
        fig.add_trace(
            go.Bar(x=exp_names, y=total_runs, name='Total Runs'),
            row=2, col=1
        )
        
        # Success rate (placeholder - would need actual data)
        success_rates = [100] * len(experiments)  # Placeholder
        fig.add_trace(
            go.Bar(x=exp_names, y=success_rates, name='Success Rate'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    @staticmethod
    def create_metrics_dashboard(experiments: List[Dict[str, Any]]) -> None:
        """
        Create a metrics dashboard.
        
        Args:
            experiments (List[Dict[str, Any]]): List of experiment data
        """
        if not experiments:
            st.warning("No experiments to display")
            return
        
        # Calculate overall metrics
        total_experiments = len(experiments)
        total_runs = sum(exp.get('total_runs', 0) for exp in experiments)
        avg_fitness = np.mean([exp.get('best_fitness', 0) for exp in experiments])
        best_fitness = max([exp.get('best_fitness', 0) for exp in experiments])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Experiments", total_experiments)
        
        with col2:
            st.metric("Total Runs", total_runs)
        
        with col3:
            st.metric("Average Fitness", f"{avg_fitness:.4f}")
        
        with col4:
            st.metric("Best Fitness", f"{best_fitness:.4f}")
    
    @staticmethod
    def plot_individual_run_with_bars(result, title: str = "Individual Run Analysis", 
                                     fitness_metric: str = 'mae', 
                                     fitness_direction: int = -1,
                                     measurement_options: Dict[str, bool] = None) -> None:
        """
        Plot individual run chart with min/max/avg bars per generation.
        
        Args:
            result: ExperimentResult object or dict with result data
            title (str): Chart title
            fitness_metric (str): Fitness metric used ('mae' or 'accuracy')
            fitness_direction (int): Fitness direction (1 for maximize, -1 for minimize)
            measurement_options (Dict[str, bool]): Which measurements to show
        """
        if measurement_options is None:
            measurement_options = {
                'train_avg': True, 'train_min': True, 'train_max': True,
                'test_avg': False, 'test_min': False, 'test_max': False
            }
        # Handle both ExperimentResult objects and dictionaries
        if hasattr(result, 'max'):
            # ExperimentResult object
            max_values = result.max
            avg_values = result.avg
            min_values = result.min
            test_values = result.fitness_test
        else:
            # Dictionary
            max_values = result.get('max', [])
            avg_values = result.get('avg', [])
            min_values = result.get('min', [])
            test_values = result.get('fitness_test', [])
        
        gens = list(range(len(max_values)))
        
        # Get fitness metric for proper labeling
        if fitness_direction == 1:  # maximize
            ylabel = 'Fitness (higher is better)'
        else:  # minimize
            ylabel = 'Fitness (lower is better)'
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add training data traces based on measurement options
        if measurement_options.get('train_avg', True) and avg_values:
            # Always show error bars (standard deviation) for average
            if hasattr(result, 'std') and result.std:
                std_values = result.std
            else:
                # Fallback: calculate std from min/max if available
                if min_values and max_values:
                    std_values = [(max_val - min_val) / 4 for max_val, min_val in zip(max_values, min_values)]
                else:
                    std_values = [0] * len(avg_values)
            
            fig.add_trace(go.Scatter(
                x=gens, y=avg_values,
                mode='lines+markers',
                name='Training Average',
                line=dict(color='blue', width=3),
                marker=dict(size=6),
                error_y=dict(
                    type='data',
                    symmetric=True,
                    array=std_values,
                    visible=True,
                    color='lightblue',
                    thickness=2
                )
            ))
        
        # Add training minimum
        if measurement_options.get('train_min', True) and min_values:
            fig.add_trace(go.Scatter(
                x=gens, y=min_values,
                mode='lines',
                name='Training Minimum',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        # Add training maximum
        if measurement_options.get('train_max', True) and max_values:
            fig.add_trace(go.Scatter(
                x=gens, y=max_values,
                mode='lines',
                name='Training Maximum',
                line=dict(color='orange', width=2, dash='dot')
            ))
        
        # Add test data traces based on measurement options
        # Note: Test data is typically a single value per generation (best individual's test performance)
        if test_values and any(v is not None for v in test_values):
            test_vals = [v if v is not None else np.nan for v in test_values]
            
            # Show separate traces for each test option selected
            if measurement_options.get('test_avg', False):
                fig.add_trace(go.Scatter(
                    x=gens, y=test_vals,
                    mode='lines+markers',
                    name='Test Average',
                    line=dict(color='red', width=3),
                    marker=dict(size=6)
                ))
            
            if measurement_options.get('test_min', False):
                fig.add_trace(go.Scatter(
                    x=gens, y=test_vals,
                    mode='lines',
                    name='Test Minimum',
                    line=dict(color='purple', width=2, dash='dash')
                ))
            
            if measurement_options.get('test_max', False):
                fig.add_trace(go.Scatter(
                    x=gens, y=test_vals,
                    mode='lines',
                    name='Test Maximum',
                    line=dict(color='brown', width=2, dash='dot')
                ))
        elif (measurement_options.get('test_avg', False) or 
              measurement_options.get('test_min', False) or 
              measurement_options.get('test_max', False)):
            # Show a message if test data is requested but not available
            st.info("ℹ️ Test data is not available for this experiment. Test fitness values are only shown when the experiment includes test set evaluation.")
        
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=ylabel,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_experiment_wide_with_bars(results, title: str = "Experiment-wide Analysis", 
                                      fitness_metric: str = 'mae',
                                      fitness_direction: int = -1,
                                      measurement_options: Dict[str, bool] = None) -> None:
        """
        Plot experiment-wide chart with min/max/avg across all runs.
        
        Args:
            results: Dictionary of ExperimentResult objects by run_id
            title (str): Chart title
            fitness_metric (str): Fitness metric used ('mae' or 'accuracy')
            fitness_direction (int): Fitness direction (1 for maximize, -1 for minimize)
            measurement_options (Dict[str, bool]): Which measurements to show
        """
        if measurement_options is None:
            measurement_options = {
                'train_avg': True, 'train_min': True, 'train_max': True,
                'test_avg': False, 'test_min': False, 'test_max': False
            }
        if not results:
            st.warning("No results to display")
            return
        
        # Get fitness metric for proper labeling
        if fitness_direction == 1:  # maximize
            ylabel = 'Fitness (higher is better)'
        else:  # minimize
            ylabel = 'Fitness (lower is better)'
        
        # Find the maximum number of generations across all runs
        max_gens = 0
        for result in results.values():
            if hasattr(result, 'max'):
                max_gens = max(max_gens, len(result.max))
            else:
                max_gens = max(max_gens, len(result.get('max', [])))
        
        gens = list(range(max_gens))
        
        # Calculate min/max/avg across all runs for each generation
        min_values = []
        max_values = []
        avg_values = []
        
        for gen in range(max_gens):
            gen_values = []
            for result in results.values():
                # Handle both ExperimentResult objects and dictionaries
                if hasattr(result, 'max'):
                    max_vals = result.max
                else:
                    max_vals = result.get('max', [])
                
                if gen < len(max_vals):
                    gen_values.append(max_vals[gen])
            
            if gen_values:
                min_values.append(min(gen_values))
                max_values.append(max(gen_values))
                avg_values.append(sum(gen_values) / len(gen_values))
            else:
                min_values.append(np.nan)
                max_values.append(np.nan)
                avg_values.append(np.nan)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add training data traces based on measurement options
        if measurement_options.get('train_avg', True) and avg_values:
            # Calculate standard deviation across runs for each generation
            std_values = []
            for gen in range(max_gens):
                gen_values = []
                for result in results.values():
                    # Handle both ExperimentResult objects and dictionaries
                    if hasattr(result, 'max'):
                        max_vals = result.max
                    else:
                        max_vals = result.get('max', [])
                    
                    if gen < len(max_vals):
                        gen_values.append(max_vals[gen])
                
                if len(gen_values) > 1:
                    # Calculate standard deviation
                    mean_val = sum(gen_values) / len(gen_values)
                    variance = sum((x - mean_val) ** 2 for x in gen_values) / len(gen_values)
                    std_values.append(variance ** 0.5)
                else:
                    std_values.append(0)
            
            fig.add_trace(go.Scatter(
                x=gens, y=avg_values,
                mode='lines+markers',
                name='Training Average Across Runs',
                line=dict(color='blue', width=3),
                marker=dict(size=6),
                error_y=dict(
                    type='data',
                    symmetric=True,
                    array=std_values,
                    visible=True,
                    color='lightblue',
                    thickness=2
                )
            ))
        
        # Add training minimum across runs
        if measurement_options.get('train_min', True) and min_values:
            fig.add_trace(go.Scatter(
                x=gens, y=min_values,
                mode='lines',
                name='Training Minimum Across Runs',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        # Add training maximum across runs
        if measurement_options.get('train_max', True) and max_values:
            fig.add_trace(go.Scatter(
                x=gens, y=max_values,
                mode='lines',
                name='Training Maximum Across Runs',
                line=dict(color='orange', width=2, dash='dot')
            ))
        
        # Add test data traces for experiment-wide analysis
        if (measurement_options.get('test_avg', False) or 
            measurement_options.get('test_min', False) or 
            measurement_options.get('test_max', False)):
            
            # Calculate test data across runs for each generation
            test_avg_values = []
            test_min_values = []
            test_max_values = []
            
            for gen in range(max_gens):
                test_gen_values = []
                for result in results.values():
                    # Handle both ExperimentResult objects and dictionaries
                    if hasattr(result, 'fitness_test'):
                        test_vals = result.fitness_test
                    else:
                        test_vals = result.get('fitness_test', [])
                    
                    if gen < len(test_vals) and test_vals[gen] is not None:
                        test_gen_values.append(test_vals[gen])
                
                if test_gen_values:
                    test_avg_values.append(sum(test_gen_values) / len(test_gen_values))
                    test_min_values.append(min(test_gen_values))
                    test_max_values.append(max(test_gen_values))
                else:
                    test_avg_values.append(np.nan)
                    test_min_values.append(np.nan)
                    test_max_values.append(np.nan)
            
            # Add test traces if data is available
            if any(v is not np.nan for v in test_avg_values):
                if measurement_options.get('test_avg', False):
                    fig.add_trace(go.Scatter(
                        x=gens, y=test_avg_values,
                        mode='lines+markers',
                        name='Test Average Across Runs',
                        line=dict(color='red', width=3),
                        marker=dict(size=6)
                    ))
                
                if measurement_options.get('test_min', False):
                    fig.add_trace(go.Scatter(
                        x=gens, y=test_min_values,
                        mode='lines',
                        name='Test Minimum Across Runs',
                        line=dict(color='purple', width=2, dash='dash')
                    ))
                
                if measurement_options.get('test_max', False):
                    fig.add_trace(go.Scatter(
                        x=gens, y=test_max_values,
                        mode='lines',
                        name='Test Maximum Across Runs',
                        line=dict(color='brown', width=2, dash='dot')
                    ))
            else:
                st.info("ℹ️ Test data is not available for this experiment. Test fitness values are only shown when the experiment includes test set evaluation.")
        
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=ylabel,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    