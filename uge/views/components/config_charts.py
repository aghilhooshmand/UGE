"""
Configuration Charts Component for UGE Application

This module provides chart and visualization components specifically for
displaying configuration parameter evolution across generations.

Classes:
- ConfigCharts: Configuration-specific chart utilities

Author: UGE Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any


class ConfigCharts:
    """
    Configuration-specific chart utilities for the UGE application.
    
    This class provides methods for creating charts that visualize
    how configuration parameters change across generations.
    """
    
    @staticmethod
    def plot_configuration_evolution(generation_configs: List[Dict[str, Any]], 
                                   config_param: str = 'elite_size',
                                   title: str = "Configuration Evolution") -> None:
        """
        Plot how a configuration parameter changes across generations.
        
        Args:
            generation_configs (List[Dict[str, Any]]): List of generation configurations
            config_param (str): Configuration parameter to plot
            title (str): Chart title
        """
        if not generation_configs:
            st.warning("⚠️ No generation configuration data available.")
            return
        
        # Extract data
        generations = [gc['generation'] for gc in generation_configs]
        values = [gc[config_param] for gc in generation_configs]
        
        # Create figure
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=generations,
            y=values,
            mode='lines+markers',
            name=f'{config_param.replace("_", " ").title()}',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=f'{config_param.replace("_", " ").title()}',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_setup_wide_configuration_stats(setup_results: Dict[str, Any], 
                                          config_param: str = 'elite_size',
                                          title: str = "Setup-wide Configuration Statistics") -> None:
        """
        Plot setup-wide configuration statistics (min/max/avg) across all runs.
        
        Args:
            setup_results (Dict[str, Any]): Dictionary of setup results by run_id
            config_param (str): Configuration parameter to analyze
            title (str): Chart title
        """
        if not setup_results:
            st.warning("⚠️ No setup results available for configuration analysis.")
            return
        
        # Collect configuration data from all runs
        all_config_data = []
        max_generations = 0
        
        for run_id, result in setup_results.items():
            if hasattr(result, 'generation_configs') and result.generation_configs:
                run_configs = []
                for gen_config in result.generation_configs:
                    if hasattr(gen_config, config_param):
                        run_configs.append(getattr(gen_config, config_param))
                    elif isinstance(gen_config, dict) and config_param in gen_config:
                        run_configs.append(gen_config[config_param])
                    else:
                        run_configs.append(None)
                
                if any(v is not None for v in run_configs):
                    all_config_data.append(run_configs)
                    max_generations = max(max_generations, len(run_configs))
        
        if not all_config_data:
            st.warning(f"⚠️ No {config_param} configuration data available across all runs.")
            return
        
        # Calculate aggregated statistics across runs for each generation
        aggregated_min = []
        aggregated_avg = []
        aggregated_max = []
        
        for gen in range(max_generations):
            gen_values = []
            for run_configs in all_config_data:
                if gen < len(run_configs) and run_configs[gen] is not None:
                    gen_values.append(run_configs[gen])
            
            if gen_values:
                aggregated_min.append(min(gen_values))
                aggregated_avg.append(sum(gen_values) / len(gen_values))
                aggregated_max.append(max(gen_values))
            else:
                aggregated_min.append(None)
                aggregated_avg.append(None)
                aggregated_max.append(None)
        
        generations = list(range(len(aggregated_avg)))
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=generations, y=aggregated_min,
            mode='lines+markers',
            name='Minimum Across Runs',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=generations, y=aggregated_avg,
            mode='lines+markers',
            name='Average Across Runs',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=generations, y=aggregated_max,
            mode='lines+markers',
            name='Maximum Across Runs',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=f'{config_param.replace("_", " ").title()}',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_configuration_comparison(setup_results: Dict[str, Dict[str, Any]], 
                                    config_param: str = 'elite_size',
                                    title: str = "Configuration Comparison Across Setups") -> None:
        """
        Plot configuration comparison across multiple setups.
        
        Args:
            setup_results (Dict[str, Dict[str, Any]]): Dictionary of setup results by setup_name
            config_param (str): Configuration parameter to compare
            title (str): Chart title
        """
        if not setup_results:
            st.warning("⚠️ No setup results available for configuration comparison.")
            return
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (setup_name, setup_data) in enumerate(setup_results.items()):
            # Get generation configs from the first run (assuming all runs have same config)
            generation_configs = None
            if 'results' in setup_data:
                # If it's a Setup object
                results = setup_data['results']
                if results:
                    first_result = list(results.values())[0]
                    if hasattr(first_result, 'generation_configs'):
                        generation_configs = first_result.generation_configs
            elif 'generation_configs' in setup_data:
                # If it's direct generation configs
                generation_configs = setup_data['generation_configs']
            
            if generation_configs:
                generations = [gc['generation'] if isinstance(gc, dict) else gc.generation 
                             for gc in generation_configs]
                values = [gc[config_param] if isinstance(gc, dict) else getattr(gc, config_param, None)
                         for gc in generation_configs]
                
                # Filter out None values
                valid_data = [(g, v) for g, v in zip(generations, values) if v is not None]
                if valid_data:
                    valid_gens, valid_values = zip(*valid_data)
                    
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=valid_gens,
                        y=valid_values,
                        mode='lines+markers',
                        name=f'{setup_name}',
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=f'{config_param.replace("_", " ").title()}',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_configuration_dashboard(setup_results: Dict[str, Any]) -> None:
        """
        Create a configuration dashboard showing statistics for all tracked parameters.
        
        Args:
            setup_results (Dict[str, Any]): Dictionary of setup results
        """
        if not setup_results:
            st.warning("⚠️ No setup results available for configuration dashboard.")
            return
        
        # Get the first result to determine available configuration parameters
        first_result = list(setup_results.values())[0]
        if not hasattr(first_result, 'generation_configs') or not first_result.generation_configs:
            st.warning("⚠️ No generation configuration data available.")
            return
        
        # Get available configuration parameters
        first_gen_config = first_result.generation_configs[0]
        if isinstance(first_gen_config, dict):
            config_params = list(first_gen_config.keys())
        else:
            config_params = [attr for attr in dir(first_gen_config) 
                           if not attr.startswith('_') and not callable(getattr(first_gen_config, attr))]
        
        # Remove non-numeric parameters
        numeric_params = []
        for param in config_params:
            if param in ['generation', 'timestamp']:
                continue
            try:
                value = first_gen_config[param] if isinstance(first_gen_config, dict) else getattr(first_gen_config, param)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_params.append(param)
            except:
                continue
        
        if not numeric_params:
            st.warning("⚠️ No numeric configuration parameters found for dashboard.")
            return
        
        st.subheader("📊 Configuration Dashboard")
        
        # Create tabs for different configuration parameters
        tab_names = [param.replace('_', ' ').title() for param in numeric_params]
        tabs = st.tabs(tab_names)
        
        for i, param in enumerate(numeric_params):
            with tabs[i]:
                # Calculate statistics for this parameter
                all_values = []
                for result in setup_results.values():
                    if hasattr(result, 'generation_configs'):
                        for gen_config in result.generation_configs:
                            value = gen_config[param] if isinstance(gen_config, dict) else getattr(gen_config, param, None)
                            if value is not None:
                                all_values.append(value)
                
                if all_values:
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Min Value", f"{min(all_values):.3f}")
                    with col2:
                        st.metric("Max Value", f"{max(all_values):.3f}")
                    with col3:
                        st.metric("Average", f"{sum(all_values)/len(all_values):.3f}")
                    with col4:
                        st.metric("Total Records", len(all_values))
                    
                    # Plot the configuration evolution
                    ConfigCharts.plot_setup_wide_configuration_stats(
                        setup_results, 
                        config_param=param,
                        title=f"{param.replace('_', ' ').title()} Evolution Across Runs"
                    )
                else:
                    st.info(f"No data available for {param.replace('_', ' ')}")
    
    @staticmethod
    def get_available_config_params(setup_results: Dict[str, Any]) -> List[str]:
        """
        Get list of available configuration parameters from setup results.
        
        Args:
            setup_results (Dict[str, Any]): Dictionary of setup results
            
        Returns:
            List[str]: List of available configuration parameter names
        """
        if not setup_results:
            return []
        
        # Get the first result to determine available configuration parameters
        first_result = list(setup_results.values())[0]
        if not hasattr(first_result, 'generation_configs') or not first_result.generation_configs:
            return []
        
        # Get available configuration parameters
        first_gen_config = first_result.generation_configs[0]
        if isinstance(first_gen_config, dict):
            config_params = list(first_gen_config.keys())
        else:
            config_params = [attr for attr in dir(first_gen_config) 
                           if not attr.startswith('_') and not callable(getattr(first_gen_config, attr))]
        
        # Remove non-numeric parameters
        numeric_params = []
        for param in config_params:
            if param in ['generation', 'timestamp']:
                continue
            try:
                value = first_gen_config[param] if isinstance(first_gen_config, dict) else getattr(first_gen_config, param)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_params.append(param)
            except:
                continue
        
        return numeric_params
