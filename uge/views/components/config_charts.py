"""
Configuration Charts Component for GE-Lab Application

This module provides chart and visualization components specifically for
displaying configuration parameter evolution across generations.

Classes:
- ConfigCharts: Configuration-specific chart utilities

Author: GE-Lab Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import pandas as pd
from sklearn.manifold import TSNE


class ConfigCharts:
    """
    Configuration-specific chart utilities for the GE-Lab application.
    
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
                                    title: str = "Configuration Comparison Across Setups",
                                    color_map: Optional[Dict[str, str]] = None) -> None:
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
                    
                    color = color_map.get(setup_name) if color_map else colors[i % len(colors)]
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
    def plot_tsne_phenotype_analysis(setup_data: Dict[str, Any],
                                     storage_service: Any,
                                     selected_setups: List[str],
                                     n_best_per_run: int = 15,
                                     selected_generations: List[int] = None,
                                     perplexity: float = 30.0,
                                     learning_rate: float = 200.0,
                                     n_iter: int = 1000,
                                     random_state: int = 42,
                                     color_map: Optional[Dict[str, str]] = None) -> None:
        """
        Plot t-SNE phenotype analysis following the paper's methodology:
        "Phenotype analysis" - comparing algorithms based on their solutions.
        
        This creates t-SNE embeddings using the best N individuals from each run,
        visualizing them with:
        - Symbol size representing error (smaller = higher error, larger = lower error)
        - Special markers for best training and best test individuals
        - Generation-specific visualization to see evolution over time
        
        Args:
            setup_data: Dictionary containing setup information and results
            storage_service: Storage service to load full setup data
            selected_setups: List of setup IDs to compare
            n_best_per_run: Number of best individuals to extract per run (default: 15)
            selected_generations: List of generations to analyze (default: all)
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            n_iter: Number of t-SNE iterations
            random_state: Random seed for reproducibility
            color_map: Dictionary mapping setup names to colors
        
        Reference:
            This implementation is based on the phenotype analysis methodology from:
            
            Lourenço, N., Assunção, F., Madureira, G., Machado, P., & Penousal Machado (2024).
            "Probabilistic Grammatical Evolution"
            In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '24).
            DOI: https://doi.org/10.1145/3712256.3726444
            
            The paper uses t-SNE to visualize best individuals from each run and fold,
            projecting them into 2D space to analyze phenotypic space exploration and
            compare how different algorithms (SGE, Co-PSGE, SGEF, PSGE) explore the
            solution space. Symbol size represents error (larger = better fitness),
            with special markers for best training (star) and best test (hexagon) individuals.
        """
        # Get setup names from session state
        setup_names = st.session_state.get('selected_setup_names', [])
        if not setup_names or not selected_setups:
            st.warning("No setups selected for t-SNE phenotype analysis.")
            return
        
        st.info(f"📊 Analyzing phenotypes from {len(selected_setups)} setups...")
        
        # Step 1: Extract best N individuals from each run of each setup
        all_individuals = []
        best_train_per_setup = {}  # Track best training individual per setup
        best_test_per_setup = {}   # Track best test individual per setup
        
        for i, setup_id in enumerate(selected_setups):
            setup_name = setup_names[i] if i < len(setup_names) else setup_id
            
            try:
                # Load full setup data
                setup = storage_service.load_setup(setup_id)
                if not setup or not setup.results:
                    continue
                
                # Track best individuals for this setup
                best_train_fitness = float('-inf')
                best_test_fitness = float('-inf')
                best_train_individual = None
                best_test_individual = None
                
                # Extract individuals from each run
                for run_id, result in setup.results.items():
                    if not hasattr(result, 'max') or not result.max:
                        continue
                    
                    # Determine which generations to analyze
                    n_gens = len(result.max)
                    if selected_generations is None:
                        gens_to_analyze = list(range(n_gens))
                    else:
                        gens_to_analyze = [g for g in selected_generations if g < n_gens]
                    
                    # For each generation, extract best N individuals
                    for gen_idx in gens_to_analyze:
                        # Since we only have aggregated data, we'll use the generation's
                        # best fitness as a proxy for the "best individual" from that generation
                        individual = {
                            'setup': setup_name,
                            'setup_id': setup_id,
                            'run_id': run_id,
                            'generation': gen_idx,
                            # Phenotype features (what the individual "looks like")
                            'fitness_train': result.max[gen_idx] if gen_idx < len(result.max) else None,
                            'fitness_test': result.fitness_test[gen_idx] if gen_idx < len(result.fitness_test) else None,
                            'nodes_length': result.nodes_length_max[gen_idx] if hasattr(result, 'nodes_length_max') and gen_idx < len(result.nodes_length_max) else None,
                            'tree_depth': result.max_tree_depth[gen_idx] if hasattr(result, 'max_tree_depth') and gen_idx < len(result.max_tree_depth) else None,
                            'invalid_count': result.invalid_count_max[gen_idx] if hasattr(result, 'invalid_count_max') and gen_idx < len(result.invalid_count_max) else None,
                            'codon_consumption': result.codon_consumption_max[gen_idx] if hasattr(result, 'codon_consumption_max') and gen_idx < len(result.codon_consumption_max) else None,
                            # Error (for sizing) - lower is better
                            'error_train': -result.max[gen_idx] if gen_idx < len(result.max) else None,
                            'error_test': -result.fitness_test[gen_idx] if gen_idx < len(result.fitness_test) else None,
                        }
                        
                        all_individuals.append(individual)
                        
                        # Track best training individual
                        if individual['fitness_train'] is not None and individual['fitness_train'] > best_train_fitness:
                            best_train_fitness = individual['fitness_train']
                            best_train_individual = len(all_individuals) - 1
                        
                        # Track best test individual
                        if individual['fitness_test'] is not None and individual['fitness_test'] > best_test_fitness:
                            best_test_fitness = individual['fitness_test']
                            best_test_individual = len(all_individuals) - 1
                
                best_train_per_setup[setup_name] = best_train_individual
                best_test_per_setup[setup_name] = best_test_individual
                
            except Exception as e:
                st.warning(f"Could not load data for setup {setup_name}: {str(e)}")
                continue
        
        if not all_individuals:
            st.error("No individuals extracted for t-SNE analysis.")
            return
        
        st.success(f"✅ Extracted {len(all_individuals)} individuals from all runs and generations")
        
        # Step 2: Create feature matrix for t-SNE
        df = pd.DataFrame(all_individuals)
        
        # Select phenotype features (characteristics of the solution)
        feature_cols = ['fitness_train', 'fitness_test', 'nodes_length', 'tree_depth', 
                       'invalid_count', 'codon_consumption']
        
        # Filter to rows with all features present
        df_features = df[['setup', 'setup_id', 'run_id', 'generation', 'error_train'] + feature_cols].dropna()
        
        if df_features.empty or len(df_features) < 2:
            st.error("Insufficient data with all features for t-SNE analysis.")
            return
        
        st.info(f"📉 Using {len(df_features)} individuals with complete phenotype data")
        
        # Step 3: Perform t-SNE
        X = df_features[feature_cols].values
        
        try:
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(X_scaled) - 1),
                learning_rate=learning_rate,
                random_state=random_state,
                init='pca'
            )
            # Try to set n_iter if supported
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(perplexity, len(X_scaled) - 1),
                    learning_rate=learning_rate,
                    random_state=random_state,
                    init='pca',
                    n_iter=n_iter
                )
            except TypeError:
                pass
            
            embedding = tsne.fit_transform(X_scaled)
            df_features['tsne_x'] = embedding[:, 0]
            df_features['tsne_y'] = embedding[:, 1]
            
        except Exception as e:
            st.error(f"t-SNE failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
        
        # Step 4: Create visualization following the paper's style
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Symbol mapping for each setup
        symbols = ['circle', 'diamond', 'square', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon']
        
        # Plot each setup with different symbol
        for idx, setup_name in enumerate(setup_names):
            setup_df = df_features[df_features['setup'] == setup_name]
            if setup_df.empty:
                continue
            
            # Get color for this setup
            color = color_map.get(setup_name) if color_map else px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
            symbol = symbols[idx % len(symbols)]
            
            # Normalize error for sizing: smaller error = larger marker
            # Map error to marker size: lower error (better fitness) = larger size
            min_error = df_features['error_train'].min()
            max_error = df_features['error_train'].max()
            error_range = max_error - min_error
            if error_range > 0:
                # Invert: lower error = larger size (5-20 range)
                sizes = 5 + 15 * (1 - (setup_df['error_train'] - min_error) / error_range)
            else:
                sizes = [10] * len(setup_df)
            
            # Regular individuals
            fig.add_trace(go.Scatter(
                x=setup_df['tsne_x'],
                y=setup_df['tsne_y'],
                mode='markers',
                name=setup_name,
                marker=dict(
                    size=sizes,
                    color=color,
                    symbol=symbol,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Setup: {setup_name}<br>Gen: {g}<br>Run: {r}<br>Train: {ft:.4f}<br>Test: {ftest:.4f}<br>Nodes: {n}" 
                      for g, r, ft, ftest, n in zip(setup_df['generation'], setup_df['run_id'], 
                                                     setup_df['fitness_train'], setup_df['fitness_test'], 
                                                     setup_df['nodes_length'])],
                hoverinfo='text',
                showlegend=True
            ))
        
        # Add special markers for best training and best test individuals
        for setup_name in setup_names:
            color = color_map.get(setup_name) if color_map else px.colors.qualitative.Plotly[setup_names.index(setup_name) % len(px.colors.qualitative.Plotly)]
            
            # Best training individual (star)
            best_train_idx = best_train_per_setup.get(setup_name)
            if best_train_idx is not None and best_train_idx < len(df_features):
                row = df_features.iloc[best_train_idx]
                fig.add_trace(go.Scatter(
                    x=[row['tsne_x']],
                    y=[row['tsne_y']],
                    mode='markers',
                    name=f'{setup_name} Best Train',
                    marker=dict(
                        size=25,
                        color=color,
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    text=f"BEST TRAIN<br>Setup: {setup_name}<br>Fitness: {row['fitness_train']:.4f}",
                    hoverinfo='text',
                    showlegend=False
                ))
            
            # Best test individual (hexagon)
            best_test_idx = best_test_per_setup.get(setup_name)
            if best_test_idx is not None and best_test_idx < len(df_features):
                row = df_features.iloc[best_test_idx]
                fig.add_trace(go.Scatter(
                    x=[row['tsne_x']],
                    y=[row['tsne_y']],
                    mode='markers',
                    name=f'{setup_name} Best Test',
                    marker=dict(
                        size=25,
                        color=color,
                        symbol='hexagon',
                        line=dict(width=2, color='black')
                    ),
                    text=f"BEST TEST<br>Setup: {setup_name}<br>Fitness: {row['fitness_test']:.4f}",
                    hoverinfo='text',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="t-SNE Phenotype Analysis: Best Individuals Across Generations<br><sub>Symbol size: larger = better fitness | ★ = Best Train | ⬡ = Best Test</sub>",
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            template='plotly_white',
            height=700,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation guide
        with st.expander("📖 How to Interpret This Visualization"):
            st.markdown("""
            ### Phenotype Analysis Interpretation Guide
            
            This visualization follows the methodology from GE literature for comparing algorithms
            based on their phenotypic characteristics (the solutions they produce).
            
            #### What You're Seeing:
            - **Each point** represents a best individual from a specific generation and run
            - **Symbol type** distinguishes different setups/algorithms
            - **Symbol size** represents fitness: **larger symbols = better fitness** (lower error)
            - **★ Star** marks the best training individual for each setup
            - **⬡ Hexagon** marks the best test individual for each setup
            - **Colors** distinguish different setups
            
            #### What to Look For:
            
            1. **Clustering**: Do individuals from the same setup cluster together?
               - Tight clusters suggest consistent solution patterns
               - Spread out points suggest diverse exploration
            
            2. **Separation**: Are different setups in different regions?
               - Distinct regions indicate different phenotypic spaces explored
               - Overlap suggests similar solutions despite different algorithms
            
            3. **Best Solutions Location**: Where are the ★ and ⬡ markers?
               - If best solutions are in a specific region, that region may contain optimal phenotypes
               - If a setup's best solutions are far from its cluster, it may indicate rare discoveries
            
            4. **Evolution Over Generations**: If analyzing specific generations:
               - Early generations may cluster in one area
               - Later generations may expand to explore new regions
               - This shows phenotypic space exploration over time
            
            #### Example Insights:
            - **"Setup A explores the left side, Setup B the right"** → Different search strategies
            - **"Best solutions are all on the right"** → Right region contains better phenotypes
            - **"Setup C's solutions are smaller/simpler"** → May generate less complex solutions
            - **"Best Train and Best Test are close"** → Similar phenotypic characteristics
            
            **Reference**: This analysis is based on the phenotype visualization methodology from:
            
            > Lourenço, N., Assunção, F., Madureira, G., Machado, P., & Penousal Machado (2024).
            > "Probabilistic Grammatical Evolution"  
            > *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '24)*  
            > DOI: [10.1145/3712256.3726444](https://doi.org/10.1145/3712256.3726444)
            
            The paper demonstrates how t-SNE visualization of best individuals reveals how different
            GE variants (SGE, Co-PSGE, SGEF, PSGE) explore distinct regions of the phenotypic space,
            helping to understand why some algorithms find better solutions than others.
            """)
        
    
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
    def plot_setup_configuration_comparison(setup_configs: Dict[str, Any], 
                                          config_param: str = 'elite_size',
                                          title: str = "Configuration Comparison Across Setups",
                                          color_map: Optional[Dict[str, str]] = None) -> None:
        """
        Plot configuration parameter comparison across multiple setups.
        This shows the average value of a configuration parameter across all runs for each setup.
        
        Args:
            setup_configs (Dict[str, Any]): Dictionary of setup configurations by setup_id
            config_param (str): Configuration parameter to compare
            title (str): Chart title
        """
        if not setup_configs:
            st.warning("⚠️ No setup configurations available for comparison.")
            return
        
        # Extract configuration values for each setup
        setup_names = []
        config_values = []
        
        for setup_id, config in setup_configs.items():
            # Get setup name from config
            setup_name = config.get('setup_name', setup_id)
            setup_names.append(setup_name)
            
            # Get the configuration parameter value
            if config_param in config:
                config_values.append(config[config_param])
            else:
                config_values.append(None)
        
        # Filter out None values
        valid_data = [(name, value) for name, value in zip(setup_names, config_values) if value is not None]
        if not valid_data:
            st.warning(f"⚠️ No valid {config_param} values found in setup configurations.")
            return
        
        valid_names, valid_values = zip(*valid_data)
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        # Determine colors for each setup
        if color_map:
            marker_colors = [color_map.get(name, 'lightblue') for name in valid_names]
        else:
            marker_colors = 'lightblue'

        fig.add_trace(go.Bar(
            x=list(valid_names),
            y=list(valid_values),
            name=f'{config_param.replace("_", " ").title()}',
            marker_color=marker_colors,
            text=[f'{v:.3f}' if isinstance(v, float) else str(v) for v in valid_values],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Setup',
            yaxis_title=f'{config_param.replace("_", " ").title()}',
            showlegend=False,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary table
        st.subheader("📊 Configuration Comparison Summary")
        
        # Create comparison table
        comparison_data = []
        for name, value in valid_data:
            comparison_data.append({
                'Setup': name,
                f'{config_param.replace("_", " ").title()}': f'{value:.3f}' if isinstance(value, float) else str(value)
            })
        
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
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
