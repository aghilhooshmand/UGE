"""
Comparison View Module

Handles the UI for setup comparison and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List
import numpy as np
from uge.services.storage_service import StorageService
from uge.views.components.config_charts import ConfigCharts
from uge.utils.statistical_tests import StatisticalTests


class ComparisonView:
    """
    Comparison View class for handling setup comparison UI.
    
    This class provides methods to render the comparison page,
    allowing users to compare multiple setups and visualize results.
    """
    
    def __init__(self, storage_service: StorageService):
        """
        Initialize the ComparisonView.
        
        Args:
            storage_service (StorageService): Storage service instance
        """
        self.storage_service = storage_service
    
    @staticmethod
    def _build_color_map(setup_names: List[str]) -> Dict[str, str]:
        """Build a deterministic unique color for each setup name.
        Uses multiple qualitative palettes, then falls back to HSV generation.
        """
        # Collect palettes
        palettes = []
        try:
            palettes.extend([
                px.colors.qualitative.Plotly,
                px.colors.qualitative.Set1,
                px.colors.qualitative.Set2,
                px.colors.qualitative.Set3,
                px.colors.qualitative.Dark24,
                px.colors.qualitative.Pastel,
            ])
        except Exception:
            palettes.append(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        # Flatten and de-duplicate while preserving order
        flat_colors: List[str] = []
        seen = set()
        for pal in palettes:
            for c in pal:
                if c not in seen:
                    seen.add(c)
                    flat_colors.append(c)

        color_map: Dict[str, str] = {}
        n = len(setup_names)
        # Assign from flat palette first
        for i, name in enumerate(setup_names):
            if i < len(flat_colors):
                color_map[name] = flat_colors[i]
            else:
                # Generate additional distinct hues in HSV
                # Evenly space hues; fixed saturation/value for readability
                import colorsys
                extra_index = i - len(flat_colors)
                hue = (extra_index / max(1, n - len(flat_colors) + 1)) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
                color_map[name] = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        return color_map
    
    def render_comparison(self, comparison_controller) -> None:
        """
        Render the comparison page.
        
        Args:
            comparison_controller: Controller for comparison logic
        """
        st.header("⚖️ Setup Comparison")
        st.markdown("Compare multiple setups")
        
        try:
            setup_paths = self.storage_service.list_setups()
            
            if len(setup_paths) >= 2:
                # Load setup objects and create options
                exp_options = {}
                for exp_path in setup_paths:
                    exp_id = exp_path.name
                    try:
                        setup = self.storage_service.load_setup(exp_id)
                        if setup and setup.config:
                            exp_name = setup.config.setup_name
                        else:
                            exp_name = exp_id
                    except:
                        exp_name = exp_id
                    exp_options[exp_name] = exp_id
                
                selected_setup_names = st.multiselect(
                    "Select Setups to Compare",
                    list(exp_options.keys()),
                    default=list(exp_options.keys())[:2] if len(exp_options) >= 2 else list(exp_options.keys())
                )
                
                # Convert selected names back to IDs for processing
                selected_setups = [exp_options[name] for name in selected_setup_names]
                # Map IDs back to their current names
                id_to_name = {v: k for k, v in exp_options.items()}
                
                if len(selected_setups) >= 2:
                    # Optional rename UI
                    with st.expander("✏️ Set Setup Names (optional)"):
                        if 'setup_display_names' not in st.session_state:
                            st.session_state.setup_display_names = {}
                        for setup_id in selected_setups:
                            current_default = st.session_state.setup_display_names.get(setup_id, id_to_name.get(setup_id, setup_id))
                            new_name = st.text_input(
                                f"Name for {id_to_name.get(setup_id, setup_id)}",
                                value=current_default,
                                key=f"rename_{setup_id}"
                            )
                            st.session_state.setup_display_names[setup_id] = new_name.strip() or current_default
                    
                    if st.button("🔍 Compare Setups"):
                        with st.spinner("Comparing setups..."):
                            comparison_results = comparison_controller.compare_setups(selected_setups)
                            
                            if comparison_results:
                                # Transform the comparison results to match expected format
                                # Build display names (custom if provided)
                                display_names = [
                                    st.session_state.setup_display_names.get(setup_id, id_to_name.get(setup_id, setup_id))
                                    for setup_id in selected_setups
                                ]
                                transformed_results = self._transform_comparison_results(comparison_results, display_names)
                                
                                
                                # Store results in session state
                                st.session_state.comparison_results = transformed_results
                                st.session_state.selected_setups = selected_setups
                                st.session_state.selected_setup_names = display_names
                            else:
                                st.error("Failed to generate comparison results. Please check that your setups have completed runs.")
                    
                    # Check if we have stored comparison results
                    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
                        comparison_results = st.session_state.comparison_results
                        selected_setups = st.session_state.selected_setups
                        
                        if comparison_results:
                            # Display comparison results
                            self._render_comparison_results(comparison_results, selected_setups)
                else:
                    st.warning("Please select at least 2 setups to compare.")
            else:
                st.info("You need at least 2 setups to perform a comparison. Create more setups using the 'Run Setup' page.")
                
        except Exception as e:
            st.error(f"Error loading setups: {str(e)}")
    
    def _transform_comparison_results(self, comparison_results: Dict[str, Any], selected_setup_names: List[str]) -> Dict[str, Any]:
        """
        Transform comparison results from controller format to chart format.
        
        Args:
            comparison_results (Dict[str, Any]): Raw comparison results from controller
            selected_setup_names (List[str]): List of selected setup names
            
        Returns:
            Dict[str, Any]: Transformed results for charting
        """
        transformed = {}
        
        # Get aggregate data from comparison results
        aggregate_data = comparison_results.get('aggregate_data', {})
        setup_configs = comparison_results.get('setup_configs', {})
        
        # Preserve setup_configs in the transformed results
        transformed['setup_configs'] = setup_configs
        
        # Also preserve the original setup analysis data for generation configs
        original_setups = comparison_results.get('setups', [])
        
        for i, setup_name in enumerate(selected_setup_names):
            setup_id = list(aggregate_data.keys())[i] if i < len(aggregate_data) else None
            
            if setup_id and setup_id in aggregate_data:
                setup_aggregate = aggregate_data[setup_id]
                
                # Transform training data
                training_data = {
                    'generations': setup_aggregate.get('generations', []),
                    'avg': setup_aggregate.get('avg_avg', []),
                    'std': setup_aggregate.get('std_avg', []),
                    'max': setup_aggregate.get('avg_max', []),
                    'min': setup_aggregate.get('avg_min', [])
                }
                
                # Transform test data
                test_data = {
                    'generations': setup_aggregate.get('generations', []),
                    'avg': setup_aggregate.get('avg_test', []),
                    'std': setup_aggregate.get('std_test', [])
                }
                
                # Transform invalid count data
                invalid_count_data = {
                    'generations': setup_aggregate.get('generations', []),
                    'avg': setup_aggregate.get('invalid_count_avg', []),
                    'std': setup_aggregate.get('invalid_count_std', []),
                    'max': setup_aggregate.get('invalid_count_max', []),
                    'min': setup_aggregate.get('invalid_count_min', [])
                }
                
                # Transform nodes length data
                nodes_length_data = {
                    'generations': setup_aggregate.get('generations', []),
                    'avg': setup_aggregate.get('nodes_length_avg', []),
                    'std': setup_aggregate.get('nodes_length_std', []),
                    'max': setup_aggregate.get('nodes_length_max', []),
                    'min': setup_aggregate.get('nodes_length_min', [])
                }
                
                transformed[setup_name] = {
                    'training': training_data,
                    'test': test_data,
                    'invalid_count': invalid_count_data,
                    'nodes_length': nodes_length_data
                }
                
                # Add the original setup analysis data for generation configs
                if i < len(original_setups):
                    transformed[setup_name]['results'] = original_setups[i].get('results', {})
        
        return transformed
    
    def _render_comparison_results(self, comparison_results: Dict[str, Any], selected_setups: List[str]) -> None:
        """
        Render comparison results with charts and statistics.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            selected_setups (List[str]): List of selected setup IDs
        """
        st.subheader("📊 Comparison Results")

        # Rebuild a deterministic unique color map for current selection
        setup_names_order = st.session_state.get('selected_setup_names', [k for k in comparison_results.keys() if k != 'setup_configs'])
        st.session_state.setup_color_map = self._build_color_map(setup_names_order)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Visual Comparison", "📊 Statistical Analysis"])
        
        with tab1:
            # Chart type selection
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Training Fitness", "Test Fitness", "Number of Invalid", "Nodes Length Evolution", "Configuration Comparison", "Configuration Evolution", "t-SNE Best Individuals", "All Metrics"],
                key="comparison_chart_type"
            )
            
            # Metric selection for detailed charts
            if chart_type in ["Training Fitness", "Test Fitness"]:
                data_type = "training" if chart_type == "Training Fitness" else "test"
                metric_type = st.selectbox(
                    f"Select {chart_type} Metric",
                    ["Average", "Maximum", "Minimum", "Average with STD Bars"],
                    key=f"comparison_metric_{data_type}"
                )
            elif chart_type in ["Number of Invalid", "Nodes Length Evolution"]:
                metric_type = st.selectbox(
                    f"Select {chart_type} Metric",
                    ["Average", "Maximum", "Minimum", "Average with STD Bars"],
                    key=f"comparison_metric_{chart_type.lower().replace(' ', '_')}"
                )
            elif chart_type == "Configuration Comparison":
                # Get available configuration parameters from setup configs
                available_config_params = self._get_available_setup_config_params(comparison_results, selected_setups)
                
                if available_config_params:
                    metric_type = st.selectbox(
                        "Select Configuration Parameter",
                        available_config_params,
                        key="comparison_config_param"
                    )
                else:
                    st.warning("No configuration parameters available for comparison.")
                    metric_type = None
            elif chart_type == "Configuration Evolution":
                # Get available configuration parameters from generation configs
                available_config_params = self._get_available_generation_config_params(comparison_results)
                
                if available_config_params:
                    metric_type = st.selectbox(
                        "Select Configuration Parameter",
                        available_config_params,
                        key="evolution_config_param"
                    )
                else:
                    st.warning("No generation configuration parameters available for evolution comparison.")
                    metric_type = None
            elif chart_type == "t-SNE Best Individuals":
                metric_type = "t-SNE"
            else:
                metric_type = "All"
            
            # Render the appropriate chart
            self._render_comparison_chart(comparison_results, chart_type, metric_type, selected_setups)
        
        with tab2:
            # Statistical comparison
            self._render_statistical_comparison(comparison_results, selected_setups)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Charts as Images"):
                st.info("Chart export functionality would be implemented here")
        
        with col2:
            if st.button("📄 Export Data as CSV"):
                csv_data = self._export_comparison_csv(comparison_results)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="setup_comparison.csv",
                    mime="text/csv"
                )
    
    def _render_comparison_chart(self, comparison_results: Dict[str, Any], chart_type: str, metric_type: str = "All", selected_setups: List[str] = None) -> None:
        """
        Render comparison chart based on chart type and metric type.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            chart_type (str): Type of chart to render
            metric_type (str): Type of metric to display
            selected_setups (List[str]): List of selected setup IDs (required for t-SNE)
        """
        if chart_type == "Training Fitness":
            self._render_fitness_chart(comparison_results, "training", metric_type)
        elif chart_type == "Test Fitness":
            self._render_fitness_chart(comparison_results, "test", metric_type)
        elif chart_type == "Number of Invalid":
            self._render_invalid_count_chart(comparison_results, metric_type)
        elif chart_type == "Nodes Length Evolution":
            self._render_nodes_length_chart(comparison_results, metric_type)
        elif chart_type == "Configuration Comparison":
            if metric_type:
                self._render_configuration_comparison_chart(comparison_results, metric_type)
        elif chart_type == "Configuration Evolution":
            if metric_type:
                self._render_configuration_evolution_chart(comparison_results, metric_type)
        elif chart_type == "t-SNE Best Individuals":
            if selected_setups:
                self._render_tsne_best_individuals(comparison_results, selected_setups)
            else:
                st.error("Setup IDs are required for t-SNE analysis.")
        elif chart_type == "All Metrics":
            self._render_all_metrics_chart(comparison_results)
    
    def _render_fitness_chart(self, comparison_results: Dict[str, Any], data_type: str, metric_type: str = "Average") -> None:
        """
        Render fitness comparison chart.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            data_type (str): "training" or "test"
            metric_type (str): "Average", "Maximum", "Minimum", or "Average with STD Bars"
        """
        fig = go.Figure()
        color_map = st.session_state.get('setup_color_map', {})
        
        if not comparison_results:
            st.warning("No comparison data available.")
            return
        
        traces_added = 0
        
        for setup_name, setup_data in comparison_results.items():
            if data_type in setup_data and setup_data[data_type]:
                generations = setup_data[data_type]['generations']
                avg_values = setup_data[data_type]['avg']
                std_values = setup_data[data_type]['std']
                max_values = setup_data[data_type].get('max', [])
                min_values = setup_data[data_type].get('min', [])
                
                if not generations:
                    st.warning(f"No {data_type} data available for {setup_name}")
                    continue
                
                # Find first valid generation (non-NaN)
                first_valid_gen = 0
                if data_type == "test":
                    # Use avg_values to find first valid generation
                    check_values = avg_values if avg_values else []
                    for i, val in enumerate(check_values):
                        if val is not None and not pd.isna(val) and val != 0:
                            first_valid_gen = i
                            break
                
                # Plot from first valid generation
                if first_valid_gen < len(generations):
                    valid_gens = generations[first_valid_gen:]
                    
                    # Select values based on metric type
                    if metric_type == "Average":
                        y_values = avg_values[first_valid_gen:] if avg_values else []
                        trace_name = f"{setup_name} ({data_type.title()} Avg)"
                    elif metric_type == "Maximum":
                        y_values = max_values[first_valid_gen:] if max_values else []
                        trace_name = f"{setup_name} ({data_type.title()} Max)"
                    elif metric_type == "Minimum":
                        y_values = min_values[first_valid_gen:] if min_values else []
                        trace_name = f"{setup_name} ({data_type.title()} Min)"
                    elif metric_type == "Average with STD Bars":
                        y_values = avg_values[first_valid_gen:] if avg_values else []
                        trace_name = f"{setup_name} ({data_type.title()} Avg)"
                    else:
                        y_values = avg_values[first_valid_gen:] if avg_values else []
                        trace_name = f"{setup_name} ({data_type.title()})"
                    
                    if not y_values:
                        st.warning(f"No {metric_type.lower()} data available for {setup_name}")
                        continue
                    
                    # Add main line
                    fig.add_trace(go.Scatter(
                        x=valid_gens,
                        y=y_values,
                        mode='lines+markers',
                        name=trace_name,
                        line=dict(width=2, color=color_map.get(setup_name)),
                        marker=dict(size=6)
                    ))
                    traces_added += 1
                    
                    # Add STD error bars if requested and available
                    if metric_type == "Average with STD Bars" and std_values:
                        valid_std = std_values[first_valid_gen:]
                        if len(valid_std) == len(y_values):
                            # Add error bars using error_y
                            fig.data[-1].update(
                                error_y=dict(
                                    type='data',
                                    symmetric=True,
                                    array=valid_std,
                                    visible=True,
                                    color='lightblue',
                                    thickness=2
                                )
                            )
        
        if traces_added == 0:
            st.warning(f"No valid {data_type} data found for any setup.")
            return
        
        # Update title based on metric type
        title = f"{data_type.title()} Fitness Comparison"
        if metric_type != "Average":
            title += f" - {metric_type}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Generation",
            yaxis_title="Fitness",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_invalid_count_chart(self, comparison_results: Dict[str, Any], metric_type: str = "Average") -> None:
        """
        Render invalid count comparison chart.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            metric_type (str): "Average", "Maximum", "Minimum", or "Average with STD Bars"
        """
        fig = go.Figure()
        color_map = st.session_state.get('setup_color_map', {})
        
        if not comparison_results:
            st.warning("No comparison data available.")
            return
        
        traces_added = 0
        
        for setup_name, setup_data in comparison_results.items():
            # Get invalid count data from aggregate data
            invalid_data = setup_data.get('invalid_count', {})
            if invalid_data:
                generations = invalid_data.get('generations', [])
                avg_values = invalid_data.get('avg', [])
                std_values = invalid_data.get('std', [])
                max_values = invalid_data.get('max', [])
                min_values = invalid_data.get('min', [])
                
                if not generations:
                    st.warning(f"No invalid count data available for {setup_name}")
                    continue
                
                # Start from second generation (index 1) instead of generation 0
                start_gen = 1 if len(generations) > 1 else 0
                valid_generations = generations[start_gen:]
                
                # Select values based on metric type
                if metric_type == "Average":
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Invalid Avg)"
                elif metric_type == "Maximum":
                    y_values = max_values[start_gen:] if max_values else []
                    trace_name = f"{setup_name} (Invalid Max)"
                elif metric_type == "Minimum":
                    y_values = min_values[start_gen:] if min_values else []
                    trace_name = f"{setup_name} (Invalid Min)"
                elif metric_type == "Average with STD Bars":
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Invalid Avg)"
                else:
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Invalid)"
                
                if not y_values:
                    st.warning(f"No {metric_type.lower()} invalid count data available for {setup_name}")
                    continue
                
                # Add main line
                fig.add_trace(go.Scatter(
                    x=valid_generations,
                    y=y_values,
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(width=2, color=color_map.get(setup_name)),
                    marker=dict(size=6)
                ))
                traces_added += 1
                
                # Add STD error bars if requested and available
                if metric_type == "Average with STD Bars" and std_values and len(std_values) > start_gen:
                    valid_std = std_values[start_gen:]
                    if len(valid_std) == len(y_values):
                        fig.data[-1].update(
                            error_y=dict(
                                type='data',
                                symmetric=True,
                                array=valid_std,
                                visible=True,
                                color='lightblue',
                                thickness=2
                            )
                        )
        
        if traces_added == 0:
            st.warning("No valid invalid count data found for any setup.")
            return
        
        # Update title based on metric type
        title = "Number of Invalid Individuals Comparison"
        if metric_type != "Average":
            title += f" - {metric_type}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Generation",
            yaxis_title="Number of Invalid Individuals",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_nodes_length_chart(self, comparison_results: Dict[str, Any], metric_type: str = "Average") -> None:
        """
        Render nodes length comparison chart.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            metric_type (str): "Average", "Maximum", "Minimum", or "Average with STD Bars"
        """
        fig = go.Figure()
        color_map = st.session_state.get('setup_color_map', {})
        
        if not comparison_results:
            st.warning("No comparison data available.")
            return
        
        traces_added = 0
        
        for setup_name, setup_data in comparison_results.items():
            # Get nodes length data from aggregate data
            nodes_data = setup_data.get('nodes_length', {})
            if nodes_data:
                generations = nodes_data.get('generations', [])
                avg_values = nodes_data.get('avg', [])
                std_values = nodes_data.get('std', [])
                max_values = nodes_data.get('max', [])
                min_values = nodes_data.get('min', [])
                
                if not generations:
                    st.warning(f"No nodes length data available for {setup_name}")
                    continue
                
                # Start from second generation (index 1) instead of generation 0
                start_gen = 1 if len(generations) > 1 else 0
                valid_generations = generations[start_gen:]
                
                # Select values based on metric type
                if metric_type == "Average":
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Nodes Avg)"
                elif metric_type == "Maximum":
                    y_values = max_values[start_gen:] if max_values else []
                    trace_name = f"{setup_name} (Nodes Max)"
                elif metric_type == "Minimum":
                    y_values = min_values[start_gen:] if min_values else []
                    trace_name = f"{setup_name} (Nodes Min)"
                elif metric_type == "Average with STD Bars":
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Nodes Avg)"
                else:
                    y_values = avg_values[start_gen:] if avg_values else []
                    trace_name = f"{setup_name} (Nodes)"
                
                if not y_values:
                    st.warning(f"No {metric_type.lower()} nodes length data available for {setup_name}")
                    continue
                
                # Add main line
                fig.add_trace(go.Scatter(
                    x=valid_generations,
                    y=y_values,
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(width=2, color=color_map.get(setup_name)),
                    marker=dict(size=6)
                ))
                traces_added += 1
                
                # Add STD error bars if requested and available
                if metric_type == "Average with STD Bars" and std_values and len(std_values) > start_gen:
                    valid_std = std_values[start_gen:]
                    if len(valid_std) == len(y_values):
                        fig.data[-1].update(
                            error_y=dict(
                                type='data',
                                symmetric=True,
                                array=valid_std,
                                visible=True,
                                color='lightblue',
                                thickness=2
                            )
                        )
        
        if traces_added == 0:
            st.warning("No valid nodes length data found for any setup.")
            return
        
        # Update title based on metric type
        title = "Nodes Length Evolution Comparison"
        if metric_type != "Average":
            title += f" - {metric_type}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Generation",
            yaxis_title="Nodes Length",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_all_metrics_chart(self, comparison_results: Dict[str, Any]) -> None:
        """
        Render all metrics comparison chart with min, max, avg, and std.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
        """
        # Create subplots for training and test data
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Fitness (Min/Max/Avg)", "Test Fitness (Min/Max/Avg)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if not comparison_results:
            st.warning("No comparison data available.")
            return
        
        color_map = st.session_state.get('setup_color_map', {})
        for setup_name, setup_data in comparison_results.items():
            # Training fitness - show min, max, avg
            if 'training' in setup_data and setup_data['training']:
                training_data = setup_data['training']
                gens = training_data['generations']
                avg = training_data['avg']
                max_vals = training_data.get('max', [])
                min_vals = training_data.get('min', [])
                std_vals = training_data.get('std', [])
                
                if gens and avg:
                    # Average line with STD bars
                    fig.add_trace(go.Scatter(
                        x=gens, y=avg, 
                        name=f"{setup_name} (Train Avg)", 
                        mode='lines+markers',
                        line=dict(width=2, color=color_map.get(setup_name)),
                        error_y=dict(
                            type='data',
                            symmetric=True,
                            array=std_vals if std_vals and len(std_vals) == len(avg) else [],
                            visible=bool(std_vals and len(std_vals) == len(avg)),
                            color='lightblue',
                            thickness=1
                        )
                    ), row=1, col=1)
                    
                    # Max line
                    if max_vals and len(max_vals) == len(avg):
                        fig.add_trace(go.Scatter(
                            x=gens, y=max_vals, 
                            name=f"{setup_name} (Train Max)", 
                            mode='lines',
                            line=dict(width=1, dash='dot', color=color_map.get(setup_name))
                        ), row=1, col=1)
                    
                    # Min line
                    if min_vals and len(min_vals) == len(avg):
                        fig.add_trace(go.Scatter(
                            x=gens, y=min_vals, 
                            name=f"{setup_name} (Train Min)", 
                            mode='lines',
                            line=dict(width=1, dash='dot', color=color_map.get(setup_name))
                        ), row=1, col=1)
            
            # Test fitness - show min, max, avg
            if 'test' in setup_data and setup_data['test']:
                test_data = setup_data['test']
                gens = test_data['generations']
                avg = test_data['avg']
                max_vals = test_data.get('max', [])
                min_vals = test_data.get('min', [])
                std_vals = test_data.get('std', [])
                
                if gens and avg:
                    # Find first valid generation for test data
                    first_valid_gen = 0
                    for i, val in enumerate(avg):
                        if val is not None and not pd.isna(val) and val != 0:
                            first_valid_gen = i
                            break
                    
                    if first_valid_gen < len(gens):
                        valid_gens = gens[first_valid_gen:]
                        valid_avg = avg[first_valid_gen:]
                        valid_max = max_vals[first_valid_gen:] if max_vals and len(max_vals) == len(avg) else []
                        valid_min = min_vals[first_valid_gen:] if min_vals and len(min_vals) == len(avg) else []
                        valid_std = std_vals[first_valid_gen:] if std_vals and len(std_vals) == len(avg) else []
                        
                        # Average line with STD bars
                        fig.add_trace(go.Scatter(
                            x=valid_gens, y=valid_avg, 
                            name=f"{setup_name} (Test Avg)", 
                            mode='lines+markers',
                            line=dict(width=2, color=color_map.get(setup_name)),
                            error_y=dict(
                                type='data',
                                symmetric=True,
                                array=valid_std,
                                visible=bool(valid_std),
                                color='lightblue',
                                thickness=1
                            )
                        ), row=1, col=2)
                        
                        # Max line
                        if valid_max:
                            fig.add_trace(go.Scatter(
                                x=valid_gens, y=valid_max, 
                                name=f"{setup_name} (Test Max)", 
                                mode='lines',
                                line=dict(width=1, dash='dot', color=color_map.get(setup_name))
                            ), row=1, col=2)
                        
                        # Min line
                        if valid_min:
                            fig.add_trace(go.Scatter(
                                x=valid_gens, y=valid_min, 
                                name=f"{setup_name} (Test Min)", 
                                mode='lines',
                                line=dict(width=1, dash='dot', color=color_map.get(setup_name))
                            ), row=1, col=2)
        
        fig.update_layout(
            height=500, 
            showlegend=True, 
            title_text="All Metrics Comparison - Min/Max/Average with STD Bars"
        )
        fig.update_xaxes(title_text="Generation")
        fig.update_yaxes(title_text="Fitness")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _export_comparison_csv(self, comparison_results: Dict[str, Any]) -> str:
        """
        Export comparison results as CSV.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            
        Returns:
            str: CSV data as string
        """
        # Create a comprehensive DataFrame
        all_data = []
        
        for setup_name, setup_data in comparison_results.items():
            # Training data
            if 'training' in setup_data and setup_data['training']:
                for i, (gen, avg, std) in enumerate(zip(
                    setup_data['training']['generations'],
                    setup_data['training']['avg'],
                    setup_data['training']['std'] or [None] * len(setup_data['training']['generations'])
                )):
                    all_data.append({
                        'Setup': setup_name,
                        'Type': 'Training',
                        'Generation': gen,
                        'Average': avg,
                        'Std': std
                    })
            
            # Test data
            if 'test' in setup_data and setup_data['test']:
                for i, (gen, avg, std) in enumerate(zip(
                    setup_data['test']['generations'],
                    setup_data['test']['avg'],
                    setup_data['test']['std'] or [None] * len(setup_data['test']['generations'])
                )):
                    all_data.append({
                        'Setup': setup_name,
                        'Type': 'Test',
                        'Generation': gen,
                        'Average': avg,
                        'Std': std
                    })
        
        df = pd.DataFrame(all_data)
        return df.to_csv(index=False)

    def _render_tsne_best_individuals(self, comparison_results: Dict[str, Any], selected_setups: List[str]) -> None:
        """
        Render t-SNE phenotype analysis following GE literature methodology.
        Visualizes best individuals from each run to show phenotypic space exploration.
        """
        st.subheader("🧬 t-SNE Phenotype Analysis")
        
        st.markdown("""
        This analysis visualizes the **phenotypic characteristics** of solutions across setups,
        following methodology from Grammatical Evolution literature. Each point represents a best
        individual from a specific generation and run, projected into 2D space using t-SNE.
        
        **Reference**: Lourenço et al. (2024). "Probabilistic Grammatical Evolution". 
        *GECCO '24*. DOI: [10.1145/3712256.3726444](https://doi.org/10.1145/3712256.3726444)
        """)

        # Configuration UI
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Generation Selection")
            gen_option = st.radio(
                "Which generations to analyze?",
                ["All Generations", "Specific Generations", "Final Generation Only"],
                key="tsne_gen_option"
            )
            
            if gen_option == "Specific Generations":
                # Get max generations from first setup
                first_setup = self.storage_service.load_setup(selected_setups[0])
                if first_setup and first_setup.results:
                    first_result = list(first_setup.results.values())[0]
                    max_gens = len(first_result.max) if hasattr(first_result, 'max') else 50
                    
                    selected_gens_str = st.text_input(
                        "Enter generations (comma-separated, e.g., 0,10,20,50)",
                        value="0,25,50",
                        key="tsne_selected_gens"
                    )
                    try:
                        selected_generations = [int(g.strip()) for g in selected_gens_str.split(',')]
                    except:
                        st.error("Invalid generation format")
                        selected_generations = None
                else:
                    selected_generations = None
            elif gen_option == "Final Generation Only":
                # Will be handled by getting the last generation
                selected_generations = [-1]  # Special marker for final generation
            else:
                selected_generations = None  # All generations
        
        with col2:
            st.markdown("#### Feature Selection")
            available_features = {
                'fitness_train': 'Fitness (Training)',
                'fitness_test': 'Fitness (Test)',
                'nodes_length': 'Nodes Length',
                'tree_depth': 'Tree Depth',
                'invalid_count': 'Invalid Count',
                'codon_consumption': 'Codon Consumption'
            }
            
            selected_features = st.multiselect(
                "Select phenotype features for t-SNE:",
                options=list(available_features.keys()),
                default=['fitness_train', 'fitness_test', 'nodes_length'],
                format_func=lambda x: available_features[x],
                help="Choose which features to use for the t-SNE embedding. More features = more dimensions considered."
            )
            
            if len(selected_features) < 2:
                st.warning("⚠️ Please select at least 2 features for meaningful t-SNE analysis")

        # t-SNE Parameters in expandable section
        with st.expander("🔧 Advanced t-SNE Parameters", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                perplexity = st.slider("Perplexity", min_value=5.0, max_value=50.0, value=30.0, step=5.0,
                                      help="Controls local vs global structure. Lower = more local detail")
                learning_rate = st.slider("Learning Rate", min_value=10.0, max_value=500.0, value=200.0, step=50.0,
                                         help="Step size for optimization")
            with col_b:
                n_iter = st.slider("Iterations", min_value=250, max_value=2000, value=1000, step=250,
                                  help="Number of optimization iterations")
                random_state = st.number_input("Random Seed", min_value=0, max_value=10000, value=42, step=1,
                                              help="For reproducibility")

        # Info about what will be visualized
        with st.expander("ℹ️ What are phenotype features?"):
            st.markdown("""
            The t-SNE embedding is created using **phenotypic features** of each individual:
            
            - **Fitness (Training & Test)**: How well the solution performs
            - **Nodes Length**: Number of terminal symbols (solution complexity)
            - **Tree Depth**: Maximum depth of the parse tree
            - **Invalid Count**: Number of invalid individuals encountered
            - **Codon Consumption**: How many codons were used
            
            These features characterize the **phenotype** (what the solution looks like and how it behaves),
            not just its fitness. This allows us to see if different setups explore different regions of
            the phenotypic space.
            
            **Tip**: Start with fitness + nodes_length, then experiment with adding other features.
            """)

        if st.button("🔬 Generate t-SNE Phenotype Analysis", type="primary"):
            # Validate feature selection
            if len(selected_features) < 2:
                st.error("❌ Please select at least 2 features for t-SNE analysis")
                return
            
            with st.spinner("Extracting individuals and computing t-SNE embedding..."):
                # Handle special case for final generation
                if selected_generations == [-1]:
                    # Get final generation index from first setup
                    first_setup = self.storage_service.load_setup(selected_setups[0])
                    if first_setup and first_setup.results:
                        first_result = list(first_setup.results.values())[0]
                        final_gen = len(first_result.max) - 1 if hasattr(first_result, 'max') else 49
                        selected_generations = [final_gen]
                
                ConfigCharts.plot_tsne_phenotype_analysis(
                    setup_data=comparison_results,
                    storage_service=self.storage_service,
                    selected_setups=selected_setups,
                    n_best_per_run=15,  # Following the paper's methodology
                    selected_generations=selected_generations,
                    selected_features=selected_features,  # Pass user-selected features
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    random_state=int(random_state),
                    color_map=st.session_state.get('setup_color_map')
                )
    
    def _get_available_config_params(self, setup_data: Dict[str, Any]) -> List[str]:
        """
        Get available configuration parameters from setup data.
        
        Args:
            setup_data (Dict[str, Any]): Setup data dictionary
            
        Returns:
            List[str]: List of available configuration parameter names
        """
        # Try to get generation configs from the setup data
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
        
        if not generation_configs:
            return []
        
        # Get available configuration parameters
        first_gen_config = generation_configs[0]
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
    
    def _render_configuration_comparison_chart(self, comparison_results: Dict[str, Any], config_param: str) -> None:
        """
        Render configuration comparison chart across setups.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            config_param (str): Configuration parameter to compare
        """
        # Get setup configs from the comparison results
        setup_configs = comparison_results.get('setup_configs', {})
        
        if not setup_configs:
            st.warning("No setup configuration data available for comparison.")
            return
        
        # Use ConfigCharts to render the comparison
        ConfigCharts.plot_setup_configuration_comparison(
            setup_configs,
            config_param=config_param,
            title=f"{config_param.replace('_', ' ').title()} Comparison Across Setups",
            color_map=st.session_state.get('setup_color_map')
        )
    
    def _get_available_setup_config_params(self, comparison_results: Dict[str, Any], selected_setups: List[str]) -> List[str]:
        """
        Get available configuration parameters from setup configurations.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            selected_setups (List[str]): List of selected setup IDs
            
        Returns:
            List[str]: List of available configuration parameter names
        """
        # Get setup configs from the comparison results
        setup_configs = comparison_results.get('setup_configs', {})
        
        if not setup_configs:
            return []
        
        # Get configuration parameters from the first setup config
        first_setup_id = selected_setups[0] if selected_setups else list(setup_configs.keys())[0]
        first_config = setup_configs.get(first_setup_id, {})
        
        if not first_config:
            return []
        
        # Get numeric configuration parameters
        numeric_params = []
        for param, value in first_config.items():
            if param in ['setup_name', 'created_at', 'completed_at']:
                continue
            try:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_params.append(param)
            except:
                continue
        
        return numeric_params
    
    def _get_available_generation_config_params(self, comparison_results: Dict[str, Any]) -> List[str]:
        """
        Get available configuration parameters from generation configurations.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            
        Returns:
            List[str]: List of available configuration parameter names
        """
        # Look for generation configs in the comparison results
        for setup_name, setup_data in comparison_results.items():
            if setup_name == 'setup_configs':
                continue
                
            # Try to get generation configs from the setup data
            generation_configs = None
            if 'results' in setup_data:
                # If it's a Setup object
                results = setup_data['results']
                if results:
                    first_result = list(results.values())[0]
                    if hasattr(first_result, 'generation_configs'):
                        generation_configs = first_result.generation_configs
                    elif isinstance(first_result, dict) and 'generation_configs' in first_result:
                        generation_configs = first_result['generation_configs']
            elif 'generation_configs' in setup_data:
                # If it's direct generation configs
                generation_configs = setup_data['generation_configs']
            
            if generation_configs:
                # Get available configuration parameters from the first generation config
                first_gen_config = generation_configs[0]
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
        
        return []
    
    def _render_configuration_evolution_chart(self, comparison_results: Dict[str, Any], config_param: str) -> None:
        """
        Render configuration evolution chart across setups.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            config_param (str): Configuration parameter to compare
        """
        # Transform comparison results to the format expected by ConfigCharts
        setup_results = {}
        
        for setup_name, setup_data in comparison_results.items():
            if setup_name == 'setup_configs':
                continue
                
            # Try to get generation configs from the setup data
            generation_configs = None
            if 'results' in setup_data:
                # If it's a Setup object
                results = setup_data['results']
                if results:
                    first_result = list(results.values())[0]
                    if hasattr(first_result, 'generation_configs'):
                        generation_configs = first_result.generation_configs
                    elif isinstance(first_result, dict) and 'generation_configs' in first_result:
                        generation_configs = first_result['generation_configs']
            elif 'generation_configs' in setup_data:
                # If it's direct generation configs
                generation_configs = setup_data['generation_configs']
            
            if generation_configs:
                setup_results[setup_name] = {'generation_configs': generation_configs}
        
        if setup_results:
            ConfigCharts.plot_configuration_comparison(
                setup_results,
                config_param=config_param,
                title=f"{config_param.replace('_', ' ').title()} Evolution Across Setups",
                color_map=st.session_state.get('setup_color_map')
            )
        else:
            st.warning("No generation configuration data available for evolution comparison.")
    
    def _render_statistical_comparison(self, comparison_results: Dict[str, Any], selected_setups: List[str]) -> None:
        """
        Render statistical significance tests for setup comparisons.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            selected_setups (List[str]): List of selected setup IDs
        """
        st.markdown("### 📊 Statistical Significance Testing")
        st.markdown("""
        Compare setups using rigorous statistical tests following best practices from the 
        Evolutionary Computation and Machine Learning literature.
        
        **References:**
        - Demšar (JMLR 2006)
        - Derrac et al. (SWARM & EC 2011)
        - García & Herrera (2008)
        """)
        
        setup_names = st.session_state.get('selected_setup_names', [k for k in comparison_results.keys() if k != 'setup_configs'])
        
        if len(setup_names) < 2:
            st.warning("Need at least 2 setups for statistical comparison.")
            return
        
        # Get raw run data for each setup
        try:
            # Load setup objects to get individual run results
            setup_data_for_stats = {}
            for i, setup_id in enumerate(selected_setups):
                setup = self.storage_service.load_setup(setup_id)
                if setup and setup.results:
                    setup_name = setup_names[i]
                    setup_data_for_stats[setup_name] = {
                        'setup_id': setup_id,
                        'setup_obj': setup,
                        'config': setup.config
                    }
            
            if len(setup_data_for_stats) < 2:
                st.error("Could not load setup data for statistical comparison.")
                return
            
            # Metric selection
            st.markdown("#### Select Metric and Generation for Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                metric_categories = {
                    "Fitness Metrics": {
                        "Best Training Fitness (Final)": "final_training",
                        "Best Test Fitness (Final)": "final_test",
                        "Average Training Fitness (Final)": "avg_training_final",
                        "Best Training Fitness (Generation)": "best_training_gen",
                        "Average Training Fitness (Generation)": "avg_training_gen"
                    },
                    "Phenotype Metrics - Per Generation": {
                        "Nodes Length - Min": "nodes_length_min_gen",
                        "Nodes Length - Avg": "nodes_length_avg_gen",
                        "Nodes Length - Max": "nodes_length_max_gen",
                        "Invalid Count - Min": "invalid_count_min_gen",
                        "Invalid Count - Avg": "invalid_count_avg_gen",
                        "Invalid Count - Max": "invalid_count_max_gen"
                    },
                    "Best Individual Characteristics - Per Run": {
                        "Tree Depth (of best individual)": "best_depth_final",
                        "Genome Length (of best individual)": "best_genome_length_final",
                        "Used Codons Ratio (of best individual)": "best_used_codons_final"
                    }
                }
                
                # Flatten options for selectbox
                all_options = {}
                for category, options in metric_categories.items():
                    for label, key in options.items():
                        all_options[f"{category} → {label}"] = key
                
                selected_metric = st.selectbox(
                    "Metric to Compare",
                    list(all_options.keys()),
                    key="stat_metric",
                    help="Choose which metric to statistically compare between setups"
                )
                metric_key = all_options[selected_metric]
            
            with col2:
                # For generation-specific comparisons
                if "Final" in selected_metric:
                    generation_idx = -1  # Last generation
                    st.info("Using final generation results")
                elif "Generation" in selected_metric:
                    # Get number of generations (assume all setups have same number)
                    first_setup = list(setup_data_for_stats.values())[0]
                    first_result = list(first_setup['setup_obj'].results.values())[0]
                    n_gens = len(first_result.max) if hasattr(first_result, 'max') else 50
                    
                    generation_idx = st.slider(
                        "Select Generation",
                        0, n_gens - 1,
                        n_gens - 1,
                        key="stat_generation",
                        help="Compare setups at a specific generation"
                    )
                else:
                    generation_idx = -1
            
            # Extract data for comparison
            st.markdown("#### Extracting Run Data...")
            extracted_data = {}
            
            for setup_name, data in setup_data_for_stats.items():
                setup_obj = data['setup_obj']
                runs_data = []
                
                for run_id, result in setup_obj.results.items():
                    value = None
                    
                    # Fitness metrics
                    if metric_key == "final_training":
                        value = result.best_training_fitness
                    elif metric_key == "final_test":
                        value = result.best_test_fitness if hasattr(result, 'best_test_fitness') else (result.fitness_test[-1] if result.fitness_test else None)
                    elif metric_key == "avg_training_final":
                        value = result.avg[-1] if result.avg else None
                    elif metric_key == "best_training_gen":
                        value = result.max[generation_idx] if hasattr(result, 'max') and generation_idx < len(result.max) else None
                    elif metric_key == "avg_training_gen":
                        value = result.avg[generation_idx] if hasattr(result, 'avg') and generation_idx < len(result.avg) else None
                    
                    # Phenotype metrics - Nodes Length (per generation)
                    elif metric_key == "nodes_length_min_gen":
                        value = result.nodes_length_min[generation_idx] if hasattr(result, 'nodes_length_min') and generation_idx < len(result.nodes_length_min) else None
                    elif metric_key == "nodes_length_avg_gen":
                        value = result.nodes_length_avg[generation_idx] if hasattr(result, 'nodes_length_avg') and generation_idx < len(result.nodes_length_avg) else None
                    elif metric_key == "nodes_length_max_gen":
                        value = result.nodes_length_max[generation_idx] if hasattr(result, 'nodes_length_max') and generation_idx < len(result.nodes_length_max) else None
                    
                    # Phenotype metrics - Invalid Count (per generation)
                    elif metric_key == "invalid_count_min_gen":
                        value = result.invalid_count_min[generation_idx] if hasattr(result, 'invalid_count_min') and generation_idx < len(result.invalid_count_min) else None
                    elif metric_key == "invalid_count_avg_gen":
                        value = result.invalid_count_avg[generation_idx] if hasattr(result, 'invalid_count_avg') and generation_idx < len(result.invalid_count_avg) else None
                    elif metric_key == "invalid_count_max_gen":
                        value = result.invalid_count_max[generation_idx] if hasattr(result, 'invalid_count_max') and generation_idx < len(result.invalid_count_max) else None
                    
                    # Phenotype metrics - Best Individual characteristics (final values)
                    elif metric_key == "best_depth_final":
                        value = result.best_depth if hasattr(result, 'best_depth') else None
                    elif metric_key == "best_genome_length_final":
                        value = result.best_genome_length if hasattr(result, 'best_genome_length') else None
                    elif metric_key == "best_used_codons_final":
                        value = result.best_used_codons if hasattr(result, 'best_used_codons') else None
                    
                    if value is not None:
                        runs_data.append(value)
                
                if runs_data:
                    extracted_data[setup_name] = np.array(runs_data)
            
            if len(extracted_data) < 2:
                st.error("Could not extract sufficient data for comparison.")
                return
            
            # Display raw data values for each setup
            st.markdown("#### 📋 Raw Data from Each Run")
            st.markdown("These are the actual values extracted from each run that will be used for statistical testing:")
            
            # Create expandable sections for each setup's raw data
            for setup_name, data in extracted_data.items():
                with st.expander(f"📊 {setup_name} - {len(data)} runs", expanded=False):
                    # Create a DataFrame with run index and values
                    raw_data_df = pd.DataFrame({
                        'Run #': [f"Run {i+1}" for i in range(len(data))],
                        'Value': data
                    })
                    
                    # Display as a table
                    st.dataframe(raw_data_df, use_container_width=True, height=min(400, len(data) * 35 + 38))
                    
                    # Quick stats for this setup
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{np.mean(data):.4f}")
                    with col2:
                        st.metric("Median", f"{np.median(data):.4f}")
                    with col3:
                        st.metric("Std Dev", f"{np.std(data, ddof=1):.4f}")
                    
                    # Show sorted values
                    st.markdown("**Sorted values (ascending):**")
                    sorted_values = sorted(data)
                    st.write(", ".join([f"{v:.4f}" for v in sorted_values]))
            
            # Display descriptive statistics summary
            st.markdown("#### 📈 Summary Statistics (All Setups)")
            desc_df = pd.DataFrame({
                'Setup': list(extracted_data.keys()),
                'N Runs': [len(data) for data in extracted_data.values()],
                'Mean': [np.mean(data) for data in extracted_data.values()],
                'Median': [np.median(data) for data in extracted_data.values()],
                'Std Dev': [np.std(data, ddof=1) for data in extracted_data.values()],
                'Min': [np.min(data) for data in extracted_data.values()],
                'Max': [np.max(data) for data in extracted_data.values()]
            })
            st.dataframe(desc_df, use_container_width=True)
            
            # Scenario A: Two setups, independent runs
            if len(extracted_data) == 2:
                st.markdown("#### 🔬 Statistical Test: Two Independent Samples")
                
                setup_names_list = list(extracted_data.keys())
                group1 = extracted_data[setup_names_list[0]]
                group2 = extracted_data[setup_names_list[1]]
                
                # Step 1: Check normality
                st.markdown("#### 📊 Step 1: Check Normality (Shapiro-Wilk Test)")
                
                with st.spinner("Checking normality..."):
                    norm1 = StatisticalTests.check_normality(group1)
                    norm2 = StatisticalTests.check_normality(group2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{setup_names_list[0]}**")
                    st.write(f"- p-value: {norm1['p_value']:.4f}")
                    if norm1['is_normal']:
                        st.success(f"✅ Normal distribution")
                    else:
                        st.warning(f"❌ NOT normal distribution")
                
                with col2:
                    st.markdown(f"**{setup_names_list[1]}**")
                    st.write(f"- p-value: {norm2['p_value']:.4f}")
                    if norm2['is_normal']:
                        st.success(f"✅ Normal distribution")
                    else:
                        st.warning(f"❌ NOT normal distribution")
                
                # Step 2: Choose and run appropriate test
                st.markdown("#### 🎯 Step 2: Run Appropriate Statistical Test")
                
                # Determine which test to use
                use_parametric = norm1['is_normal'] and norm2['is_normal']
                
                if use_parametric:
                    st.info("**Both distributions are normal** → Using **Welch's t-test** (parametric)")
                    with st.spinner("Running Welch's t-test..."):
                        test_result = StatisticalTests.welch_t_test(group1, group2)
                else:
                    st.info("**At least one distribution is non-normal** → Using **Mann-Whitney U test** (non-parametric)")
                    with st.spinner("Running Mann-Whitney U test..."):
                        test_result = StatisticalTests.mann_whitney_u(group1, group2)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Used", test_result['test'])
                with col2:
                    st.metric("p-value", f"{test_result['p_value']:.4f}")
                with col3:
                    if test_result['significant']:
                        st.metric("Result", "✅ Significant", help="p < 0.05")
                    else:
                        st.metric("Result", "❌ Not Significant", help="p ≥ 0.05")
                
                # Interpretation
                st.markdown("#### 💡 Interpretation")
                if test_result['significant']:
                    mean1 = np.mean(group1)
                    mean2 = np.mean(group2)
                    if mean1 > mean2:
                        better = setup_names_list[0]
                        worse = setup_names_list[1]
                    else:
                        better = setup_names_list[1]
                        worse = setup_names_list[0]
                    
                    st.success(f"""
                    **{better}** performs **significantly better** than **{worse}** 
                    (p = {test_result['p_value']:.4f}).
                    
                    This means the difference is statistically significant and unlikely due to random chance.
                    """)
                else:
                    st.info(f"""
                    **No significant difference** between {setup_names_list[0]} and {setup_names_list[1]} 
                    (p = {test_result['p_value']:.4f}).
                    
                    The observed difference could be due to random variation.
                    """)
                
                
                # Visualization
                st.markdown("#### 📊 Distribution Comparison")
                fig = go.Figure()
                
                for setup_name, data in extracted_data.items():
                    color = st.session_state.get('setup_color_map', {}).get(setup_name, '#1f77b4')
                    fig.add_trace(go.Box(
                        y=data,
                        name=setup_name,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title=f"Distribution of {selected_metric}",
                    yaxis_title=selected_metric,
                    xaxis_title="Setup",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Scenario B: More than 2 setups, independent runs
            elif len(extracted_data) > 2:
                st.markdown("#### 🔬 Statistical Test: Multiple Independent Samples")
                st.info("**Scenario B**: One problem, 3+ algorithms, independent runs")
                
                groups = list(extracted_data.values())
                setup_names_list = list(extracted_data.keys())
                
                # Kruskal-Wallis test
                with st.spinner("Running Kruskal-Wallis test..."):
                    kw_results = StatisticalTests.kruskal_wallis(groups)
                
                st.markdown("#### 🎯 Kruskal-Wallis H Test")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("H Statistic", f"{kw_results['statistic']:.4f}")
                with col2:
                    st.metric("p-value", f"{kw_results['p_value']:.4f}")
                with col3:
                    sig_emoji = "✅" if kw_results['significant'] else "❌"
                    st.metric("Significant", f"{sig_emoji} {kw_results['significant']}")
                with col4:
                    st.metric("η²", f"{kw_results['effect_size_eta_squared']:.3f}")
                
                if kw_results['significant']:
                    st.success("✅ Significant difference detected among setups!")
                    st.markdown("#### 🔍 Post-hoc Pairwise Comparisons (Mann-Whitney U)")
                    
                    # Perform pairwise comparisons
                    pairwise_results = []
                    for i in range(len(setup_names_list)):
                        for j in range(i + 1, len(setup_names_list)):
                            mw_result = StatisticalTests.mann_whitney_u(groups[i], groups[j])
                            pairwise_results.append({
                                'Setup 1': setup_names_list[i],
                                'Setup 2': setup_names_list[j],
                                'p-value': f"{mw_result['p_value']:.4f}",
                                'Significant': '✅ Yes' if mw_result['p_value'] < 0.05 else '❌ No'
                            })
                    
                    pairwise_df = pd.DataFrame(pairwise_results)
                    st.dataframe(pairwise_df, use_container_width=True)
                    
                    st.caption("⚠️ Note: For multiple comparisons, consider applying Holm-Bonferroni correction")
                else:
                    st.info("No significant difference detected among setups (p > 0.05)")
                
                # Visualization
                st.markdown("#### 📊 Distribution Comparison")
                fig = go.Figure()
                
                for setup_name, data in extracted_data.items():
                    color = st.session_state.get('setup_color_map', {}).get(setup_name, '#1f77b4')
                    fig.add_trace(go.Box(
                        y=data,
                        name=setup_name,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title=f"Distribution of {selected_metric}",
                    yaxis_title=selected_metric,
                    xaxis_title="Setup",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Guidelines
            with st.expander("📚 Simple Guidelines for Statistical Testing"):
                st.markdown("""
                ### Statistical Testing for Evolutionary Algorithms
                
                #### 🎯 The Standard Approach:
                
                **Step 1: Check Normality**
                - Use **Shapiro-Wilk test**
                - **p > 0.05** → Data is normal ✅
                - **p ≤ 0.05** → Data is NOT normal ❌
                
                **Step 2: Choose Test Based on Normality**
                - **Both normal** → Use **Welch's t-test**
                - **At least one non-normal** → Use **Mann-Whitney U test** ⭐ (most common in EA)
                
                **Step 3: Interpret p-value**
                - **p < 0.05** → Significant difference ✅
                - **p ≥ 0.05** → No significant difference ❌
                
                ---
                
                #### 📊 For 3+ Setups:
                
                1. Use **Kruskal-Wallis H test** (overall test)
                2. If significant → Do **pairwise Mann-Whitney U tests**
                3. Apply **Holm-Bonferroni correction** for multiple comparisons
                
                ---
                
                #### ⚠️ Important Notes:
                
                - **Always run 30+ times** per setup (minimum 10)
                - **Use independent random seeds** for each run
                - EA/GE results are often **non-normal** → Mann-Whitney U is safer
                - **Report both**: normality test results + final p-value
                
                ---
                
                #### 📚 References:
                - Demšar (2006): Statistical Comparisons - JMLR
                - Derrac et al. (2011): Nonparametric Tests - SWARM & EC
                """)
        
        except Exception as e:
            st.error(f"Error in statistical comparison: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
