"""
Comparison View Module

Handles the UI for setup comparison and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List
from uge.services.storage_service import StorageService
from uge.views.components.config_charts import ConfigCharts


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
                
                if len(selected_setups) >= 2:
                    if st.button("🔍 Compare Setups"):
                        with st.spinner("Comparing setups..."):
                            comparison_results = comparison_controller.compare_setups(selected_setups)
                            
                            if comparison_results:
                                # Transform the comparison results to match expected format
                                transformed_results = self._transform_comparison_results(comparison_results, selected_setup_names)
                                
                                # Debug information
                                st.write(f"Debug: Found {len(transformed_results)} setups to compare")
                                for setup_name, data in transformed_results.items():
                                    training_gen = len(data['training']['generations']) if data['training']['generations'] else 0
                                    test_gen = len(data['test']['generations']) if data['test']['generations'] else 0
                                    st.write(f"Debug: {setup_name} - Training: {training_gen} gens, Test: {test_gen} gens")
                                
                                # Store results in session state
                                st.session_state.comparison_results = transformed_results
                                st.session_state.selected_setups = selected_setups
                                st.session_state.selected_setup_names = selected_setup_names
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
        
        return transformed
    
    def _render_comparison_results(self, comparison_results: Dict[str, Any], selected_setups: List[str]) -> None:
        """
        Render comparison results with charts and statistics.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            selected_setups (List[str]): List of selected setup IDs
        """
        st.subheader("📊 Comparison Results")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Training Fitness", "Test Fitness", "Number of Invalid", "Nodes Length Evolution", "Configuration Comparison", "All Metrics"],
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
            # Get available configuration parameters from the first setup
            first_setup_data = list(comparison_results.values())[0]
            available_config_params = self._get_available_config_params(first_setup_data)
            
            if available_config_params:
                metric_type = st.selectbox(
                    "Select Configuration Parameter",
                    available_config_params,
                    key="comparison_config_param"
                )
            else:
                st.warning("No configuration parameters available for comparison.")
                metric_type = None
        else:
            metric_type = "All"
        
        # Render the appropriate chart
        self._render_comparison_chart(comparison_results, chart_type, metric_type)
        
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
    
    def _render_comparison_chart(self, comparison_results: Dict[str, Any], chart_type: str, metric_type: str = "All") -> None:
        """
        Render comparison chart based on chart type and metric type.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            chart_type (str): Type of chart to render
            metric_type (str): Type of metric to display
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
                        line=dict(width=2),
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
                
                # Select values based on metric type
                if metric_type == "Average":
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Invalid Avg)"
                elif metric_type == "Maximum":
                    y_values = max_values if max_values else []
                    trace_name = f"{setup_name} (Invalid Max)"
                elif metric_type == "Minimum":
                    y_values = min_values if min_values else []
                    trace_name = f"{setup_name} (Invalid Min)"
                elif metric_type == "Average with STD Bars":
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Invalid Avg)"
                else:
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Invalid)"
                
                if not y_values:
                    st.warning(f"No {metric_type.lower()} invalid count data available for {setup_name}")
                    continue
                
                # Add main line
                fig.add_trace(go.Scatter(
                    x=generations,
                    y=y_values,
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                traces_added += 1
                
                # Add STD error bars if requested and available
                if metric_type == "Average with STD Bars" and std_values and len(std_values) == len(y_values):
                    fig.data[-1].update(
                        error_y=dict(
                            type='data',
                            symmetric=True,
                            array=std_values,
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
                
                # Select values based on metric type
                if metric_type == "Average":
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Nodes Avg)"
                elif metric_type == "Maximum":
                    y_values = max_values if max_values else []
                    trace_name = f"{setup_name} (Nodes Max)"
                elif metric_type == "Minimum":
                    y_values = min_values if min_values else []
                    trace_name = f"{setup_name} (Nodes Min)"
                elif metric_type == "Average with STD Bars":
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Nodes Avg)"
                else:
                    y_values = avg_values if avg_values else []
                    trace_name = f"{setup_name} (Nodes)"
                
                if not y_values:
                    st.warning(f"No {metric_type.lower()} nodes length data available for {setup_name}")
                    continue
                
                # Add main line
                fig.add_trace(go.Scatter(
                    x=generations,
                    y=y_values,
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                traces_added += 1
                
                # Add STD error bars if requested and available
                if metric_type == "Average with STD Bars" and std_values and len(std_values) == len(y_values):
                    fig.data[-1].update(
                        error_y=dict(
                            type='data',
                            symmetric=True,
                            array=std_values,
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
                        line=dict(width=2),
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
                            line=dict(width=1, dash='dot')
                        ), row=1, col=1)
                    
                    # Min line
                    if min_vals and len(min_vals) == len(avg):
                        fig.add_trace(go.Scatter(
                            x=gens, y=min_vals, 
                            name=f"{setup_name} (Train Min)", 
                            mode='lines',
                            line=dict(width=1, dash='dot')
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
                            line=dict(width=2),
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
                                line=dict(width=1, dash='dot')
                            ), row=1, col=2)
                        
                        # Min line
                        if valid_min:
                            fig.add_trace(go.Scatter(
                                x=valid_gens, y=valid_min, 
                                name=f"{setup_name} (Test Min)", 
                                mode='lines',
                                line=dict(width=1, dash='dot')
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
        # Transform comparison results to the format expected by ConfigCharts
        setup_results = {}
        
        for setup_name, setup_data in comparison_results.items():
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
            
            if generation_configs:
                setup_results[setup_name] = {'generation_configs': generation_configs}
        
        if setup_results:
            ConfigCharts.plot_configuration_comparison(
                setup_results,
                config_param=config_param,
                title=f"{config_param.replace('_', ' ').title()} Comparison Across Setups"
            )
        else:
            st.warning("No configuration data available for comparison.")
