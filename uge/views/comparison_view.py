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
                        comparison_results = comparison_controller.compare_setups(selected_setups)
                        
                        if comparison_results:
                            # Store results in session state
                            st.session_state.comparison_results = comparison_results
                            st.session_state.selected_setups = selected_setups
                            st.session_state.selected_setup_names = selected_setup_names
                    
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
            ["Training Fitness", "Test Fitness", "All Metrics"],
            key="comparison_chart_type"
        )
        
        # Render the appropriate chart
        self._render_comparison_chart(comparison_results, chart_type)
        
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
    
    def _render_comparison_chart(self, comparison_results: Dict[str, Any], chart_type: str) -> None:
        """
        Render comparison chart based on chart type.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            chart_type (str): Type of chart to render
        """
        if chart_type == "Training Fitness":
            self._render_fitness_chart(comparison_results, "training")
        elif chart_type == "Test Fitness":
            self._render_fitness_chart(comparison_results, "test")
        elif chart_type == "All Metrics":
            self._render_all_metrics_chart(comparison_results)
    
    def _render_fitness_chart(self, comparison_results: Dict[str, Any], data_type: str) -> None:
        """
        Render fitness comparison chart.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
            data_type (str): "training" or "test"
        """
        fig = go.Figure()
        
        for setup_name, setup_data in comparison_results.items():
            if data_type in setup_data and setup_data[data_type]:
                generations = setup_data[data_type]['generations']
                avg_values = setup_data[data_type]['avg']
                std_values = setup_data[data_type]['std']
                
                # Find first valid generation (non-NaN)
                first_valid_gen = 0
                if data_type == "test":
                    for i, val in enumerate(avg_values):
                        if val is not None and not pd.isna(val) and val != 0:
                            first_valid_gen = i
                            break
                
                # Plot from first valid generation
                if first_valid_gen < len(generations):
                    valid_gens = generations[first_valid_gen:]
                    valid_avg = avg_values[first_valid_gen:]
                    valid_std = std_values[first_valid_gen:] if std_values else None
                    
                    # Add main line
                    fig.add_trace(go.Scatter(
                        x=valid_gens,
                        y=valid_avg,
                        mode='lines+markers',
                        name=f"{setup_name} ({data_type.title()})",
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Add error bars if available
                    if valid_std:
                        fig.add_trace(go.Scatter(
                            x=valid_gens,
                            y=valid_avg,
                            mode='lines',
                            name=f"{setup_name} ({data_type.title()}) ±1σ",
                            line=dict(width=1, dash='dash'),
                            showlegend=False
                        ))
        
        fig.update_layout(
            title=f"{data_type.title()} Fitness Comparison",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_all_metrics_chart(self, comparison_results: Dict[str, Any]) -> None:
        """
        Render all metrics comparison chart.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results data
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Training Fitness", "Test Fitness", "Training Accuracy", "Test Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for setup_name, setup_data in comparison_results.items():
            # Training fitness
            if 'training' in setup_data and setup_data['training']:
                gens = setup_data['training']['generations']
                avg = setup_data['training']['avg']
                fig.add_trace(go.Scatter(x=gens, y=avg, name=f"{setup_name} (Train)", mode='lines+markers'), row=1, col=1)
            
            # Test fitness
            if 'test' in setup_data and setup_data['test']:
                gens = setup_data['test']['generations']
                avg = setup_data['test']['avg']
                # Find first valid generation for test data
                first_valid_gen = 0
                for i, val in enumerate(avg):
                    if val is not None and not pd.isna(val) and val != 0:
                        first_valid_gen = i
                        break
                if first_valid_gen < len(gens):
                    valid_gens = gens[first_valid_gen:]
                    valid_avg = avg[first_valid_gen:]
                    fig.add_trace(go.Scatter(x=valid_gens, y=valid_avg, name=f"{setup_name} (Test)", mode='lines+markers'), row=1, col=2)
            
            # Add accuracy plots if available
            # (This would be implemented based on available accuracy data)
        
        fig.update_layout(height=600, showlegend=True, title_text="All Metrics Comparison")
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
