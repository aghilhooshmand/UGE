"""
Home View Module

Handles the UI for the home/welcome page.
"""

import streamlit as st
from typing import Optional
from uge.views.components.base_view import BaseView


class HomeView(BaseView):
    """
    Home View class for handling the welcome/home page UI.
    
    This class provides methods to render the home page,
    showing project overview, features, and getting started information.
    """
    
    def __init__(self):
        """
        Initialize the HomeView.
        """
        super().__init__(
            title="🧬 GE-Lab - Grammatical Evolution Laboratory",
            description="Welcome to GE-Lab - A Laboratory for learning how Grammatical Evolution works"
        )
    
    def render(self) -> None:
        """Render the view (required by BaseView)."""
        self.render_home()
    
    def render_home(self) -> None:
        """Render the home page."""
        # Hero Section
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 1rem;">🧬 GE-Lab</h1>
            <h2 style="color: #666; font-size: 1.5rem; margin-bottom: 2rem;">Grammatical Evolution Laboratory</h2>
            <p style="font-size: 1.2rem; color: #888; max-width: 800px; margin: 0 auto;">
                A Laboratory for learning how Grammatical Evolution works. A sophisticated web application that provides a user-friendly interface for conducting 
                Grammatical Evolution setups with advanced visualization and analysis capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Start Section
        st.markdown("---")
        st.subheader("🚀 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Run Setup
            - Select a dataset
            - Choose a grammar
            - Configure parameters
            - Execute your setup
            """)
            if st.button("🏃 Go to Run Setup", use_container_width=True):
                st.session_state.current_page = "🏃 Run Setup"
                st.rerun()
        
        with col2:
            st.markdown("""
            ### 2️⃣ View Results
            - Analyze fitness evolution
            - Track invalid individuals
            - Monitor nodes length
            - Export data
            """)
            if st.button("📊 Go to Analysis", use_container_width=True):
                st.session_state.current_page = "📊 Analysis"
                st.rerun()
        
        with col3:
            st.markdown("""
            ### 3️⃣ Compare Setups
            - Select multiple setups
            - Compare performance
            - Visualize differences
            - Statistical analysis
            """)
            if st.button("⚖️ Go to Comparison", use_container_width=True):
                st.session_state.current_page = "⚖️ Setup Comparison"
                st.rerun()
        
        # Features Section
        st.markdown("---")
        st.subheader("🌟 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🖥️ Interactive Web Interface
            - Built with Streamlit for easy use
            - Intuitive navigation and controls
            - Real-time progress tracking
            
            #### 🧬 Grammatical Evolution
            - Full GE implementation using GRAPE library
            - Multiple grammar support
            - Configurable parameters
            
            #### 📊 Advanced Visualization
            - Interactive charts with Plotly
            - Min/Max/Average tracking
            - Standard deviation error bars
            
            #### 📁 Dataset Management
            - Support for multiple dataset formats
            - Automatic preprocessing
            - Dataset validation
            """)
        
        with col2:
            st.markdown("""
            #### 🔄 Multiple Runs
            - Automated execution of multiple independent runs
            - Statistical aggregation across runs
            - Comprehensive performance metrics
            
            #### 🚫 Invalid Individuals Tracking
            - Monitor invalid individuals across generations
            - Track evolution of solution validity
            - Statistical analysis of invalidity
            
            #### 🌳 Nodes Length Tracking
            - Track evolution of terminal symbols (nodes)
            - Monitor solution complexity
            - Analyze structural changes
            
            #### 📝 Professional Grammar Editor
            - Full CRUD operations for BNF grammar management
            - Syntax validation and safety features
            - Template loading and backup options
            """)
        
        # Technologies Section
        st.markdown("---")
        st.subheader("🛠️ Technologies Used")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.markdown("""
            **Frontend**
            - Streamlit (Python web framework)
            - Interactive UI components
            - Real-time updates
            """)
        
        with tech_col2:
            st.markdown("""
            **Backend**
            - Python with DEAP
            - Evolutionary algorithms
            - Data processing
            """)
        
        with tech_col3:
            st.markdown("""
            **GE Library**
            - DEAP and GRAPE framework
            - Grammatical Evolution
            - BNF grammar parsing
            """)
        
        with tech_col4:
            st.markdown("""
            **Visualization**
            - Plotly (Interactive charts)
            - Pandas, NumPy
            - scikit-learn
            """)
        
        # Getting Started Section
        st.markdown("---")
        st.subheader("📚 Getting Started")
        
        st.markdown("""
        ### Prerequisites
        - Python 3.8 or higher
        - pip (Python package manager)
        
        ### Installation
        1. **Clone the repository**:
           ```bash
           git clone <repository-url>
           cd GE-Lab
           ```
        
        2. **Create virtual environment**:
           ```bash
           python -m venv GE-Lab_env
           source GE-Lab_env/bin/activate  # On Windows: GE-Lab_env\\Scripts\\activate
           ```
        
        3. **Install dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        
        4. **Run the application**:
           ```bash
           streamlit run app.py
           ```
        
        5. **Open your browser** to `http://localhost:8501`
        """)
        
        # Architecture Section
        st.markdown("---")
        st.subheader("🏗️ Architecture")
        
        st.markdown("""
        The GE-Lab application follows the **Model-View-Controller (MVC)** architectural pattern:
        
        - **Models**: Data structures and business entities (`uge/models/`)
        - **Views**: User interface and presentation logic (`uge/views/`)
        - **Controllers**: Business logic orchestration (`uge/controllers/`)
        - **Services**: Business logic and external integrations (`uge/services/`)
        """)
        
        # Navigation Section
        st.markdown("---")
        st.subheader("🧭 Navigation")
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            st.markdown("""
            **Setup Management**
            - 🏃 Run Setup
            - 🧪 Setup Manager
            - 📁 Dataset Management
            """)
        
        with nav_col2:
            st.markdown("""
            **Analysis & Visualization**
            - 📊 Analysis
            - ⚖️ Setup Comparison
            - 📈 Interactive Charts
            """)
        
        with nav_col3:
            st.markdown("""
            **Grammar & Configuration**
            - 📝 Grammar Editor
            - ⚙️ Parameter Configuration
            - 🛠️ System Settings
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>Built with ❤️ using Streamlit and Python</p>
            <p><strong>Happy Experimenting with GE-Lab! 🧬✨</strong></p>
        </div>
        """, unsafe_allow_html=True)
