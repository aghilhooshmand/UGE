# 🧬 UGE (Unified Grammatical Evolution) Application

## 📖 Complete Project Documentation

Welcome to the UGE application - a comprehensive Streamlit-based web application for running and analyzing Grammatical Evolution experiments. This document provides complete documentation for understanding, using, and developing with the UGE system.

## 🎯 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Class Documentation](#class-documentation)
5. [API Reference](#api-reference)
6. [Development Guide](#development-guide)
7. [Troubleshooting](#troubleshooting)

## 🌟 Project Overview

### What is UGE?

UGE is a sophisticated web application that provides a user-friendly interface for conducting Grammatical Evolution (GE) experiments. It combines the power of evolutionary algorithms with modern web technologies to make GE research accessible and efficient.

### Key Features

- 🖥️ **Interactive Web Interface**: Built with Streamlit for easy use
- 🧬 **Grammatical Evolution**: Full GE implementation using GRAPE library
- 📊 **Advanced Visualization**: Interactive charts with Plotly
- 📁 **Dataset Management**: Support for multiple dataset formats
- 🔄 **Multiple Runs**: Automated execution of multiple independent runs
- 📈 **Comprehensive Analysis**: Detailed performance metrics and comparisons
- 🚫 **Invalid Individuals Tracking**: Monitor and analyze invalid individuals across generations
- 🌳 **Nodes Length Tracking**: Track evolution of terminal symbols (nodes) across generations
- 💾 **Persistent Storage**: Automatic saving and loading of experiments

### Technologies Used

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with DEAP (Distributed Evolutionary Algorithms)
- **GE Library**: GRAPE (Grammatical Evolution framework)
- **Visualization**: Plotly (Interactive charts)
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Architecture**: Model-View-Controller (MVC) pattern

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd UGE
```

2. **Create virtual environment**:

```bash
python -m venv UGE_env
source UGE_env/bin/activate  # On Windows: UGE_env\Scripts\activate
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

### First Experiment

1. **Navigate to "Run Experiment"** page
2. **Select a dataset** (e.g., `clinical_breast_cancer_RFC.csv`)
3. **Choose a grammar** (e.g., `UGE_Classification.bnf`)
4. **Set experiment name** (auto-generated)
5. **Configure parameters** (or use defaults)
6. **Click "🚀 Run Experiment"**
7. **View results** in the "Analysis" page

## 🏗️ Architecture Deep Dive

### MVC Pattern Implementation

The UGE application follows the Model-View-Controller (MVC) architectural pattern:

#### Models (`uge/models/`)

**Purpose**: Define data structures and business entities

```python
# Core Models
- Dataset: Represents datasets with loading and preprocessing capabilities
- DatasetInfo: Metadata about datasets (columns, rows, file info)
- Experiment: Complete experiment with multiple runs
- ExperimentConfig: Configuration parameters for experiments
- ExperimentResult: Results from individual experiment runs
```

#### Views (`uge/views/`)

**Purpose**: Handle user interface and presentation logic

```python
# View Components
- BaseView: Common functionality for all views
- ExperimentView: Experiment configuration and execution interface
- DatasetView: Dataset management and browsing
- AnalysisView: Results visualization and analysis
- Forms: Reusable form components
- Charts: Interactive chart generation
```

#### Controllers (`uge/controllers/`)

**Purpose**: Orchestrate business logic and coordinate between views and services

```python
# Controllers
- ExperimentController: Manages experiment lifecycle
- DatasetController: Handles dataset operations
- AnalysisController: Coordinates analysis operations
```

#### Services (`uge/services/`)

**Purpose**: Implement business logic and external integrations

```python
# Services
- GEService: Grammatical Evolution algorithm execution
- DatasetService: Dataset loading, preprocessing, and validation
- StorageService: File system operations and persistence
```

### Data Flow Architecture

```
User Input → Views → Controllers → Services → Models → External Libraries
     ↑                                                      ↓
User Interface ← Views ← Controllers ← Services ← Results Storage
```

## 📚 Class Documentation

### Core Models

#### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    """
    Configuration parameters for Grammatical Evolution experiments.
  
    This class encapsulates all parameters needed to run a GE experiment,
    including GA parameters, GE parameters, dataset settings, and options.
  
    Key Parameters:
    - experiment_name: Human-readable experiment identifier
    - dataset: Dataset file name (e.g., 'clinical_breast_cancer_RFC.csv')
    - grammar: BNF grammar file (e.g., 'UGE_Classification.bnf')
    - fitness_metric: Optimization target ('mae' or 'accuracy')
    - n_runs: Number of independent runs (typically 3-10)
    - generations: Evolution duration (typically 50-200)
    - population: Population size (typically 100-500)
    """
```

#### ExperimentResult

```python
@dataclass
class ExperimentResult:
    """
    Results from a single experiment run.
  
    Contains performance metrics tracked across generations:
    - max: Best fitness values per generation
    - avg: Average fitness values per generation
    - min: Worst fitness values per generation
    - fitness_test: Test set performance per generation
    - best_phenotype: The best solution found
    """
```

#### Dataset

```python
class Dataset:
    """
    Main dataset class with comprehensive data handling capabilities.
  
    Features:
    - Lazy loading: Data loaded only when needed
    - Multiple formats: CSV, DATA file support
    - Preprocessing: Automated data cleaning and preparation
    - Validation: Dataset compatibility checking
    """
```

### View Components

#### BaseView

```python
class BaseView(ABC):
    """
    Abstract base class for all views.
  
    Provides common functionality:
    - Session state management
    - Error/success message handling
    - Header rendering
    - Consistent UI patterns
    """
```

#### Forms

```python
class Forms:
    """
    Static class for creating form components.
  
    Methods:
    - create_experiment_form(): Main experiment configuration form
    - create_dataset_form(): Dataset selection and preview
    - create_analysis_form(): Analysis options and filters
    """
```

#### Charts

```python
class Charts:
    """
    Static class for creating interactive visualizations.
  
    Chart Types:
    - plot_fitness_evolution(): Individual run performance
    - plot_experiment_wide(): Multi-run aggregated analysis
    - plot_comparison_chart(): Cross-experiment comparisons
    """
```

### Service Classes

#### GEService

```python
class GEService:
    """
    Core service for Grammatical Evolution execution.
  
    Responsibilities:
    - Algorithm execution using GRAPE/DEAP
    - Population management and evolution
    - Fitness evaluation and selection
    - Progress tracking and UI updates
    """
```

#### DatasetService

```python
class DatasetService:
    """
    Service for dataset management operations.
  
    Capabilities:
    - Dataset discovery and listing
    - Metadata extraction and validation
    - Data loading and preprocessing
    - Compatibility checking
    """
```

## 🔌 API Reference

### Experiment Execution

#### Running an Experiment

```python
# Through UI
experiment_view = ExperimentView(experiment_controller)
experiment_view.render()

# Programmatically
config = ExperimentConfig(
    experiment_name="My Experiment",
    dataset="clinical_breast_cancer_RFC.csv",
    grammar="UGE_Classification.bnf",
    n_runs=3,
    generations=50
)
experiment = experiment_controller.run_experiment(config)
```

#### Accessing Results

```python
# Get experiment by ID
experiment = experiment_controller.get_experiment("exp_20250915_143849_001d4e52")

# Get best result across all runs
best_result = experiment.get_best_result()

# Get average fitness
avg_fitness = experiment.get_average_fitness()

# Access individual run results
for run_id, result in experiment.results.items():
    print(f"Run {run_id}: Best fitness = {result.best_training_fitness}")
```

### Dataset Operations

#### Loading and Preprocessing

```python
# List available datasets
datasets = dataset_service.list_datasets()

# Get dataset information
info = dataset_service.get_dataset_info("clinical_breast_cancer_RFC.csv")

# Load and preprocess
X_train, Y_train, X_test, Y_test = dataset_service.preprocess_dataset(
    "clinical_breast_cancer_RFC.csv",
    label_column="class",
    test_size=0.3
)
```

### Visualization

#### Creating Charts

```python
# Individual run chart
Charts.plot_individual_run_with_bars(
    result=experiment_result,
    title="Fitness Evolution",
    fitness_metric="mae"
)

# Experiment-wide comparison
Charts.plot_experiment_wide_with_bars(
    results=experiment.results,
    title="Multi-Run Analysis"
)
```

## 🛠️ Development Guide

### Project Structure

```
UGE/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Basic project information
├── ARCHITECTURE.md          # Detailed architecture documentation
├── README_COMPLETE.md       # This comprehensive guide
├── uge/                     # Main application package
│   ├── __init__.py
│   ├── models/              # Data models
│   │   ├── dataset.py
│   │   ├── experiment.py
│   │   └── grammar.py
│   ├── views/               # User interface
│   │   ├── components/
│   │   │   ├── base_view.py
│   │   │   ├── forms.py
│   │   │   └── charts.py
│   │   ├── dataset_view.py
│   │   ├── experiment_view.py
│   │   └── analysis_view.py
│   ├── controllers/         # Business logic orchestration
│   │   ├── base_controller.py
│   │   ├── dataset_controller.py
│   │   └── experiment_controller.py
│   ├── services/            # Business logic services
│   │   ├── dataset_service.py
│   │   ├── ge_service.py
│   │   └── storage_service.py
│   └── utils/               # Utilities and constants
│       ├── constants.py
│       ├── helpers.py
│       ├── logger.py
│       └── tooltip_manager.py
├── datasets/                # Dataset files
│   ├── clinical_breast_cancer_RFC.csv
│   └── processed.cleveland.data
├── grammars/                # BNF grammar files
│   ├── UGE_Classification.bnf
│   ├── heartDisease.bnf
│   └── Your_Grammar.bnf
└── results/                 # Experiment results
    └── experiments/
        └── exp_*/
```

### Adding New Features

#### 1. Adding a New View

```python
from uge.views.components.base_view import BaseView

class MyNewView(BaseView):
    def __init__(self):
        super().__init__(
            title="My New View",
            description="Description of what this view does"
        )
  
    def render(self):
        self.render_header()
        # Your view implementation here
```

#### 2. Adding a New Service

```python
class MyNewService:
    def __init__(self, dependency_service=None):
        self.dependency_service = dependency_service
  
    def my_service_method(self, parameter):
        # Your service logic here
        return result
```

#### 3. Adding a New Chart Type

```python
class Charts:
    @staticmethod
    def plot_my_new_chart(data, title="My Chart"):
        fig = go.Figure()
        # Create your Plotly figure
        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)
```

### Testing

#### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_experiment.py

# Run with coverage
python -m pytest --cov=uge tests/
```

#### Writing Tests

```python
import pytest
from uge.models.experiment import ExperimentConfig

def test_experiment_config_creation():
    config = ExperimentConfig(
        experiment_name="Test Experiment",
        dataset="test.csv",
        grammar="test.bnf"
    )
    assert config.experiment_name == "Test Experiment"
    assert config.dataset == "test.csv"
```

### Code Style

The project follows Python best practices:

- **PEP 8**: Python style guide compliance
- **Type Hints**: All functions include type annotations
- **Docstrings**: Comprehensive documentation for all classes and methods
- **Error Handling**: Proper exception handling with meaningful messages
- **Logging**: Structured logging for debugging and monitoring

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'streamlit'
# Solution: Activate virtual environment
source UGE_env/bin/activate
pip install -r requirements.txt
```

#### 2. Dataset Loading Issues

```bash
# Error: Dataset file not found
# Solution: Check dataset file exists in datasets/ folder
ls datasets/
```

#### 3. Experiment Execution Errors

```bash
# Error: Index out of bounds
# Solution: Check dataset preprocessing compatibility
# Ensure label column exists in dataset
```

#### 4. Memory Issues

```bash
# Error: Out of memory during experiment
# Solution: Reduce population size or generations
# Or increase system memory
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Reduce Population Size**: Lower `population` parameter
2. **Fewer Generations**: Reduce `generations` parameter
3. **Smaller Datasets**: Use smaller test datasets
4. **Fewer Runs**: Reduce `n_runs` parameter

## 📞 Support

For questions, issues, or contributions:

1. **Check Documentation**: Review this guide and ARCHITECTURE.md
2. **Search Issues**: Look for similar problems in issue tracker
3. **Create Issue**: Provide detailed error messages and steps to reproduce
4. **Contribute**: Submit pull requests with improvements

## 🎓 Learning Resources

### Grammatical Evolution

- [Grammatical Evolution: Tutorial](https://www.grammatical-evolution.org/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [GRAPE Library](https://github.com/bdsul/grape)

### Streamlit

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Tutorials](https://docs.streamlit.io/library/get-started)

### Python Best Practices

- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---

**Happy Experimenting with UGE! 🧬✨**

This comprehensive documentation should provide everything needed to understand, use, and develop with the UGE application. The modular architecture makes it easy to extend and customize for specific research needs.
