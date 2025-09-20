# UGE Application Architecture Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture Pattern](#architecture-pattern)
3. [Class Hierarchy](#class-hierarchy)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Class Relationships](#class-relationships)
7. [Mermaid Diagrams](#mermaid-diagrams)

## 🎯 Overview

The UGE (Unified Grammatical Evolution) application is a Streamlit-based web application for running and analyzing Grammatical Evolution setups. It follows the Model-View-Controller (MVC) architectural pattern and provides a comprehensive interface for:

- **Dataset Management**: Loading and preprocessing datasets
- **Setup Configuration**: Setting up GE parameters with dynamic parameter support
- **Setup Execution**: Running multiple independent runs with unified algorithm
- **Results Analysis**: Visualizing and comparing results with comprehensive statistics
- **Dynamic Parameter System**: Per-parameter configuration for adaptive evolution

## 🏗️ Architecture Pattern

The application follows the **Model-View-Controller (MVC)** pattern:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      VIEWS      │    │   CONTROLLERS   │    │    SERVICES     │
│   (User UI)     │◄──►│  (Orchestration)│◄──►│  (Business      │
│                 │    │                 │    │    Logic)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MODELS      │    │     UTILS       │    │   EXTERNAL      │
│ (Data Structures)│    │  (Helpers &     │    │   LIBRARIES     │
│                 │    │   Constants)    │    │ (GRAPE, DEAP)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Principles:
- **Separation of Concerns**: Each layer has distinct responsibilities
- **Loose Coupling**: Components interact through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together
- **Testability**: Business logic can be tested independently

## 🏛️ Class Hierarchy

### Core Components:

```
uge/
├── models/           # Data Models (M in MVC)
│   ├── dataset.py    # Dataset and DatasetInfo classes
│   ├── setup.py # Setup, SetupConfig, SetupResult
│   └── grammar.py    # Grammar model (if exists)
├── views/            # User Interface (V in MVC)
│   ├── components/   # Reusable UI components
│   │   ├── base_view.py    # Base class for all views
│   │   ├── forms.py        # Form components
│   │   └── charts.py       # Chart components
│   ├── dataset_view.py     # Dataset management view
│   ├── setup_view.py       # Setup configuration view
│   ├── analysis_view.py    # Results analysis view
│   ├── grammar_view.py     # BNF grammar editor with CRUD operations
│   ├── setup_manager_view.py # Setup management and monitoring view
│   └── comparison_view.py  # Setup comparison and analysis view
├── controllers/      # Business Logic Orchestration (C in MVC)
│   ├── base_controller.py     # Base controller class
│   ├── dataset_controller.py  # Dataset operations
│   └── setup_controller.py # Setup execution
├── services/         # Business Logic Services
│   ├── dataset_service.py  # Dataset loading and preprocessing
│   ├── ge_service.py       # Grammatical Evolution execution
│   └── storage_service.py  # File system operations
└── utils/            # Utilities and Constants
    ├── constants.py      # Application constants
    ├── helpers.py        # Helper functions
    ├── logger.py         # Logging utilities
    └── tooltip_manager.py # UI tooltip management
```

## 🔧 Component Details

### 1. Models Layer

#### Dataset Model (`uge/models/dataset.py`)
```python
@dataclass
class DatasetInfo:
    """Metadata about a dataset"""
    name: str
    path: Path
    file_type: str
    size_bytes: int
    columns: list
    rows: int
    features: int
    has_labels: bool = False
    label_column: Optional[str] = None

class Dataset:
    """Main dataset class with loading and preprocessing capabilities"""
    def __init__(self, info: DatasetInfo)
    def load(self) -> pd.DataFrame
    def preprocess_cleveland_data(self, random_seed: int) -> Tuple[np.ndarray, ...]
    def preprocess_csv_data(self, label_column: str, test_size: float, random_seed: int) -> Tuple[np.ndarray, ...]
```

**Responsibilities:**
- Store dataset metadata and statistics
- Load data from files (CSV, DATA formats)
- Preprocess data for machine learning
- Handle different dataset types (Cleveland heart disease, generic CSV)

#### Setup Models (`uge/models/setup.py`)
```python
@dataclass
class SetupConfig:
    """Configuration for GE setups"""
    setup_name: str
    dataset: str
    grammar: str
    fitness_metric: str
    fitness_direction: int
    n_runs: int
    generations: int
    population: int
    # ... many more parameters

@dataclass
class SetupResult:
    """Results from a single setup run"""
    config: SetupConfig
    max: List[float]  # Best fitness per generation
    avg: List[float]  # Average fitness per generation
    min: List[float]  # Worst fitness per generation
    fitness_test: List[Optional[float]]  # Test performance
    best_phenotype: Optional[str]  # Best solution found

@dataclass
class Setup:
    """Complete setup with multiple runs"""
    id: str
    config: SetupConfig
    results: Dict[str, SetupResult]
    status: str  # 'created', 'running', 'completed', 'failed'
    created_at: str
    completed_at: Optional[str]
```

**Responsibilities:**
- Define setup configuration structure
- Store individual run results
- Aggregate multiple run results
- Track setup lifecycle

### 2. Views Layer

#### Base View (`uge/views/components/base_view.py`)
```python
class BaseView(ABC):
    """Base class for all views"""
    def __init__(self, title: str, description: str = "")
    def render_header(self)
    def show_error(self, message: str)
    def show_success(self, message: str)
    def get_session_state(self, key: str, default: Any = None) -> Any
    def set_session_state(self, key: str, value: Any)
    @abstractmethod
    def render(self)
```

**Responsibilities:**
- Provide common UI functionality
- Handle session state management
- Standardize error/success messages
- Enforce render method implementation

#### Forms Component (`uge/views/components/forms.py`)
```python
class Forms:
    """Form components and input utilities"""
    @staticmethod
    def create_setup_form(help_texts: Dict[str, str] = None, 
                              datasets: List[str] = None, 
                              grammars: List[str] = None) -> Tuple[bool, Dict[str, Any]]
```

**Responsibilities:**
- Create setup configuration forms
- Validate user input
- Provide help texts and tooltips
- Return form data in structured format

#### Charts Component (`uge/views/components/charts.py`)
```python
class Charts:
    """Chart and visualization utilities"""
    @staticmethod
    def plot_fitness_evolution(result: Dict[str, Any], title: str = "Fitness Evolution")
    @staticmethod
    def plot_individual_run_with_bars(result, title: str = "Individual Run Analysis")
    @staticmethod
    def plot_setup_wide_with_bars(results: Dict[str, Any], title: str = "Setup-wide Analysis")
```

**Responsibilities:**
- Create interactive Plotly charts
- Visualize fitness evolution
- Compare multiple setups
- Provide detailed run analysis

#### Grammar View (`uge/views/grammar_view.py`)
```python
class GrammarView:
    """
    Professional BNF grammar editor with full CRUD operations.
    
    Features:
    - View existing grammars (read-only)
    - Add new grammar files with validation
    - Edit existing grammar content
    - Delete grammars with safety confirmation
    - Template loading and form management
    """
    def render_grammar_editor(self, grammars: List[str]) -> None
    def _render_view_grammar(self, grammars: List[str]) -> None
    def _render_add_grammar(self) -> None
    def _render_edit_grammar(self, grammars: List[str]) -> None
    def _render_delete_grammar(self, grammars: List[str]) -> None
    def _validate_grammar_input(self, name: str, content: str) -> bool
    def _save_new_grammar(self, name: str, content: str) -> bool
    def _delete_grammar(self, name: str) -> bool
```

**Responsibilities:**
- Provide professional-grade grammar editing interface
- Implement full CRUD operations for BNF files
- Validate grammar input and file operations
- Manage session state for grammar editing
- Provide safety features and confirmation dialogs

### 3. Controllers Layer

#### Setup Controller (`uge/controllers/setup_controller.py`)
```python
class SetupController:
    """Controller for setup operations"""
    def __init__(self, dataset_service: DatasetService, ge_service: GEService, storage_service: StorageService)
    def run_setup(self, config: SetupConfig, **ui_elements) -> Optional[Setup]
    def get_setup(self, setup_id: str) -> Optional[Setup]
    def list_setups(self) -> List[Setup]
```

**Responsibilities:**
- Orchestrate setup execution
- Coordinate between services
- Handle UI updates during execution
- Manage setup lifecycle

### 4. Services Layer

#### Dataset Service (`uge/services/dataset_service.py`)
```python
class DatasetService:
    """Service for dataset management operations"""
    def list_datasets(self) -> List[str]
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]
    def load_dataset(self, dataset_name: str) -> Optional[Dataset]
    def preprocess_dataset(self, dataset_name: str, label_column: Optional[str] = None, 
                          test_size: float = 0.3, random_seed: int = 42) -> Tuple[np.ndarray, ...]
    def check_dataset_compatibility(self, dataset_name: str, label_column: Optional[str] = None) -> List[str]
```

**Responsibilities:**
- Manage dataset files
- Load and validate datasets
- Preprocess data for setups
- Check dataset compatibility

#### GE Service (`uge/services/ge_service.py`)
```python
class GEService:
    """Service for Grammatical Evolution operations"""
    def run_setup(self, config: SetupConfig, **ui_elements) -> Optional[SetupResult]
    def _run_single_setup(self, config: SetupConfig, run_number: int, **ui_elements) -> Optional[SetupResult]
```

**Responsibilities:**
- Execute Grammatical Evolution algorithms
- Integrate with GRAPE library
- Handle DEAP evolutionary algorithms
- Manage setup execution flow

## 🔄 Data Flow

### Setup Execution Flow:
```
1. User fills form (SetupView)
   ↓
2. Form data → SetupConfig (SetupView)
   ↓
3. SetupConfig → SetupController
   ↓
4. SetupController → DatasetService (load data)
   ↓
5. SetupController → GEService (run algorithm)
   ↓
6. GEService → DEAP/GRAPE (evolutionary computation)
   ↓
7. Results → SetupResult → Setup
   ↓
8. Setup → AnalysisView (display charts)
```

### Data Preprocessing Flow:
```
1. Dataset file → DatasetService.load_dataset()
   ↓
2. Dataset → Dataset.preprocess_*_data()
   ↓
3. Raw data → Cleaned data → Train/Test split
   ↓
4. Preprocessed data → GEService
   ↓
5. GEService → DEAP algorithm
```

## 🔗 Class Relationships

### Inheritance Relationships:
- `BaseView` ← `SetupView`, `DatasetView`, `AnalysisView`
- `BaseController` ← `SetupController`, `DatasetController`
- `DatasetInfo` ← `Dataset` (composition)
- `SetupConfig` ← `SetupResult` ← `Setup` (composition)

### Dependency Relationships:
- `SetupView` → `SetupController` → `GEService`, `DatasetService`
- `AnalysisView` → `Charts` → `Plotly`
- `Forms` → `DEFAULT_CONFIG`, `UI_CONSTANTS`
- `GEService` → `GRAPE`, `DEAP` (external libraries)

### Composition Relationships:
- `Setup` contains `SetupConfig` and multiple `SetupResult`
- `Dataset` contains `DatasetInfo`
- `SetupResult` contains `SetupConfig`

## 📊 Mermaid Diagrams

### 1. Overall Architecture
```mermaid
graph TB
    subgraph "Presentation Layer"
        UI[Streamlit UI]
        Forms[Forms Component]
        Charts[Charts Component]
    end
    
    subgraph "Application Layer"
        Views[Views]
        Controllers[Controllers]
    end
    
    subgraph "Business Layer"
        Services[Services]
        Models[Models]
    end
    
    subgraph "Data Layer"
        Files[File System]
        Results[Results Storage]
    end
    
    subgraph "External Libraries"
        GRAPE[GRAPE Library]
        DEAP[DEAP Library]
        Plotly[Plotly Charts]
    end
    
    UI --> Views
    Views --> Controllers
    Controllers --> Services
    Services --> Models
    Services --> Files
    Services --> GRAPE
    Services --> DEAP
    Charts --> Plotly
    Models --> Results
```

### 2. Class Hierarchy
```mermaid
classDiagram
    class BaseView {
        <<abstract>>
        +title: str
        +description: str
        +render_header()
        +show_error(message: str)
        +show_success(message: str)
        +get_session_state(key: str)
        +set_session_state(key: str, value: Any)
        +render()* void
    }
    
    class SetupView {
        -setup_controller: SetupController
        -on_setup_submit: Callable
        -on_setup_cancel: Callable
        +render()
        +_handle_setup_submission()
        +_show_setup_results()
    }
    
    class DatasetView {
        -dataset_controller: DatasetController
        +render()
        +_render_dataset_list()
        +_render_dataset_info()
    }
    
    class AnalysisView {
        -setup_controller: SetupController
        +render()
        +_render_setup_selection()
        +_render_analysis_charts()
    }
    
    BaseView <|-- SetupView
    BaseView <|-- DatasetView
    BaseView <|-- AnalysisView
    
    class SetupController {
        -dataset_service: DatasetService
        -ge_service: GEService
        -storage_service: StorageService
        +run_setup(config: SetupConfig) Setup
        +get_setup(setup_id: str) Setup
        +list_setups() List[Setup]
    }
    
    class DatasetService {
        -datasets_dir: Path
        +list_datasets() List[str]
        +get_dataset_info(dataset_name: str) DatasetInfo
        +load_dataset(dataset_name: str) Dataset
        +preprocess_dataset(dataset_name: str) Tuple
    }
    
    class GEService {
        +run_setup(config: SetupConfig) SetupResult
        -_run_single_setup(config: SetupConfig) SetupResult
    }
    
    SetupView --> SetupController
    SetupController --> DatasetService
    SetupController --> GEService
    
    class SetupConfig {
        +setup_name: str
        +dataset: str
        +grammar: str
        +fitness_metric: str
        +n_runs: int
        +generations: int
        +population: int
        +to_dict() Dict
    }
    
    class SetupResult {
        +config: SetupConfig
        +max: List[float]
        +avg: List[float]
        +min: List[float]
        +fitness_test: List[Optional[float]]
        +best_phenotype: Optional[str]
        +to_dict() Dict
    }
    
    class Setup {
        +id: str
        +config: SetupConfig
        +results: Dict[str, SetupResult]
        +status: str
        +created_at: str
        +completed_at: Optional[str]
        +add_result(run_id: str, result: SetupResult)
        +get_best_result() SetupResult
        +get_average_fitness() float
    }
    
    SetupConfig <-- SetupResult
    SetupResult <-- Setup
    Setup --> SetupConfig
```

### 3. Data Flow Diagram
```mermaid
sequenceDiagram
    participant User
    participant SetupView
    participant SetupController
    participant DatasetService
    participant GEService
    participant DEAP
    participant AnalysisView
    
    User->>SetupView: Fill setup form
    SetupView->>SetupView: Validate form data
    SetupView->>SetupView: Create SetupConfig
    SetupView->>SetupController: run_setup(config)
    
    SetupController->>DatasetService: load_dataset(dataset_name)
    DatasetService->>DatasetService: preprocess_data()
    DatasetService-->>SetupController: X_train, Y_train, X_test, Y_test
    
    loop For each run (n_runs)
        SetupController->>GEService: run_single_setup(config, run_number)
        GEService->>DEAP: Initialize population
        loop For each generation
            GEService->>DEAP: Evaluate fitness
            GEService->>DEAP: Selection, crossover, mutation
            GEService->>SetupView: Update progress
        end
        GEService-->>SetupController: SetupResult
    end
    
    SetupController->>SetupController: Create Setup object
    SetupController-->>SetupView: Setup
    SetupView->>User: Show success message and results
    
    User->>AnalysisView: View setup results
    AnalysisView->>AnalysisView: Load setup data
    AnalysisView->>AnalysisView: Create charts
    AnalysisView->>User: Display interactive charts
```

### 4. Service Dependencies
```mermaid
graph TD
    subgraph "Controllers"
        EC[SetupController]
        DC[DatasetController]
    end
    
    subgraph "Services"
        DS[DatasetService]
        GS[GEService]
        SS[StorageService]
    end
    
    subgraph "Models"
        D[Dataset]
        DI[DatasetInfo]
        ECONF[SetupConfig]
        ER[SetupResult]
        E[Setup]
    end
    
    subgraph "External"
        GRAPE[GRAPE Library]
        DEAP[DEAP Library]
        PANDAS[Pandas]
        NUMPY[NumPy]
    end
    
    EC --> DS
    EC --> GS
    EC --> SS
    DC --> DS
    
    DS --> D
    DS --> DI
    DS --> PANDAS
    DS --> NUMPY
    
    GS --> ECONF
    GS --> ER
    GS --> GRAPE
    GS --> DEAP
    
    SS --> E
    SS --> PANDAS
```

## 🚫 Invalid Individuals Tracking Integration

### Purpose
The tracking features monitor key evolutionary metrics across generations during GE evolution:

**Invalid Individuals Tracking:**
- Population quality over time
- Grammar effectiveness
- Evolution progress

**Nodes Length Tracking:**
- Complexity evolution (number of terminal symbols)
- Solution sophistication over generations
- Grammar utilization patterns

### Changes Made by Aghil

#### 1. GRAPE Library Modifications (`grape/algorithms.py`)
```python
# ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
# Added invalid individuals statistics calculation for UGE integration
# For each generation, we only have one invalid count, so min=max=avg=invalid
invalid_count_min = invalid
invalid_count_max = invalid
invalid_count_avg = float(invalid)
# ===== END MODIFICATION BY AGHIL =====

# ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
# Added nodes length statistics calculation for UGE integration
nodes_length_min = min(nodes) if nodes else 0
nodes_length_max = max(nodes) if nodes else 0
nodes_length_avg = avg_nodes
# ===== END MODIFICATION BY AGHIL =====
```

**Changes include:**
- Header comment documenting all modifications by Aghil
- Invalid individuals statistics calculation (min, max, avg)
- Nodes length statistics calculation (min, max, avg)
- Updated logbook headers to include `invalid_count_min`, `invalid_count_avg`, `invalid_count_max`, `nodes_length_min`, `nodes_length_avg`, `nodes_length_max`
- Updated logbook records to include both invalid individuals and nodes length data

#### 2. UGE Integration Points

**SetupResult Model (`uge/models/setup.py`):**
- Added `invalid_count_min: List[int]`, `invalid_count_avg: List[float]`, `invalid_count_max: List[int]`
- Added `nodes_length_min: List[int]`, `nodes_length_avg: List[float]`, `nodes_length_max: List[int]`
- Updated serialization/deserialization methods

**GE Service (`uge/services/ge_service.py`):**
- Extract invalid individuals and nodes length data from GRAPE logbook
- Convert and store in SetupResult objects

**Charts Component (`uge/views/components/charts.py`):**
- `plot_invalid_count_evolution()`: Individual run invalid count charts
- `plot_setup_wide_invalid_count()`: Setup-wide invalid count charts
- `plot_nodes_length_evolution()`: Individual run nodes length charts
- `plot_setup_wide_nodes_length()`: Setup-wide nodes length charts

**Analysis View (`uge/views/analysis_view.py`):**
- Added "Number of Invalid Individuals" and "Nodes Length Evolution" analysis type selections
- Individual run charts with run selection for both metrics
- Setup-wide aggregated analysis for both metrics

**Constants (`uge/utils/constants.py`):**
- Added invalid individuals and nodes length fields to `default_report_items`

### Data Flow for Tracking Features
```
GRAPE Evolution → Invalid Count & Nodes Length Calculation → Logbook Storage → 
UGE Service Extraction → SetupResult Storage → 
Analysis View Display → Interactive Charts

Invalid Individuals: Population Quality & Grammar Effectiveness
Nodes Length: Solution Complexity & Terminal Symbol Evolution
```

This comprehensive documentation provides:
1. ✅ **Self-documenting code** with detailed docstrings
2. ✅ **Complete architecture overview** with MVC pattern explanation
3. ✅ **Detailed class relationships** and responsibilities
4. ✅ **Multiple Mermaid diagrams** showing different aspects of the architecture
5. ✅ **Data flow documentation** showing how information moves through the system

The documentation serves as both a learning resource and a reference for understanding the entire UGE application architecture.
