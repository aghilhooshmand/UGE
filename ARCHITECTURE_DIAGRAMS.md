# 🏗️ UGE Application Architecture Diagrams

This document contains comprehensive Mermaid diagrams showing the object-oriented architecture of the UGE application.

## 📊 1. Overall System Architecture

```mermaid
graph TB
    subgraph "🌐 Presentation Layer"
        UI[Streamlit Web Interface]
        Forms[Forms Component]
        Charts[Charts Component]
        Views[View Components]
    end
    
    subgraph "🎮 Application Layer"
        EC[Setup Controller]
        DC[Dataset Controller]
        AC[Analysis Controller]
    end
    
    subgraph "⚙️ Business Logic Layer"
        GES[GE Service]
        DS[Dataset Service]
        SS[Storage Service]
    end
    
    subgraph "📊 Data Layer"
        Models[Data Models]
        Files[File System]
        Results[Results Storage]
    end
    
    subgraph "🔧 External Libraries"
        GRAPE[GRAPE Library]
        DEAP[DEAP Library]
        Plotly[Plotly Charts]
        Streamlit[Streamlit Framework]
    end
    
    UI --> Views
    Views --> EC
    Views --> DC
    Views --> AC
    EC --> GES
    EC --> DS
    EC --> SS
    DC --> DS
    DS --> Models
    GES --> Models
    GES --> GRAPE
    GES --> DEAP
    Charts --> Plotly
    Views --> Streamlit
    Models --> Files
    SS --> Results
```

## 🏛️ 2. Class Hierarchy and Inheritance

```mermaid
classDiagram
    %% Abstract Base Classes
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
    
    class BaseController {
        <<abstract>>
        +initialize()
        +validate_input()
        +handle_error()
    }
    
    %% View Classes
    class SetupView {
        -experiment_controller: SetupController
        -on_experiment_submit: Callable
        -on_experiment_cancel: Callable
        +render()
        +_handle_experiment_submission()
        +_show_experiment_results()
    }
    
    class DatasetView {
        -dataset_controller: DatasetController
        +render()
        +_render_dataset_list()
        +_render_dataset_info()
        +_render_dataset_preview()
    }
    
    class AnalysisView {
        -experiment_controller: SetupController
        +render()
        +_render_experiment_selection()
        +_render_analysis_charts()
        +_render_individual_run_charts()
    }
    
    %% Controller Classes
    class SetupController {
        -dataset_service: DatasetService
        -ge_service: GEService
        -storage_service: StorageService
        +run_experiment(config: SetupConfig) Setup
        +get_experiment(experiment_id: str) Setup
        +list_experiments() List[Setup]
        +cancel_experiment(experiment_id: str)
    }
    
    class DatasetController {
        -dataset_service: DatasetService
        +list_datasets() List[str]
        +get_dataset_info(dataset_name: str) DatasetInfo
        +validate_dataset(dataset_name: str) List[str]
    }
    
    %% Service Classes
    class GEService {
        +run_experiment(config: SetupConfig) SetupResult
        -_run_single_experiment(config: SetupConfig) SetupResult
        -_setup_grape_algorithm()
        -_evaluate_fitness()
    }
    
    class DatasetService {
        -datasets_dir: Path
        +list_datasets() List[str]
        +get_dataset_info(dataset_name: str) DatasetInfo
        +load_dataset(dataset_name: str) Dataset
        +preprocess_dataset(dataset_name: str) Tuple
        +check_dataset_compatibility() List[str]
    }
    
    class StorageService {
        -results_dir: Path
        +save_experiment(experiment: Setup)
        +load_experiment(experiment_id: str) Setup
        +list_experiments() List[str]
    }
    
    %% Model Classes
    class DatasetInfo {
        +name: str
        +path: Path
        +file_type: str
        +size_bytes: int
        +columns: list
        +rows: int
        +features: int
        +has_labels: bool
        +label_column: Optional[str]
    }
    
    class Dataset {
        +info: DatasetInfo
        +data: Optional[DataFrame]
        +X_train: Optional[ndarray]
        +Y_train: Optional[ndarray]
        +X_test: Optional[ndarray]
        +Y_test: Optional[ndarray]
        +load() DataFrame
        +preprocess_cleveland_data() Tuple
        +preprocess_csv_data() Tuple
    }
    
    class SetupConfig {
        +experiment_name: str
        +dataset: str
        +grammar: str
        +fitness_metric: str
        +fitness_direction: int
        +n_runs: int
        +generations: int
        +population: int
        +p_crossover: float
        +p_mutation: float
        +elite_size: int
        +tournsize: int
        +halloffame_size: int
        +max_tree_depth: int
        +min_init_tree_depth: int
        +max_init_tree_depth: int
        +min_init_genome_length: int
        +max_init_genome_length: int
        +codon_size: int
        +codon_consumption: str
        +genome_representation: str
        +initialisation: str
        +random_seed: int
        +label_column: str
        +test_size: float
        +report_items: List[str]
        +created_at: str
        +to_dict() Dict
    }
    
    class SetupResult {
        +config: SetupConfig
        +report_items: List[str]
        +max: List[float]
        +avg: List[float]
        +min: List[float]
        +std: List[float]
        +fitness_test: List[Optional[float]]
        +best_phenotype: Optional[str]
        +best_training_fitness: Optional[float]
        +best_depth: Optional[int]
        +best_genome_length: Optional[int]
        +best_used_codons: Optional[float]
        +timestamp: str
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
        +to_dict() Dict
    }
    
    %% Inheritance Relationships
    BaseView <|-- SetupView
    BaseView <|-- DatasetView
    BaseView <|-- AnalysisView
    BaseController <|-- SetupController
    BaseController <|-- DatasetController
    
    %% Composition Relationships
    Dataset *-- DatasetInfo : contains
    Setup *-- SetupConfig : contains
    Setup *-- SetupResult : contains multiple
    SetupResult *-- SetupConfig : references
    
    %% Dependency Relationships
    SetupView --> SetupController
    DatasetView --> DatasetController
    AnalysisView --> SetupController
    SetupController --> GEService
    SetupController --> DatasetService
    SetupController --> StorageService
    DatasetController --> DatasetService
    GEService --> SetupConfig
    GEService --> SetupResult
    DatasetService --> Dataset
    DatasetService --> DatasetInfo
    StorageService --> Setup
```

## 🔄 3. Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant SetupView
    participant SetupController
    participant DatasetService
    participant GEService
    participant DEAP
    participant GRAPE
    participant StorageService
    participant AnalysisView
    
    User->>SetupView: 1. Fill experiment form
    SetupView->>SetupView: 2. Validate form data
    SetupView->>SetupView: 3. Create SetupConfig
    SetupView->>SetupController: 4. run_experiment(config)
    
    SetupController->>DatasetService: 5. load_dataset(dataset_name)
    DatasetService->>DatasetService: 6. preprocess_data()
    DatasetService-->>SetupController: 7. X_train, Y_train, X_test, Y_test
    
    SetupController->>GEService: 8. run_experiment(config, ui_elements)
    
    loop For each run (n_runs)
        GEService->>GEService: 9. Initialize experiment run
        GEService->>GRAPE: 10. Setup GE algorithm
        GRAPE->>DEAP: 11. Initialize population
        
        loop For each generation
            DEAP->>DEAP: 12. Evaluate fitness
            DEAP->>DEAP: 13. Selection, crossover, mutation
            DEAP-->>GRAPE: 14. Updated population
            GRAPE-->>GEService: 15. Generation results
            GEService->>SetupView: 16. Update progress UI
        end
        
        GEService->>GEService: 17. Collect run results
        GEService-->>SetupController: 18. SetupResult
    end
    
    SetupController->>SetupController: 19. Create Setup object
    SetupController->>StorageService: 20. save_experiment(experiment)
    SetupController-->>SetupView: 21. Setup object
    SetupView->>User: 22. Show success message and results
    
    User->>AnalysisView: 23. Navigate to Analysis page
    AnalysisView->>StorageService: 24. load_experiment(experiment_id)
    StorageService-->>AnalysisView: 25. Setup data
    AnalysisView->>AnalysisView: 26. Create charts and visualizations
    AnalysisView->>User: 27. Display interactive analysis
```

## 🏗️ 4. Component Dependencies

```mermaid
graph TD
    subgraph "🎨 UI Layer"
        UI[Streamlit App]
        Forms[Forms Component]
        Charts[Charts Component]
    end
    
    subgraph "📱 View Layer"
        EV[SetupView]
        DV[DatasetView]
        AV[AnalysisView]
        BV[BaseView]
    end
    
    subgraph "🎮 Controller Layer"
        EC[SetupController]
        DC[DatasetController]
        BC[BaseController]
    end
    
    subgraph "⚙️ Service Layer"
        GES[GEService]
        DS[DatasetService]
        SS[StorageService]
    end
    
    subgraph "📊 Model Layer"
        E[Setup]
        ECONF[SetupConfig]
        ER[SetupResult]
        D[Dataset]
        DI[DatasetInfo]
    end
    
    subgraph "🔧 External Dependencies"
        GRAPE[GRAPE Library]
        DEAP[DEAP Library]
        PLOTLY[Plotly]
        PANDAS[Pandas]
        NUMPY[NumPy]
        SKLEARN[Scikit-learn]
    end
    
    %% UI to Views
    UI --> EV
    UI --> DV
    UI --> AV
    
    %% Views inheritance
    BV <|-- EV
    BV <|-- DV
    BV <|-- AV
    
    %% Views to Controllers
    EV --> EC
    DV --> DC
    
    %% Controllers inheritance
    BC <|-- EC
    BC <|-- DC
    
    %% Controllers to Services
    EC --> GES
    EC --> DS
    EC --> SS
    DC --> DS
    
    %% Services to Models
    GES --> ECONF
    GES --> ER
    DS --> D
    DS --> DI
    SS --> E
    
    %% Model relationships
    E *-- ECONF
    E *-- ER
    ER *-- ECONF
    D *-- DI
    
    %% External dependencies
    GES --> GRAPE
    GES --> DEAP
    Charts --> PLOTLY
    DS --> PANDAS
    DS --> NUMPY
    DS --> SKLEARN
    Forms --> PLOTLY
```

## 📊 5. Setup Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: User creates experiment
    
    Created --> Validating: Form submission
    Validating --> Valid: Validation passed
    Validating --> Invalid: Validation failed
    Invalid --> Created: Fix errors
    
    Valid --> Running: Start experiment
    Running --> RunInProgress: Execute individual run
    
    RunInProgress --> RunCompleted: Run finished
    RunCompleted --> RunInProgress: More runs needed
    RunCompleted --> Completed: All runs finished
    
    Running --> Failed: Error occurred
    Failed --> [*]
    
    Completed --> Analyzing: View results
    Analyzing --> [*]
    
    note right of Created
        SetupConfig created
        Form validation
        Parameter checking
    end note
    
    note right of Running
        Multiple runs executed
        Progress tracking
        UI updates
    end note
    
    note right of Completed
        All results collected
        Setup saved
        Analysis available
    end note
```

## 🔧 6. Service Architecture

```mermaid
graph LR
    subgraph "🎮 Controllers"
        EC[Setup<br/>Controller]
        DC[Dataset<br/>Controller]
    end
    
    subgraph "⚙️ Core Services"
        GES[GE Service<br/>Algorithm Execution]
        DS[Dataset Service<br/>Data Management]
        SS[Storage Service<br/>Persistence]
    end
    
    subgraph "🔧 Utility Services"
        LS[Logger Service<br/>Logging]
        CS[Config Service<br/>Configuration]
        VS[Validation Service<br/>Input Validation]
    end
    
    subgraph "📊 External Integrations"
        GRAPE[GRAPE<br/>GE Library]
        DEAP[DEAP<br/>Evolutionary Algorithms]
        FS[File System<br/>Storage]
    end
    
    EC --> GES
    EC --> DS
    EC --> SS
    DC --> DS
    
    GES --> GRAPE
    GES --> DEAP
    DS --> FS
    SS --> FS
    
    GES --> LS
    DS --> VS
    SS --> CS
```

## 🎯 7. User Interaction Flow

```mermaid
journey
    title User Journey Through UGE Application
    section Initial Setup
      Open Application: 5: User
      Select Dataset: 4: User
      Configure Setup: 3: User
    section Setup Execution
      Submit Setup: 5: User
      Monitor Progress: 4: User
      Wait for Completion: 2: User
    section Results Analysis
      View Results: 5: User
      Analyze Charts: 5: User
      Compare Runs: 4: User
      Export Data: 3: User
```

## 📈 8. Performance Monitoring

```mermaid
graph TB
    subgraph "📊 Metrics Collection"
        EM[Setup Metrics]
        PM[Performance Metrics]
        SM[System Metrics]
    end
    
    subgraph "🔍 Analysis Layer"
        DA[Data Analysis]
        PA[Performance Analysis]
        RA[Resource Analysis]
    end
    
    subgraph "📈 Visualization"
        CD[Charts & Dashboards]
        AL[Alerts & Notifications]
        RP[Reports]
    end
    
    EM --> DA
    PM --> PA
    SM --> RA
    
    DA --> CD
    PA --> AL
    RA --> RP
```

---

These diagrams provide a comprehensive view of the UGE application's object-oriented architecture, showing:

1. **System Architecture**: Overall system structure and layers
2. **Class Hierarchy**: Inheritance and composition relationships
3. **Data Flow**: How data moves through the system
4. **Component Dependencies**: What depends on what
5. **Setup Lifecycle**: States and transitions
6. **Service Architecture**: Service organization and responsibilities
7. **User Journey**: User interaction flow
8. **Performance Monitoring**: Metrics and analysis structure

Each diagram can be rendered using any Mermaid-compatible tool or viewer to provide visual understanding of the system architecture.
