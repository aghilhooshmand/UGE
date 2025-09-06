# 🧬 UGE: Universal Grammatical Evolution Platform

A comprehensive Streamlit web application for running, analyzing, and comparing Grammatical Evolution (GE) experiments with advanced visualization and statistical analysis capabilities.

## ✨ Key Features

### 🚀 **Experiment Management**
- **Multi-experiment support** - Run and manage multiple GE experiments
- **Flexible configuration** - Comprehensive parameter tuning for GA and GE algorithms
- **Experiment tracking** - Automatic saving and organization of experiment results
- **Batch processing** - Run multiple independent runs per experiment

### 📊 **Advanced Analysis & Visualization**
- **Individual experiment analysis** - Detailed performance metrics and visualizations
- **Multi-experiment comparison** - Compare any number of experiments side-by-side
- **Aggregate statistics** - Average performance across runs and generations
- **Interactive charts** - Plotly-based visualizations with confidence intervals
- **Generation-by-generation analysis** - Track fitness evolution over time

### 🎯 **Professional Interface**
- **Modern UI** - Clean, professional interface with intuitive navigation
- **Real-time monitoring** - Live progress tracking during experiment execution
- **Export functionality** - Download comprehensive analysis data as CSV
- **Responsive design** - Optimized for different screen sizes

## 🛠️ Technical Features

### **Algorithm Support**
- **Grammatical Evolution** using DEAP framework
- **Multiple fitness metrics** (MAE, Accuracy)
- **Flexible initialization** (Sensible, Random)
- **Advanced genetic operators** (Tournament selection, One-point crossover, Mutation)

### **Data Management**
- **Multiple dataset formats** (CSV, .data files)
- **Automatic data preprocessing** (Normalization, encoding)
- **Train/test splitting** with configurable ratios
- **Label column selection** for CSV datasets

### **Analysis Capabilities**
- **Cross-run aggregation** - Average performance across multiple runs
- **Cross-generation analysis** - Fitness evolution over generations
- **Statistical measures** - Mean, standard deviation, confidence intervals
- **Performance ranking** - Automatic ranking of experiments
- **Best individual tracking** - Store and display best solutions

## 📁 Project Structure

```
UGE/
├── app.py                    # Main Streamlit application
├── algorithms.py             # GE algorithm implementations
├── functions.py              # Primitive functions for GE
├── grape.py                  # GRAPE framework core
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── config_help.json          # Configuration help text
├── datasets/                 # Input datasets
│   ├── processed.cleveland.data
│   ├── clinical_breast_cancer_RFC.csv
│   └── ...
├── grammars/                 # BNF grammar files
│   ├── heartDisease.bnf
│   ├── UGE_Classification.bnf
│   └── ...
├── results/                  # Experiment results
│   ├── experiments/          # Organized by experiment
│   └── runs/                 # Individual run results
└── UGE_env/                  # Virtual environment
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Navigate to project directory
cd UGE

# Activate virtual environment
# For Bash:
source UGE_env/bin/activate

# For PowerShell:
& "./UGE_env/bin/Activate.ps1"

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. **Run the Application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` (or next available port).

### 3. **Create Your First Experiment**
1. Navigate to **"🏃 Run Experiment"** page
2. Configure your experiment parameters
3. Select dataset and grammar
4. Set GA and GE parameters
5. Click **"🚀 Run Experiment"**

## 📖 Usage Guide

### **Running Experiments**
1. **Experiment Configuration**:
   - Choose experiment name and dataset
   - Select appropriate grammar file
   - Configure GA parameters (population, generations, operators)
   - Set GE-specific parameters (tree depth, genome length, etc.)

2. **Multiple Runs**:
   - Set number of independent runs
   - Each run uses different random seeds
   - Results are automatically aggregated

### **Analysis & Comparison**
1. **Individual Analysis**:
   - View detailed statistics for single experiments
   - Analyze fitness evolution across generations
   - Export run data and configurations

2. **Multi-Experiment Comparison**:
   - Select multiple experiments to compare
   - View aggregate performance across runs and generations
   - See performance rankings and best experiment identification
   - Export comprehensive comparison data

### **Data Management**
1. **Dataset Manager**:
   - Upload new datasets
   - Preview existing datasets
   - Manage dataset files

2. **Grammar Editor**:
   - Create new BNF grammars
   - Edit existing grammars
   - Preview grammar files

## ⚙️ Configuration Options

### **GA Parameters**
- **Population Size**: Number of individuals per generation
- **Generations**: Number of evolutionary cycles
- **Crossover Probability**: Likelihood of crossover operation
- **Mutation Probability**: Likelihood of mutation operation
- **Elite Size**: Number of best individuals preserved
- **Tournament Size**: Selection pressure for tournament selection

### **GE Parameters**
- **Tree Depth**: Maximum depth of evolved trees
- **Genome Length**: Length of genetic representation
- **Codon Size**: Size of genetic codons
- **Initialization**: Sensible (guided) or Random initialization
- **Codon Consumption**: Lazy or Eager consumption strategy

### **Fitness Metrics**
- **MAE (Mean Absolute Error)**: Lower is better (minimization)
- **Accuracy**: Higher is better (maximization)

## 📊 Analysis Features

### **Performance Metrics**
- **Training Fitness**: Performance on training data
- **Test Fitness**: Performance on test data
- **Best Individual**: Best solution found
- **Statistical Measures**: Mean, standard deviation, confidence intervals

### **Visualizations**
- **Fitness Evolution**: Line charts showing performance over generations
- **Aggregate Charts**: Average performance across runs with confidence intervals
- **Comparison Charts**: Side-by-side comparison of multiple experiments
- **Interactive Features**: Hover details, zoom, pan capabilities

### **Export Options**
- **Summary Statistics**: High-level performance metrics
- **Detailed Data**: Generation-by-generation results
- **Comparison Reports**: Multi-experiment analysis
- **Configuration Data**: Complete experiment settings

## 🔧 Extending the Platform

### **Adding New Functions**
Extend `functions.py` to add new primitive functions for your grammars:
```python
def new_function(x, y):
    # Your function implementation
    return result
```

### **Adding New Datasets**
1. Place dataset files in `datasets/` directory
2. For CSV files, ensure proper format with label column
3. Use Dataset Manager in the app to preview and manage

### **Adding New Grammars**
1. Create BNF grammar files in `grammars/` directory
2. Use Grammar Editor in the app to create and edit
3. Ensure grammar uses available primitive functions

## 🐛 Troubleshooting

### **Common Issues**
1. **Port already in use**: Streamlit will automatically use the next available port
2. **Dataset loading errors**: Check file format and column names
3. **Grammar parsing errors**: Validate BNF syntax and function availability
4. **Memory issues**: Reduce population size or generations for large experiments

### **Performance Tips**
1. **Smaller populations** for quick testing
2. **Fewer generations** for initial parameter tuning
3. **Multiple runs** for statistical significance
4. **Export data** for external analysis

## 📈 Example Workflows

### **Parameter Tuning**
1. Create baseline experiment with default parameters
2. Create variations with different parameter values
3. Compare experiments to identify best settings
4. Use aggregate analysis to see parameter effects

### **Algorithm Comparison**
1. Run experiments with different grammars
2. Compare performance across different approaches
3. Analyze which grammar works best for your dataset
4. Export results for publication or further analysis

## 🤝 Contributing

This platform is designed for research and educational purposes. Feel free to:
- Add new analysis features
- Implement additional genetic operators
- Create new visualization types
- Extend the framework capabilities

## 📄 License

This project is part of the GA Course curriculum and is intended for educational and research purposes.

## 🎓 Educational Context

This UGE platform was developed as part of the Grammatical Evolution course, demonstrating:
- **Evolutionary Computation** principles
- **Grammatical Evolution** implementation
- **Statistical Analysis** of evolutionary algorithms
- **Web Application** development for research tools
- **Data Visualization** and analysis techniques

---

**Happy Evolving! 🧬✨**