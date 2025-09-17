# UGE - Grammatical Evolution for Classification
## Installation and Usage Guide

### 🚀 Quick Start with Docker (Recommended)

The easiest way to run UGE is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd UGE

# Start the application
docker-compose up -d

# Access the application
open http://localhost:8501
```

### 📋 Prerequisites

#### For Docker Installation:
- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

#### For Local Installation:
- Python 3.13+
- pip (Python package manager)

### 🐳 Docker Installation (Recommended)

#### 1. Install Docker and Docker Compose
- **Windows/Mac**: Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow [Docker installation guide](https://docs.docker.com/engine/install/)

#### 2. Clone and Run
```bash
# Clone the repository
git clone <repository-url>
cd UGE

# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

#### 3. Access the Application
- Open your browser and go to: `http://localhost:8501`
- The application will be ready to use!

### 🐍 Local Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd UGE
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv UGE_env

# Activate virtual environment
# On Windows:
UGE_env\Scripts\activate
# On macOS/Linux:
source UGE_env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the Application
```bash
streamlit run app.py
```

#### 5. Access the Application
- Open your browser and go to: `http://localhost:8501`

### 📁 Directory Structure

```
UGE/
├── app.py                    # Main application entry point
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── datasets/                # Sample datasets
│   ├── clinical_breast_cancer_RFC.csv
│   └── processed.cleveland.data
├── grammars/                # BNF grammar files
│   ├── heartDisease.bnf
│   └── UGE_Classification.bnf
├── grape/                   # Core GE implementation
├── uge/                     # Application framework
│   ├── controllers/         # MVC Controllers
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   ├── utils/              # Utilities & configuration
│   └── views/              # UI components
└── results/                 # Setup results (auto-created)
```

### 🎯 Usage Guide

#### 1. **Run Setup**
- Go to "Run Setup" page
- Select a dataset (or upload your own)
- Choose a grammar file
- Configure setup parameters
- Click "Start Setup"

#### 2. **Analyze Results**
- Go to "Analysis" page
- Select an setup to analyze
- View detailed statistics and charts
- Export results in JSON/CSV format

#### 3. **Compare Setups**
- Go to "Setup Comparison" page
- Select multiple setups
- Compare performance metrics
- Generate comparison charts

### ⚙️ Configuration

#### Customizing Tooltips and Help Text
Edit `uge/utils/tooltip_config.json` to customize all help text and tooltips:

```json
{
  "setup_parameters": {
    "population": "Your custom explanation here...",
    "generations": "Your custom explanation here..."
  }
}
```

#### Adding Custom Datasets
1. Place your CSV files in the `datasets/` directory
2. Ensure your dataset has proper headers
3. Select your dataset in the "Run Setup" page

#### Adding Custom Grammars
1. Create BNF grammar files in the `grammars/` directory
2. Follow the existing grammar format
3. Select your grammar in the "Run Setup" page

### 🔧 Advanced Configuration

#### Environment Variables
```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

#### Docker Compose Profiles
```bash
# Run with nginx reverse proxy
docker-compose --profile with-nginx up -d

# Run in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 🐛 Troubleshooting

#### Common Issues

**1. Port Already in Use**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

**2. Permission Issues (Linux/Mac)**
```bash
# Fix directory permissions
sudo chown -R $USER:$USER results/ datasets/ grammars/
```

**3. Memory Issues**
```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory
```

**4. Import Errors**
```bash
# Rebuild Docker image
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 📊 Performance Tips

1. **For Large Datasets**: Increase Docker memory allocation
2. **For Long Setups**: Use `docker-compose up -d` to run in background
3. **For Multiple Users**: Consider using nginx profile for load balancing

### 🔒 Security Considerations

- The application runs on `0.0.0.0` by default (accessible from any IP)
- For production use, consider:
  - Using a reverse proxy (nginx)
  - Implementing authentication
  - Using HTTPS
  - Restricting network access

### 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the application logs: `docker-compose logs -f`
3. Create an issue in the repository
4. Contact the development team

### 🚀 Production Deployment

For production deployment:
1. Use the nginx profile: `docker-compose --profile with-nginx up -d`
2. Configure proper SSL certificates
3. Set up monitoring and logging
4. Implement backup strategies for results directory
5. Consider using Docker Swarm or Kubernetes for scaling

---

**Version**: 1.0.0  
**Last Updated**: $(date)  
**Compatibility**: Python 3.13+, Docker 20.10+
