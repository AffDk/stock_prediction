# Stock Prediction System

A comprehensive machine learning system for predicting stock prices using neural networks, technical indicators, and news sentiment analysis. The system provides both local development capabilities and cloud deployment options.

## �️ Architecture Overview

```
stock_prediction/
├── src/                        # Core application code
│   ├── api/                   # FastAPI backend services
│   ├── dashboard/             # Streamlit web interface
│   ├── data_collection/       # Data gathering modules
│   ├── feature_engineering/   # Feature processing
│   ├── training/              # ML model training
│   └── utils/                 # Shared utilities
├── config/                    # Configuration files
├── deployment/                # Kubernetes deployment manifests
├── models/                    # Trained model artifacts
├── data/                      # Dataset storage
├── notebooks/                 # Jupyter analysis notebooks
└── simple_api.py             # Simplified production API
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- uv package manager
- Docker (for containerized deployment)
- Google Cloud SDK (for cloud deployment)

### Local Development Setup

1. **Clone and Setup Environment**
   ```bash
   cd stock_prediction
   uv sync
   ```

2. **Start Local Services**
   ```bash
   # Terminal 1: Start API Server
   uv run python simple_api.py

   # Terminal 2: Start Dashboard
   uv run streamlit run src/dashboard/app.py --server.port 8503
   ```

3. **Access Applications**
   - **Dashboard**: http://localhost:8503
   - **API**: http://localhost:8889
   - **API Docs**: http://localhost:8889/docs
- **Local Development**: Efficient training on local hardware with cloud deployment
- **Real-time API**: FastAPI server with prediction endpoints
- **Interactive Dashboard**: Streamlit web app for monitoring and visualization

### 🏗️ Hybrid Architecture (Local Training + Cloud Deployment)

**Local Training Benefits**:
- **Cost Efficiency**: Zero training costs using local hardware
- **Development Speed**: Fast iteration and debugging on local machine
- **Data Privacy**: Sensitive financial data stays on your machine
- **Full Control**: Complete control over training process and hyperparameters
- **Learning Focus**: Understand ML fundamentals before scaling to cloud

**Cloud Deployment Benefits**:
- **Scalability**: Auto-scaling API server handles multiple users
- **Reliability**: Managed Cloud Run infrastructure with 99.9% uptime
- **Global Access**: API accessible from anywhere
- **Cost Optimization**: Pay-per-request pricing model

**Architecture Philosophy**:
- **Phase 1 (Current)**: Master local training and model development
- **Phase 2 (Future)**: Scale to cloud training with Vertex AI and GPU acceleration
- **Best of Both**: Local development efficiency + cloud production reliability

## 📊 Local Training + Cloud Deployment Architecture

```
┌────────────────── LOCAL DEVELOPMENT ──────────────────────┐
│  ┌─ Data Collection ─┐    ┌─ Feature Engineering ─┐       │
│  │ NewsAPI Client    │    │ FinBERT Embeddings    │       │  
│  │ yFinance Client   │ → │ Technical Indicators  │       │
│  │ Data Validation   │    │ Feature Scaling       │       │
│  └───────────────────┘    └───────────────────────┘       │
│                                    ↓                     │
│  ┌─ Model Training ──┐    ┌─ Model Evaluation ────┐       │
│  │ PyTorch MLP       │    │ Validation Metrics    │       │
│  │ Early Stopping    │ ← │ Performance Analysis  │       │
│  │ Model Checkpoints │    │ Backtesting Results   │       │
│  └───────────────────┘    └───────────────────────┘       │
└───────────────────────────────────────────────────────────┘
                              ↓ (Upload trained models)
┌────────────────── CLOUD DEPLOYMENT ───────────────────────┐
│  ┌─ Cloud Storage ──┐    ┌─ Cloud Run API ───────┐        │
│  │ Model Artifacts  │    │ FastAPI Server        │        │
│  │ Training Data    │ → │ Auto-scaling          │        │
│  │ Logs & Metrics   │    │ Load Balancing        │        │
│  └──────────────────┘    └───────────────────────┘        │
│                                    ↓                     │
│  ┌─ Streamlit Dashboard ─┐  ┌─ Monitoring ────────┐       │
│  │ Real-time Predictions │  │ Cloud Logging       │       │
│  │ Performance Tracking  │  │ Error Tracking      │       │
│  │ Interactive Charts    │  │ Cost Monitoring     │       │
│  └───────────────────────┘  └─────────────────────┘       │
└───────────────────────────────────────────────────────────┘
```

**Key Benefits of This Architecture**:
- 🔧 **Local Training**: Zero cloud compute costs, full control over training
- ☁️ **Cloud Serving**: Scalable API with global availability  
- 💰 **Cost Effective**: Only pay for serving, not training compute
- 🚀 **Fast Iteration**: Train and test locally, deploy when ready
- 📈 **Production Ready**: Cloud Run handles scaling and reliability

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- GCP account (free tier)
- NewsAPI account

### Installation

1. **Install uv (if not already installed):**
   ```bash
   # Windows PowerShell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Linux/Mac
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd stock_prediction
   
   # Option 1: Use the setup script (recommended)
   python setup_uv.py
   
   # Option 2: Manual setup
   uv sync --dev
   ```

2. **Set up API keys:**
   ```bash
   # Create .env file
   echo "NEWS_API_KEY=your_newsapi_key" > .env
   echo "GCP_PROJECT_ID=your_gcp_project_id" >> .env
   ```

3. **Set up GCP Service Account & Download Credentials:**

   **Step 1: Create a Service Account in GCP Console**
   1. Go to [Google Cloud Console](https://console.cloud.google.com/)
   2. Select your project: `proud-curve-473118-h6`
   3. Navigate to **IAM & Admin** → **Service Accounts**
   4. Click **+ CREATE SERVICE ACCOUNT**
   5. Fill in the details:
      - **Service account name**: `stock-prediction-service`
      - **Service account ID**: `stock-prediction-service` (auto-generated)
      - **Description**: `Service account for stock prediction ML pipeline`
   6. Click **CREATE AND CONTINUE**

   **Step 2: Grant Required Permissions**
   Add these roles to the service account:
   - **Cloud Run Admin** (for API deployment)
   - **Cloud Storage Admin** (for model artifacts)
   - **Vertex AI User** (for future ML training)
   - **Monitoring Viewer** (for logging/monitoring)
   - **BigQuery User** (for data analysis)
   
   Click **CONTINUE** → **DONE**

   **Step 3: Create and Download the Key**
   1. In the Service Accounts list, click on your newly created service account
   2. Go to the **KEYS** tab
   3. Click **ADD KEY** → **Create new key**
   4. Select **JSON** format
   5. Click **CREATE**
   6. The key file will automatically download to your computer
   7. **Important**: Rename the downloaded file to `gcp-credentials.json`
   8. Move it to your project root directory

   **Step 4: Set Environment Variable (Windows)**
   ```powershell
   # Windows PowerShell - Set for current session
   $env:GOOGLE_APPLICATION_CREDENTIALS = "gcp-credentials.json"
   
   # Windows PowerShell - Set permanently (recommended)
   [Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "gcp-credentials.json", "User")
   
   # Or add to your .env file (recommended for uv)
   Add-Content .env "GOOGLE_APPLICATION_CREDENTIALS=gcp-credentials.json"
   ```

   **Step 5: Install & Configure gcloud CLI (Windows)**
   ```powershell
   # Install gcloud CLI if not already installed
   # Download from: https://cloud.google.com/sdk/docs/install-sdk#windows
   
   # Or install via Chocolatey (if you have it)
   choco install gcloudsdk
   
   # Initialize gcloud and authenticate
   gcloud init
   ```

   **Step 6: Authenticate & Set Project (Windows)**
   ```powershell
   # Authenticate with your Google account
   gcloud auth login
   
   # Set the project ID
   gcloud config set project proud-curve-473118-h6
   
   # Set up application default credentials (for service account)
   gcloud auth application-default login
   
   # Verify everything is set up correctly
   gcloud auth list
   gcloud config list project
   ```

   **Alternative: Use Service Account Key Directly (Windows)**
   ```powershell
   # If you prefer to use the service account key file directly
   gcloud auth activate-service-account --key-file=gcp-credentials.json
   gcloud config set project proud-curve-473118-h6
   
   # Verify authentication
   gcloud auth list --filter=status:ACTIVE --format="table(account)"
   ```

   **Security Note**: 
   - ⚠️ **Never commit `gcp-credentials.json` to version control**
   - The file is already in `.gitignore`
   - Store it securely and rotate keys regularly

## 🧠 Model Architecture & Fine-Tuning Strategy

### Multi-Stage Model Architecture

Our system uses a **two-stage approach** combining pre-trained transformers with custom prediction layers:

#### Stage 1: FinBERT Fine-Tuning for Financial News Understanding

**Base Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **Architecture**: BERT-base (12 layers, 768 hidden units, 12 attention heads)
- **Pre-training**: Pre-trained on financial texts (earnings calls, news, SEC filings)
- **Parameters**: 110M parameters total

**Fine-Tuning Strategy**:
```python
# Layers to Fine-tune (Selective Unfreezing)
├── Embedding Layer: FROZEN (preserve financial vocabulary)
├── Encoder Layers 0-8: FROZEN (preserve low-level financial patterns)  
├── Encoder Layers 9-11: UNFROZEN (adapt to stock-specific context)
├── Pooler Layer: UNFROZEN (learn stock price relevance)
└── Custom Classification Head: TRAINABLE (news → price impact)
```

**Fine-Tuning Configuration**:
- **Trainable Parameters**: ~13M (12% of total model)
- **Learning Rate**: 2e-5 (BERT layers), 1e-4 (custom head)
- **Batch Size**: 16-32 (depends on GPU memory)
- **Max Sequence Length**: 512 tokens
- **Training Objective**: Regression (news → 7-day price change)
- **Regularization**: Layer-wise learning rate decay, gradient clipping

#### Stage 2: Multi-Modal Fusion Network for Price Prediction

**Architecture**: Custom PyTorch Neural Network
```python
class StockPredictionModel(nn.Module):
    def __init__(self):
        # Input Processing
        self.news_projection = nn.Linear(768, 256)      # FinBERT → compressed
        self.technical_projection = nn.Linear(19, 64)   # Technical indicators
        
        # Fusion Layer
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=320,  # 256 + 64
            num_heads=8,
            dropout=0.1
        )
        
        # Prediction Network
        self.predictor = nn.Sequential(
            nn.Linear(320, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1)  # Price change prediction
        )
```

**Model Components**:
1. **News Encoder**: Fine-tuned FinBERT (768 → 256 dimensions)
2. **Technical Encoder**: Dense layers for financial indicators (19 → 64 dimensions)
3. **Fusion Layer**: Multi-head attention to combine modalities (320 dimensions)
4. **Prediction Head**: 4-layer MLP with layer normalization and dropout

**Training Strategy**:
- **Two-Stage Training**:
  1. **Stage 1**: Fine-tune FinBERT on news-price correlation (2-3 days)
  2. **Stage 2**: Train fusion network with frozen FinBERT (1-2 days)
- **Progressive Unfreezing**: Gradually unfreeze FinBERT layers during Stage 2
- **Mixed Precision**: FP16 training to fit larger batches on GPU
- **Gradient Accumulation**: Simulate larger batch sizes on limited VRAM

### GCP Training Infrastructure

**Recommended Training Setup** (using your $300 credit):

#### Option 1: AI Platform Training (Managed)
```bash
# High-performance managed training
Machine Type: n1-highmem-8 (8 vCPUs, 52GB RAM)
Accelerator: 1x NVIDIA V100 (16GB VRAM)
Estimated Cost: $3-5/hour
Training Time: 12-24 hours total
Total Cost: $50-120
```

#### Option 2: Compute Engine + Custom Setup (More Control)
```bash
# Custom VM with flexibility
Machine Type: n1-standard-8 (8 vCPUs, 30GB RAM)
Accelerator: 1-2x NVIDIA T4 (16GB VRAM each)
Disk: 500GB SSD for fast data loading
Estimated Cost: $1.5-3/hour
Training Time: 24-48 hours total
Total Cost: $40-150
```

#### Option 3: Vertex AI Workbench (Best for Learning)
```bash
# Jupyter notebook environment with managed dependencies
Machine Type: n1-standard-4 (4 vCPUs, 15GB RAM)
Accelerator: 1x NVIDIA T4 (16GB VRAM)
Estimated Cost: $1-2/hour
Training Time: Interactive development + 12-24h training
Total Cost: $50-100
```

### Hyperparameter Tuning Strategy

**Automated Hyperparameter Search** (using GCP Vizier):
```yaml
# Search Space Configuration
parameters:
  learning_rate_bert:
    type: DOUBLE
    min_value: 1e-6
    max_value: 1e-4
    scale: LOG
    
  learning_rate_fusion:
    type: DOUBLE  
    min_value: 1e-4
    max_value: 1e-2
    scale: LOG
    
  dropout_rate:
    type: DOUBLE
    min_value: 0.1
    max_value: 0.5
    
  batch_size:
    type: DISCRETE
    feasible_points: [16, 32, 64]
    
  news_lookback_days:
    type: INTEGER
    min_value: 3
    max_value: 14

# Search Algorithm: Bayesian Optimization
# Trials: 20-50 experiments
# Estimated Cost: $20-50 for tuning
```

### Expected Model Performance at Scale

**Dataset Scale** (with $300 budget):
- **Stocks**: 50-100 major stocks (S&P 500 subset)
- **Time Range**: 2-3 years of historical data
- **News Articles**: 500K-1M financial news articles
- **Training Samples**: 100K-500K stock-news-price triplets
- **Total Dataset Size**: 50-200GB

**Performance Targets**:
- **Mean Absolute Error (MAE)**: <2% on 7-day price predictions
- **Directional Accuracy**: >75% (better than random 50%)
- **Sharpe Ratio**: >1.5 (risk-adjusted returns)
- **Information Ratio**: >0.8 (alpha generation capability)

**Production Serving Performance**:
- **Prediction Latency**: <200ms per request
- **Throughput**: 1000+ requests/minute
- **Model Size**: ~500MB (optimized for serving)
- **Daily Retraining**: Automated pipeline for fresh predictions

## 🗄️ Data Storage & Training Details

### Where Data is Stored

**Local Storage (Development & Training)**:
```
data/
├── raw/                    # Original collected data
│   ├── sample/            # 30-day sample data for testing
│   └── full/              # Complete 1-year dataset
├── processed/             # Feature-engineered datasets  
│   ├── training_dataset.parquet
│   └── embeddings/        # FinBERT news embeddings
└── training/              # Structured training data
```

**Cloud Storage (GCS Bucket: `stock-prediction-data-bucket`)**:
```
gs://stock-prediction-data-bucket/
├── raw_data/              # Backup of collected data
│   ├── stocks/           # Daily stock data (Parquet files)
│   └── news/             # News articles with timestamps
├── processed_data/        # Feature-engineered datasets
│   ├── embeddings/       # FinBERT embeddings (768-dim vectors)
│   └── training_sets/    # Ready-to-train datasets
├── models/               # Trained model artifacts
│   ├── stock_predictor/  # PyTorch model + scalers
│   └── checkpoints/      # Training checkpoints
└── logs/                 # Training logs and metrics
```

### Training Process

**Local Training Pipeline (Current Implementation)**:

1. **Data Collection**:
   ```bash
   # Collect sample data (30 days) for testing
   uv run python collect_sample_data.py
   
   # Or collect full dataset (1 year) for production training
   uv run python train_pipeline.py --symbols AAPL GOOGL MSFT --start-date 2024-01-01
   ```
   - **NewsAPI**: Fetches financial news articles with relevance filtering
   - **yFinance**: Downloads OHLCV data + technical indicators  
   - **Data Validation**: Checks for missing data, outliers, and quality issues
   - **Storage**: Saves raw data as Parquet files in `data/raw/`

2. **Feature Engineering**:
   ```bash
   # FinBERT embeddings are computed during training pipeline
   uv run python -c "from src.feature_engineering.feature_engineer import FeatureEngineer; fe = FeatureEngineer(); fe.process_features()"
   ```
   - **News Embeddings**: FinBERT (ProsusAI/finbert) processes financial news
     - Pre-trained on financial text for domain-specific understanding
     - 768-dimensional embeddings per news article
     - Aggregated over 7-day lookback window with relevance weighting
   - **Technical Indicators**: 20+ indicators calculated locally
     - Moving averages (SMA, EMA), RSI, MACD, Bollinger Bands
     - Volume indicators, volatility measures, momentum oscillators
   - **Feature Scaling**: StandardScaler normalizes all features
   - **Storage**: Processed features saved as Parquet in `data/processed/`

3. **Model Training (PyTorch)**:
   ```bash
   # Full training pipeline with all steps
   uv run python train_pipeline.py
   
   # Or train with existing processed data
   uv run python -m src.training.stock_predictor
   ```
   
   **Neural Network Architecture**:
   - **Input Layer**: 773 features (768 FinBERT + 5 technical indicators)
   - **Hidden Layers**: [512, 256, 128] neurons with ReLU activation
   - **Dropout**: 0.3 between layers to prevent overfitting
   - **Output Layer**: 1 neuron for price prediction (regression)
   - **Loss Function**: Mean Squared Error (MSE)
   - **Optimizer**: Adam with learning rate 0.001
   
   **Training Configuration**:
   - **Batch Size**: 32 samples
   - **Epochs**: 50 (with early stopping patience=10)
   - **Train/Validation Split**: 80/20
   - **Hardware**: Runs on CPU or local GPU (if available)
   - **Training Time**: ~5-15 minutes depending on dataset size

4. **Model Evaluation & Validation**:
   ```bash
   # Training includes automatic evaluation
   # Check results in logs/ directory
   ```
   - **Validation Metrics**: MAE, RMSE, R² score calculated on validation set
   - **Backtesting**: Walk-forward validation on out-of-sample data  
   - **Performance Tracking**: Training/validation loss plots saved to `logs/`
   - **Model Checkpoints**: Best model saved based on validation loss

**Why Local Training?**:
- **Cost Efficiency**: Zero cloud compute costs during development
- **Development Speed**: Fast iteration and debugging on local machine  
- **Data Privacy**: Financial data stays on your local machine
- **Full Control**: Complete control over training hyperparameters and process
- **Hardware Agnostic**: Runs on CPU or GPU, automatically detects available hardware
- **Learning Focus**: Master ML fundamentals before scaling to cloud infrastructure

### Usage

#### Quick Start (Local Development)
```bash
# 1. Set up environment
uv shell

# 2. Run sample training pipeline (30 days data)
uv run python train_pipeline.py --sample-only --symbols AAPL GOOGL MSFT

# 3. Start API server locally
uv run python -m src.api.main

# 4. Start dashboard (in another terminal)
uv run streamlit run src/dashboard/app.py

# 5. Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 7}'
```

#### Complete Local Training Pipeline
```bash
# Full 1-year dataset collection and training
uv run python train_pipeline.py \
  --symbols AAPL GOOGL MSFT TSLA AMZN NVDA META NFLX \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Or step by step:
# 1. Collect data only
uv run python collect_sample_data.py --full-year

# 2. Train model on existing data  
uv run python train_pipeline.py --skip-data-collection

# 3. Deploy to cloud (upload models, start services)
uv run python deploy_gcp.py
```

#### Development & Testing (Local Focus)
```bash
# Quick local development cycle
uv run python collect_sample_data.py        # Get sample data (30 days)
uv run python train_pipeline.py --debug    # Train with debug logging
uv run python -m src.api.main              # Test API locally
curl http://localhost:8000/predict -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 7}'  # Test prediction
```

## 🚀 Deployment Options

### Local Development (Recommended for Training)
```bash
# Run everything locally - no cloud costs for training
uv run python train_pipeline.py               # Train locally
uv run python -m src.api.main &               # API on localhost:8000
uv run streamlit run src/dashboard/app.py      # Dashboard on localhost:8501
```
**Benefits**: Free training, fast iteration, full control, data privacy
**Use Case**: Development, model experimentation, learning ML fundamentals

### Hybrid Deployment (Recommended for Production)
```bash
# Train locally, deploy API to cloud
uv run python train_pipeline.py           # Local training (free)
uv run python deploy_gcp.py              # Deploy trained model to Cloud Run
```
**Benefits**: Cost-efficient training + scalable serving
**Use Case**: Production system with controlled costs
**Cost**: ~$10-30/month (serving only, no training costs)

### Full Cloud Deployment (For Team Collaboration)
```bash
# Everything in the cloud (future enhancement)
gcloud builds submit --tag gcr.io/proud-curve-473118-h6/stock-prediction
gcloud run deploy --image gcr.io/proud-curve-473118-h6/stock-prediction
```
**Benefits**: Fully managed, team collaboration, auto-scaling
**Use Case**: Multi-user production applications
**Cost**: ~$50-200/month depending on usage and training frequency

# 3. Set up monitoring and alerting
gcloud monitoring policies create --policy-from-file=monitoring/alerting.yaml
```
**Benefits**: Full observability, automated CI/CD, enterprise features
**Use Case**: Large-scale production with team collaboration
**Cost**: ~$100-500/month depending on scale

### Architecture Decision Matrix

| Component | Development | Production | Enterprise |
|-----------|-------------|------------|------------|
| **Training** | Vertex AI (basic) | Vertex AI Pipelines | Vertex AI + KFP |
| **Serving** | Vertex AI Endpoints | Cloud Run | GKE + Istio |
| **Data** | Cloud Storage | Cloud Storage + BigQuery | Multi-region + CDN |
| **Monitoring** | Basic logging | Cloud Monitoring | Full observability stack |
| **Cost/Month** | $10-30 | $30-100 | $100-500+ |

## 💰 Cost Analysis & Local Training Benefits

### Local Training Costs (Current Implementation)
```
Hardware Requirements:      $0   (uses existing laptop/desktop)
Training Compute:          $0   (local CPU/GPU)
Data Collection:           $0   (NewsAPI free tier: 1000 requests/day)
Storage:                   $0   (local storage)
Development Tools:         $0   (open source: Python, PyTorch, etc.)
Total Training Cost:       $0
```

### Cloud Deployment Costs (After Training)
```
Model Storage (GCS):       $1   (5GB model artifacts)
API Serving (Cloud Run):  $5   (light usage, auto-scaling)
Dashboard Hosting:        $3   (Streamlit on Cloud Run)
Monitoring:               $2   (basic Cloud Monitoring)
Total Monthly Cost:       ~$11/month
```

### Cost Comparison: Local vs Cloud Training

**Local Training Approach (Current)**:
- **Training**: FREE (uses your hardware)
- **Development**: FAST (no cloud setup/authentication delays)
- **Privacy**: HIGH (data stays local)
- **Learning**: OPTIMAL (understand fundamentals first)
- **Deployment**: $10-20/month (serving only)

**Cloud Training Approach (Future Phase)**:
- **Training**: $50-200/month (GPU instances)
- **Development**: SLOWER (cloud provisioning overhead)
- **Privacy**: MEDIUM (data in cloud)
- **Learning**: ADVANCED (production ML infrastructure)
- **Deployment**: $50-500/month (full MLOps pipeline)

## 🛣️ Development Roadmap

### Phase 1: Local Training Mastery (Current) ✅
**Goal**: Master ML fundamentals with zero cloud costs
- [x] Local environment setup with uv package manager
- [x] Data collection pipeline (NewsAPI + yFinance)
- [x] FinBERT news embedding integration
- [x] Technical indicators feature engineering
- [x] PyTorch neural network implementation
- [x] Training pipeline with validation and checkpointing
- [x] FastAPI serving infrastructure
- [x] Streamlit dashboard for monitoring
- [x] Model evaluation and backtesting

**Outcomes**: Fully functional stock prediction system running locally

### Phase 2: Cloud Deployment (Next Priority) 🔄
**Goal**: Deploy trained models to cloud for scalable serving
- [ ] GCP authentication and project setup
- [ ] Cloud Storage integration for model artifacts
- [ ] Cloud Run deployment for FastAPI
- [ ] Cloud Run deployment for Streamlit dashboard
- [ ] Production monitoring and logging
- [ ] CI/CD pipeline with automated deployments

**Expected Timeline**: 1-2 weeks
**Expected Cost**: $10-30/month

### Phase 3: Cloud Training Migration (Future) 📅
**Goal**: Scale to cloud training for larger datasets and faster iteration
- [ ] Vertex AI custom training jobs
- [ ] Containerized training pipeline
- [ ] Hyperparameter tuning with Vizier
- [ ] Distributed training for large datasets
- [ ] MLOps pipeline with experiment tracking
- [ ] A/B testing and model versioning

**Expected Timeline**: 2-4 weeks
**Expected Cost**: $100-300/month (during active development)

### Phase 4: Production Scale (Advanced) 🚀
**Goal**: Enterprise-grade system with full MLOps
- [ ] Multi-region deployment
- [ ] Real-time data streaming
- [ ] Advanced monitoring and alerting
- [ ] Automated retraining pipelines
- [ ] Load testing and performance optimization
- [ ] Security hardening and compliance

**Expected Timeline**: 1-2 months
**Expected Cost**: $200-500/month

**Tesla V100 (For production-scale)**:
```
Base Cost:              $2.48/hour
+ Vertex AI overhead:   $0.52/hour
Total:                  $3.00/hour
Faster training:        ~8 hours = $24.00
Better performance:     Worth it for final model
```

### Local Development Optimization Strategies

1. **Hardware Utilization**: 
   ```bash
   # Check if GPU is available for training
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   
   # Use all available CPU cores for data processing
   export OMP_NUM_THREADS=$(nproc)
   ```

2. **Efficient Data Handling**:
   - Use Parquet format for 10x faster I/O compared to CSV
   - Implement data caching to avoid re-downloading news/stock data
   - Process features incrementally and cache embeddings

3. **Memory Management**: 
   - Batch size optimization based on available RAM/VRAM
   - Gradient accumulation for larger effective batch sizes
   - Clear PyTorch cache between training runs

4. **Development Workflow**:
   ```bash
   # Fast iteration cycle
   uv run python collect_sample_data.py     # Quick data for testing
   uv run python train_pipeline.py --debug  # Train with debug mode
   uv run python -m src.api.main &          # Test API locally
   ```

5. **Model Experimentation**:
   - Save training metrics to `logs/` for comparison
   - Use early stopping to avoid wasted training time
   - Implement model checkpointing for resuming training

### Local Development Learning Outcomes

**Week 1 - Environment & Data Pipeline**:
- uv package management mastery
- NewsAPI and yFinance data collection
- Data validation and preprocessing
- Understanding financial data structures

**Week 2 - Feature Engineering & ML**:
- FinBERT transformer model integration
- Technical indicators calculation
- PyTorch neural network implementation
- Training loop and validation strategies

**Week 3 - API & Dashboard Development**:
- FastAPI server implementation
- RESTful API design patterns
- Streamlit dashboard creation
- Real-time prediction serving

**Week 4 - Testing & Documentation** (Current Phase):
- Comprehensive testing and validation
- Performance optimization
- Documentation and code organization
- Preparation for cloud deployment

## 🛠️ Local Development Deep Dive

### Local Training Pipeline Implementation

**Training Script Structure**:
```python
# train_pipeline.py - Main training orchestrator
from src.data_collection.orchestrator import DataOrchestrator
from src.feature_engineering.feature_engineer import FeatureEngineer  
from src.training.stock_predictor import StockPredictor

def main():
    # 1. Data Collection
    orchestrator = DataOrchestrator()
    orchestrator.collect_full_dataset(symbols=['AAPL', 'GOOGL', 'MSFT'])
    
    # 2. Feature Engineering
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.process_features()
    
    # 3. Model Training
    predictor = StockPredictor()
    model, metrics = predictor.train(features_df)
    
    print(f"Training completed. Validation MAE: {metrics['mae']:.3f}")
```

**Hardware Detection & Optimization**:
```python
# src/training/stock_predictor.py
import torch
import multiprocessing as mp

class StockPredictor:
    def __init__(self):
        # Automatically detect and use available hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(mp.cpu_count(), 8)  # Optimize for data loading
        print(f"Using device: {self.device}")
        print(f"Data loading workers: {self.num_workers}")
    
    def train(self, data):
        # Configure batch size based on available memory
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.batch_size = min(64, max(16, gpu_memory // (1024**3) * 8))
        else:
            self.batch_size = 32
```

### Local Hyperparameter Optimization

**Manual Hyperparameter Search**:
```python
# hyperparameter_search.py
import itertools
from src.training.stock_predictor import StockPredictor

def grid_search():
    # Define parameter grid for local experimentation
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4], 
        'hidden_sizes': [[256, 128], [512, 256, 128], [1024, 512, 256]]
    }
    
    best_score = float('inf')
    best_params = None
    
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Train model with these parameters
        predictor = StockPredictor(**param_dict)
        model, metrics = predictor.train(train_data)
        
        if metrics['val_loss'] < best_score:
            best_score = metrics['val_loss']
            best_params = param_dict
            
        print(f"Params: {param_dict}, Val Loss: {metrics['val_loss']:.4f}")
    
    print(f"Best parameters: {best_params}")
    return best_params
```

### Local Data Pipeline Optimization

**Efficient Data Loading**:
```python
# src/data_collection/orchestrator.py
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

class DataOrchestrator:
    def __init__(self):
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def collect_with_caching(self, symbols, start_date, end_date):
        """Collect data with intelligent caching to avoid redundant API calls"""
        cached_data = []
        symbols_to_fetch = []
        
        for symbol in symbols:
            cache_file = f"{self.cache_dir}/{symbol}_{start_date}_{end_date}.parquet"
            if os.path.exists(cache_file):
                cached_data.append(pd.read_parquet(cache_file))
                print(f"Loaded {symbol} from cache")
            else:
                symbols_to_fetch.append(symbol)
        
        # Fetch only missing data
        if symbols_to_fetch:
            with ThreadPoolExecutor(max_workers=4) as executor:
                new_data = list(executor.map(self.fetch_symbol_data, symbols_to_fetch))
                
            # Cache new data
            for symbol, data in zip(symbols_to_fetch, new_data):
                cache_file = f"{self.cache_dir}/{symbol}_{start_date}_{end_date}.parquet"
                data.to_parquet(cache_file)
        
        return pd.concat(cached_data + new_data, ignore_index=True)
```
```

**Cloud Workflow for Training Pipeline**:
```yaml
# training_workflow.yaml
main:
  steps:
  - data_validation:
      call: validate_data_quality
      args:
        bucket: "stock-data-raw"
  - feature_engineering:
      call: process_features
      args:
        input_bucket: "stock-data-raw" 
        output_bucket: "stock-features"
  - model_training:
      call: submit_vertex_training
      args:
        config_file: "vertex_training_config.yaml"
  - model_evaluation:
      call: evaluate_model_performance
      args:
        model_path: ${model_training.model_path}
  - deployment:
      call: deploy_to_endpoint
      args:
        model_path: ${model_training.model_path}
      condition: ${model_evaluation.accuracy > 0.70}
```

### Production Monitoring & Observability

**Key Metrics to Track**:
```python
# Cloud Monitoring metrics for production
TRAINING_METRICS = {
    'training_loss': 'custom.googleapis.com/ml/training_loss',
    'validation_loss': 'custom.googleapis.com/ml/validation_loss', 
    'gpu_utilization': 'custom.googleapis.com/compute/gpu_utilization',
    'training_duration': 'custom.googleapis.com/ml/training_duration_minutes'
}

SERVING_METRICS = {
    'prediction_latency': 'custom.googleapis.com/api/prediction_latency_ms',
    'prediction_accuracy': 'custom.googleapis.com/ml/prediction_accuracy',
    'error_rate': 'custom.googleapis.com/api/error_rate',
    'qps': 'custom.googleapis.com/api/queries_per_second'
}
```

**Alerting Configuration**:
```yaml
# alerting_policy.yaml
displayName: "Stock Prediction Training Alerts"
conditions:
- displayName: "Training Job Failed"
  conditionThreshold:
    filter: 'resource.type="vertex_ai_custom_job"'
    comparison: COMPARISON_EQ
    thresholdValue: 1
    duration: 60s
- displayName: "High Prediction Error Rate"
  conditionThreshold:
    filter: 'metric.type="custom.googleapis.com/api/error_rate"'
    comparison: COMPARISON_GT
    thresholdValue: 0.05
    duration: 300s
```

### Model Versioning & A/B Testing

**Model Registry Strategy**:
```python
# Vertex AI Model Registry integration
from google.cloud import aiplatform

def deploy_model_version(model_path: str, version: str):
    """Deploy new model version with A/B testing"""
    model = aiplatform.Model.upload(
        display_name=f"stock-predictor-{version}",
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-cpu.1-9:latest"
    )
    
    # Deploy with traffic split for A/B testing
    endpoint = aiplatform.Endpoint.create(display_name="stock-prediction-endpoint")
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"model-{version}",
        machine_type="n1-standard-2",
        traffic_percentage=20  # Start with 20% traffic
    )
    
    return endpoint
```

#### API Usage Examples
```bash
# Health check
curl http://localhost:8000/

# Get prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 7}'

# Start training
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "GOOGL"], "start_date": "2024-09-01", "end_date": "2025-08-31"}'
```

## � Technical Implementation Details

### Data Processing Pipeline

**Step 1: Data Collection (Local)**
```
NewsAPI + yfinance → Raw Data (JSON/CSV) → Local Storage (data/raw/)
├── Stock Data: OHLCV + Technical Indicators (20 features)
├── News Data: Articles with timestamps, relevance scores
└── Validation: Data quality checks, missing value handling
```

**Step 2: Feature Engineering (Local)**
```
Raw Data → FinBERT Embeddings + Technical Features → Training Dataset
├── News Processing: Title + Description → FinBERT → 768-dim vectors
├── Financial Features: SMA, RSI, MACD, Bollinger Bands, etc.
├── Target Generation: 7-day forward returns (percentage change)
└── Feature Scaling: StandardScaler for model input
```

**Step 3: Model Training (Local)**
```
Training Dataset → PyTorch MLP → Trained Model → Local Storage
├── Architecture: [Input(773) → Hidden(512,256,128) → Output(1)]
├── Training: Adam optimizer, MSE loss, early stopping
├── Validation: 80/20 split, MAE, R², directional accuracy
└── Artifacts: model.pth, scalers.pkl, config.json
```

**Step 4: Deployment (Cloud)**
```
Local Model → GCS Upload → Cloud Run → FastAPI Serving
├── Container: Docker image with model + dependencies
├── Scaling: Auto-scale 0-10 instances based on traffic  
├── Storage: Model artifacts served from GCS
└── Monitoring: Request logs, model performance tracking
```

### Model Architecture Details

**Input Features (773 dimensions)**:
- **Financial Indicators (19 features)**:
  - Price data: Open, High, Low, Close, Volume
  - Moving averages: SMA_20, SMA_50
  - Momentum: RSI_14, MACD, MACD_Signal
  - Volatility: Bollinger Bands, 20-day volatility
  - Returns: 1-day, 5-day, 20-day percentage changes
  
- **News Embeddings (768 features)**:
  - FinBERT embeddings of financial news
  - Aggregated over 7-day lookback window
  - Weighted by relevance scores
  
- **Derived Features (4 features)**:
  - Volume ratios, price momentum
  - News sentiment scores
  - Market regime indicators

**Model Training Process**:
1. **Data Preparation**: ~50,000 samples from 1-year historical data
2. **Feature Scaling**: StandardScaler for inputs, targets
3. **Network Architecture**: 
   - Input Layer: 773 neurons
   - Hidden Layers: [512, 256, 128] with ReLU activation
   - Dropout: 0.3 between layers for regularization
   - Output Layer: 1 neuron (regression target)
4. **Training Configuration**:
   - Batch Size: 32 samples
   - Learning Rate: 0.001 (Adam optimizer)
   - Epochs: 50 (with early stopping patience=10)
   - Validation Split: 20%

**Expected Performance Metrics**:
- **Mean Absolute Error (MAE)**: ~2.3% price prediction error
- **R² Score**: ~0.74 (explains 74% of variance)
- **Directional Accuracy**: ~72% (correct up/down prediction)
- **Sharpe Ratio**: ~1.45 (risk-adjusted returns)

### Data Storage Strategy

**Local Development Storage**:
```
data/
├── raw/                          (~2-5 GB)
│   ├── AAPL_stock_data.parquet  # Daily OHLCV + indicators
│   ├── AAPL_news_data.parquet   # News articles with metadata
│   └── ...                      # Other symbols
├── processed/                    (~1-3 GB)
│   ├── embeddings/              # FinBERT embeddings cache
│   └── training_dataset.parquet # Final training dataset
└── training/                     (~500 MB)
    └── features_targets.pkl     # Ready-to-train data
```

**Cloud Storage (GCS) Strategy**:
```
gs://stock-prediction-data-bucket/
├── raw_data/                    # Partitioned by symbol/date
│   ├── year=2024/month=09/     # Hive-style partitioning
│   └── year=2025/month=08/     # Efficient querying
├── processed_data/              # Compressed Parquet files
│   ├── embeddings.parquet.gzip # News embeddings archive
│   └── features.parquet.gzip   # Combined feature dataset
├── models/                      # Model versioning
│   ├── v1.0/                   # Timestamp-based versions
│   └── latest/                 # Symlink to current model
└── logs/                        # Training and inference logs
```

**Why This Architecture?**:
- **Cost Efficiency**: Local training avoids cloud compute costs
- **Scalability**: Cloud storage and serving handle production load
- **Flexibility**: Easy to switch between local and cloud training
- **Reliability**: Model artifacts backed up in cloud storage
- **Performance**: Local NVMe storage for fast data access during training

## �📁 Project Structure

```
stock_prediction/
├── src/
│   ├── data_collection/      # News and stock data collection
│   ├── feature_engineering/  # Data preprocessing and embeddings
│   ├── training/             # Model training and fine-tuning
│   ├── evaluation/           # Model evaluation and metrics
│   ├── api/                  # FastAPI deployment
│   ├── dashboard/            # Streamlit dashboard
│   └── utils/                # Common utilities
├── config/                   # Configuration files
├── data/                     # Local data storage (gitignored)
├── models/                   # Trained models (gitignored)
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
└── deployment/               # Cloud deployment configurations
```

## 🔧 Configuration

Key configuration files:
- `config/config.yaml`: Main configuration
- `config/stocks.yaml`: Stock symbols and parameters
- `config/gcp.yaml`: GCP settings
- `.env`: Environment variables (create from template)

## 💰 Cost Optimization Strategy

### Free Tier Usage:
- **GCP**: 90-day trial + always-free tier
- **Colab**: Free GPU for training (limited)
- **NewsAPI**: 1000 requests/day (free)
- **Hugging Face**: Free model hosting

### Cost Monitoring:
- Billing alerts configured
- Spot instances for training
- Cloud Run scales to zero
- Efficient data partitioning

## 📈 Features

### Data Collection
- **Multi-source**: NewsAPI, yfinance, alternative data
- **Time-aware**: Week-over-week prediction windows
- **Robust**: Handles market holidays and data gaps
- **Scalable**: Parallel processing with rate limiting

### Model Architecture
- **Embeddings**: FinBERT for news sentiment (768-dim)
- **Features**: Technical indicators (5-dim)
- **Architecture**: Multi-layer perceptron (MLP)
- **Training**: GPU-accelerated with hyperparameter tuning

### Deployment
- **API**: FastAPI with async endpoints
- **Frontend**: Streamlit dashboard
- **Monitoring**: Real-time accuracy tracking
- **Alerts**: Performance degradation notifications

## 📊 Performance Metrics

- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Sharpe Ratio**: Risk-adjusted returns
- **Directional Accuracy**: Up/down prediction accuracy

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🚀 Deployment

### Local Development
```bash
# API server
uvicorn src.api.main:app --reload

# Dashboard
streamlit run src/dashboard/app.py
```

### Cloud Deployment
```bash
# Deploy to Cloud Run
gcloud run deploy stock-prediction-api --source .

# Deploy dashboard to Hugging Face Spaces
# (See deployment/huggingface/ for configuration)
```

## 📝 Development Roadmap

### Week 1: Prototype & Data Collection
- [x] Project setup with UV
- [ ] Sample data collection (1 month, 5 stocks)
- [ ] Basic news embedding pipeline
- [ ] GCS storage integration

### Week 2: Full Dataset & Model Training
- [ ] Complete 1-year dataset collection
- [ ] Feature engineering pipeline
- [ ] Model training and evaluation
- [ ] Hyperparameter optimization

### Week 3-4: Deployment & Monitoring
- [ ] FastAPI deployment
- [ ] Streamlit dashboard
- [ ] Cloud Run deployment
- [ ] Performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without proper risk management and regulatory compliance.