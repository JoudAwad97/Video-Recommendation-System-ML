# Video Recommendation System

A production-ready video recommendation system using a Two-Tower architecture for candidate retrieval and CatBoost Ranker for personalized ranking. Built with TensorFlow, designed for AWS deployment with fully automated ML training pipelines.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [ML Pipeline](#ml-pipeline)
- [Development](#development)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This system implements a 4-stage recommendation pipeline:

1. **Candidate Generation**: Two-Tower neural network retrieves ~100-500 candidates from millions of videos using approximate nearest neighbor search
2. **Filtering**: Business rules filter out watched/irrelevant items, apply content policies
3. **Ranking**: CatBoost Ranker scores candidates using rich user-video interaction features
4. **Ordering**: Final ordering with business logic (diversity, freshness, promotional content)

### Key Features

- **Two-Tower Architecture**: Separate user and video encoders with shared embedding space
- **Multi-Query Retrieval**: Generate diverse candidates using category-based query contexts
- **CatBoost Ranker**: Gradient boosting model with categorical feature support
- **Real-time Serving**: AWS Lambda + API Gateway with Redis caching
- **Feature Store Integration**: DynamoDB for online features, Glue for offline features
- **A/B Testing Framework**: Built-in experimentation and metric tracking
- **Automated ML Pipeline**: End-to-end training, evaluation, and deployment via Step Functions
- **Production-Ready**: Health checks, circuit breakers, structured logging, secrets management

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Video Recommendation System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Candidate  │    │   Filtering  │    │   Ranking    │    │  Final    │ │
│  │  Generation  │───▶│   Service    │───▶│   Service    │───▶│  Ordering │ │
│  │  (Two-Tower) │    │              │    │  (CatBoost)  │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Vector Store │    │    Redis     │    │   Feature    │    │    API    │ │
│  │   (FAISS)    │    │    Cache     │    │    Store     │    │  Gateway  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML Training Pipeline                               │
│                        (AWS Step Functions)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐          │
│  │ Preprocessing │───▶│   Parallel Training  │───▶│  Evaluation  │          │
│  │    Lambda    │    │                      │    │    Lambda    │          │
│  └──────────────┘    │ ┌──────────────────┐ │    └──────────────┘          │
│         │            │ │ Two-Tower Train  │ │           │                   │
│         ▼            │ │     Lambda       │ │           ▼                   │
│  ┌──────────────┐    │ └──────────────────┘ │    ┌──────────────┐          │
│  │ Generate     │    │ ┌──────────────────┐ │    │  Deployment  │          │
│  │ Synthetic    │    │ │  Ranker Train    │ │    │    Lambda    │          │
│  │ Data         │    │ │     Lambda       │ │    │ (Conditional)│          │
│  └──────────────┘    │ └──────────────────┘ │    └──────────────┘          │
│                      └──────────────────────┘           │                   │
│                                                         ▼                   │
│                                              ┌──────────────────┐           │
│                                              │ Update DynamoDB  │           │
│                                              │ + S3 Vector Store│           │
│                                              └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
video-recommendation-system/
├── src/
│   ├── config/                     # Configuration management
│   │   ├── feature_config.py       # Feature definitions, embedding dims
│   │   └── settings.py             # Centralized settings with validation
│   │
│   ├── lambdas/                    # Lambda function entry points
│   │   ├── recommendations.py      # Real-time serving handler
│   │   ├── preprocessing.py        # Data preprocessing handler
│   │   ├── training.py             # Model training handlers
│   │   ├── evaluation.py           # Model evaluation handler
│   │   └── deployment.py           # Model deployment handler
│   │
│   ├── data/                       # Data schemas and loaders
│   │   ├── schemas.py              # User, Video, Channel, Interaction dataclasses
│   │   ├── synthetic_generator.py  # Generate synthetic test data
│   │   └── data_loader.py          # Load data from various sources
│   │
│   ├── preprocessing/              # Data preprocessing components
│   │   ├── vocabulary_builder.py   # Categorical vocabulary builders
│   │   ├── normalizers.py          # Numeric normalization
│   │   ├── text_embedder.py        # BERT title embeddings
│   │   ├── tag_embedder.py         # CBOW-style tag embeddings
│   │   └── artifacts.py            # Save/load preprocessing artifacts
│   │
│   ├── feature_engineering/        # Feature transformations
│   │   ├── user_features.py        # User tower feature engineering
│   │   ├── video_features.py       # Video tower feature engineering
│   │   ├── interaction_features.py # Interaction-based features
│   │   └── ranker_features.py      # Ranker model features
│   │
│   ├── models/                     # Model definitions
│   │   ├── two_tower.py            # Two-Tower retrieval model
│   │   ├── ranker.py               # CatBoost ranking model
│   │   ├── trainers.py             # Training loops and callbacks
│   │   └── metrics.py              # Custom evaluation metrics
│   │
│   ├── serving/                    # Model serving components
│   │   ├── recommendation_service.py   # Main recommendation API
│   │   ├── vector_store.py             # FAISS vector store
│   │   ├── feature_store_client.py     # DynamoDB feature client
│   │   ├── redis_cache_client.py       # Redis caching layer
│   │   ├── filtering_service.py        # Business rule filtering
│   │   ├── offline_pipeline.py         # Batch embedding generation
│   │   └── ranker_service_v2.py        # Enhanced ranking service
│   │
│   ├── ml_pipeline/                # MLOps pipeline (Step Functions)
│   │   ├── ml_pipeline.py              # Main pipeline orchestrator
│   │   ├── pipeline_config.py          # Pipeline configuration
│   │   ├── training_orchestrator.py    # Training job orchestration
│   │   ├── preprocessing_job.py        # Data preprocessing jobs
│   │   ├── deployment_pipeline.py      # Model deployment automation
│   │   ├── model_registry.py           # Model versioning
│   │   └── data_versioning.py          # Data versioning and splits
│   │
│   ├── monitoring/                 # Monitoring and observability
│   │   ├── online_metrics.py       # Real-time metric tracking
│   │   ├── performance_monitor.py  # Latency monitoring
│   │   └── ab_testing.py           # A/B testing framework
│   │
│   ├── data_collection/            # Continuous learning
│   │   ├── ground_truth_collector.py   # Collect user feedback
│   │   ├── inference_tracker.py        # Track inference results
│   │   ├── merge_job.py                # Merge predictions with feedback
│   │   └── feedback_loop.py            # Feedback processing
│   │
│   └── utils/                      # Utilities
│       ├── logging_utils.py        # Structured JSON/text logging
│       ├── health.py               # Health checks, circuit breakers
│       └── io_utils.py             # File I/O utilities
│
├── tests/                          # Test suite (250+ tests)
├── scripts/                        # Entry point scripts
│   ├── deploy.sh                   # Full deployment automation
│   ├── bootstrap_data.py           # Generate and populate data
│   ├── run_processing.py           # Run processing pipeline
│   ├── train_two_tower.py          # Train Two-Tower model
│   └── train_ranker.py             # Train Ranker model
│
├── deployment/                     # Infrastructure as Code
│   ├── cdk/                        # AWS CDK (TypeScript)
│   │   └── lib/
│   │       ├── stacks/
│   │       │   └── video-recommendation-stack.ts
│   │       └── constructs/         # CDK constructs
│   │           ├── storage.ts          # S3, DynamoDB
│   │           ├── compute.ts          # Lambda
│   │           ├── api.ts              # API Gateway
│   │           ├── ml-pipeline.ts      # Step Functions ML Pipeline
│   │           ├── data-collection.ts  # Kinesis, Firehose
│   │           ├── feature-store.ts    # Feature Store
│   │           ├── secrets.ts          # AWS Secrets Manager
│   │           └── monitoring.ts       # CloudWatch Dashboards
│   └── docker/                     # Docker configurations
│       └── Dockerfile.lambda       # Lambda container image
│
├── configs/                        # Configuration files
│   └── processing_config.yaml
│
├── .env.example                    # Environment variables template
├── docker-compose.yml              # Local development environment
├── Makefile                        # Development commands
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
└── .pre-commit-config.yaml         # Git hooks
```

## Quick Start

### One-Command Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/video-recommendation-system.git
cd video-recommendation-system

# Deploy to AWS (runs tests, deploys infrastructure, trains models, populates data)
./scripts/deploy.sh
```

This single command will:
1. ✅ Check prerequisites (AWS CLI, Node.js, Python, Docker)
2. ✅ Install Python and CDK dependencies
3. ✅ Run test suite
4. ✅ Deploy AWS infrastructure via CDK
5. ✅ **Automatically run ML training pipeline**:
   - Generate synthetic training data (1000 users, 500 videos, 10000 interactions)
   - Train Two-Tower model and generate video embeddings
   - Train CatBoost Ranker model
   - Evaluate models against quality thresholds
   - Deploy models if evaluation passes
   - Update DynamoDB with video embeddings
6. ✅ Bootstrap test data
7. ✅ Validate the API

**After deployment, you'll have a fully operational recommendation system with trained models!**

### Deploy to Different Environments

```bash
# Development (default)
./scripts/deploy.sh

# Staging
./scripts/deploy.sh staging

# Production
./scripts/deploy.sh prod
```

## ML Pipeline

### Overview

The ML pipeline is fully automated and runs as an AWS Step Functions state machine. It handles:

1. **Preprocessing**: Generates synthetic data, builds vocabularies, creates training datasets
2. **Parallel Training**: Trains Two-Tower and Ranker models concurrently
3. **Evaluation**: Validates model quality against configurable thresholds
4. **Conditional Deployment**: Only deploys models that meet quality criteria

### Pipeline Flow

```
deploy.sh triggers → Step Functions → Preprocessing Lambda
                                              ↓
                          ┌───────────────────┴───────────────────┐
                          ↓                                       ↓
                   Two-Tower Training               Ranker Training
                   (Video Embeddings)               (CatBoost Model)
                          ↓                                       ↓
                          └───────────────────┬───────────────────┘
                                              ↓
                                      Evaluation Lambda
                                     (Quality Checks)
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                              Metrics PASS         Metrics FAIL
                                    ↓                   ↓
                            Deployment Lambda      Pipeline Ends
                                    ↓              (No Deployment)
                          ┌─────────┴─────────┐
                          ↓                   ↓
                   Update DynamoDB      Upload to S3
                   (Video Features)     (Embeddings)
```

### Evaluation Thresholds

Models are evaluated against these default thresholds (configurable):

| Model | Metric | Threshold |
|-------|--------|-----------|
| Two-Tower | Recall@100 | ≥ 0.1 |
| Two-Tower | NDCG | ≥ 0.3 |
| Ranker | AUC | ≥ 0.6 |
| Ranker | NDCG | ≥ 0.3 |

### Manual Pipeline Trigger

```bash
# Trigger pipeline manually
aws stepfunctions start-execution \
  --state-machine-arn <ML_PIPELINE_ARN> \
  --input '{"trigger": "manual"}'

# Check pipeline status
aws stepfunctions describe-execution \
  --execution-arn <EXECUTION_ARN>
```

### Model Artifacts

After successful pipeline execution, artifacts are stored:

```
s3://{model-bucket}/
├── models/
│   ├── two_tower/{version}/
│   │   ├── video_embeddings.npz    # Video embeddings
│   │   ├── model_config.json       # Model configuration
│   │   └── vocabularies.json       # Feature vocabularies
│   └── ranker/{version}/
│       └── ranker_model.cbm        # CatBoost model
├── vector_store/
│   └── video_embeddings.npz        # Production embeddings
└── preprocessing/{job_id}/
    ├── vocabularies.json           # Shared vocabularies
    ├── video_catalog.parquet       # Video metadata
    └── ranker_train.parquet        # Training data
```

### How Inference Uses Trained Models

**Online Inference (Real-time API)**:
- Lambda loads embeddings from `s3://{model-bucket}/vector_store/video_embeddings.npz`
- FAISS index performs similarity search against trained embeddings
- Video features (with embeddings) stored in DynamoDB

**Offline Inference (Batch)**:
- `OfflineInferencePipeline` loads Two-Tower model weights
- Processes videos in batches to generate new embeddings
- Updates vector store for online serving

## Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for local development)
- Node.js 18+ (for AWS CDK deployment)
- AWS CLI configured (for AWS deployment)

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.13.0 | Deep learning framework |
| catboost | >=1.2.0 | Gradient boosting ranker |
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0,<2.0.0 | Numerical computing |
| pyarrow | >=12.0.0,<18.0.0 | Parquet file support |
| boto3 | >=1.28.0 | AWS SDK |
| redis | >=4.5.0,<6.0.0 | Redis client |

Install with optional dependencies:

```bash
# Core dependencies only
pip install -r requirements.txt

# With ML training dependencies
pip install -e ".[ml]"

# With development dependencies
pip install -e ".[dev]"

# All dependencies
pip install -e ".[all]"
```

## Development

### Available Make Commands

```bash
make help                 # Show all available commands

# Installation
make install              # Install production dependencies
make install-dev          # Install dev dependencies + pre-commit hooks
make install-cdk          # Install CDK dependencies

# Testing
make test                 # Run tests
make test-cov             # Run tests with coverage report
make test-fast            # Run tests in parallel

# Code Quality
make lint                 # Run linters (ruff, mypy)
make format               # Format code with ruff
make pre-commit           # Run pre-commit hooks

# Docker
make docker-build         # Build Lambda Docker image
make docker-run           # Run Lambda locally

# Local Development
make local-serve          # Run local recommendation server
make bootstrap-data       # Generate and populate data (AWS)
make bootstrap-data-local # Generate data locally (no AWS)

# Training
make train-two-tower      # Train Two-Tower model
make train-ranker         # Train Ranker model
make process-data         # Run data processing pipeline

# Deployment
make deploy-dev           # Deploy to dev environment
make deploy-staging       # Deploy to staging
make deploy-prod          # Deploy to production (with confirmation)
make deploy-diff          # Preview deployment changes

# Utilities
make clean                # Clean build artifacts
make logs                 # Tail Lambda logs
```

### Local Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# Services available:
# - DynamoDB Local: http://localhost:8000
# - Redis: localhost:6379
# - LocalStack (S3): http://localhost:4566
# - Lambda: http://localhost:8080

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset all data
docker-compose down -v
```

### Pre-commit Hooks

Pre-commit hooks are automatically installed with `make install-dev`. They run:

- **ruff**: Linting and formatting
- **mypy**: Type checking
- **bandit**: Security scanning
- **detect-secrets**: Secrets detection

```bash
# Run manually on all files
pre-commit run --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify
```

## Deployment

### Full Deployment (Recommended)

```bash
# Deploy everything including ML pipeline
./scripts/deploy.sh

# Deploy to specific environment
./scripts/deploy.sh staging
./scripts/deploy.sh prod
```

### CDK-Only Deployment

```bash
# Navigate to CDK directory
cd deployment/cdk

# Install dependencies
npm install

# Bootstrap CDK (first time only)
npx cdk bootstrap

# Deploy to dev
npx cdk deploy --all -c environment=dev

# Or use Make
make deploy-dev
```

### Post-Deployment Validation

After deployment, validate the system:

```bash
# Health check
curl https://<api-url>/health

# Get recommendations
curl https://<api-url>/recommendations/123?n=5

# POST recommendations
curl -X POST https://<api-url>/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 123, "num_recommendations": 10}'
```

### Stack Outputs

After deployment, the following outputs are available:

| Output | Description |
|--------|-------------|
| `ApiUrl` | API Gateway endpoint URL |
| `ModelBucketName` | S3 bucket for model artifacts |
| `DataBucketName` | S3 bucket for training data |
| `ArtifactsBucketName` | S3 bucket for preprocessing artifacts |
| `UserFeaturesTableName` | DynamoDB table for user features |
| `VideoFeaturesTableName` | DynamoDB table for video features |
| `MLPipelineStateMachineArn` | Step Functions ML pipeline ARN |
| `EventStreamName` | Kinesis stream for event ingestion |

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with dependency status |
| `/recommendations/{user_id}` | GET | Get recommendations for user |
| `/recommendations` | POST | Get recommendations with full request |

### Get Recommendations

```http
GET /recommendations/12345?n=20
```

**Response:**
```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "video_id": 101,
      "score": 0.95,
      "category": "gaming"
    }
  ],
  "total_latency_ms": 45
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": [
    {"name": "dynamodb:user-features", "status": "healthy", "latency_ms": 12.5},
    {"name": "redis:cache", "status": "healthy", "latency_ms": 2.3},
    {"name": "vector_store", "status": "healthy", "latency_ms": 0.1}
  ]
}
```

## Configuration

### Environment Variables

All configuration is managed through environment variables. See [.env.example](.env.example) for the complete list.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `dev` | Environment (dev/staging/prod) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `MODEL_BUCKET` | - | S3 bucket for models |
| `ARTIFACTS_BUCKET` | - | S3 bucket for artifacts |
| `USER_FEATURES_TABLE` | - | DynamoDB user features table |
| `VIDEO_FEATURES_TABLE` | - | DynamoDB video features table |
| `USE_REDIS` | `false` | Enable Redis caching |
| `REDIS_HOST` | `localhost` | Redis host |

### Centralized Settings

The application uses a centralized settings system with validation:

```python
from src.config import get_settings, validate_settings

# Get cached settings
settings = get_settings()

# Validate settings at startup
errors = validate_settings(raise_on_error=True)

# Access settings
print(settings.aws.model_bucket)
print(settings.redis.enabled)
print(settings.serving.default_num_recommendations)
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_serving.py -v

# Run tests matching pattern
pytest tests/ -k "ranker" -v

# Run tests in parallel
make test-fast
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| Preprocessing | 85% |
| Feature Engineering | 82% |
| Models | 78% |
| Serving | 80% |
| Monitoring | 75% |

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| P99 Latency | <100ms | End-to-end recommendation |
| Cold Start | <3s | Lambda with ARM64 |
| Throughput | 1000 RPS | Per Lambda instance |
| Two-Tower Inference | <5ms | User embedding generation |
| Vector Search | <10ms | FAISS ANN (100 candidates) |
| Ranker Inference | <20ms | CatBoost scoring |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make check`)
5. Commit your changes (uses [Conventional Commits](https://www.conventionalcommits.org/))
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
