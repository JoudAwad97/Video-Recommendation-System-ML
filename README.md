# Video Recommendation System

A production-ready video recommendation system using a Two-Tower architecture for candidate retrieval and CatBoost Ranker for personalized ranking. Built with TensorFlow, designed for AWS deployment.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Data Processing](#data-processing)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
  - [Training Pipeline](#training-pipeline)
  - [Serving](#serving)
  - [Monitoring](#monitoring)
  - [Data Collection](#data-collection)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
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
- **Feature Store Integration**: SageMaker Feature Store for online/offline feature serving
- **A/B Testing Framework**: Built-in experimentation and metric tracking
- **MLOps Pipeline**: Automated training, versioning, and deployment

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

## Project Structure

```
video-recommendation-system/
├── src/
│   ├── config/                     # Feature and model configurations
│   │   ├── feature_config.py       # Feature definitions, embedding dims
│   │   └── __init__.py
│   │
│   ├── data/                       # Data schemas and loaders
│   │   ├── schemas.py              # User, Video, Channel, Interaction dataclasses
│   │   ├── synthetic_generator.py  # Generate synthetic test data
│   │   ├── data_loader.py          # Load data from various sources
│   │   └── __init__.py
│   │
│   ├── preprocessing/              # Data preprocessing components
│   │   ├── vocabulary_builder.py   # Categorical vocabulary builders
│   │   ├── normalizers.py          # Numeric normalization (z-score, log, buckets)
│   │   ├── text_embedder.py        # BERT title embeddings via TF Hub
│   │   ├── tag_embedder.py         # CBOW-style tag embeddings
│   │   ├── artifacts.py            # Save/load preprocessing artifacts
│   │   └── __init__.py
│   │
│   ├── feature_engineering/        # Feature transformations
│   │   ├── user_features.py        # User tower feature engineering
│   │   ├── video_features.py       # Video tower feature engineering
│   │   ├── interaction_features.py # Interaction-based features
│   │   ├── ranker_features.py      # Ranker model features
│   │   └── __init__.py
│   │
│   ├── dataset/                    # Dataset generation
│   │   ├── two_tower_dataset.py    # Two-Tower training pairs
│   │   ├── ranker_dataset.py       # Ranker pos/neg samples
│   │   ├── tf_dataset_builder.py   # TensorFlow Dataset builders
│   │   └── __init__.py
│   │
│   ├── models/                     # Model definitions
│   │   ├── two_tower.py            # Two-Tower retrieval model
│   │   ├── ranker.py               # CatBoost ranking model
│   │   ├── trainers.py             # Training loops and callbacks
│   │   ├── metrics.py              # Custom evaluation metrics
│   │   ├── model_config.py         # Model hyperparameters
│   │   └── __init__.py
│   │
│   ├── pipeline/                   # Data processing pipeline
│   │   ├── processing_pipeline.py  # Main orchestration
│   │   ├── pipeline_config.py      # Pipeline configuration
│   │   └── __init__.py
│   │
│   ├── ml_pipeline/                # MLOps pipeline
│   │   ├── ml_pipeline.py          # End-to-end ML pipeline
│   │   ├── training_orchestrator.py# Training job orchestration
│   │   ├── preprocessing_job.py    # Data preprocessing jobs
│   │   ├── deployment_pipeline.py  # Model deployment automation
│   │   ├── model_registry.py       # Model versioning and registry
│   │   ├── data_versioning.py      # Dataset versioning
│   │   ├── pipeline_config.py      # Pipeline configuration
│   │   └── __init__.py
│   │
│   ├── serving/                    # Model serving components
│   │   ├── recommendation_service.py   # Main recommendation API
│   │   ├── lambda_handler.py           # AWS Lambda entry point
│   │   ├── feature_store_client.py     # SageMaker Feature Store client
│   │   ├── redis_cache_client.py       # Redis caching layer
│   │   ├── multi_query_generator.py    # Diverse candidate generation
│   │   ├── filtering_service.py        # Business rule filtering
│   │   ├── ranker_service_v2.py        # Enhanced ranking service
│   │   ├── vector_store.py             # FAISS vector store
│   │   ├── offline_pipeline.py         # Batch embedding generation
│   │   ├── candidate_retrieval.py      # ANN candidate retrieval
│   │   ├── query_encoder.py            # User query encoding
│   │   ├── ranking_service.py          # Ranking service
│   │   ├── orchestrator.py             # Service orchestration
│   │   ├── sagemaker_deployment.py     # SageMaker deployment
│   │   ├── serving_config.py           # Serving configuration
│   │   └── __init__.py
│   │
│   ├── monitoring/                 # Monitoring and observability
│   │   ├── online_metrics.py       # Real-time metric tracking
│   │   ├── performance_monitor.py  # Latency and throughput monitoring
│   │   ├── data_quality_monitor.py # Data drift detection
│   │   ├── data_capture.py         # Inference data logging
│   │   ├── ab_testing.py           # A/B testing framework
│   │   ├── ranker_sampler.py       # Ranker performance sampling
│   │   ├── monitoring_config.py    # Monitoring configuration
│   │   └── __init__.py
│   │
│   ├── data_collection/            # Continuous learning
│   │   ├── ground_truth_collector.py   # Collect user feedback
│   │   ├── inference_tracker.py        # Track inference results
│   │   ├── feedback_loop.py            # Feedback processing
│   │   ├── merge_job.py                # Merge feedback with training data
│   │   ├── collection_config.py        # Collection configuration
│   │   └── __init__.py
│   │
│   └── utils/                      # Utilities
│       ├── io_utils.py             # File I/O (JSON, Parquet, TFRecord)
│       ├── logging_utils.py        # Structured logging
│       └── __init__.py
│
├── tests/                          # Unit and integration tests (252 tests)
│   ├── test_vocabulary_builder.py
│   ├── test_normalizers.py
│   ├── test_text_embedder.py
│   ├── test_two_tower.py
│   ├── test_ranker.py
│   ├── test_serving.py
│   ├── test_monitoring.py
│   └── ...
│
├── configs/                        # Configuration files
│   └── processing_config.yaml
│
├── scripts/                        # Entry point scripts
│   └── run_processing.py
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/video-recommendation-system.git
cd video-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For AWS deployment (optional)
pip install boto3 sagemaker

# For vector search (optional)
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.13.0 | Deep learning framework |
| tensorflow-hub | >=0.14.0 | Pre-trained BERT models |
| catboost | >=1.2.0 | Gradient boosting ranker |
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| pyarrow | >=12.0.0 | Parquet file support |
| scikit-learn | >=1.3.0 | ML utilities |
| pytest | >=7.0.0 | Testing framework |

## Quick Start

### 1. Generate and Process Synthetic Data

```bash
# Run with synthetic data (for testing/development)
python scripts/run_processing.py --synthetic

# With custom parameters
python scripts/run_processing.py --synthetic \
    --users 5000 \
    --videos 2000 \
    --interactions 50000 \
    --negative-ratio 3
```

### 2. Train Models

```python
from src.models.two_tower import TwoTowerModel
from src.models.ranker import CatBoostRanker
from src.models.trainers import TwoTowerTrainer, RankerTrainer

# Train Two-Tower model
two_tower = TwoTowerModel(config)
trainer = TwoTowerTrainer(two_tower, train_dataset, val_dataset)
trainer.train(epochs=10)

# Train Ranker model
ranker = CatBoostRanker(config)
ranker_trainer = RankerTrainer(ranker, train_df, val_df)
ranker_trainer.train()
```

### 3. Serve Recommendations

```python
from src.serving import RecommendationService, RecommendationServiceConfig

# Initialize service
config = RecommendationServiceConfig()
service = RecommendationService(config)
service.initialize(
    model_path="models/two_tower",
    artifacts_path="artifacts/"
)

# Get recommendations
response = service.get_recommendations(
    user_id=12345,
    num_recommendations=20
)

for rec in response.recommendations:
    print(f"Video {rec.video_id}: {rec.score:.3f}")
```

## Modules

### Data Processing

The data processing module handles raw data ingestion and transformation.

```python
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.schemas import User, Video, Interaction

# Generate synthetic data
generator = SyntheticDataGenerator(seed=42)
users = generator.generate_users(num_users=1000)
videos = generator.generate_videos(num_videos=500)
interactions = generator.generate_interactions(
    users=users,
    videos=videos,
    num_interactions=10000
)
```

**Input Data Schemas:**

| Entity | Required Fields |
|--------|-----------------|
| User | user_id, country, language, age |
| Video | video_id, title, category, duration, tags |
| Channel | channel_id, subscriber_count |
| Interaction | user_id, video_id, timestamp, interaction_type |

### Feature Engineering

#### Two-Tower Features

**User Tower:**
| Feature | Transformation | Output |
|---------|----------------|--------|
| user_id | IntegerLookup | Embedding index |
| country | StringLookup | Embedding index |
| user_language | StringLookup (shared) | Embedding index |
| age | Normalize + Bucket | [normalized, bucket_idx] |
| previously_watched_category | StringLookup | Embedding index |

**Video Tower:**
| Feature | Transformation | Output |
|---------|----------------|--------|
| video_id | IntegerLookup | Embedding index |
| category | StringLookup | Embedding index |
| title | BERT (TF Hub) | float[384] |
| video_duration | Log + Normalize + Bucket | [log_norm, bucket_idx] |
| popularity | One-hot | float[4] |
| video_language | StringLookup (shared) | Embedding index |
| tags | CBOW embedding | float[100] |

#### Ranker Features

**Categorical Features (CatBoost native):**
- country, user_language, category, child_categories
- video_language, interaction_time_day, device

**Numeric Features:**
| Feature | Transformation |
|---------|----------------|
| age | normalize + bucket |
| video_duration | log + normalize |
| view_count | log transform |
| like_count | log + ratio (likes/subscribers) |
| comment_count | log + ratio |
| channel_subscriber_count | log + tier bin |
| interaction_time_hour | cyclical (sin/cos) |

### Models

#### Two-Tower Model

The Two-Tower architecture creates separate encoders for users and videos that project them into a shared embedding space.

```python
from src.models.two_tower import TwoTowerModel, TwoTowerConfig

config = TwoTowerConfig(
    user_embedding_dim=64,
    video_embedding_dim=64,
    hidden_layers=[256, 128],
    output_dim=64,
    temperature=0.05,
)

model = TwoTowerModel(config)
```

**Architecture:**
```
User Features                    Video Features
      │                               │
      ▼                               ▼
┌─────────────┐               ┌─────────────┐
│  User Tower │               │ Video Tower │
│  (MLP)      │               │  (MLP)      │
└─────────────┘               └─────────────┘
      │                               │
      ▼                               ▼
 User Embedding              Video Embedding
      │                               │
      └───────────┬───────────────────┘
                  ▼
            Dot Product
                  │
                  ▼
           Similarity Score
```

#### CatBoost Ranker

The ranker model scores candidate videos using rich interaction features.

```python
from src.models.ranker import CatBoostRanker, RankerConfig

config = RankerConfig(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
)

ranker = CatBoostRanker(config)
ranker.train(train_df, val_df, categorical_features=[...])
```

### Training Pipeline

The ML pipeline provides end-to-end training orchestration.

```python
from src.ml_pipeline import MLPipeline, MLPipelineConfig

config = MLPipelineConfig(
    experiment_name="recommendation_v1",
    data_version="2024-01-15",
    two_tower_epochs=10,
    ranker_iterations=1000,
)

pipeline = MLPipeline(config)
pipeline.run(
    train_data_path="data/train",
    val_data_path="data/val",
    output_path="models/"
)
```

**Pipeline Stages:**
1. Data preprocessing and validation
2. Feature engineering
3. Two-Tower model training
4. Video embedding generation
5. Vector index building
6. Ranker model training
7. Model evaluation
8. Model registration
9. Deployment (optional)

### Serving

#### RecommendationService

The main serving component that orchestrates the recommendation flow.

```python
from src.serving import (
    RecommendationService,
    RecommendationServiceConfig,
    ServingConfig,
    FeatureStoreConfig,
    RedisCacheConfig,
)

config = RecommendationServiceConfig(
    serving=ServingConfig(
        num_candidates=200,
        num_recommendations=20,
    ),
    feature_store=FeatureStoreConfig(
        user_feature_group="user-features",
        video_feature_group="video-features",
    ),
    cache=RedisCacheConfig(
        host="localhost",
        port=6379,
    ),
)

service = RecommendationService(config)
service.initialize()

# Get recommendations
response = service.get_recommendations(
    user_id=12345,
    num_recommendations=20,
    excluded_video_ids={101, 102, 103},
    user_preferences={"preferred_categories": ["gaming", "music"]},
)
```

#### Lambda Handler

AWS Lambda entry point for API Gateway integration.

```python
# lambda_handler.py is automatically invoked by AWS Lambda
# Configure via environment variables:
#   - USE_SAGEMAKER_FEATURE_STORE=true
#   - REDIS_HOST=your-elasticache-endpoint
#   - MODEL_PATH=/opt/ml/model
```

**Supported Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/{user_id}` | GET | Get recommendations for user |
| `/recommendations` | POST | Get recommendations with full request |
| `/interactions` | POST | Record user interaction |
| `/health` | GET | Health check |
| `/cached/{user_id}` | GET | Get cached recommendations |

### Monitoring

#### Online Metrics

Track real-time recommendation performance.

```python
from src.monitoring import OnlineMetricsTracker, MetricsConfig

tracker = OnlineMetricsTracker(MetricsConfig())

# Record events
tracker.record_impression(user_id, video_ids)
tracker.record_click(user_id, video_id, position)
tracker.record_watch(user_id, video_id, watch_duration, video_duration)

# Get metrics
metrics = tracker.get_metrics(window_minutes=60)
print(f"CTR: {metrics['click_through_rate']:.3%}")
print(f"Watch Rate: {metrics['watch_rate']:.3%}")
```

#### A/B Testing

Run experiments to compare model versions.

```python
from src.monitoring import ABTestingFramework, Experiment

framework = ABTestingFramework()

# Create experiment
experiment = Experiment(
    name="ranker_v2_test",
    variants=["control", "treatment"],
    traffic_split=[0.5, 0.5],
)
framework.register_experiment(experiment)

# Assign user to variant
variant = framework.get_variant("ranker_v2_test", user_id)

# Record outcome
framework.record_outcome("ranker_v2_test", user_id, clicked=True)

# Analyze results
results = framework.analyze("ranker_v2_test")
print(f"Treatment lift: {results['treatment_lift']:.2%}")
print(f"P-value: {results['p_value']:.4f}")
```

### Data Collection

#### Ground Truth Collector

Collect user feedback for continuous learning.

```python
from src.data_collection import GroundTruthCollector, CollectionConfig

collector = GroundTruthCollector(CollectionConfig())

# Log inference
collector.log_inference(
    request_id="req-123",
    user_id=12345,
    recommended_videos=[101, 102, 103],
    scores=[0.95, 0.87, 0.82],
)

# Log user feedback
collector.log_feedback(
    request_id="req-123",
    user_id=12345,
    video_id=101,
    feedback_type="click",
)
```

## API Reference

### REST API

#### Get Recommendations

```http
GET /recommendations/{user_id}?n=20&exclude_watched=true
```

**Response:**
```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "video_id": 101,
      "score": 0.95,
      "category": "gaming",
      "source": "personalized"
    }
  ],
  "metadata": {
    "latency_ms": 45,
    "candidates_retrieved": 200,
    "candidates_after_filtering": 150
  }
}
```

#### Record Interaction

```http
POST /interactions
Content-Type: application/json

{
  "user_id": 12345,
  "video_id": 101,
  "category": "gaming",
  "interaction_type": "watch",
  "duration_watched": 180.5
}
```

## Configuration

### Processing Configuration

```yaml
# configs/processing_config.yaml
embeddings:
  user_id_dim: 32
  video_id_dim: 32
  category_dim: 32
  country_dim: 16
  language_dim: 16

buckets:
  age: [18, 25, 35, 45, 55, 65]
  duration: [60, 300, 600, 1800, 3600]
  subscriber_tiers: [1000, 10000, 100000, 1000000]

ranker:
  negative_sample_ratio: 3
  min_watch_ratio: 0.4

two_tower:
  hidden_layers: [256, 128]
  output_dim: 64
  temperature: 0.05
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/opt/ml/model` | Path to model artifacts |
| `ARTIFACTS_PATH` | `/opt/ml/artifacts` | Path to preprocessing artifacts |
| `USE_SAGEMAKER_FEATURE_STORE` | `false` | Enable SageMaker Feature Store |
| `USE_REDIS` | `false` | Enable Redis caching |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

Run the test suite:

```bash
# Run all tests (252 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_two_tower.py -v

# Run tests matching pattern
pytest tests/ -k "ranker" -v
```

### Test Categories

| Module | Tests | Description |
|--------|-------|-------------|
| Preprocessing | 45 | Vocabulary, normalizers, embedders |
| Feature Engineering | 38 | User, video, ranker features |
| Models | 52 | Two-Tower, Ranker training |
| Serving | 67 | API, caching, filtering |
| Monitoring | 35 | Metrics, A/B testing |
| Data Collection | 15 | Feedback loop |

## Deployment

### AWS Lambda Deployment

1. **Package the application:**
```bash
# Install dependencies to package directory
pip install -r requirements.txt -t package/
cp -r src package/
cd package && zip -r ../deployment.zip .
```

2. **Create Lambda function:**
```bash
aws lambda create-function \
    --function-name video-recommendations \
    --runtime python3.9 \
    --handler src.serving.lambda_handler.handler \
    --role arn:aws:iam::ACCOUNT:role/lambda-role \
    --zip-file fileb://deployment.zip \
    --timeout 30 \
    --memory-size 1024
```

3. **Configure API Gateway:**
```bash
# Create REST API with Lambda proxy integration
aws apigateway create-rest-api --name "recommendations-api"
```

### SageMaker Deployment

```python
from src.serving import SageMakerDeployer

deployer = SageMakerDeployer(
    role="arn:aws:iam::ACCOUNT:role/sagemaker-role",
    instance_type="ml.m5.xlarge",
)

# Deploy Two-Tower model
deployer.deploy_two_tower(
    model_path="s3://bucket/models/two_tower",
    endpoint_name="two-tower-endpoint",
)

# Deploy Ranker model
deployer.deploy_ranker(
    model_path="s3://bucket/models/ranker",
    endpoint_name="ranker-endpoint",
)
```

### Infrastructure Requirements

| Component | AWS Service | Specifications |
|-----------|-------------|----------------|
| API | API Gateway + Lambda | 1024MB memory, 30s timeout |
| Feature Store | SageMaker Feature Store | Online + Offline store |
| Cache | ElastiCache (Redis) | cache.r6g.large |
| Vector Store | OpenSearch | r6g.large.search |
| Model Hosting | SageMaker Endpoints | ml.m5.xlarge |
| Storage | S3 | Standard tier |

## Positive Interaction Rules

An interaction is considered positive for training if any of these conditions are met:

| Condition | Weight | Description |
|-----------|--------|-------------|
| Liked | 1.0 | User explicitly liked the video |
| Commented | 1.0 | User left a comment |
| Shared | 1.0 | User shared the video |
| Clicked | 0.8 | User clicked on the video |
| Watched >40% | 0.7 | User watched more than 40% of duration |
| Impression | 0.3 | Video was shown to user |

## Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| P99 Latency | <100ms | End-to-end recommendation |
| Throughput | 1000 RPS | Per Lambda instance |
| Two-Tower Inference | <5ms | User embedding generation |
| Vector Search | <10ms | FAISS ANN search (100 candidates) |
| Ranker Inference | <20ms | CatBoost scoring (100 items) |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.
