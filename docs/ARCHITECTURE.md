# Architecture Overview

This document provides a detailed overview of the Video Recommendation System architecture.

## Table of Contents

- [System Overview](#system-overview)
- [Recommendation Pipeline](#recommendation-pipeline)
- [Model Architecture](#model-architecture)
- [Data Flow](#data-flow)
- [AWS Infrastructure](#aws-infrastructure)
- [Key Design Decisions](#key-design-decisions)

## System Overview

The Video Recommendation System is a production-ready machine learning system that provides personalized video recommendations. It uses a 4-stage pipeline architecture optimized for both quality and latency.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Video Recommendation System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Candidate  │    │   Filtering  │    │   Ranking    │    │  Final    │ │
│  │  Generation  │───▶│   Service    │───▶│   Service    │───▶│  Ordering │ │
│  │  (Two-Tower) │    │              │    │  (CatBoost)  │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│    100-500 items       50-200 items        20-50 items         10-20 items │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Vector Store │    │    Redis     │    │   Feature    │    │    API    │ │
│  │   (FAISS)    │    │    Cache     │    │    Store     │    │  Gateway  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Recommendation Pipeline

### Stage 1: Candidate Generation

**Purpose**: Efficiently retrieve a manageable set of candidates from millions of videos.

**Implementation**: Two-Tower Neural Network
- **User Tower**: Encodes user features into a dense embedding
- **Video Tower**: Encodes video features into a dense embedding
- **Similarity**: Dot product similarity in shared embedding space

**Key Components**:
- [src/models/two_tower.py](../src/models/two_tower.py) - Model definition
- [src/serving/vector_store.py](../src/serving/vector_store.py) - FAISS index for ANN search

**Performance**:
- Input: All videos (~millions)
- Output: 100-500 candidates
- Latency: <10ms (ANN search)

### Stage 2: Filtering

**Purpose**: Apply business rules and content policies.

**Filters Applied**:
1. **Watched Filter**: Remove videos user has already watched
2. **Content Policy**: Remove age-restricted content if needed
3. **Availability**: Remove unavailable or deleted videos
4. **Freshness**: Apply recency requirements

**Key Components**:
- [src/serving/filtering_service.py](../src/serving/filtering_service.py) - Business rule filters

**Performance**:
- Input: 100-500 candidates
- Output: 50-200 candidates
- Latency: <5ms

### Stage 3: Ranking

**Purpose**: Score candidates using rich features for final ordering.

**Implementation**: CatBoost Ranker
- Gradient boosting model optimized for ranking
- Native categorical feature support
- Fast inference (<20ms for 200 items)

**Features Used**:
- User features: demographics, watch history, preferences
- Video features: category, duration, popularity, engagement metrics
- Context features: time of day, device, location
- Interaction features: historical user-video interactions

**Key Components**:
- [src/models/ranker.py](../src/models/ranker.py) - CatBoost model
- [src/serving/ranker_service_v2.py](../src/serving/ranker_service_v2.py) - Ranking service
- [src/feature_engineering/ranker_features.py](../src/feature_engineering/ranker_features.py) - Feature transformations

**Performance**:
- Input: 50-200 candidates
- Output: 20-50 ranked items
- Latency: <20ms

### Stage 4: Final Ordering

**Purpose**: Apply business logic for final presentation.

**Optimizations Applied**:
1. **Diversity**: Ensure category diversity in top results
2. **Freshness Boost**: Promote new content
3. **Promotional Slots**: Insert sponsored content
4. **A/B Testing**: Apply experiment variants

**Key Components**:
- [src/serving/recommendation_service.py](../src/serving/recommendation_service.py) - Final ordering logic

**Performance**:
- Input: 20-50 ranked items
- Output: 10-20 final recommendations
- Latency: <5ms

## Model Architecture

### Two-Tower Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      Two-Tower Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Tower                          Video Tower                 │
│  ┌─────────────────┐                ┌─────────────────┐         │
│  │   user_id       │                │   video_id      │         │
│  │   country       │                │   category      │         │
│  │   language      │                │   language      │         │
│  │   age           │                │   duration      │         │
│  │   watch_history │                │   title (BERT)  │         │
│  └────────┬────────┘                │   tags          │         │
│           │                         │   popularity    │         │
│           ▼                         └────────┬────────┘         │
│  ┌─────────────────┐                         │                  │
│  │  Embedding      │                         ▼                  │
│  │  Layers         │                ┌─────────────────┐         │
│  └────────┬────────┘                │  Embedding      │         │
│           │                         │  Layers         │         │
│           ▼                         └────────┬────────┘         │
│  ┌─────────────────┐                         │                  │
│  │  Dense Layers   │                         ▼                  │
│  │  (256 → 128)    │                ┌─────────────────┐         │
│  └────────┬────────┘                │  Dense Layers   │         │
│           │                         │  (256 → 128)    │         │
│           ▼                         └────────┬────────┘         │
│  ┌─────────────────┐                         │                  │
│  │  User Embedding │                         ▼                  │
│  │  (64-dim)       │                ┌─────────────────┐         │
│  └────────┬────────┘                │  Video Embedding│         │
│           │                         │  (64-dim)       │         │
│           │                         └────────┬────────┘         │
│           │                                  │                  │
│           └──────────────┬───────────────────┘                  │
│                          │                                      │
│                          ▼                                      │
│                 ┌─────────────────┐                             │
│                 │   Dot Product   │                             │
│                 │   Similarity    │                             │
│                 └─────────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Training**:
- Loss: Contrastive loss with in-batch negatives
- Positive samples: User interactions (watch, like, comment)
- Negative samples: Random videos from same batch

### CatBoost Ranker

**Model Configuration**:
```python
CatBoostRanker(
    loss_function='YetiRank',
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
)
```

**Feature Categories**:

| Category | Features | Type |
|----------|----------|------|
| User Demographics | age, country, language | Categorical/Numeric |
| User Behavior | watch_count, avg_watch_time, favorite_categories | Numeric |
| Video Static | category, duration, upload_age | Categorical/Numeric |
| Video Engagement | views, likes, comments, ctr | Numeric |
| Context | hour, day_of_week, device | Categorical |
| Cross Features | user_category_affinity, recency_score | Numeric |

## Data Flow

### Online Serving Flow

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Client  │───▶│ API Gateway │───▶│   Lambda    │───▶│   Response  │
└──────────┘    └─────────────┘    └──────┬──────┘    └─────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
            │   Redis     │       │  DynamoDB   │       │  S3/FAISS   │
            │   Cache     │       │  Features   │       │  Vector DB  │
            └─────────────┘       └─────────────┘       └─────────────┘
```

### Offline Training Flow

The ML pipeline is **fully automated** and runs during deployment via `./scripts/deploy.sh`. It can also be triggered manually.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML Pipeline (Step Functions)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌─────────────────────────┐    ┌────────────┐ │
│  │ Preprocessing│    │  Training (Parallel)    │    │ Evaluation │ │
│  │    Lambda    │───▶│  ┌─────────────────┐    │───▶│   Lambda   │ │
│  │              │    │  │ Two-Tower Lambda│    │    │            │ │
│  │  • Synthetic │    │  │ (Embeddings)    │    │    │ • Metrics  │ │
│  │    Data Gen  │    │  └─────────────────┘    │    │ • Threshold│ │
│  │  • Vocabularies   │  ┌─────────────────┐    │    │ • Decision │ │
│  │  • Training  │    │  │ Ranker Lambda   │    │    └─────┬──────┘ │
│  │    Datasets  │    │  │ (CatBoost)      │    │          │        │
│  └──────────────┘    │  └─────────────────┘    │          │        │
│                      └─────────────────────────┘          │        │
│                                                           │        │
│                                              ┌────────────▼──────┐ │
│                                              │    Deployment     │ │
│                                              │      Lambda       │ │
│                                              │  (Conditional)    │ │
│                                              │                   │ │
│                                              │ • Copy to prod S3 │ │
│                                              │ • Update DynamoDB │ │
│                                              │ • Update registry │ │
│                                              └───────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline Trigger Methods**:
- **Automatic**: Runs during `./scripts/deploy.sh`
- **Manual**: `aws stepfunctions start-execution --state-machine-arn <ARN> --input '{"trigger": "manual"}'`
- **Scheduled**: Weekly via EventBridge (production only)

**Evaluation Thresholds** (configurable):
| Model | Metric | Threshold |
|-------|--------|-----------|
| Two-Tower | Recall@100 | ≥ 0.1 |
| Two-Tower | NDCG | ≥ 0.3 |
| Ranker | AUC | ≥ 0.6 |
| Ranker | NDCG | ≥ 0.3 |

### Data Collection Flow

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Client  │───▶│   Kinesis   │───▶│  Firehose   │───▶│     S3      │
│  Events  │    │   Stream    │    │  Delivery   │    │  (Parquet)  │
└──────────┘    └──────┬──────┘    └─────────────┘    └─────────────┘
                       │
                       ▼
               ┌─────────────┐
               │  Processing │
               │   Lambda    │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  DynamoDB   │
               │  Tracking   │
               └─────────────┘
```

## AWS Infrastructure

### Resource Overview

| Layer | Resources | Purpose |
|-------|-----------|---------|
| **API** | API Gateway, Lambda | Request handling |
| **Compute** | Lambda (ARM64) | Model inference |
| **Storage** | S3, DynamoDB | Models, features, data |
| **Cache** | ElastiCache Redis | Response caching |
| **ML Pipeline** | Step Functions, SageMaker | Training orchestration |
| **Data** | Kinesis, Firehose, Glue | Event collection |
| **Monitoring** | CloudWatch, X-Ray | Observability |
| **Security** | Secrets Manager, IAM | Credentials |

### Environment Configurations

| Config | Dev | Staging | Prod |
|--------|-----|---------|------|
| Lambda Memory | 1024 MB | 1024 MB | 2048 MB |
| Lambda Architecture | ARM64 | ARM64 | ARM64 |
| VPC | No | Yes | Yes |
| Redis | No | t3.small | r6g.large |
| DynamoDB | On-Demand | On-Demand | Provisioned |
| Alarms | No | No | Yes |

## Key Design Decisions

### 1. Two-Stage Retrieval

**Decision**: Use Two-Tower for candidate generation, CatBoost for ranking.

**Rationale**:
- Two-Tower enables efficient ANN search over millions of items
- CatBoost provides accurate ranking with rich features
- Separation allows independent optimization of each stage

**Trade-offs**:
- Two models to train and maintain
- Potential for cascade errors between stages

### 2. ARM64 Lambda

**Decision**: Use ARM64 (Graviton2) for Lambda functions.

**Rationale**:
- 20% better price/performance
- Lower cold start times
- Native TensorFlow Lite support

**Trade-offs**:
- Some dependencies may not have ARM builds
- Requires ARM-compatible Docker images

### 3. FAISS for Vector Search

**Decision**: Use FAISS with IVF index for approximate nearest neighbor search.

**Rationale**:
- Proven performance at scale
- Good accuracy/latency trade-off
- Fits in Lambda memory (<500MB index)

**Trade-offs**:
- Index must be rebuilt for new videos
- Not a managed service (vs. OpenSearch, Pinecone)

### 4. DynamoDB for Feature Store

**Decision**: Use DynamoDB for online feature serving.

**Rationale**:
- Single-digit millisecond latency
- Serverless, scales automatically
- Cost-effective for read-heavy workloads

**Trade-offs**:
- Limited query flexibility
- No joins (denormalized data)

### 5. Step Functions for ML Pipeline

**Decision**: Use Step Functions for training orchestration.

**Rationale**:
- Built-in parallel execution
- Error handling and retries
- Visual workflow monitoring
- Native AWS service integration

**Trade-offs**:
- AWS-specific (not portable)
- Limited compute in state transitions

### 6. Centralized Configuration

**Decision**: Use environment variables with validation and Secrets Manager.

**Rationale**:
- 12-factor app compliance
- Easy environment-specific configuration
- Secure credential management

**Trade-offs**:
- Requires restart for config changes
- More complex than hard-coded values

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| P50 Latency | <50ms | ~45ms |
| P99 Latency | <100ms | ~95ms |
| Cold Start | <3s | ~2.5s |
| Throughput | 1000 RPS | 1000+ RPS |
| Availability | 99.9% | 99.9% |

## Future Improvements

1. **Real-time Feature Updates**: Stream processing for instant feature updates
2. **Multi-Objective Ranking**: Optimize for multiple objectives (engagement, diversity)
3. **Contextual Bandits**: Online learning for exploration/exploitation
4. **Embedding Updates**: Incremental vector index updates
5. **Edge Caching**: CloudFront integration for global latency reduction
