# Video Recommendation System - Feature Engineering & Data Processing

A comprehensive feature engineering and data processing pipeline for building a video recommendation system using the Two-Tower architecture and a Ranker model.

## Architecture Overview

This pipeline implements a 4-stage recommendation system:

1. **Stage 1 - Candidate Generation**: Two-Tower model retrieves ~100s candidates from millions
2. **Stage 2 - Filtering**: Business rules filter out seen/irrelevant items
3. **Stage 3 - Ranking**: Ranker model scores candidates with rich features
4. **Stage 4 - Ordering**: Final ordering based on scores + business logic

## Project Structure

```
video-recommendation-system/
├── src/
│   ├── config/                    # Feature configurations
│   ├── data/                      # Data schemas and loaders
│   ├── preprocessing/             # Vocabularies, normalizers, embedders
│   ├── feature_engineering/       # User, video, ranker transformers
│   ├── dataset/                   # Dataset generators
│   ├── pipeline/                  # Pipeline orchestration
│   └── utils/                     # I/O and logging utilities
├── tests/                         # Unit tests
├── configs/                       # YAML configuration files
├── scripts/                       # Entry point scripts
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run with Synthetic Data (for testing)

```bash
python scripts/run_processing.py --synthetic
```

### Run with Custom Parameters

```bash
python scripts/run_processing.py --synthetic \
    --users 5000 \
    --videos 2000 \
    --interactions 50000 \
    --negative-ratio 3
```

### Run with Real Data

```bash
python scripts/run_processing.py --data-dir data/raw
```

Expected input files:
- `users.parquet` - User data
- `videos.parquet` - Video data
- `channels.parquet` - Channel data
- `interactions.parquet` - User-video interactions

## Features

### Two-Tower Model Features

**User Tower:**
- `user_id` → IntegerLookup + Embedding
- `country` → StringLookup + Embedding
- `user_language` → StringLookup + Embedding (shared)
- `age` → Normalized + Bucket embedding
- `previously_watched_category` → StringLookup + Embedding

**Video Tower:**
- `video_id` → IntegerLookup + Embedding
- `category` → StringLookup + Embedding
- `title` → Pre-computed BERT embedding
- `video_duration` → Log + Normalize + Bucket
- `popularity` → One-hot encoding
- `video_language` → StringLookup + Embedding (shared)
- `tags` → CBOW-style embedding

### Ranker Model Features

**Categorical (CatBoost native):**
- country, user_language, category, child_categories
- video_language, interaction_time_day, device

**Numeric (with transformations):**
- age, video_duration, view_count, like_count, comment_count
- channel_subscriber_count, interaction_time_hour (cyclical)
- Derived: like_ratio, comment_ratio, engagement_rate

## Output

### Processed Datasets

```
processed_data/
├── two_tower/
│   ├── train.parquet      # Positive user-video pairs
│   ├── val.parquet
│   └── test.parquet
└── ranker/
    ├── train.parquet      # Labeled pos/neg samples
    ├── val.parquet
    └── test.parquet
```

### Artifacts

```
artifacts/
├── vocabularies/          # Categorical feature vocabularies
├── normalizers/           # Normalization statistics
├── buckets/               # Bucket boundaries
└── embeddings/            # Pre-computed embeddings
```

## Running Tests

```bash
pytest tests/ -v
```

## Usage Example

```python
from src.pipeline.processing_pipeline import ProcessingPipeline
from src.pipeline.pipeline_config import PipelineConfig

# Configure pipeline
config = PipelineConfig(
    artifacts_dir="artifacts",
    processed_data_dir="processed_data",
    negative_ratio=3,
)

# Run pipeline with synthetic data
pipeline = ProcessingPipeline(pipeline_config=config)
results = pipeline.run_synthetic(
    num_users=1000,
    num_videos=500,
    num_interactions=10000
)

# Access results
train_df = results["two_tower_train"]
ranker_train = results["ranker_train"]

# Get feature specs for model building
feature_specs = pipeline.get_feature_specs()
```

## Configuration

See `configs/processing_config.yaml` for all available options:

```yaml
embeddings:
  user_id_dim: 32
  video_id_dim: 32
  category_dim: 32

buckets:
  age: [18, 25, 35, 45, 55, 65]
  duration: [60, 300, 600, 1800, 3600]

ranker:
  negative_sample_ratio: 3
  min_watch_ratio: 0.4
```

## Positive Interaction Rules

An interaction is considered positive if:
- User **liked** the video
- User **commented** on the video
- User **clicked** on the video
- User **shared** the video
- User **watched > 40%** of the video duration
- User had an **impression** (light engagement)

## Next Steps

After processing, the outputs can be used to:
1. Train a Two-Tower retrieval model (TensorFlow/Keras)
2. Train a Ranker model (CatBoost/XGBoost/Neural Network)
3. Build a vector index for candidate retrieval (FAISS/Pinecone)
4. Set up a feature store for real-time inference

## License

MIT
