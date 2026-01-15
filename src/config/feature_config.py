"""
Feature configuration for Two-Tower and Ranker models.

This module defines all feature names, embedding dimensions, bucket boundaries,
and other configuration constants used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =============================================================================
# Constants
# =============================================================================

# Categorical value constants
POPULARITY_LEVELS = ["low", "medium", "high", "viral"]
DEVICE_TYPES = ["Mobile", "Desktop", "Laptop", "Tablet"]
DAYS_OF_WEEK = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

# Interaction types and positive interaction rules
INTERACTION_TYPES = ["like", "dislike", "comment", "watch", "click", "impression", "share"]
POSITIVE_INTERACTION_TYPES = ["like", "comment", "click", "share"]  # watch requires duration check

# Special tokens
START_TOKEN = "[START]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding dimensions."""

    # ID embeddings (high cardinality)
    user_id_dim: int = 32
    video_id_dim: int = 32

    # Categorical embeddings (low-medium cardinality)
    country_dim: int = 16
    language_dim: int = 16  # Shared between user and video
    category_dim: int = 32
    child_categories_dim: int = 32
    popularity_dim: int = 4  # One-hot, so equals number of levels
    device_dim: int = 8
    day_dim: int = 8

    # Text embeddings
    title_embedding_dim: int = 768  # BERT output dimension
    tags_embedding_dim: int = 100   # CBOW-style embedding

    # Age bucket embedding
    age_bucket_dim: int = 8
    duration_bucket_dim: int = 8


@dataclass
class BucketConfig:
    """Configuration for bucket boundaries."""

    # Age buckets: [0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+]
    age_boundaries: List[int] = field(default_factory=lambda: [18, 25, 35, 45, 55, 65])

    # Duration buckets in seconds: [0-60, 60-300, 300-600, 600-1800, 1800-3600, 3600+]
    duration_boundaries: List[int] = field(default_factory=lambda: [60, 300, 600, 1800, 3600])

    # Subscriber tier boundaries: [micro (<10K), small (10K-100K), mid (100K-1M), large (1M+)]
    subscriber_tier_boundaries: List[int] = field(default_factory=lambda: [10000, 100000, 1000000])

    # View count buckets
    view_count_boundaries: List[int] = field(default_factory=lambda: [1000, 10000, 100000, 1000000])


@dataclass
class TwoTowerFeatureConfig:
    """Feature configuration for Two-Tower model."""

    # User tower features
    user_features: List[str] = field(default_factory=lambda: [
        "user_id",
        "country",
        "user_language",
        "age",
        "previously_watched_category",
    ])

    # Video tower features
    video_features: List[str] = field(default_factory=lambda: [
        "video_id",
        "category",
        "title",
        "video_duration",
        "popularity",
        "video_language",
        "tags",
    ])

    # Features requiring vocabulary lookup
    categorical_features: List[str] = field(default_factory=lambda: [
        "user_id",
        "video_id",
        "country",
        "user_language",
        "video_language",
        "category",
        "previously_watched_category",
        "popularity",
    ])

    # Features requiring normalization
    numeric_features: List[str] = field(default_factory=lambda: [
        "age",
        "video_duration",
    ])

    # Features requiring text embedding
    text_features: List[str] = field(default_factory=lambda: [
        "title",
        "tags",
    ])

    # Minimum watch ratio for positive interaction
    min_watch_ratio: float = 0.4


@dataclass
class RankerFeatureConfig:
    """Feature configuration for Ranker model."""

    # All features for ranker
    all_features: List[str] = field(default_factory=lambda: [
        # User features
        "user_id",
        "country",
        "user_language",
        "age",
        # Video features
        "video_id",
        "category",
        "child_categories",
        "video_duration",
        "popularity",
        "tags",
        "comment_count",
        "like_count",
        "view_count",
        "video_language",
        "channel_subscriber_count",
        # Interaction context features
        "interaction_time_day",
        "interaction_time_hour",
        "device",
    ])

    # Categorical features (CatBoost handles directly)
    categorical_features: List[str] = field(default_factory=lambda: [
        "country",
        "user_language",
        "category",
        "child_categories",
        "video_language",
        "interaction_time_day",
        "device",
    ])

    # Numeric features requiring transformation
    numeric_features: List[str] = field(default_factory=lambda: [
        "age",
        "video_duration",
        "comment_count",
        "like_count",
        "view_count",
        "channel_subscriber_count",
        "interaction_time_hour",
    ])

    # Features requiring log transformation
    log_transform_features: List[str] = field(default_factory=lambda: [
        "video_duration",
        "comment_count",
        "like_count",
        "view_count",
        "channel_subscriber_count",
    ])

    # Negative sampling configuration
    negative_sample_ratio: int = 3  # 1 positive : 3 negatives

    # Minimum watch ratio for positive interaction
    min_watch_ratio: float = 0.4


@dataclass
class FeatureConfig:
    """Main configuration class combining all feature configs."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    buckets: BucketConfig = field(default_factory=BucketConfig)
    two_tower: TwoTowerFeatureConfig = field(default_factory=TwoTowerFeatureConfig)
    ranker: RankerFeatureConfig = field(default_factory=RankerFeatureConfig)

    # Paths for artifacts
    artifacts_dir: str = "artifacts"
    vocabularies_dir: str = "artifacts/vocabularies"
    normalizers_dir: str = "artifacts/normalizers"
    buckets_dir: str = "artifacts/buckets"
    embeddings_dir: str = "artifacts/embeddings"

    # Paths for processed data
    processed_data_dir: str = "processed_data"
    two_tower_data_dir: str = "processed_data/two_tower"
    ranker_data_dir: str = "processed_data/ranker"
    tfrecords_dir: str = "processed_data/tfrecords"

    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Random seed for reproducibility
    random_seed: int = 42


# Default configuration instance
DEFAULT_CONFIG = FeatureConfig()
