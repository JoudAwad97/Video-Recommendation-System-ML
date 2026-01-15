"""
Model configuration for Two-Tower and Ranker models.

This module defines hyperparameters and architecture configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TwoTowerModelConfig:
    """Configuration for Two-Tower model architecture and training."""

    # Output embedding dimension (shared by both towers)
    embedding_dim: int = 16

    # Hidden layer dimensions for dense networks
    user_tower_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    video_tower_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Feature embedding dimensions (used in embedding layers)
    user_id_embedding_dim: int = 32
    video_id_embedding_dim: int = 32
    country_embedding_dim: int = 16
    language_embedding_dim: int = 16
    category_embedding_dim: int = 16

    # Numeric feature processing
    age_bucket_embedding_dim: int = 8
    duration_bucket_embedding_dim: int = 8

    # Pre-computed embedding dimensions (from preprocessing)
    title_embedding_dim: int = 512  # Universal Sentence Encoder output
    tags_embedding_dim: int = 100   # CBOW-style tag embeddings

    # Popularity one-hot dimension
    popularity_dim: int = 4

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10

    # Regularization
    dropout_rate: float = 0.2
    l2_regularization: float = 0.0001

    # Temperature for softmax in in-batch negatives
    temperature: float = 0.05

    # Validation metrics
    precision_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    # Model checkpointing
    checkpoint_dir: str = "models/checkpoints/two_tower"
    save_best_only: bool = True


@dataclass
class RankerModelConfig:
    """Configuration for CatBoost Ranker model."""

    # CatBoost hyperparameters
    iterations: int = 1000
    learning_rate: float = 0.1
    depth: int = 6
    l2_leaf_reg: float = 3.0

    # Loss function
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"

    # Early stopping
    early_stopping_rounds: int = 50

    # Threading
    thread_count: int = -1  # Use all available cores

    # Random seed
    random_seed: int = 42

    # Verbose
    verbose: int = 100

    # Feature handling
    cat_features: List[str] = field(default_factory=lambda: [
        "country",
        "user_language",
        "category",
        "child_categories",
        "video_language",
        "interaction_time_day",
        "device",
        "popularity",
    ])

    # Model saving
    model_dir: str = "models/checkpoints/ranker"


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    two_tower: TwoTowerModelConfig = field(default_factory=TwoTowerModelConfig)
    ranker: RankerModelConfig = field(default_factory=RankerModelConfig)

    # Data paths
    processed_data_dir: str = "processed_data"
    artifacts_dir: str = "artifacts"

    # Output paths
    models_dir: str = "models"
    logs_dir: str = "logs"

    # Random seed for reproducibility
    random_seed: int = 42

    # Device configuration
    use_gpu: bool = True
    mixed_precision: bool = False


# Default configuration instances
DEFAULT_TWO_TOWER_CONFIG = TwoTowerModelConfig()
DEFAULT_RANKER_CONFIG = RankerModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
