"""
Configuration for ML Pipeline components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class PipelineExecutionMode(Enum):
    """Mode of pipeline execution."""

    FULL_TRAINING = "full_training"  # Train on all historical data
    CONTINUOUS_TRAINING = "continuous_training"  # Train on new data only
    INFERENCE_ONLY = "inference_only"  # Just run inference pipeline


class PipelineStep(Enum):
    """Steps in the ML pipeline."""

    PREPROCESSING = "preprocessing"
    DATA_SPLITTING = "data_splitting"
    TRAINING = "training"
    EVALUATION = "evaluation"
    MODEL_REGISTRATION = "model_registration"
    DEPLOYMENT = "deployment"
    MONITORING_SETUP = "monitoring_setup"


class ModelType(Enum):
    """Types of models in the pipeline."""

    TWO_TOWER = "two_tower"
    RANKER = "ranker"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing job."""

    # Input paths
    raw_interactions_path: str = "data/raw/interactions"
    raw_users_path: str = "data/raw/users"
    raw_videos_path: str = "data/raw/videos"

    # Output paths
    processed_data_path: str = "data/processed"
    vocabularies_path: str = "artifacts/vocabularies"
    feature_store_path: str = "data/feature_store"

    # Feature engineering settings
    compute_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # SageMaker settings
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    max_runtime_seconds: int = 3600


@dataclass
class DataSplittingConfig:
    """Configuration for data splitting job."""

    # Split ratios
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    # Output format
    output_format: str = "parquet"  # parquet, tfrecord

    # Versioning
    version_prefix: str = "v"

    # Paths
    versioned_data_base_path: str = "data/versioned"


@dataclass
class TrainingConfig:
    """Configuration for training job."""

    # Model settings
    model_type: str = "two_tower"  # two_tower, ranker

    # Hyperparameters (Two-Tower)
    embedding_dim: int = 16
    user_tower_hidden_dims: List[int] = field(
        default_factory=lambda: [64, 32]
    )
    video_tower_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 64]
    )
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10

    # Hyperparameters (Ranker)
    ranker_iterations: int = 1000
    ranker_depth: int = 6
    ranker_learning_rate: float = 0.1

    # SageMaker settings
    instance_type: str = "ml.p3.2xlarge"  # GPU for Two-Tower
    instance_count: int = 1
    use_spot_instances: bool = True
    max_wait_seconds: int = 7200

    # Distributed training
    enable_distributed: bool = False
    distribution_strategy: str = "data_parallel"

    # Continuous learning
    enable_continuous_learning: bool = False
    base_model_path: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Metrics to compute
    metrics: List[str] = field(
        default_factory=lambda: [
            "precision_at_k",
            "recall_at_k",
            "ndcg",
            "mrr",
            "diversity",
        ]
    )

    # Thresholds for promotion
    min_precision_at_10: float = 0.1
    min_ndcg: float = 0.3
    max_latency_ms: float = 100

    # Comparison settings
    compare_with_baseline: bool = True
    improvement_threshold: float = 0.05  # 5% improvement required


@dataclass
class DeploymentTargetConfig:
    """Configuration for deployment target."""

    # Endpoint settings
    endpoint_name_prefix: str = "video-rec"

    # Instance settings
    use_serverless: bool = True
    serverless_memory_mb: int = 2048
    serverless_max_concurrency: int = 10

    # Provisioned settings
    instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1

    # Auto-scaling
    enable_auto_scaling: bool = True
    min_capacity: int = 1
    max_capacity: int = 4

    # Traffic shifting
    enable_canary_deployment: bool = True
    canary_traffic_percentage: float = 0.1
    canary_duration_minutes: int = 30


@dataclass
class MLPipelineConfig:
    """Main configuration for the ML pipeline."""

    # Pipeline identification
    pipeline_name: str = "video-recommendation-pipeline"
    pipeline_version: str = "1.0"

    # Execution mode
    execution_mode: PipelineExecutionMode = PipelineExecutionMode.FULL_TRAINING
    model_type: ModelType = ModelType.TWO_TOWER

    # Sub-configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    data_splitting: DataSplittingConfig = field(default_factory=DataSplittingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentTargetConfig = field(default_factory=DeploymentTargetConfig)

    # AWS settings
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    role_arn: str = ""

    # Storage paths
    artifacts_path: str = "artifacts"
    models_path: str = "models"
    logs_path: str = "logs/pipeline"

    # Notification settings
    enable_notifications: bool = True
    sns_topic_arn: str = ""

    # Steps to run
    steps_to_run: List[PipelineStep] = field(
        default_factory=lambda: [
            PipelineStep.PREPROCESSING,
            PipelineStep.DATA_SPLITTING,
            PipelineStep.TRAINING,
            PipelineStep.EVALUATION,
            PipelineStep.MODEL_REGISTRATION,
            PipelineStep.DEPLOYMENT,
        ]
    )

    def should_run_step(self, step: PipelineStep) -> bool:
        """Check if a step should be run.

        Args:
            step: Pipeline step to check.

        Returns:
            True if step should be run.
        """
        return step in self.steps_to_run

    def get_s3_path(self, *parts: str) -> str:
        """Get full S3 path.

        Args:
            *parts: Path components.

        Returns:
            Full S3 URI.
        """
        path = "/".join(parts)
        return f"s3://{self.s3_bucket}/{path}"
