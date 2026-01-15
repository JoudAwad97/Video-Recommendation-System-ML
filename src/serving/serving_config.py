"""
Configuration for serving components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""

    # Vector store type
    store_type: str = "faiss"  # Options: "faiss", "pinecone", "weaviate"

    # FAISS-specific settings
    index_type: str = "IVFFlat"  # Options: "Flat", "IVFFlat", "IVFPQ", "HNSW"
    nlist: int = 100  # Number of clusters for IVF indexes
    nprobe: int = 10  # Number of clusters to search
    m: int = 8  # Number of sub-quantizers for PQ

    # Index paths
    index_path: Optional[str] = None
    id_mapping_path: Optional[str] = None

    # Embedding dimension
    embedding_dim: int = 16

    # Search settings
    default_top_k: int = 100


@dataclass
class ServingConfig:
    """Configuration for the serving pipeline."""

    # Model paths
    two_tower_model_path: str = "models/two_tower"
    ranker_model_path: str = "models/ranker"
    artifacts_path: str = "artifacts"

    # Vector database config
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)

    # Candidate retrieval settings
    num_candidates: int = 100  # Number of candidates from vector search

    # Filtering settings
    enable_business_rules: bool = True
    max_video_age_days: int = 365
    blocked_categories: List[str] = field(default_factory=list)

    # Ranking settings
    top_k_final: int = 20  # Final number of recommendations

    # Batch processing settings
    batch_size: int = 256

    # Caching settings
    enable_video_cache: bool = True
    video_cache_ttl_seconds: int = 3600

    # Logging
    log_predictions: bool = True

    # Performance settings
    use_gpu: bool = False
    num_workers: int = 4

    def get_model_paths(self) -> dict:
        """Get all model paths as a dictionary."""
        return {
            "two_tower": self.two_tower_model_path,
            "ranker": self.ranker_model_path,
            "artifacts": self.artifacts_path,
        }


@dataclass
class SageMakerConfig:
    """Configuration for SageMaker deployment."""

    # Endpoint settings
    endpoint_name: str = "video-rec-endpoint"
    instance_type: str = "ml.m5.large"

    # Serverless settings (for serverless endpoints)
    use_serverless: bool = True
    memory_size_mb: int = 2048  # 1024, 2048, 3072, 4096, 5120, or 6144
    max_concurrency: int = 10

    # Model settings
    model_name: str = "video-rec-model"
    model_data_url: str = ""  # S3 path to model artifacts

    # Container settings
    image_uri: str = ""  # ECR image URI
    framework: str = "tensorflow"
    framework_version: str = "2.13.0"

    # Auto-scaling settings (for non-serverless)
    min_instances: int = 1
    max_instances: int = 4
    target_invocations_per_instance: int = 100
    scale_in_cooldown: int = 300
    scale_out_cooldown: int = 60

    # IAM role
    role_arn: str = ""

    # Tags
    tags: dict = field(default_factory=lambda: {
        "Project": "VideoRecommendation",
        "Environment": "Development",
    })
