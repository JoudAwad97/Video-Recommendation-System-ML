"""
Centralized application settings with environment validation.

This module provides:
- Type-safe settings management using dataclasses
- Environment-specific defaults (dev/staging/prod)
- Validation of required environment variables
- Secrets management integration (AWS Secrets Manager)
- Startup validation to catch configuration errors early
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environment."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    LOCAL = "local"
    TEST = "test"


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass
class AWSSettings:
    """AWS-specific settings."""

    region: str = "us-east-2"
    account_id: str = ""

    # S3 Buckets
    data_bucket: str = ""
    model_bucket: str = ""
    artifacts_bucket: str = ""

    # DynamoDB Tables
    user_features_table: str = ""
    video_features_table: str = ""
    recommendations_cache_table: str = ""

    # SageMaker
    sagemaker_role_arn: str = ""

    @classmethod
    def from_env(cls) -> "AWSSettings":
        """Load AWS settings from environment variables."""
        return cls(
            region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-2")),
            account_id=os.environ.get("AWS_ACCOUNT_ID", ""),
            data_bucket=os.environ.get("DATA_BUCKET", ""),
            model_bucket=os.environ.get("MODEL_BUCKET", ""),
            artifacts_bucket=os.environ.get("ARTIFACTS_BUCKET", ""),
            user_features_table=os.environ.get("USER_FEATURES_TABLE", ""),
            video_features_table=os.environ.get("VIDEO_FEATURES_TABLE", ""),
            recommendations_cache_table=os.environ.get("RECOMMENDATIONS_CACHE_TABLE", ""),
            sagemaker_role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        )


@dataclass
class RedisSettings:
    """Redis/ElastiCache settings."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    password: str = ""  # Should come from Secrets Manager in production
    ssl: bool = False
    db: int = 0

    # Connection pooling
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    # ElastiCache specific
    elasticache_endpoint: str = ""
    cluster_mode: bool = False

    @classmethod
    def from_env(cls) -> "RedisSettings":
        """Load Redis settings from environment variables."""
        return cls(
            enabled=os.environ.get("USE_REDIS", "false").lower() == "true",
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD", ""),
            ssl=os.environ.get("REDIS_SSL", "false").lower() == "true",
            db=int(os.environ.get("REDIS_DB", "0")),
            max_connections=int(os.environ.get("REDIS_MAX_CONNECTIONS", "10")),
            socket_timeout=float(os.environ.get("REDIS_SOCKET_TIMEOUT", "5.0")),
            socket_connect_timeout=float(os.environ.get("REDIS_CONNECT_TIMEOUT", "5.0")),
            elasticache_endpoint=os.environ.get("ELASTICACHE_ENDPOINT", ""),
            cluster_mode=os.environ.get("REDIS_CLUSTER_MODE", "false").lower() == "true",
        )


@dataclass
class FeatureStoreSettings:
    """Feature store settings (SageMaker Feature Store / DynamoDB)."""

    use_sagemaker: bool = False
    region: str = "us-east-2"
    user_feature_group: str = ""
    video_feature_group: str = ""

    # DynamoDB fallback
    use_dynamodb: bool = True
    dynamodb_user_table: str = ""
    dynamodb_video_table: str = ""

    @classmethod
    def from_env(cls) -> "FeatureStoreSettings":
        """Load feature store settings from environment variables."""
        return cls(
            use_sagemaker=os.environ.get("USE_SAGEMAKER_FEATURE_STORE", "false").lower() == "true",
            region=os.environ.get("FEATURE_STORE_REGION", os.environ.get("AWS_REGION", "us-east-2")),
            user_feature_group=os.environ.get("USER_FEATURE_GROUP", ""),
            video_feature_group=os.environ.get("VIDEO_FEATURE_GROUP", ""),
            use_dynamodb=os.environ.get("USE_DYNAMODB_FEATURE_STORE", "true").lower() == "true",
            dynamodb_user_table=os.environ.get("USER_FEATURES_TABLE", ""),
            dynamodb_video_table=os.environ.get("VIDEO_FEATURES_TABLE", ""),
        )


@dataclass
class VectorStoreSettings:
    """Vector store settings for embedding search."""

    type: str = "in_memory"  # in_memory, faiss, pinecone, opensearch
    endpoint: str = ""
    api_key: str = ""  # For Pinecone

    # S3 location for vector index
    s3_bucket: str = ""
    s3_key: str = "vector_store/vector_index.npz"
    video_features_s3_key: str = "vector_store/video_features.json"

    # Index parameters
    embedding_dim: int = 16
    index_type: str = "Flat"  # Flat, IVF, HNSW
    nlist: int = 100  # For IVF
    nprobe: int = 10  # For IVF search

    @classmethod
    def from_env(cls) -> "VectorStoreSettings":
        """Load vector store settings from environment variables."""
        return cls(
            type=os.environ.get("VECTOR_STORE_TYPE", "in_memory"),
            endpoint=os.environ.get("VECTOR_STORE_ENDPOINT", ""),
            api_key=os.environ.get("VECTOR_STORE_API_KEY", ""),
            s3_bucket=os.environ.get("MODEL_BUCKET", ""),
            s3_key=os.environ.get("VECTOR_STORE_S3_KEY", "vector_store/vector_index.npz"),
            video_features_s3_key=os.environ.get("VIDEO_FEATURES_S3_KEY", "vector_store/video_features.json"),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "16")),
            index_type=os.environ.get("VECTOR_INDEX_TYPE", "Flat"),
            nlist=int(os.environ.get("VECTOR_INDEX_NLIST", "100")),
            nprobe=int(os.environ.get("VECTOR_INDEX_NPROBE", "10")),
        )


@dataclass
class ServingSettings:
    """Recommendation serving settings."""

    # Recommendation parameters
    default_num_recommendations: int = 20
    max_num_recommendations: int = 100
    min_candidates: int = 50
    candidate_multiplier: int = 5

    # Diversity settings
    enable_diversity: bool = True
    diversity_weight: float = 0.3
    category_diversity_weight: float = 0.5

    # Filtering settings
    enable_filtering: bool = True
    filter_watched: bool = True
    filter_blocked: bool = True

    # Model paths
    model_path: str = "/opt/ml/model"
    artifacts_path: str = "/opt/ml/artifacts"

    # Timeouts
    timeout_ms: int = 5000
    retrieval_timeout_ms: int = 1000
    ranking_timeout_ms: int = 2000

    @classmethod
    def from_env(cls) -> "ServingSettings":
        """Load serving settings from environment variables."""
        return cls(
            default_num_recommendations=int(os.environ.get("DEFAULT_NUM_RECOMMENDATIONS", "20")),
            max_num_recommendations=int(os.environ.get("MAX_NUM_RECOMMENDATIONS", "100")),
            min_candidates=int(os.environ.get("MIN_CANDIDATES", "50")),
            candidate_multiplier=int(os.environ.get("CANDIDATE_MULTIPLIER", "5")),
            enable_diversity=os.environ.get("ENABLE_DIVERSITY", "true").lower() == "true",
            diversity_weight=float(os.environ.get("DIVERSITY_WEIGHT", "0.3")),
            category_diversity_weight=float(os.environ.get("CATEGORY_DIVERSITY_WEIGHT", "0.5")),
            enable_filtering=os.environ.get("ENABLE_FILTERING", "true").lower() == "true",
            filter_watched=os.environ.get("FILTER_WATCHED", "true").lower() == "true",
            filter_blocked=os.environ.get("FILTER_BLOCKED", "true").lower() == "true",
            model_path=os.environ.get("MODEL_PATH", "/opt/ml/model"),
            artifacts_path=os.environ.get("ARTIFACTS_PATH", "/opt/ml/artifacts"),
            timeout_ms=int(os.environ.get("SERVING_TIMEOUT_MS", "5000")),
            retrieval_timeout_ms=int(os.environ.get("RETRIEVAL_TIMEOUT_MS", "1000")),
            ranking_timeout_ms=int(os.environ.get("RANKING_TIMEOUT_MS", "2000")),
        )


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # json or text
    include_timestamp: bool = True
    include_request_id: bool = True
    mask_sensitive_fields: bool = True

    # Fields to mask in logs
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "password", "api_key", "token", "secret", "authorization"
    ])

    @classmethod
    def from_env(cls) -> "LoggingSettings":
        """Load logging settings from environment variables."""
        return cls(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format=os.environ.get("LOG_FORMAT", "json"),
            include_timestamp=os.environ.get("LOG_INCLUDE_TIMESTAMP", "true").lower() == "true",
            include_request_id=os.environ.get("LOG_INCLUDE_REQUEST_ID", "true").lower() == "true",
            mask_sensitive_fields=os.environ.get("LOG_MASK_SENSITIVE", "true").lower() == "true",
        )


@dataclass
class Settings:
    """Main settings class combining all configuration."""

    # Environment
    environment: Environment = Environment.DEV
    service_name: str = "video-recommendation-service"
    version: str = "1.0.0"
    debug: bool = False

    # Component settings
    aws: AWSSettings = field(default_factory=AWSSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    feature_store: FeatureStoreSettings = field(default_factory=FeatureStoreSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    serving: ServingSettings = field(default_factory=ServingSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def __post_init__(self):
        """Apply environment-specific defaults after initialization."""
        self._apply_environment_defaults()

    def _apply_environment_defaults(self) -> None:
        """Apply defaults based on environment."""
        if self.environment == Environment.PROD:
            # Production defaults
            self.debug = False
            self.logging.level = "INFO"
            self.logging.format = "json"
        elif self.environment == Environment.DEV:
            # Development defaults
            self.debug = True
            self.logging.level = "DEBUG"
        elif self.environment == Environment.LOCAL:
            # Local development defaults
            self.debug = True
            self.logging.level = "DEBUG"
            self.logging.format = "text"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load all settings from environment variables."""
        env_str = os.environ.get("ENVIRONMENT", "dev").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            environment = Environment.DEV

        return cls(
            environment=environment,
            service_name=os.environ.get("SERVICE_NAME", "video-recommendation-service"),
            version=os.environ.get("SERVICE_VERSION", "1.0.0"),
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            aws=AWSSettings.from_env(),
            redis=RedisSettings.from_env(),
            feature_store=FeatureStoreSettings.from_env(),
            vector_store=VectorStoreSettings.from_env(),
            serving=ServingSettings.from_env(),
            logging=LoggingSettings.from_env(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (with sensitive fields masked)."""
        def _mask_sensitive(obj: Any, mask_fields: List[str]) -> Any:
            if isinstance(obj, dict):
                return {
                    k: "***MASKED***" if any(s in k.lower() for s in mask_fields)
                    else _mask_sensitive(v, mask_fields)
                    for k, v in obj.items()
                }
            elif hasattr(obj, "__dataclass_fields__"):
                return _mask_sensitive(
                    {f: getattr(obj, f) for f in obj.__dataclass_fields__},
                    mask_fields
                )
            return obj

        mask_fields = ["password", "secret", "key", "token", "api_key"]
        return _mask_sensitive(self, mask_fields)

    def validate(self) -> List[str]:
        """Validate settings and return list of errors."""
        errors = []

        # Validate based on environment
        if self.environment in (Environment.STAGING, Environment.PROD):
            # Production/Staging requires AWS configuration
            if not self.aws.model_bucket:
                errors.append("MODEL_BUCKET is required in staging/production")
            if not self.aws.user_features_table:
                errors.append("USER_FEATURES_TABLE is required in staging/production")
            if not self.aws.video_features_table:
                errors.append("VIDEO_FEATURES_TABLE is required in staging/production")

        # Validate Redis if enabled
        if self.redis.enabled:
            if not self.redis.host and not self.redis.elasticache_endpoint:
                errors.append("REDIS_HOST or ELASTICACHE_ENDPOINT required when Redis is enabled")

        # Validate vector store
        if self.vector_store.type == "pinecone" and not self.vector_store.api_key:
            errors.append("VECTOR_STORE_API_KEY required for Pinecone")

        # Validate serving settings
        if self.serving.max_num_recommendations < self.serving.default_num_recommendations:
            errors.append("MAX_NUM_RECOMMENDATIONS must be >= DEFAULT_NUM_RECOMMENDATIONS")

        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance loaded from environment.
    """
    return Settings.from_env()


def validate_settings(raise_on_error: bool = True) -> List[str]:
    """Validate current settings.

    Args:
        raise_on_error: If True, raise ConfigurationError on validation failure.

    Returns:
        List of validation error messages.

    Raises:
        ConfigurationError: If validation fails and raise_on_error is True.
    """
    settings = get_settings()
    errors = settings.validate()

    if errors:
        error_msg = f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)

        if raise_on_error:
            raise ConfigurationError(error_msg)

    return errors


def load_secrets_from_aws(secret_name: str, region: str = "us-east-2") -> Dict[str, str]:
    """Load secrets from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret in Secrets Manager.
        region: AWS region.

    Returns:
        Dictionary of secret key-value pairs.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)

        if "SecretString" in response:
            return json.loads(response["SecretString"])
        else:
            import base64
            return json.loads(base64.b64decode(response["SecretBinary"]))

    except ClientError as e:
        logger.warning(f"Could not load secrets from AWS: {e}")
        return {}
    except ImportError:
        logger.warning("boto3 not available, skipping secrets loading")
        return {}


def apply_secrets_to_env(secret_name: str, region: str = "us-east-2") -> None:
    """Load secrets from AWS and apply to environment variables.

    Args:
        secret_name: Name of the secret in Secrets Manager.
        region: AWS region.
    """
    secrets = load_secrets_from_aws(secret_name, region)

    for key, value in secrets.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug(f"Applied secret {key} to environment")
