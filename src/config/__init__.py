"""
Configuration module for Video Recommendation System.

This module provides centralized configuration management with:
- Environment variable validation
- Type-safe configuration dataclasses
- Environment-specific defaults (dev/staging/prod)
- Secrets management integration
"""

from .feature_config import (
    FeatureConfig,
    TwoTowerFeatureConfig,
    RankerFeatureConfig,
    EmbeddingConfig,
    BucketConfig,
    DEFAULT_CONFIG,
    POPULARITY_LEVELS,
    DEVICE_TYPES,
    DAYS_OF_WEEK,
    INTERACTION_TYPES,
    POSITIVE_INTERACTION_TYPES,
    START_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
)

from .settings import (
    Settings,
    Environment,
    AWSSettings,
    RedisSettings,
    FeatureStoreSettings,
    VectorStoreSettings,
    ServingSettings,
    LoggingSettings,
    ConfigurationError,
    get_settings,
    validate_settings,
    load_secrets_from_aws,
    apply_secrets_to_env,
)

__all__ = [
    # Feature configs
    "FeatureConfig",
    "TwoTowerFeatureConfig",
    "RankerFeatureConfig",
    "EmbeddingConfig",
    "BucketConfig",
    "DEFAULT_CONFIG",
    # Constants
    "POPULARITY_LEVELS",
    "DEVICE_TYPES",
    "DAYS_OF_WEEK",
    "INTERACTION_TYPES",
    "POSITIVE_INTERACTION_TYPES",
    "START_TOKEN",
    "UNK_TOKEN",
    "PAD_TOKEN",
    # Settings
    "Settings",
    "Environment",
    "AWSSettings",
    "RedisSettings",
    "FeatureStoreSettings",
    "VectorStoreSettings",
    "ServingSettings",
    "LoggingSettings",
    "ConfigurationError",
    "get_settings",
    "validate_settings",
    "load_secrets_from_aws",
    "apply_secrets_to_env",
]
