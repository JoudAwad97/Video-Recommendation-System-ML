"""
Serving module for video recommendation system.

This module provides components for deploying and serving the recommendation models:
- Offline inference pipeline for batch video embedding generation
- Online inference pipeline for real-time recommendations
- Vector database integration for efficient similarity search
- SageMaker deployment utilities
- Feature Store client for online feature serving
- Redis cache client for user interaction data
- Multi-query candidate generation for diverse recommendations
- Filtering service for business rules
- Enhanced ranker service with feature enrichment
- Lambda handler for API Gateway integration
"""

from .serving_config import ServingConfig, VectorDBConfig
from .offline_pipeline import OfflineInferencePipeline
from .vector_store import VectorStore, FAISSVectorStore, InMemoryVectorStore
from .query_encoder import QueryEncoderService
from .candidate_retrieval import CandidateRetrievalService
from .ranking_service import RankingService
from .orchestrator import RecommendationOrchestrator

# Phase 8: Enhanced serving components
from .feature_store_client import (
    FeatureStoreConfig,
    FeatureStoreClient,
    InMemoryFeatureStore,
    SageMakerFeatureStoreClient,
    create_feature_store_client,
)
from .redis_cache_client import (
    RedisCacheConfig,
    UserInteraction,
    CacheClient,
    InMemoryCacheClient,
    RedisCacheClient,
    create_cache_client,
)
from .multi_query_generator import (
    MultiQueryConfig,
    QueryResult,
    MultiQueryResult,
    MultiQueryGenerator,
    DiversitySampler,
)
from .filtering_service import (
    FilteringConfig,
    FilterResult,
    FilteringService,
    FilteringPipeline,
)
from .ranker_service_v2 import (
    RankerServiceConfig,
    RankedCandidate,
    RankingResult,
    EnhancedRankerService,
    SageMakerRankerClient,
)
from .recommendation_service import (
    RecommendationServiceConfig,
    VideoRecommendation,
    RecommendationServiceResponse,
    RecommendationService,
)
from .lambda_handler import (
    LambdaConfig,
    handler as lambda_handler,
)

__all__ = [
    # Core serving components
    "ServingConfig",
    "VectorDBConfig",
    "OfflineInferencePipeline",
    "VectorStore",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "QueryEncoderService",
    "CandidateRetrievalService",
    "RankingService",
    "RecommendationOrchestrator",
    # Feature Store
    "FeatureStoreConfig",
    "FeatureStoreClient",
    "InMemoryFeatureStore",
    "SageMakerFeatureStoreClient",
    "create_feature_store_client",
    # Redis Cache
    "RedisCacheConfig",
    "UserInteraction",
    "CacheClient",
    "InMemoryCacheClient",
    "RedisCacheClient",
    "create_cache_client",
    # Multi-Query Generator
    "MultiQueryConfig",
    "QueryResult",
    "MultiQueryResult",
    "MultiQueryGenerator",
    "DiversitySampler",
    # Filtering
    "FilteringConfig",
    "FilterResult",
    "FilteringService",
    "FilteringPipeline",
    # Enhanced Ranker
    "RankerServiceConfig",
    "RankedCandidate",
    "RankingResult",
    "EnhancedRankerService",
    "SageMakerRankerClient",
    # Recommendation Service
    "RecommendationServiceConfig",
    "VideoRecommendation",
    "RecommendationServiceResponse",
    "RecommendationService",
    # Lambda Handler
    "LambdaConfig",
    "lambda_handler",
]
