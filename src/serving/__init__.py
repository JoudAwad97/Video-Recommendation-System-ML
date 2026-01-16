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

Note: Heavy imports (numpy, tensorflow, etc.) are lazy-loaded to support
Lambda cold starts without requiring all dependencies at import time.
"""

from typing import TYPE_CHECKING

# Light imports that don't require numpy/tensorflow
from .serving_config import ServingConfig, VectorDBConfig
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
from .filtering_service import (
    FilteringConfig,
    FilterResult,
    FilteringService,
    FilteringPipeline,
)

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from .offline_pipeline import OfflineInferencePipeline
    from .vector_store import VectorStore, FAISSVectorStore, InMemoryVectorStore
    from .query_encoder import QueryEncoderService
    from .candidate_retrieval import CandidateRetrievalService
    from .ranking_service import RankingService
    from .orchestrator import RecommendationOrchestrator
    from .multi_query_generator import (
        MultiQueryConfig,
        QueryResult,
        MultiQueryResult,
        MultiQueryGenerator,
        DiversitySampler,
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


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    # Offline pipeline (requires numpy)
    if name == "OfflineInferencePipeline":
        from .offline_pipeline import OfflineInferencePipeline
        return OfflineInferencePipeline

    # Vector stores (may require numpy/faiss)
    if name in ("VectorStore", "FAISSVectorStore", "InMemoryVectorStore"):
        from . import vector_store
        return getattr(vector_store, name)

    # Query encoder (requires numpy)
    if name == "QueryEncoderService":
        from .query_encoder import QueryEncoderService
        return QueryEncoderService

    # Candidate retrieval (requires numpy)
    if name == "CandidateRetrievalService":
        from .candidate_retrieval import CandidateRetrievalService
        return CandidateRetrievalService

    # Ranking service (requires numpy)
    if name == "RankingService":
        from .ranking_service import RankingService
        return RankingService

    # Orchestrator (requires numpy)
    if name == "RecommendationOrchestrator":
        from .orchestrator import RecommendationOrchestrator
        return RecommendationOrchestrator

    # Multi-query generator (requires numpy)
    if name in ("MultiQueryConfig", "QueryResult", "MultiQueryResult",
                "MultiQueryGenerator", "DiversitySampler"):
        from . import multi_query_generator
        return getattr(multi_query_generator, name)

    # Enhanced ranker (requires numpy/catboost)
    if name in ("RankerServiceConfig", "RankedCandidate", "RankingResult",
                "EnhancedRankerService", "SageMakerRankerClient"):
        from . import ranker_service_v2
        return getattr(ranker_service_v2, name)

    # Recommendation service (requires numpy)
    if name in ("RecommendationServiceConfig", "VideoRecommendation",
                "RecommendationServiceResponse", "RecommendationService"):
        from . import recommendation_service
        return getattr(recommendation_service, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core serving components (lazy loaded)
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
    # Feature Store (eager)
    "FeatureStoreConfig",
    "FeatureStoreClient",
    "InMemoryFeatureStore",
    "SageMakerFeatureStoreClient",
    "create_feature_store_client",
    # Redis Cache (eager)
    "RedisCacheConfig",
    "UserInteraction",
    "CacheClient",
    "InMemoryCacheClient",
    "RedisCacheClient",
    "create_cache_client",
    # Multi-Query Generator (lazy loaded)
    "MultiQueryConfig",
    "QueryResult",
    "MultiQueryResult",
    "MultiQueryGenerator",
    "DiversitySampler",
    # Filtering (eager)
    "FilteringConfig",
    "FilterResult",
    "FilteringService",
    "FilteringPipeline",
    # Enhanced Ranker (lazy loaded)
    "RankerServiceConfig",
    "RankedCandidate",
    "RankingResult",
    "EnhancedRankerService",
    "SageMakerRankerClient",
    # Recommendation Service (lazy loaded)
    "RecommendationServiceConfig",
    "VideoRecommendation",
    "RecommendationServiceResponse",
    "RecommendationService",
]
