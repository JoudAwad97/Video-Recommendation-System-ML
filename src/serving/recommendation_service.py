"""
Complete recommendation service orchestrator.

Implements the full recommendation pipeline:
1. Candidate Generation (multi-query with diversity)
2. Filtering (business rules + watched history)
3. Ranking (feature enrichment + CatBoost)
4. Final ordering and diversification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import time
import numpy as np

from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig, VectorDBConfig
from .feature_store_client import (
    FeatureStoreClient,
    FeatureStoreConfig,
    InMemoryFeatureStore,
    create_feature_store_client,
)
from .redis_cache_client import (
    CacheClient,
    RedisCacheConfig,
    InMemoryCacheClient,
    create_cache_client,
    UserInteraction,
)
from .multi_query_generator import (
    MultiQueryGenerator,
    MultiQueryConfig,
    MultiQueryResult,
    QueryResult,
    DiversitySampler,
)
from .filtering_service import (
    FilteringService,
    FilteringConfig,
    FilterResult,
)
from .ranker_service_v2 import (
    EnhancedRankerService,
    RankerServiceConfig,
    RankingResult,
    RankedCandidate,
)
from .vector_store import VectorStore, InMemoryVectorStore

logger = get_logger(__name__)


@dataclass
class RecommendationServiceConfig:
    """Configuration for the recommendation service."""

    # Component configs
    serving: ServingConfig = field(default_factory=ServingConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    cache: RedisCacheConfig = field(default_factory=RedisCacheConfig)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    ranker: RankerServiceConfig = field(default_factory=RankerServiceConfig)

    # Pipeline settings
    enable_multi_query: bool = True
    enable_filtering: bool = True
    enable_ranking: bool = True
    enable_diversification: bool = True

    # Performance settings
    max_latency_ms: float = 100  # Target latency
    enable_caching: bool = True

    # Output settings
    default_num_recommendations: int = 20
    max_num_recommendations: int = 100


@dataclass
class VideoRecommendation:
    """A single video recommendation."""

    video_id: int
    score: float
    rank: int
    source: str  # "retrieval", "ranking"
    metadata: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "metadata": self.metadata,
            "explanation": self.explanation,
        }


@dataclass
class RecommendationServiceResponse:
    """Response from the recommendation service."""

    request_id: str
    user_id: int
    recommendations: List[VideoRecommendation]
    total_latency_ms: float
    stage_latencies: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": self.stage_latencies,
            "metadata": self.metadata,
        }


class RecommendationService:
    """Complete recommendation service.

    Orchestrates the full recommendation pipeline as shown in the
    architecture diagrams:

    1. Candidate Generation:
       - Fetch user features from SageMaker Feature Store
       - Fetch recent interactions from Redis ElastiCache
       - Generate multiple user embeddings for diversity
       - Query vector database for each embedding

    2. Filtering:
       - Load watched videos from Redis
       - Apply business rules (duration, category, etc.)

    3. Ranking:
       - Add extra features from Feature Store
       - Call ranker model for scoring

    4. Final Ordering:
       - Apply diversification
       - Return top-k results

    Example:
        >>> service = RecommendationService(config)
        >>> service.initialize()
        >>> response = service.get_recommendations(user_id=123)
    """

    def __init__(self, config: RecommendationServiceConfig):
        """Initialize the recommendation service.

        Args:
            config: Service configuration.
        """
        self.config = config

        # Components (initialized lazily)
        self.feature_store: Optional[FeatureStoreClient] = None
        self.cache_client: Optional[CacheClient] = None
        self.multi_query_generator: Optional[MultiQueryGenerator] = None
        self.filtering_service: Optional[FilteringService] = None
        self.ranker_service: Optional[EnhancedRankerService] = None
        self.vector_store: Optional[VectorStore] = None
        self.diversity_sampler: Optional[DiversitySampler] = None

        # Query encoder
        self._query_encoder = None

        # Video metadata
        self._video_metadata: Dict[int, Dict[str, Any]] = {}

        # State
        self._is_initialized = False
        self._request_counter = 0

    def initialize(
        self,
        use_sagemaker_feature_store: bool = False,
        use_redis: bool = False,
        model_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ) -> "RecommendationService":
        """Initialize all service components.

        Args:
            use_sagemaker_feature_store: Use SageMaker Feature Store.
            use_redis: Use Redis for caching.
            model_path: Path to model artifacts.
            artifacts_path: Path to preprocessing artifacts.

        Returns:
            Self for method chaining.
        """
        logger.info("Initializing recommendation service...")

        # Initialize feature store
        self.feature_store = create_feature_store_client(
            self.config.feature_store,
            use_sagemaker=use_sagemaker_feature_store,
        )

        # Initialize cache client
        self.cache_client = create_cache_client(
            self.config.cache,
            use_redis=use_redis,
        )

        # Initialize multi-query generator
        self.multi_query_generator = MultiQueryGenerator(
            self.config.multi_query,
            self.cache_client,
            self.feature_store,
        )

        # Initialize filtering service
        self.filtering_service = FilteringService(
            self.config.filtering,
            self.cache_client,
        )

        # Initialize ranker service
        self.ranker_service = EnhancedRankerService(
            self.config.ranker,
            self.feature_store,
        )

        # Try to load ranker model
        if model_path:
            try:
                self.ranker_service.load_model(model_path)
            except Exception as e:
                logger.warning(f"Could not load ranker model: {e}")

        # Initialize vector store
        self.vector_store = InMemoryVectorStore(self.config.serving.vector_db)

        # Initialize diversity sampler
        self.diversity_sampler = DiversitySampler(
            min_categories=self.config.multi_query.min_category_diversity,
        )

        self._is_initialized = True
        logger.info("Recommendation service initialized")

        return self

    def load_video_data(
        self,
        video_metadata: Dict[int, Dict[str, Any]],
        video_embeddings: Optional[Dict[int, np.ndarray]] = None,
    ) -> "RecommendationService":
        """Load video data for serving.

        Args:
            video_metadata: Video metadata dictionary.
            video_embeddings: Video embedding dictionary.

        Returns:
            Self for method chaining.
        """
        self._video_metadata = video_metadata

        # Load into feature store for enrichment
        if isinstance(self.feature_store, InMemoryFeatureStore):
            self.feature_store.load_video_features(video_metadata)

        # Load embeddings into vector store
        if video_embeddings and self.vector_store:
            video_ids = list(video_embeddings.keys())
            embeddings = np.array([video_embeddings[vid] for vid in video_ids])
            self.vector_store.add(video_ids, embeddings)

        logger.info(f"Loaded {len(video_metadata)} videos")
        return self

    def get_recommendations(
        self,
        user_id: int,
        num_recommendations: int = 20,
        user_features: Optional[Dict[str, Any]] = None,
        excluded_video_ids: Optional[Set[int]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        interaction_context: Optional[Dict[str, Any]] = None,
    ) -> RecommendationServiceResponse:
        """Get recommendations for a user.

        Args:
            user_id: User identifier.
            num_recommendations: Number of recommendations.
            user_features: Pre-fetched user features.
            excluded_video_ids: Videos to exclude.
            user_preferences: User preference overrides.
            interaction_context: Interaction context.

        Returns:
            RecommendationServiceResponse.
        """
        if not self._is_initialized:
            raise RuntimeError("Service not initialized")

        start_time = time.time()
        stage_latencies = {}

        self._request_counter += 1
        request_id = f"rec_{self._request_counter}_{int(time.time())}"

        # Limit num_recommendations
        num_recommendations = min(
            num_recommendations,
            self.config.max_num_recommendations,
        )

        # Stage 1: Get user features
        stage_start = time.time()
        if user_features is None:
            user_features = self.feature_store.get_user_features(user_id) or {}
        user_features["user_id"] = user_id
        stage_latencies["feature_fetch"] = (time.time() - stage_start) * 1000

        # Stage 2: Candidate Generation (Multi-Query)
        stage_start = time.time()
        candidates = self._generate_candidates(
            user_id=user_id,
            user_features=user_features,
            num_candidates=self.config.serving.num_candidates,
        )
        stage_latencies["candidate_generation"] = (time.time() - stage_start) * 1000

        if not candidates:
            return RecommendationServiceResponse(
                request_id=request_id,
                user_id=user_id,
                recommendations=[],
                total_latency_ms=(time.time() - start_time) * 1000,
                stage_latencies=stage_latencies,
                metadata={"reason": "no_candidates"},
            )

        # Stage 3: Filtering
        stage_start = time.time()
        if self.config.enable_filtering:
            filter_result = self.filtering_service.filter(
                video_ids=[vid for vid, _ in candidates],
                user_id=user_id,
                video_metadata=self._video_metadata,
                user_preferences=user_preferences,
                additional_exclusions=excluded_video_ids,
            )
            # Update candidates to only passed videos
            passed_set = set(filter_result.passed_video_ids)
            candidates = [(vid, score) for vid, score in candidates if vid in passed_set]
        stage_latencies["filtering"] = (time.time() - stage_start) * 1000

        # Stage 4: Ranking
        stage_start = time.time()
        if self.config.enable_ranking and candidates:
            ranking_result = self.ranker_service.rank(
                user_id=user_id,
                candidates=candidates,
                user_features=user_features,
                video_metadata=self._video_metadata,
                interaction_context=interaction_context,
                top_k=num_recommendations * 2,  # Extra for diversification
            )
            ranked_candidates = ranking_result.ranked_candidates
        else:
            # Use retrieval scores directly
            ranked_candidates = [
                RankedCandidate(
                    video_id=vid,
                    ranker_score=score,
                    retrieval_score=score,
                    final_score=score,
                    rank=i + 1,
                    metadata=self._video_metadata.get(vid),
                )
                for i, (vid, score) in enumerate(
                    sorted(candidates, key=lambda x: x[1], reverse=True)
                )
            ]
        stage_latencies["ranking"] = (time.time() - stage_start) * 1000

        # Stage 5: Diversification and Final Ordering
        stage_start = time.time()
        if self.config.enable_diversification:
            final_candidates = self._diversify(
                ranked_candidates, num_recommendations
            )
        else:
            final_candidates = ranked_candidates[:num_recommendations]
        stage_latencies["diversification"] = (time.time() - stage_start) * 1000

        # Build response
        recommendations = []
        for i, candidate in enumerate(final_candidates):
            rec = VideoRecommendation(
                video_id=candidate.video_id,
                score=candidate.final_score,
                rank=i + 1,
                source="ranking" if self.config.enable_ranking else "retrieval",
                metadata=candidate.metadata,
            )
            recommendations.append(rec)

        total_latency = (time.time() - start_time) * 1000

        return RecommendationServiceResponse(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
            metadata={
                "num_candidates_generated": len(candidates) if candidates else 0,
                "num_after_filtering": len(ranked_candidates),
            },
        )

    def _generate_candidates(
        self,
        user_id: int,
        user_features: Dict[str, Any],
        num_candidates: int,
    ) -> List[Tuple[int, float]]:
        """Generate candidates using multi-query approach.

        Args:
            user_id: User identifier.
            user_features: User features.
            num_candidates: Number of candidates to generate.

        Returns:
            List of (video_id, score) tuples.
        """
        if self.vector_store is None or self.vector_store.size == 0:
            return []

        if self.config.enable_multi_query:
            # Generate multiple query contexts
            query_contexts = self.multi_query_generator.generate_query_contexts(
                user_id, user_features
            )

            # Execute queries
            query_results = []
            for context in query_contexts:
                # Generate user embedding for this context
                embedding = self._encode_user(context)
                if embedding is None:
                    continue

                # Query vector store
                video_ids, scores = self.vector_store.search(
                    embedding,
                    top_k=self.config.multi_query.candidates_per_query,
                )

                result = self.multi_query_generator.create_query_result(
                    context, video_ids, scores, embedding
                )
                query_results.append(result)

            # Merge results
            if query_results:
                merged = self.multi_query_generator.merge_results(
                    query_results, num_candidates
                )
                return list(zip(merged.merged_video_ids, merged.merged_scores))
            else:
                return []
        else:
            # Single query
            embedding = self._encode_user(user_features)
            if embedding is None:
                return []

            video_ids, scores = self.vector_store.search(embedding, top_k=num_candidates)
            return list(zip(video_ids, scores))

    def _encode_user(self, user_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode user features to embedding.

        Args:
            user_features: User feature dictionary.

        Returns:
            User embedding or None.
        """
        if self._query_encoder is not None:
            return self._query_encoder.encode(user_features)

        # Fallback: return random embedding for testing
        return np.random.randn(self.config.serving.vector_db.embedding_dim).astype(np.float32)

    def _diversify(
        self,
        candidates: List[RankedCandidate],
        num_results: int,
    ) -> List[RankedCandidate]:
        """Apply diversification to ranked candidates.

        Args:
            candidates: Ranked candidates.
            num_results: Number of results.

        Returns:
            Diversified candidate list.
        """
        if len(candidates) <= num_results:
            return candidates

        video_ids = [c.video_id for c in candidates]
        scores = [c.final_score for c in candidates]

        diverse_ids, diverse_scores = self.diversity_sampler.sample(
            video_ids, scores, self._video_metadata, num_results
        )

        # Map back to candidates
        id_to_candidate = {c.video_id: c for c in candidates}
        return [id_to_candidate[vid] for vid in diverse_ids]

    def record_interaction(
        self,
        user_id: int,
        video_id: int,
        category: str,
        interaction_type: str = "watch",
        duration_watched: float = 0.0,
    ) -> None:
        """Record a user interaction for future recommendations.

        Updates the user's recent interactions in Redis for
        multi-query diversity.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            category: Video category.
            interaction_type: Type of interaction.
            duration_watched: Duration watched in seconds.
        """
        interaction = UserInteraction(
            category=category,
            video_id=video_id,
            timestamp=datetime.utcnow().isoformat(),
            interaction_type=interaction_type,
            duration_watched=duration_watched,
        )
        self.cache_client.add_interaction(user_id, interaction)

    def store_recommendations(
        self,
        user_id: int,
        recommendations: List[VideoRecommendation],
    ) -> None:
        """Store recommendations for later retrieval.

        Useful for showing different subsets of recommendations
        over time.

        Args:
            user_id: User identifier.
            recommendations: Recommendations to store.
        """
        video_ids = [r.video_id for r in recommendations]
        scores = [r.score for r in recommendations]
        self.cache_client.set_recommendations(user_id, video_ids, scores)

    def get_cached_recommendations(
        self,
        user_id: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendations for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of recommendation dicts or None.
        """
        return self.cache_client.get_recommendations(user_id)

    def warmup(self) -> None:
        """Warm up all service components."""
        if not self._is_initialized:
            raise RuntimeError("Service not initialized")

        # Warm up ranker
        if self.ranker_service:
            self.ranker_service.warmup()

        logger.info("Recommendation service warmed up")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Statistics dictionary.
        """
        stats = {
            "is_initialized": self._is_initialized,
            "request_count": self._request_counter,
            "num_videos": len(self._video_metadata),
        }

        if self.vector_store:
            stats["vector_store_size"] = self.vector_store.size

        if isinstance(self.cache_client, InMemoryCacheClient):
            stats["cache_stats"] = self.cache_client.get_stats()

        return stats
