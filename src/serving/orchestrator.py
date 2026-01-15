"""
Recommendation orchestrator for end-to-end recommendation serving.

Coordinates the 4-stage recommendation pipeline:
1. Candidate Generation (Two-Tower + Vector Search)
2. Filtering (Business Rules)
3. Ranking (CatBoost Ranker)
4. Ordering (Final reranking and diversification)
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig, VectorDBConfig
from .query_encoder import QueryEncoderService
from .vector_store import VectorStore, FAISSVectorStore, InMemoryVectorStore
from .candidate_retrieval import CandidateRetrievalService, RetrievalResult
from .ranking_service import RankingService, RankingResult, RankedVideo

logger = get_logger(__name__)


@dataclass
class RecommendationRequest:
    """Request for video recommendations."""

    user_data: Dict[str, Any]
    num_recommendations: int = 20
    excluded_video_ids: Optional[Set[int]] = None
    filters: Optional[Dict[str, Any]] = None
    interaction_context: Optional[Dict[str, Any]] = None
    enable_diversification: bool = True
    diversity_weight: float = 0.1


@dataclass
class Recommendation:
    """A single video recommendation."""

    video_id: int
    score: float
    rank: int
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RecommendationResponse:
    """Response containing video recommendations."""

    recommendations: List[Recommendation]
    request_id: str
    user_id: Optional[int] = None
    latency_ms: float = 0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecommendationOrchestrator:
    """Orchestrates the full recommendation pipeline.

    This class coordinates all stages of the recommendation pipeline:
    1. Query Encoding: Convert user features to embedding
    2. Candidate Retrieval: Find similar videos from index
    3. Filtering: Apply business rules
    4. Ranking: Score candidates with ranker model
    5. Ordering: Final diversification and reranking

    Example:
        >>> orchestrator = RecommendationOrchestrator(config)
        >>> orchestrator.initialize()
        >>> response = orchestrator.recommend(request)
    """

    def __init__(self, config: ServingConfig):
        """Initialize the orchestrator.

        Args:
            config: Serving configuration.
        """
        self.config = config
        self.query_encoder: Optional[QueryEncoderService] = None
        self.vector_store: Optional[VectorStore] = None
        self.retrieval_service: Optional[CandidateRetrievalService] = None
        self.ranking_service: Optional[RankingService] = None
        self.video_data: Dict[int, Dict[str, Any]] = {}
        self._is_initialized = False
        self._request_counter = 0

    def initialize(
        self,
        model_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
        vector_index_path: Optional[str] = None,
    ) -> "RecommendationOrchestrator":
        """Initialize all services.

        Args:
            model_path: Path to model directory.
            artifacts_path: Path to artifacts directory.
            vector_index_path: Path to vector index.

        Returns:
            Self for method chaining.
        """
        model_path = model_path or self.config.two_tower_model_path
        artifacts_path = artifacts_path or self.config.artifacts_path

        logger.info("Initializing recommendation orchestrator...")

        # Initialize query encoder
        self.query_encoder = QueryEncoderService(self.config)
        self.query_encoder.load_models(model_path, artifacts_path)

        # Initialize vector store
        if self.config.vector_db.store_type == "faiss":
            try:
                self.vector_store = FAISSVectorStore(self.config.vector_db)
            except ImportError:
                logger.warning("FAISS not available, using InMemoryVectorStore")
                self.vector_store = InMemoryVectorStore(self.config.vector_db)
        else:
            self.vector_store = InMemoryVectorStore(self.config.vector_db)

        # Load vector index if path provided
        if vector_index_path:
            self.vector_store.load(vector_index_path)

        # Initialize retrieval service
        self.retrieval_service = CandidateRetrievalService(self.config)
        self.retrieval_service.set_query_encoder(self.query_encoder)
        self.retrieval_service.set_vector_store(self.vector_store)

        # Initialize ranking service
        self.ranking_service = RankingService(self.config)
        try:
            self.ranking_service.load_models(
                self.config.ranker_model_path,
                artifacts_path,
            )
        except Exception as e:
            logger.warning(f"Could not load ranker model: {e}")
            logger.warning("Ranking will use retrieval scores only")
            self.ranking_service = None

        self._is_initialized = True
        logger.info("Orchestrator initialized successfully")

        return self

    def load_video_data(
        self,
        video_data: Dict[int, Dict[str, Any]],
    ) -> "RecommendationOrchestrator":
        """Load video metadata for ranking and filtering.

        Args:
            video_data: Dictionary mapping video_id to features.

        Returns:
            Self for method chaining.
        """
        self.video_data = video_data
        if self.retrieval_service:
            self.retrieval_service.load_video_metadata(video_data)
        logger.info(f"Loaded data for {len(video_data)} videos")
        return self

    def load_video_embeddings(
        self,
        video_ids: List[int],
        embeddings: np.ndarray,
    ) -> "RecommendationOrchestrator":
        """Load video embeddings into the vector store.

        Args:
            video_ids: List of video IDs.
            embeddings: Video embeddings array.

        Returns:
            Self for method chaining.
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")

        self.vector_store.add(video_ids, embeddings)
        logger.info(f"Loaded {len(video_ids)} video embeddings")
        return self

    def recommend(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Generate recommendations for a user.

        Args:
            request: Recommendation request.

        Returns:
            RecommendationResponse with ranked videos.
        """
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized")

        import time
        start_time = time.time()
        stage_latencies = {}

        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{int(time.time())}"

        # Stage 1: Candidate Retrieval
        stage_start = time.time()
        retrieval_result = self.retrieval_service.retrieve(
            user_data=request.user_data,
            num_candidates=self.config.num_candidates,
            excluded_video_ids=request.excluded_video_ids,
            filters=request.filters,
        )
        stage_latencies["retrieval"] = (time.time() - stage_start) * 1000

        if not retrieval_result.candidates:
            return RecommendationResponse(
                recommendations=[],
                request_id=request_id,
                latency_ms=(time.time() - start_time) * 1000,
                stage_latencies=stage_latencies,
            )

        # Stage 2: Ranking
        stage_start = time.time()
        if self.ranking_service:
            ranking_result = self.ranking_service.rank(
                user_data=request.user_data,
                candidates=retrieval_result.candidates,
                video_data=self.video_data,
                interaction_context=request.interaction_context,
                top_k=request.num_recommendations * 2,  # Get extra for diversity
            )
            ranked_videos = ranking_result.ranked_videos
        else:
            # Use retrieval scores only
            ranked_videos = [
                RankedVideo(
                    video_id=c.video_id,
                    retrieval_score=c.similarity_score,
                    ranker_score=c.similarity_score,
                    final_score=c.similarity_score,
                    rank=i + 1,
                    metadata=c.metadata,
                )
                for i, c in enumerate(
                    sorted(retrieval_result.candidates,
                           key=lambda x: x.similarity_score,
                           reverse=True)
                )
            ]
        stage_latencies["ranking"] = (time.time() - stage_start) * 1000

        # Stage 3: Diversification and Final Ordering
        stage_start = time.time()
        if request.enable_diversification:
            final_videos = self._diversify(
                ranked_videos,
                request.num_recommendations,
                request.diversity_weight,
            )
        else:
            final_videos = ranked_videos[:request.num_recommendations]
        stage_latencies["ordering"] = (time.time() - stage_start) * 1000

        # Create response
        recommendations = []
        for i, video in enumerate(final_videos, start=1):
            rec = Recommendation(
                video_id=video.video_id,
                score=video.final_score,
                rank=i,
                metadata=video.metadata,
            )
            recommendations.append(rec)

        total_latency = (time.time() - start_time) * 1000

        user_id = request.user_data.get("user_id") or request.user_data.get("id")

        return RecommendationResponse(
            recommendations=recommendations,
            request_id=request_id,
            user_id=user_id,
            latency_ms=total_latency,
            stage_latencies=stage_latencies,
            metadata={
                "num_candidates": len(retrieval_result.candidates),
                "num_filtered": retrieval_result.num_filtered,
            },
        )

    def _diversify(
        self,
        ranked_videos: List[RankedVideo],
        num_results: int,
        diversity_weight: float,
    ) -> List[RankedVideo]:
        """Apply diversification to ranked results.

        Uses Maximal Marginal Relevance (MMR) style diversification
        based on video categories.

        Args:
            ranked_videos: Ranked video list.
            num_results: Number of results to return.
            diversity_weight: Weight for diversity vs relevance.

        Returns:
            Diversified video list.
        """
        if len(ranked_videos) <= num_results:
            return ranked_videos

        selected = []
        remaining = list(ranked_videos)
        category_counts: Dict[str, int] = {}

        while len(selected) < num_results and remaining:
            best_score = float("-inf")
            best_idx = 0

            for idx, video in enumerate(remaining):
                # Relevance score
                relevance = video.final_score

                # Diversity penalty based on category frequency
                category = video.metadata.get("category", "Unknown") if video.metadata else "Unknown"
                category_count = category_counts.get(category, 0)
                diversity_penalty = category_count * diversity_weight

                # Combined score
                mmr_score = relevance - diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Select best video
            selected_video = remaining.pop(best_idx)
            selected.append(selected_video)

            # Update category count
            category = selected_video.metadata.get("category", "Unknown") if selected_video.metadata else "Unknown"
            category_counts[category] = category_counts.get(category, 0) + 1

        return selected

    def recommend_simple(
        self,
        user_data: Dict[str, Any],
        num_recommendations: int = 20,
        excluded_video_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Simple interface for getting recommendations.

        Args:
            user_data: User features dictionary.
            num_recommendations: Number of videos to return.
            excluded_video_ids: Videos to exclude.

        Returns:
            List of recommendation dictionaries.
        """
        request = RecommendationRequest(
            user_data=user_data,
            num_recommendations=num_recommendations,
            excluded_video_ids=excluded_video_ids,
        )

        response = self.recommend(request)

        return [
            {
                "video_id": rec.video_id,
                "score": rec.score,
                "rank": rec.rank,
            }
            for rec in response.recommendations
        ]

    def warmup(self) -> None:
        """Warm up all services."""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized")

        logger.info("Warming up orchestrator...")

        # Warm up query encoder
        self.query_encoder.warmup()

        # Warm up ranking service
        if self.ranking_service:
            self.ranking_service.warmup()

        logger.info("Orchestrator warmed up")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "is_initialized": self._is_initialized,
            "request_count": self._request_counter,
            "vector_store_size": self.vector_store.size if self.vector_store else 0,
            "video_data_count": len(self.video_data),
            "ranking_enabled": self.ranking_service is not None,
        }
