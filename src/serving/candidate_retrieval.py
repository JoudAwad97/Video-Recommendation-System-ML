"""
Candidate retrieval service for generating video candidates.

Retrieves candidate videos using similarity search against the
video embedding index, with support for filtering and business rules.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig
from .vector_store import VectorStore
from .query_encoder import QueryEncoderService

logger = get_logger(__name__)


@dataclass
class VideoCandidate:
    """A video candidate with metadata."""

    video_id: int
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Result of candidate retrieval."""

    candidates: List[VideoCandidate]
    user_embedding: np.ndarray
    retrieval_time_ms: float
    num_filtered: int = 0


class CandidateRetrievalService:
    """Service for retrieving video candidates using similarity search.

    This service:
    1. Encodes user features to an embedding
    2. Performs similarity search against video index
    3. Applies business rule filtering
    4. Returns ranked candidates for further ranking

    Example:
        >>> service = CandidateRetrievalService(config)
        >>> service.set_query_encoder(query_encoder)
        >>> service.set_vector_store(vector_store)
        >>> result = service.retrieve(user_data, num_candidates=100)
    """

    def __init__(self, config: ServingConfig):
        """Initialize the candidate retrieval service.

        Args:
            config: Serving configuration.
        """
        self.config = config
        self.query_encoder: Optional[QueryEncoderService] = None
        self.vector_store: Optional[VectorStore] = None
        self.video_metadata: Dict[int, Dict] = {}
        self._blocked_videos: Set[int] = set()
        self._is_ready = False

    def set_query_encoder(
        self,
        query_encoder: QueryEncoderService,
    ) -> "CandidateRetrievalService":
        """Set the query encoder service.

        Args:
            query_encoder: Initialized query encoder.

        Returns:
            Self for method chaining.
        """
        self.query_encoder = query_encoder
        self._check_ready()
        return self

    def set_vector_store(
        self,
        vector_store: VectorStore,
    ) -> "CandidateRetrievalService":
        """Set the vector store.

        Args:
            vector_store: Initialized vector store with video embeddings.

        Returns:
            Self for method chaining.
        """
        self.vector_store = vector_store
        self._check_ready()
        return self

    def load_video_metadata(
        self,
        metadata: Dict[int, Dict],
    ) -> "CandidateRetrievalService":
        """Load video metadata for filtering.

        Args:
            metadata: Dictionary mapping video_id to metadata dict.
                Expected keys: category, upload_date, is_active, etc.

        Returns:
            Self for method chaining.
        """
        self.video_metadata = metadata
        logger.info(f"Loaded metadata for {len(metadata)} videos")
        return self

    def set_blocked_videos(
        self,
        video_ids: Set[int],
    ) -> "CandidateRetrievalService":
        """Set videos that should be blocked from recommendations.

        Args:
            video_ids: Set of video IDs to block.

        Returns:
            Self for method chaining.
        """
        self._blocked_videos = video_ids
        logger.info(f"Blocked {len(video_ids)} videos")
        return self

    def _check_ready(self) -> None:
        """Check if service is ready for retrieval."""
        self._is_ready = (
            self.query_encoder is not None and
            self.vector_store is not None
        )

    def retrieve(
        self,
        user_data: Dict[str, Any],
        num_candidates: Optional[int] = None,
        excluded_video_ids: Optional[Set[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieve video candidates for a user.

        Args:
            user_data: User features dictionary.
            num_candidates: Number of candidates to retrieve.
            excluded_video_ids: Video IDs to exclude (e.g., already watched).
            filters: Additional filters (category, language, etc.).

        Returns:
            RetrievalResult with candidates and metadata.
        """
        if not self._is_ready:
            raise RuntimeError(
                "Service not ready. Set query_encoder and vector_store first."
            )

        import time
        start_time = time.time()

        num_candidates = num_candidates or self.config.num_candidates
        excluded_video_ids = excluded_video_ids or set()

        # Encode user to embedding
        user_embedding = self.query_encoder.encode_user(user_data)

        # Retrieve more candidates to account for filtering
        retrieval_multiplier = 2  # Retrieve 2x to have buffer for filtering
        search_k = min(
            num_candidates * retrieval_multiplier,
            self.vector_store.size,
        )

        # Search vector store
        video_ids, scores = self.vector_store.search(user_embedding, top_k=search_k)

        # Apply filters
        candidates = []
        num_filtered = 0

        for video_id, score in zip(video_ids, scores):
            # Check exclusions
            if video_id in excluded_video_ids:
                num_filtered += 1
                continue

            # Check blocked videos
            if video_id in self._blocked_videos:
                num_filtered += 1
                continue

            # Apply business rules if enabled
            if self.config.enable_business_rules:
                if not self._passes_business_rules(video_id, filters):
                    num_filtered += 1
                    continue

            # Create candidate
            metadata = self.video_metadata.get(video_id)
            candidate = VideoCandidate(
                video_id=video_id,
                similarity_score=score,
                metadata=metadata,
            )
            candidates.append(candidate)

            # Stop when we have enough candidates
            if len(candidates) >= num_candidates:
                break

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Retrieved {len(candidates)} candidates in {elapsed_ms:.2f}ms "
            f"({num_filtered} filtered)"
        )

        return RetrievalResult(
            candidates=candidates,
            user_embedding=user_embedding,
            retrieval_time_ms=elapsed_ms,
            num_filtered=num_filtered,
        )

    def retrieve_batch(
        self,
        users_data: List[Dict[str, Any]],
        num_candidates: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve video candidates for multiple users.

        Args:
            users_data: List of user data dictionaries.
            num_candidates: Number of candidates per user.

        Returns:
            List of RetrievalResults.
        """
        if not self._is_ready:
            raise RuntimeError(
                "Service not ready. Set query_encoder and vector_store first."
            )

        import time
        start_time = time.time()

        num_candidates = num_candidates or self.config.num_candidates

        # Batch encode users
        user_embeddings = self.query_encoder.encode_users_batch(users_data)

        # Batch search
        all_video_ids, all_scores = self.vector_store.batch_search(
            user_embeddings,
            top_k=num_candidates * 2,
        )

        # Process results
        results = []
        for i, (video_ids, scores, user_emb) in enumerate(
            zip(all_video_ids, all_scores, user_embeddings)
        ):
            candidates = []
            for video_id, score in zip(video_ids, scores):
                if video_id in self._blocked_videos:
                    continue
                metadata = self.video_metadata.get(video_id)
                candidate = VideoCandidate(
                    video_id=int(video_id),
                    similarity_score=float(score),
                    metadata=metadata,
                )
                candidates.append(candidate)
                if len(candidates) >= num_candidates:
                    break

            results.append(RetrievalResult(
                candidates=candidates,
                user_embedding=user_emb,
                retrieval_time_ms=0,  # Set below
                num_filtered=0,
            ))

        elapsed_ms = (time.time() - start_time) * 1000
        avg_time = elapsed_ms / len(results)
        for result in results:
            result.retrieval_time_ms = avg_time

        return results

    def _passes_business_rules(
        self,
        video_id: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if video passes business rules.

        Args:
            video_id: Video ID to check.
            filters: Additional filter criteria.

        Returns:
            True if video passes all rules.
        """
        metadata = self.video_metadata.get(video_id, {})

        # Check if video is active
        if not metadata.get("is_active", True):
            return False

        # Check video age
        upload_date = metadata.get("upload_date")
        if upload_date and self.config.max_video_age_days > 0:
            if isinstance(upload_date, str):
                upload_date = datetime.fromisoformat(upload_date)
            max_age = timedelta(days=self.config.max_video_age_days)
            if datetime.now() - upload_date > max_age:
                return False

        # Check blocked categories
        category = metadata.get("category", "")
        if category in self.config.blocked_categories:
            return False

        # Apply custom filters
        if filters:
            # Category filter
            if "category" in filters:
                if category != filters["category"]:
                    return False

            # Language filter
            if "language" in filters:
                if metadata.get("language") != filters["language"]:
                    return False

            # Minimum duration filter
            if "min_duration" in filters:
                if metadata.get("duration", 0) < filters["min_duration"]:
                    return False

            # Maximum duration filter
            if "max_duration" in filters:
                if metadata.get("duration", float("inf")) > filters["max_duration"]:
                    return False

        return True

    def get_similar_videos(
        self,
        video_id: int,
        num_similar: int = 10,
    ) -> List[VideoCandidate]:
        """Get videos similar to a given video.

        Args:
            video_id: Source video ID.
            num_similar: Number of similar videos to return.

        Returns:
            List of similar video candidates.
        """
        # This requires video embeddings to be stored
        # For now, return empty list - would need separate video index
        logger.warning("get_similar_videos not implemented for user-based index")
        return []
