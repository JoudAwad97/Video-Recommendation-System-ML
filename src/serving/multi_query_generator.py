"""
Multi-query candidate generation for diverse recommendations.

Generates multiple user embeddings based on recent interaction categories
to provide diverse and fresh recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import numpy as np

from ..utils.logging_utils import get_logger
from .redis_cache_client import CacheClient, InMemoryCacheClient, RedisCacheConfig
from .feature_store_client import FeatureStoreClient, InMemoryFeatureStore, FeatureStoreConfig

logger = get_logger(__name__)


@dataclass
class MultiQueryConfig:
    """Configuration for multi-query candidate generation."""

    # Number of parallel queries
    max_queries: int = 5

    # Candidates per query
    candidates_per_query: int = 50

    # Deduplication
    enable_deduplication: bool = True

    # Weighting for different queries
    primary_query_weight: float = 0.5  # Weight for main user embedding
    category_query_weight: float = 0.3  # Weight for category-based queries
    recent_interaction_weight: float = 0.2  # Weight for recent interaction queries

    # Category diversity
    min_category_diversity: int = 3  # Minimum different categories in results


@dataclass
class QueryResult:
    """Result from a single query."""

    query_id: str
    query_type: str  # "primary", "category", "recent"
    category: Optional[str]
    video_ids: List[int]
    scores: List[float]
    embedding: Optional[np.ndarray] = None


@dataclass
class MultiQueryResult:
    """Combined results from multiple queries."""

    query_results: List[QueryResult]
    merged_video_ids: List[int]
    merged_scores: List[float]
    video_sources: Dict[int, List[str]]  # video_id -> list of query sources
    total_candidates: int
    unique_candidates: int
    category_coverage: int


class MultiQueryGenerator:
    """Generates multiple queries for diverse candidate retrieval.

    Uses recent user interactions to generate multiple user embeddings,
    each representing a different aspect of user interest.

    Example:
        >>> generator = MultiQueryGenerator(config, cache_client, feature_store)
        >>> queries = generator.generate_queries(user_id=123, user_data={...})
        >>> merged = generator.merge_results(query_results)
    """

    def __init__(
        self,
        config: MultiQueryConfig,
        cache_client: Optional[CacheClient] = None,
        feature_store: Optional[FeatureStoreClient] = None,
    ):
        """Initialize the multi-query generator.

        Args:
            config: Multi-query configuration.
            cache_client: Cache client for user interactions.
            feature_store: Feature store for user/video features.
        """
        self.config = config
        self.cache_client = cache_client or InMemoryCacheClient(RedisCacheConfig())
        self.feature_store = feature_store or InMemoryFeatureStore(FeatureStoreConfig())

    def generate_query_contexts(
        self,
        user_id: int,
        base_user_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate multiple query contexts for a user.

        Each context represents a different user embedding based on
        recent categories the user has interacted with.

        Args:
            user_id: User identifier.
            base_user_data: Base user feature dictionary.

        Returns:
            List of user data dictionaries for each query.
        """
        query_contexts = []

        # Query 1: Primary query with base user features
        primary_context = base_user_data.copy()
        primary_context["_query_type"] = "primary"
        primary_context["_query_id"] = "q_primary"
        query_contexts.append(primary_context)

        # Get recent categories from cache
        recent_categories = self.cache_client.get_recent_categories(
            user_id, limit=self.config.max_queries - 1
        )

        # Generate category-based queries
        for i, category in enumerate(recent_categories):
            category_context = base_user_data.copy()
            # Override the previously_watched_category to create diverse embeddings
            category_context["previously_watched_category"] = category
            category_context["_query_type"] = "category"
            category_context["_query_id"] = f"q_cat_{i}"
            category_context["_category"] = category
            query_contexts.append(category_context)

            if len(query_contexts) >= self.config.max_queries:
                break

        logger.debug(
            f"Generated {len(query_contexts)} query contexts for user {user_id}"
        )

        return query_contexts

    def merge_results(
        self,
        query_results: List[QueryResult],
        max_results: int = 100,
    ) -> MultiQueryResult:
        """Merge results from multiple queries.

        Combines candidates from all queries, handles deduplication,
        and assigns weighted scores.

        Args:
            query_results: Results from individual queries.
            max_results: Maximum number of merged results.

        Returns:
            MultiQueryResult with merged candidates.
        """
        # Track all video scores from different sources
        video_scores: Dict[int, List[Tuple[float, str, float]]] = {}  # video_id -> [(score, source, weight)]
        video_sources: Dict[int, List[str]] = {}

        for result in query_results:
            # Determine weight based on query type
            if result.query_type == "primary":
                weight = self.config.primary_query_weight
            elif result.query_type == "category":
                weight = self.config.category_query_weight
            else:
                weight = self.config.recent_interaction_weight

            for vid, score in zip(result.video_ids, result.scores):
                if vid not in video_scores:
                    video_scores[vid] = []
                    video_sources[vid] = []

                video_scores[vid].append((score, result.query_id, weight))
                video_sources[vid].append(result.query_id)

        # Calculate merged scores
        merged_scores_dict = {}
        for vid, score_list in video_scores.items():
            # Weighted average with bonus for appearing in multiple queries
            total_weight = sum(w for _, _, w in score_list)
            weighted_score = sum(s * w for s, _, w in score_list) / total_weight if total_weight > 0 else 0

            # Bonus for diversity (appearing in multiple queries)
            diversity_bonus = min(len(score_list) - 1, 3) * 0.05
            merged_scores_dict[vid] = weighted_score + diversity_bonus

        # Sort by merged score
        sorted_videos = sorted(
            merged_scores_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_results]

        merged_video_ids = [vid for vid, _ in sorted_videos]
        merged_scores = [score for _, score in sorted_videos]

        # Calculate category coverage
        categories_seen = set()
        for result in query_results:
            if result.category:
                categories_seen.add(result.category)

        total_candidates = sum(len(r.video_ids) for r in query_results)

        return MultiQueryResult(
            query_results=query_results,
            merged_video_ids=merged_video_ids,
            merged_scores=merged_scores,
            video_sources=video_sources,
            total_candidates=total_candidates,
            unique_candidates=len(merged_video_ids),
            category_coverage=len(categories_seen),
        )

    def create_query_result(
        self,
        query_context: Dict[str, Any],
        video_ids: List[int],
        scores: List[float],
        embedding: Optional[np.ndarray] = None,
    ) -> QueryResult:
        """Create a QueryResult from query context and results.

        Args:
            query_context: Query context dictionary.
            video_ids: Retrieved video IDs.
            scores: Similarity scores.
            embedding: User embedding used for query.

        Returns:
            QueryResult object.
        """
        return QueryResult(
            query_id=query_context.get("_query_id", "unknown"),
            query_type=query_context.get("_query_type", "unknown"),
            category=query_context.get("_category"),
            video_ids=video_ids,
            scores=scores,
            embedding=embedding,
        )


class DiversitySampler:
    """Samples candidates to ensure category diversity.

    Ensures that final recommendations include videos from
    diverse categories even if some categories have lower scores.

    Example:
        >>> sampler = DiversitySampler(min_categories=3)
        >>> diverse_videos = sampler.sample(candidates, video_metadata, n=20)
    """

    def __init__(self, min_categories: int = 3, category_cap: int = 5):
        """Initialize the diversity sampler.

        Args:
            min_categories: Minimum number of categories to include.
            category_cap: Maximum videos from any single category.
        """
        self.min_categories = min_categories
        self.category_cap = category_cap

    def sample(
        self,
        video_ids: List[int],
        scores: List[float],
        video_metadata: Dict[int, Dict[str, Any]],
        n: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """Sample diverse candidates.

        Args:
            video_ids: Candidate video IDs (sorted by score).
            scores: Corresponding scores.
            video_metadata: Video metadata with category info.
            n: Number of results to return.

        Returns:
            Tuple of (video_ids, scores).
        """
        if len(video_ids) <= n:
            return video_ids, scores

        # Group by category
        category_videos: Dict[str, List[Tuple[int, float]]] = {}
        for vid, score in zip(video_ids, scores):
            category = video_metadata.get(vid, {}).get("category", "Unknown")
            if category not in category_videos:
                category_videos[category] = []
            category_videos[category].append((vid, score))

        # Ensure minimum category coverage first
        selected_videos = []
        selected_scores = []
        categories_covered = set()

        # First pass: take top video from each category
        for category in sorted(
            category_videos.keys(),
            key=lambda c: max(s for _, s in category_videos[c]),
            reverse=True,
        ):
            if len(selected_videos) >= n:
                break

            if category_videos[category]:
                vid, score = category_videos[category][0]
                selected_videos.append(vid)
                selected_scores.append(score)
                categories_covered.add(category)
                category_videos[category] = category_videos[category][1:]

        # Second pass: fill remaining slots with best remaining videos
        remaining = []
        for category, videos in category_videos.items():
            # Apply category cap
            videos_added = sum(
                1 for v in selected_videos
                if video_metadata.get(v, {}).get("category") == category
            )
            for vid, score in videos:
                if videos_added < self.category_cap:
                    remaining.append((vid, score, category))
                    videos_added += 1

        remaining.sort(key=lambda x: x[1], reverse=True)

        for vid, score, category in remaining:
            if len(selected_videos) >= n:
                break
            if vid not in selected_videos:
                selected_videos.append(vid)
                selected_scores.append(score)

        return selected_videos, selected_scores

    def ensure_category_coverage(
        self,
        video_ids: List[int],
        scores: List[float],
        video_metadata: Dict[int, Dict[str, Any]],
        required_categories: List[str],
        n: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """Ensure specific categories are represented.

        Args:
            video_ids: Candidate video IDs.
            scores: Corresponding scores.
            video_metadata: Video metadata.
            required_categories: Categories that must be represented.
            n: Number of results.

        Returns:
            Tuple of (video_ids, scores).
        """
        # First ensure required categories
        selected = []
        selected_scores = []

        for category in required_categories:
            for vid, score in zip(video_ids, scores):
                if vid in selected:
                    continue
                if video_metadata.get(vid, {}).get("category") == category:
                    selected.append(vid)
                    selected_scores.append(score)
                    break

        # Fill remaining with top scores
        for vid, score in zip(video_ids, scores):
            if len(selected) >= n:
                break
            if vid not in selected:
                selected.append(vid)
                selected_scores.append(score)

        return selected, selected_scores
