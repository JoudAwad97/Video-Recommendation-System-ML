"""
Redis cache client for user interaction data.

Provides fast access to:
- Latest user interactions (categories watched)
- Recently watched videos for filtering
- Pre-computed recommendations for fast serving
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RedisCacheConfig:
    """Configuration for Redis cache."""

    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # AWS ElastiCache settings
    use_elasticache: bool = False
    elasticache_endpoint: str = ""

    # Key prefixes
    user_interactions_prefix: str = "user:interactions:"
    user_watched_prefix: str = "user:watched:"
    user_recommendations_prefix: str = "user:recs:"

    # TTL settings
    interactions_ttl_hours: int = 24
    watched_ttl_days: int = 30
    recommendations_ttl_minutes: int = 60

    # Limits
    max_recent_interactions: int = 50
    max_watched_history: int = 1000


@dataclass
class UserInteraction:
    """A user interaction event."""

    category: str
    video_id: int
    timestamp: str
    interaction_type: str = "watch"  # watch, like, click
    duration_watched: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "interaction_type": self.interaction_type,
            "duration_watched": self.duration_watched,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInteraction":
        """Create from dictionary."""
        return cls(**data)


class CacheClient(ABC):
    """Abstract base class for cache clients."""

    @abstractmethod
    def get_recent_interactions(self, user_id: int) -> List[UserInteraction]:
        """Get recent interactions for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of recent interactions.
        """
        pass

    @abstractmethod
    def get_recent_categories(self, user_id: int, limit: int = 10) -> List[str]:
        """Get recent categories a user interacted with.

        Args:
            user_id: User identifier.
            limit: Maximum number of categories.

        Returns:
            List of category names (most recent first).
        """
        pass

    @abstractmethod
    def get_watched_videos(self, user_id: int, days: int = 30) -> Set[int]:
        """Get videos watched by user in recent days.

        Args:
            user_id: User identifier.
            days: Number of days to look back.

        Returns:
            Set of video IDs.
        """
        pass

    @abstractmethod
    def add_interaction(self, user_id: int, interaction: UserInteraction) -> None:
        """Add an interaction to user's history.

        Args:
            user_id: User identifier.
            interaction: Interaction to add.
        """
        pass

    @abstractmethod
    def set_recommendations(
        self, user_id: int, video_ids: List[int], scores: Optional[List[float]] = None
    ) -> None:
        """Store pre-computed recommendations for a user.

        Args:
            user_id: User identifier.
            video_ids: List of recommended video IDs.
            scores: Optional list of scores.
        """
        pass

    @abstractmethod
    def get_recommendations(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get stored recommendations for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of recommendation dicts or None.
        """
        pass


class InMemoryCacheClient(CacheClient):
    """In-memory cache client for local development.

    Provides the same interface as Redis but stores data in memory.

    Example:
        >>> cache = InMemoryCacheClient(config)
        >>> cache.add_interaction(user_id=1, interaction=interaction)
        >>> categories = cache.get_recent_categories(user_id=1)
    """

    def __init__(self, config: RedisCacheConfig):
        """Initialize the in-memory cache.

        Args:
            config: Cache configuration.
        """
        self.config = config
        self._interactions: Dict[int, List[UserInteraction]] = {}
        self._watched: Dict[int, Set[int]] = {}
        self._recommendations: Dict[int, Dict[str, Any]] = {}

    def get_recent_interactions(self, user_id: int) -> List[UserInteraction]:
        """Get recent interactions for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of recent interactions.
        """
        if user_id not in self._interactions:
            return []

        # Return most recent first
        interactions = self._interactions[user_id]
        return interactions[-self.config.max_recent_interactions:][::-1]

    def get_recent_categories(self, user_id: int, limit: int = 10) -> List[str]:
        """Get recent categories a user interacted with.

        Args:
            user_id: User identifier.
            limit: Maximum number of categories.

        Returns:
            List of unique category names (most recent first).
        """
        interactions = self.get_recent_interactions(user_id)

        # Get unique categories preserving order
        seen = set()
        categories = []
        for interaction in interactions:
            if interaction.category not in seen:
                seen.add(interaction.category)
                categories.append(interaction.category)
                if len(categories) >= limit:
                    break

        return categories

    def get_watched_videos(self, user_id: int, days: int = 30) -> Set[int]:
        """Get videos watched by user.

        Args:
            user_id: User identifier.
            days: Number of days to look back (not used in memory impl).

        Returns:
            Set of video IDs.
        """
        return self._watched.get(user_id, set())

    def add_interaction(self, user_id: int, interaction: UserInteraction) -> None:
        """Add an interaction to user's history.

        Args:
            user_id: User identifier.
            interaction: Interaction to add.
        """
        if user_id not in self._interactions:
            self._interactions[user_id] = []

        self._interactions[user_id].append(interaction)

        # Trim if too long
        if len(self._interactions[user_id]) > self.config.max_recent_interactions * 2:
            self._interactions[user_id] = self._interactions[user_id][
                -self.config.max_recent_interactions:
            ]

        # Also track watched videos
        if user_id not in self._watched:
            self._watched[user_id] = set()
        self._watched[user_id].add(interaction.video_id)

        # Trim watched set if too large - convert to list to maintain FIFO order
        if len(self._watched[user_id]) > self.config.max_watched_history:
            # Keep only the most recent entries by converting to list
            watched_list = list(self._watched[user_id])
            # Keep the last max_watched_history items
            self._watched[user_id] = set(watched_list[-self.config.max_watched_history:])

    def set_recommendations(
        self, user_id: int, video_ids: List[int], scores: Optional[List[float]] = None
    ) -> None:
        """Store pre-computed recommendations.

        Args:
            user_id: User identifier.
            video_ids: List of recommended video IDs.
            scores: Optional list of scores.
        """
        self._recommendations[user_id] = {
            "video_ids": video_ids,
            "scores": scores or [1.0] * len(video_ids),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_recommendations(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get stored recommendations.

        Args:
            user_id: User identifier.

        Returns:
            List of recommendation dicts or None.
        """
        if user_id not in self._recommendations:
            return None

        data = self._recommendations[user_id]

        # Check if expired
        stored_time = datetime.fromisoformat(data["timestamp"])
        age_minutes = (datetime.utcnow() - stored_time).total_seconds() / 60
        if age_minutes > self.config.recommendations_ttl_minutes:
            return None

        return [
            {"video_id": vid, "score": score}
            for vid, score in zip(data["video_ids"], data["scores"])
        ]

    def add_watched_video(self, user_id: int, video_id: int) -> None:
        """Add a video to user's watched history.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
        """
        if user_id not in self._watched:
            self._watched[user_id] = set()
        self._watched[user_id].add(video_id)

    def clear_user_data(self, user_id: int) -> None:
        """Clear all data for a user.

        Args:
            user_id: User identifier.
        """
        self._interactions.pop(user_id, None)
        self._watched.pop(user_id, None)
        self._recommendations.pop(user_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "num_users_with_interactions": len(self._interactions),
            "num_users_with_watched": len(self._watched),
            "num_users_with_recommendations": len(self._recommendations),
            "total_interactions": sum(len(v) for v in self._interactions.values()),
        }


class RedisCacheClient(CacheClient):
    """Redis cache client for production use.

    Uses Redis for fast caching of user interaction data.
    Supports both standalone Redis and AWS ElastiCache.

    Example:
        >>> client = RedisCacheClient(config)
        >>> categories = client.get_recent_categories(user_id=123)
    """

    def __init__(self, config: RedisCacheConfig):
        """Initialize the Redis cache client.

        Args:
            config: Cache configuration.
        """
        self.config = config
        self._redis = None
        self._connected = False

        try:
            import redis
            if config.use_elasticache and config.elasticache_endpoint:
                self._redis = redis.Redis(
                    host=config.elasticache_endpoint,
                    port=config.port,
                    db=config.db,
                    decode_responses=True,
                )
            else:
                self._redis = redis.Redis(
                    host=config.host,
                    port=config.port,
                    db=config.db,
                    password=config.password,
                    decode_responses=True,
                )
            # Test connection
            self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis")
        except ImportError:
            logger.warning("redis package not available")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")

    def _get_interactions_key(self, user_id: int) -> str:
        """Get Redis key for user interactions."""
        return f"{self.config.user_interactions_prefix}{user_id}"

    def _get_watched_key(self, user_id: int) -> str:
        """Get Redis key for user watched videos."""
        return f"{self.config.user_watched_prefix}{user_id}"

    def _get_recommendations_key(self, user_id: int) -> str:
        """Get Redis key for user recommendations."""
        return f"{self.config.user_recommendations_prefix}{user_id}"

    def get_recent_interactions(self, user_id: int) -> List[UserInteraction]:
        """Get recent interactions from Redis.

        Args:
            user_id: User identifier.

        Returns:
            List of recent interactions.
        """
        if not self._connected:
            return []

        try:
            key = self._get_interactions_key(user_id)
            # Use LRANGE to get most recent items
            items = self._redis.lrange(key, 0, self.config.max_recent_interactions - 1)

            interactions = []
            for item in items:
                data = json.loads(item)
                interactions.append(UserInteraction.from_dict(data))

            return interactions

        except Exception as e:
            logger.warning(f"Failed to get interactions for user {user_id}: {e}")
            return []

    def get_recent_categories(self, user_id: int, limit: int = 10) -> List[str]:
        """Get recent categories from Redis.

        Args:
            user_id: User identifier.
            limit: Maximum number of categories.

        Returns:
            List of unique category names.
        """
        interactions = self.get_recent_interactions(user_id)

        seen = set()
        categories = []
        for interaction in interactions:
            if interaction.category not in seen:
                seen.add(interaction.category)
                categories.append(interaction.category)
                if len(categories) >= limit:
                    break

        return categories

    def get_watched_videos(self, user_id: int, days: int = 30) -> Set[int]:
        """Get watched videos from Redis.

        Args:
            user_id: User identifier.
            days: Number of days to look back.

        Returns:
            Set of video IDs.
        """
        if not self._connected:
            return set()

        try:
            key = self._get_watched_key(user_id)
            # Use HGETALL to get all watched videos
            watched = self._redis.hgetall(key)

            # Filter by timestamp if needed
            cutoff = datetime.utcnow() - timedelta(days=days)
            result = set()

            for video_id_str, timestamp_str in watched.items():
                try:
                    watched_time = datetime.fromisoformat(timestamp_str)
                    if watched_time >= cutoff:
                        result.add(int(video_id_str))
                except (ValueError, TypeError):
                    result.add(int(video_id_str))

            return result

        except Exception as e:
            logger.warning(f"Failed to get watched videos for user {user_id}: {e}")
            return set()

    def add_interaction(self, user_id: int, interaction: UserInteraction) -> None:
        """Add an interaction to Redis.

        Args:
            user_id: User identifier.
            interaction: Interaction to add.
        """
        if not self._connected:
            return

        try:
            # Add to interactions list (LPUSH for most recent first)
            interactions_key = self._get_interactions_key(user_id)
            self._redis.lpush(interactions_key, json.dumps(interaction.to_dict()))
            self._redis.ltrim(interactions_key, 0, self.config.max_recent_interactions - 1)
            self._redis.expire(
                interactions_key,
                self.config.interactions_ttl_hours * 3600,
            )

            # Add to watched videos (HSET with timestamp)
            watched_key = self._get_watched_key(user_id)
            self._redis.hset(
                watched_key,
                str(interaction.video_id),
                interaction.timestamp,
            )
            self._redis.expire(
                watched_key,
                self.config.watched_ttl_days * 86400,
            )

        except Exception as e:
            logger.warning(f"Failed to add interaction for user {user_id}: {e}")

    def set_recommendations(
        self, user_id: int, video_ids: List[int], scores: Optional[List[float]] = None
    ) -> None:
        """Store recommendations in Redis.

        Args:
            user_id: User identifier.
            video_ids: List of recommended video IDs.
            scores: Optional list of scores.
        """
        if not self._connected:
            return

        try:
            key = self._get_recommendations_key(user_id)
            data = {
                "video_ids": video_ids,
                "scores": scores or [1.0] * len(video_ids),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._redis.set(
                key,
                json.dumps(data),
                ex=self.config.recommendations_ttl_minutes * 60,
            )

        except Exception as e:
            logger.warning(f"Failed to set recommendations for user {user_id}: {e}")

    def get_recommendations(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get stored recommendations from Redis.

        Args:
            user_id: User identifier.

        Returns:
            List of recommendation dicts or None.
        """
        if not self._connected:
            return None

        try:
            key = self._get_recommendations_key(user_id)
            data_str = self._redis.get(key)

            if not data_str:
                return None

            data = json.loads(data_str)
            return [
                {"video_id": vid, "score": score}
                for vid, score in zip(data["video_ids"], data["scores"])
            ]

        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            return None

    def update_watched_categories(
        self, user_id: int, category: str
    ) -> None:
        """Update the latest watched categories for a user.

        This is specifically for the multi-query diversity feature.

        Args:
            user_id: User identifier.
            category: Category name.
        """
        if not self._connected:
            return

        interaction = UserInteraction(
            category=category,
            video_id=0,  # Not needed for category tracking
            timestamp=datetime.utcnow().isoformat(),
            interaction_type="category_view",
        )
        self.add_interaction(user_id, interaction)


def create_cache_client(
    config: RedisCacheConfig,
    use_redis: bool = False,
) -> CacheClient:
    """Factory function to create a cache client.

    Args:
        config: Cache configuration.
        use_redis: Whether to use Redis.

    Returns:
        CacheClient instance.
    """
    if use_redis:
        return RedisCacheClient(config)
    else:
        return InMemoryCacheClient(config)
