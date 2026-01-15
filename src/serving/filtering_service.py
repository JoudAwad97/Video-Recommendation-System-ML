"""
Filtering service for video recommendations.

Applies business rules to filter out inappropriate or irrelevant videos
before the ranking stage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta

from ..utils.logging_utils import get_logger
from .redis_cache_client import CacheClient, InMemoryCacheClient, RedisCacheConfig

logger = get_logger(__name__)


@dataclass
class FilteringConfig:
    """Configuration for filtering service."""

    # Watch history filtering
    filter_watched_videos: bool = True
    watched_video_lookback_days: int = 30

    # Video age filtering
    filter_old_videos: bool = False
    max_video_age_days: int = 365

    # Duration filtering
    enable_duration_filter: bool = True
    min_duration_seconds: int = 0
    max_duration_seconds: int = 7200  # 2 hours

    # Category filtering
    blocked_categories: List[str] = field(default_factory=list)

    # Content filtering
    filter_nsfw: bool = True
    filter_low_quality: bool = True
    min_quality_score: float = 0.3

    # Engagement filtering
    min_views: int = 0
    min_likes: int = 0

    # Channel filtering
    blocked_channel_ids: Set[int] = field(default_factory=set)

    # Custom filter settings
    enable_custom_filters: bool = True


@dataclass
class FilterResult:
    """Result of filtering operation."""

    passed_video_ids: List[int]
    filtered_video_ids: List[int]
    filter_reasons: Dict[int, str]  # video_id -> reason for filtering
    total_input: int
    total_passed: int
    total_filtered: int
    filter_stats: Dict[str, int]  # filter_name -> count filtered


class FilteringService:
    """Service for filtering video candidates.

    Applies various filtering rules based on:
    - User watch history (filter recently watched)
    - Video metadata (age, duration, category)
    - Content quality (NSFW, quality score)
    - User preferences (blocked categories, channels)

    Example:
        >>> service = FilteringService(config, cache_client)
        >>> result = service.filter(
        ...     video_ids=[1, 2, 3],
        ...     user_id=123,
        ...     video_metadata={...},
        ... )
    """

    def __init__(
        self,
        config: FilteringConfig,
        cache_client: Optional[CacheClient] = None,
    ):
        """Initialize the filtering service.

        Args:
            config: Filtering configuration.
            cache_client: Cache client for user data.
        """
        self.config = config
        self.cache_client = cache_client or InMemoryCacheClient(RedisCacheConfig())

        # Custom filters
        self._custom_filters: List[Callable[[int, Dict[str, Any]], Optional[str]]] = []

    def filter(
        self,
        video_ids: List[int],
        user_id: int,
        video_metadata: Dict[int, Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None,
        additional_exclusions: Optional[Set[int]] = None,
    ) -> FilterResult:
        """Apply filters to video candidates.

        Args:
            video_ids: List of candidate video IDs.
            user_id: User identifier.
            video_metadata: Dictionary of video metadata.
            user_preferences: Optional user preference overrides.
            additional_exclusions: Additional video IDs to exclude.

        Returns:
            FilterResult with passed and filtered videos.
        """
        passed = []
        filtered = []
        filter_reasons: Dict[int, str] = {}
        filter_stats: Dict[str, int] = {}

        # Get user's watched videos
        watched_videos = set()
        if self.config.filter_watched_videos:
            watched_videos = self.cache_client.get_watched_videos(
                user_id, days=self.config.watched_video_lookback_days
            )

        # Merge with additional exclusions
        exclusions = watched_videos.copy()
        if additional_exclusions:
            exclusions.update(additional_exclusions)

        # Get user preferences
        user_prefs = user_preferences or {}
        min_duration = user_prefs.get("min_duration", self.config.min_duration_seconds)
        max_duration = user_prefs.get("max_duration", self.config.max_duration_seconds)
        blocked_cats = set(user_prefs.get("blocked_categories", self.config.blocked_categories))

        for video_id in video_ids:
            metadata = video_metadata.get(video_id, {})
            reason = self._check_filters(
                video_id,
                metadata,
                exclusions,
                blocked_cats,
                min_duration,
                max_duration,
            )

            if reason:
                filtered.append(video_id)
                filter_reasons[video_id] = reason

                # Track stats
                filter_type = reason.split(":")[0]
                filter_stats[filter_type] = filter_stats.get(filter_type, 0) + 1
            else:
                passed.append(video_id)

        logger.debug(
            f"Filtered {len(filtered)}/{len(video_ids)} videos for user {user_id}"
        )

        return FilterResult(
            passed_video_ids=passed,
            filtered_video_ids=filtered,
            filter_reasons=filter_reasons,
            total_input=len(video_ids),
            total_passed=len(passed),
            total_filtered=len(filtered),
            filter_stats=filter_stats,
        )

    def _check_filters(
        self,
        video_id: int,
        metadata: Dict[str, Any],
        exclusions: Set[int],
        blocked_categories: Set[str],
        min_duration: int,
        max_duration: int,
    ) -> Optional[str]:
        """Check all filters for a video.

        Args:
            video_id: Video identifier.
            metadata: Video metadata.
            exclusions: Set of excluded video IDs.
            blocked_categories: Set of blocked categories.
            min_duration: Minimum video duration.
            max_duration: Maximum video duration.

        Returns:
            Reason string if filtered, None if passed.
        """
        # Exclusion check
        if video_id in exclusions:
            return "exclusion: recently watched or blocked"

        # Category check
        category = metadata.get("category", "")
        if category in blocked_categories:
            return f"category: blocked ({category})"

        # Duration check
        if self.config.enable_duration_filter:
            duration = metadata.get("video_duration", 0)
            if duration > 0:
                if duration < min_duration:
                    return f"duration: too short ({duration}s)"
                if duration > max_duration:
                    return f"duration: too long ({duration}s)"

        # Video age check
        if self.config.filter_old_videos:
            upload_date = metadata.get("upload_date")
            if upload_date:
                try:
                    upload_datetime = datetime.fromisoformat(upload_date)
                    age_days = (datetime.utcnow() - upload_datetime).days
                    if age_days > self.config.max_video_age_days:
                        return f"age: too old ({age_days} days)"
                except (ValueError, TypeError):
                    pass

        # NSFW check
        if self.config.filter_nsfw:
            is_nsfw = metadata.get("is_nsfw", False)
            if is_nsfw:
                return "content: NSFW flagged"

        # Quality check
        if self.config.filter_low_quality:
            quality_score = metadata.get("quality_score", 1.0)
            if quality_score < self.config.min_quality_score:
                return f"quality: low score ({quality_score:.2f})"

        # Engagement checks
        if self.config.min_views > 0:
            views = metadata.get("view_count", 0)
            if views < self.config.min_views:
                return f"engagement: low views ({views})"

        if self.config.min_likes > 0:
            likes = metadata.get("like_count", 0)
            if likes < self.config.min_likes:
                return f"engagement: low likes ({likes})"

        # Channel check
        channel_id = metadata.get("channel_id")
        if channel_id and channel_id in self.config.blocked_channel_ids:
            return f"channel: blocked ({channel_id})"

        # Custom filters
        if self.config.enable_custom_filters:
            for custom_filter in self._custom_filters:
                reason = custom_filter(video_id, metadata)
                if reason:
                    return f"custom: {reason}"

        return None

    def add_custom_filter(
        self,
        filter_func: Callable[[int, Dict[str, Any]], Optional[str]],
    ) -> None:
        """Add a custom filter function.

        Args:
            filter_func: Function that takes (video_id, metadata) and
                        returns a reason string if filtered, None otherwise.
        """
        self._custom_filters.append(filter_func)

    def filter_by_duration_preference(
        self,
        video_ids: List[int],
        video_metadata: Dict[int, Dict[str, Any]],
        duration_preference: str,  # "short", "medium", "long", "any"
    ) -> List[int]:
        """Filter videos by user's duration preference.

        Args:
            video_ids: Candidate video IDs.
            video_metadata: Video metadata.
            duration_preference: Duration preference.

        Returns:
            Filtered video IDs.
        """
        duration_ranges = {
            "short": (0, 300),  # 0-5 minutes
            "medium": (300, 1200),  # 5-20 minutes
            "long": (1200, float("inf")),  # 20+ minutes
            "any": (0, float("inf")),
        }

        min_dur, max_dur = duration_ranges.get(duration_preference, (0, float("inf")))

        filtered = []
        for vid in video_ids:
            duration = video_metadata.get(vid, {}).get("video_duration", 0)
            if min_dur <= duration <= max_dur:
                filtered.append(vid)

        return filtered

    def get_blocked_videos_for_user(self, user_id: int) -> Set[int]:
        """Get all videos that should be blocked for a user.

        Combines:
        - Recently watched videos
        - Explicitly blocked videos
        - Videos from blocked channels

        Args:
            user_id: User identifier.

        Returns:
            Set of blocked video IDs.
        """
        blocked = set()

        # Add watched videos
        blocked.update(
            self.cache_client.get_watched_videos(
                user_id, days=self.config.watched_video_lookback_days
            )
        )

        return blocked


class FilteringPipeline:
    """Pipeline for applying multiple filter stages.

    Applies filters in a specific order for efficiency:
    1. Exclusions (fastest)
    2. Category filters
    3. Duration filters
    4. Content quality filters
    5. Engagement filters

    Example:
        >>> pipeline = FilteringPipeline(config)
        >>> result = pipeline.run(video_ids, user_id, metadata)
    """

    def __init__(
        self,
        config: FilteringConfig,
        cache_client: Optional[CacheClient] = None,
    ):
        """Initialize the filtering pipeline.

        Args:
            config: Filtering configuration.
            cache_client: Cache client.
        """
        self.config = config
        self.service = FilteringService(config, cache_client)

    def run(
        self,
        video_ids: List[int],
        user_id: int,
        video_metadata: Dict[int, Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> FilterResult:
        """Run the filtering pipeline.

        Args:
            video_ids: Input video IDs.
            user_id: User identifier.
            video_metadata: Video metadata dictionary.
            user_preferences: User preferences.

        Returns:
            FilterResult with passed videos.
        """
        return self.service.filter(
            video_ids=video_ids,
            user_id=user_id,
            video_metadata=video_metadata,
            user_preferences=user_preferences,
        )

    def get_filter_stats_summary(
        self,
        filter_results: List[FilterResult],
    ) -> Dict[str, Any]:
        """Get summary statistics from multiple filter results.

        Args:
            filter_results: List of filter results.

        Returns:
            Summary statistics.
        """
        total_input = sum(r.total_input for r in filter_results)
        total_passed = sum(r.total_passed for r in filter_results)
        total_filtered = sum(r.total_filtered for r in filter_results)

        combined_stats: Dict[str, int] = {}
        for result in filter_results:
            for key, count in result.filter_stats.items():
                combined_stats[key] = combined_stats.get(key, 0) + count

        return {
            "total_input": total_input,
            "total_passed": total_passed,
            "total_filtered": total_filtered,
            "pass_rate": total_passed / total_input if total_input > 0 else 0,
            "filter_breakdown": combined_stats,
        }
