"""
Feature Store client for online feature retrieval.

Provides interfaces for fetching user and video features from
SageMaker Feature Store or local cache.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""

    # Feature group names
    user_feature_group: str = "user-features"
    video_feature_group: str = "video-features"

    # AWS settings
    aws_region: str = "us-east-1"

    # Cache settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    # Feature selection
    user_features: List[str] = field(default_factory=lambda: [
        "user_id", "country", "user_language", "age",
    ])
    video_features: List[str] = field(default_factory=lambda: [
        "video_id", "category", "video_language", "video_duration",
        "popularity", "view_count", "like_count", "comment_count",
        "channel_subscriber_count",
    ])

    # Timeout settings
    timeout_ms: int = 100  # 100ms timeout for feature store calls


class FeatureStoreClient(ABC):
    """Abstract base class for feature store clients."""

    @abstractmethod
    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary of user features or None.
        """
        pass

    @abstractmethod
    def get_video_features(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a video.

        Args:
            video_id: Video identifier.

        Returns:
            Dictionary of video features or None.
        """
        pass

    @abstractmethod
    def batch_get_video_features(
        self, video_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get features for multiple videos.

        Args:
            video_ids: List of video identifiers.

        Returns:
            Dictionary mapping video_id to features.
        """
        pass


class InMemoryFeatureStore(FeatureStoreClient):
    """In-memory feature store for local development and testing.

    Stores features in memory with optional TTL-based caching.

    Example:
        >>> store = InMemoryFeatureStore(config)
        >>> store.load_user_features({1: {"age": 25, "country": "US"}})
        >>> features = store.get_user_features(1)
    """

    def __init__(self, config: FeatureStoreConfig):
        """Initialize the in-memory feature store.

        Args:
            config: Feature store configuration.
        """
        self.config = config
        self._user_features: Dict[int, Dict[str, Any]] = {}
        self._video_features: Dict[int, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def load_user_features(
        self, features: Dict[int, Dict[str, Any]]
    ) -> None:
        """Load user features into memory.

        Args:
            features: Dictionary mapping user_id to features.
        """
        self._user_features.update(features)
        logger.info(f"Loaded features for {len(features)} users")

    def load_video_features(
        self, features: Dict[int, Dict[str, Any]]
    ) -> None:
        """Load video features into memory.

        Args:
            features: Dictionary mapping video_id to features.
        """
        self._video_features.update(features)
        logger.info(f"Loaded features for {len(features)} videos")

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary of user features or None.
        """
        return self._user_features.get(user_id)

    def get_video_features(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a video.

        Args:
            video_id: Video identifier.

        Returns:
            Dictionary of video features or None.
        """
        return self._video_features.get(video_id)

    def batch_get_video_features(
        self, video_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get features for multiple videos.

        Args:
            video_ids: List of video identifiers.

        Returns:
            Dictionary mapping video_id to features.
        """
        return {
            vid: self._video_features[vid]
            for vid in video_ids
            if vid in self._video_features
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "num_users": len(self._user_features),
            "num_videos": len(self._video_features),
        }


class SageMakerFeatureStoreClient(FeatureStoreClient):
    """SageMaker Feature Store client for production use.

    Fetches features from SageMaker Feature Store with caching.

    Example:
        >>> client = SageMakerFeatureStoreClient(config)
        >>> features = client.get_user_features(user_id=123)
    """

    def __init__(self, config: FeatureStoreConfig):
        """Initialize the SageMaker Feature Store client.

        Args:
            config: Feature store configuration.
        """
        self.config = config
        self._boto3_available = False
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        try:
            import boto3
            self._boto3_available = True
            self._featurestore_client = boto3.client(
                "sagemaker-featurestore-runtime",
                region_name=config.aws_region,
            )
        except ImportError:
            logger.warning("boto3 not available. SageMaker Feature Store will not work.")

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user from Feature Store.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary of user features or None.
        """
        cache_key = f"user_{user_id}"

        # Check cache
        if self.config.enable_cache and cache_key in self._cache:
            if self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        if not self._boto3_available:
            return None

        try:
            response = self._featurestore_client.get_record(
                FeatureGroupName=self.config.user_feature_group,
                RecordIdentifierValueAsString=str(user_id),
                FeatureNames=self.config.user_features,
            )

            features = self._parse_record(response.get("Record", []))

            # Update cache
            if self.config.enable_cache:
                self._cache[cache_key] = features
                self._cache_timestamps[cache_key] = datetime.utcnow()

            return features

        except Exception as e:
            logger.warning(f"Failed to get user features for {user_id}: {e}")
            return None

    def get_video_features(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a video from Feature Store.

        Args:
            video_id: Video identifier.

        Returns:
            Dictionary of video features or None.
        """
        cache_key = f"video_{video_id}"

        # Check cache
        if self.config.enable_cache and cache_key in self._cache:
            if self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        if not self._boto3_available:
            return None

        try:
            response = self._featurestore_client.get_record(
                FeatureGroupName=self.config.video_feature_group,
                RecordIdentifierValueAsString=str(video_id),
                FeatureNames=self.config.video_features,
            )

            features = self._parse_record(response.get("Record", []))

            # Update cache
            if self.config.enable_cache:
                self._cache[cache_key] = features
                self._cache_timestamps[cache_key] = datetime.utcnow()

            return features

        except Exception as e:
            logger.warning(f"Failed to get video features for {video_id}: {e}")
            return None

    def batch_get_video_features(
        self, video_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get features for multiple videos.

        Args:
            video_ids: List of video identifiers.

        Returns:
            Dictionary mapping video_id to features.
        """
        results = {}

        # Check cache first
        uncached_ids = []
        for vid in video_ids:
            cache_key = f"video_{vid}"
            if self.config.enable_cache and cache_key in self._cache:
                if self._is_cache_valid(cache_key):
                    results[vid] = self._cache[cache_key]
                    continue
            uncached_ids.append(vid)

        if not uncached_ids or not self._boto3_available:
            return results

        # Batch fetch uncached features
        try:
            # SageMaker FS batch get
            identifiers = [
                {
                    "FeatureGroupName": self.config.video_feature_group,
                    "RecordIdentifiersValueAsString": [str(vid) for vid in uncached_ids],
                }
            ]

            response = self._featurestore_client.batch_get_record(
                Identifiers=identifiers
            )

            for record in response.get("Records", []):
                features = self._parse_record(record.get("Record", []))
                video_id = int(features.get("video_id", 0))
                if video_id:
                    results[video_id] = features

                    # Update cache
                    if self.config.enable_cache:
                        cache_key = f"video_{video_id}"
                        self._cache[cache_key] = features
                        self._cache_timestamps[cache_key] = datetime.utcnow()

        except Exception as e:
            logger.warning(f"Failed to batch get video features: {e}")

        return results

    def _parse_record(self, record: List[Dict]) -> Dict[str, Any]:
        """Parse a Feature Store record.

        Args:
            record: List of feature name-value pairs.

        Returns:
            Parsed feature dictionary.
        """
        features = {}
        for item in record:
            name = item.get("FeatureName", "")
            value = item.get("ValueAsString", "")

            # Try to parse numeric values
            try:
                if "." in value:
                    features[name] = float(value)
                else:
                    features[name] = int(value)
            except ValueError:
                features[name] = value

        return features

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid.

        Args:
            cache_key: Cache key.

        Returns:
            True if cache is valid.
        """
        if cache_key not in self._cache_timestamps:
            return False

        age = (datetime.utcnow() - self._cache_timestamps[cache_key]).total_seconds()
        return age < self.config.cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Feature cache cleared")


def create_feature_store_client(
    config: FeatureStoreConfig,
    use_sagemaker: bool = False,
) -> FeatureStoreClient:
    """Factory function to create a feature store client.

    Args:
        config: Feature store configuration.
        use_sagemaker: Whether to use SageMaker Feature Store.

    Returns:
        FeatureStoreClient instance.
    """
    if use_sagemaker:
        return SageMakerFeatureStoreClient(config)
    else:
        return InMemoryFeatureStore(config)
