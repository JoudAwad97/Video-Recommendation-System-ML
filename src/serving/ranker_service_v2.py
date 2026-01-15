"""
Enhanced ranker service with feature enrichment.

Fetches additional features from Feature Store and applies
the CatBoost ranker model to score video candidates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from ..utils.logging_utils import get_logger
from .feature_store_client import (
    FeatureStoreClient,
    InMemoryFeatureStore,
    FeatureStoreConfig,
)

logger = get_logger(__name__)


@dataclass
class RankerServiceConfig:
    """Configuration for ranker service."""

    # Model settings
    model_path: str = "models/ranker"

    # Feature settings
    user_features: List[str] = field(default_factory=lambda: [
        "user_id", "country", "user_language", "age",
    ])
    video_features: List[str] = field(default_factory=lambda: [
        "video_id", "category", "video_language", "video_duration",
        "popularity", "view_count", "like_count", "comment_count",
        "channel_subscriber_count",
    ])
    interaction_features: List[str] = field(default_factory=lambda: [
        "interaction_time_hour", "interaction_time_day", "device",
    ])

    # Scoring settings
    batch_size: int = 100
    score_normalization: bool = True

    # Feature enrichment
    enrich_from_feature_store: bool = True

    # Fallback settings
    default_score: float = 0.5


@dataclass
class RankedCandidate:
    """A ranked video candidate."""

    video_id: int
    ranker_score: float
    retrieval_score: float
    final_score: float
    rank: int
    features_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RankingResult:
    """Result of ranking operation."""

    ranked_candidates: List[RankedCandidate]
    total_scored: int
    scoring_latency_ms: float
    feature_fetch_latency_ms: float


class EnhancedRankerService:
    """Enhanced ranker service with feature enrichment.

    Provides scoring of video candidates using:
    1. Feature Store for user and video features
    2. CatBoost model for relevance scoring
    3. Score normalization and blending

    Example:
        >>> service = EnhancedRankerService(config, feature_store)
        >>> result = service.rank(
        ...     user_id=123,
        ...     candidates=[(1, 0.9), (2, 0.8)],
        ... )
    """

    def __init__(
        self,
        config: RankerServiceConfig,
        feature_store: Optional[FeatureStoreClient] = None,
    ):
        """Initialize the enhanced ranker service.

        Args:
            config: Ranker service configuration.
            feature_store: Feature store client.
        """
        self.config = config
        self.feature_store = feature_store or InMemoryFeatureStore(FeatureStoreConfig())
        self._model = None
        self._is_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the ranker model.

        Args:
            model_path: Path to model file.
        """
        model_path = model_path or self.config.model_path

        try:
            from catboost import CatBoostClassifier
            self._model = CatBoostClassifier()
            self._model.load_model(model_path)
            self._is_loaded = True
            logger.info(f"Loaded ranker model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load ranker model: {e}")
            self._is_loaded = False

    def rank(
        self,
        user_id: int,
        candidates: List[Tuple[int, float]],  # (video_id, retrieval_score)
        user_features: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        interaction_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """Rank video candidates for a user.

        Args:
            user_id: User identifier.
            candidates: List of (video_id, retrieval_score) tuples.
            user_features: Pre-fetched user features.
            video_metadata: Pre-fetched video metadata.
            interaction_context: Interaction context (time, device, etc.).
            top_k: Number of top results to return.

        Returns:
            RankingResult with ranked candidates.
        """
        import time
        start_time = time.time()
        feature_fetch_start = time.time()

        # Fetch user features if not provided
        if user_features is None and self.config.enrich_from_feature_store:
            user_features = self.feature_store.get_user_features(user_id) or {}

        user_features = user_features or {}

        # Fetch video features
        video_ids = [vid for vid, _ in candidates]
        if video_metadata is None and self.config.enrich_from_feature_store:
            video_metadata = self.feature_store.batch_get_video_features(video_ids)

        video_metadata = video_metadata or {}

        feature_fetch_latency = (time.time() - feature_fetch_start) * 1000

        # Prepare features and score
        scoring_start = time.time()
        ranked = self._score_candidates(
            user_id=user_id,
            candidates=candidates,
            user_features=user_features,
            video_metadata=video_metadata,
            interaction_context=interaction_context or {},
        )
        scoring_latency = (time.time() - scoring_start) * 1000

        # Sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        # Apply rank
        for i, candidate in enumerate(ranked):
            candidate.rank = i + 1

        # Limit to top_k
        if top_k:
            ranked = ranked[:top_k]

        return RankingResult(
            ranked_candidates=ranked,
            total_scored=len(candidates),
            scoring_latency_ms=scoring_latency,
            feature_fetch_latency_ms=feature_fetch_latency,
        )

    def _score_candidates(
        self,
        user_id: int,
        candidates: List[Tuple[int, float]],
        user_features: Dict[str, Any],
        video_metadata: Dict[int, Dict[str, Any]],
        interaction_context: Dict[str, Any],
    ) -> List[RankedCandidate]:
        """Score candidates using the ranker model.

        Args:
            user_id: User identifier.
            candidates: Candidate list.
            user_features: User features.
            video_metadata: Video metadata.
            interaction_context: Interaction context.

        Returns:
            List of RankedCandidate objects.
        """
        ranked = []

        # Prepare features for all candidates
        feature_rows = []
        for video_id, retrieval_score in candidates:
            video_features = video_metadata.get(video_id, {})

            # Combine features
            combined = self._prepare_features(
                user_id=user_id,
                video_id=video_id,
                user_features=user_features,
                video_features=video_features,
                interaction_context=interaction_context,
            )

            feature_rows.append((video_id, retrieval_score, combined))

        # Score with model if available
        if self._is_loaded and self._model is not None:
            # Batch prediction
            X = [row[2] for row in feature_rows]
            try:
                # Get probability of positive class
                scores = self._model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                scores = [self.config.default_score] * len(feature_rows)
        else:
            # Use retrieval scores as fallback
            scores = [row[1] for row in feature_rows]

        # Create ranked candidates
        for i, (video_id, retrieval_score, features) in enumerate(feature_rows):
            ranker_score = scores[i]

            # Blend retrieval and ranker scores
            final_score = self._blend_scores(retrieval_score, ranker_score)

            ranked.append(RankedCandidate(
                video_id=video_id,
                ranker_score=ranker_score,
                retrieval_score=retrieval_score,
                final_score=final_score,
                rank=0,  # Will be set after sorting
                features_used=features,
                metadata=video_metadata.get(video_id),
            ))

        return ranked

    def _prepare_features(
        self,
        user_id: int,
        video_id: int,
        user_features: Dict[str, Any],
        video_features: Dict[str, Any],
        interaction_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare feature dictionary for ranker model.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            user_features: User features.
            video_features: Video features.
            interaction_context: Interaction context.

        Returns:
            Combined feature dictionary.
        """
        features = {}

        # User features
        for feat in self.config.user_features:
            features[feat] = user_features.get(feat, 0)

        # Video features
        for feat in self.config.video_features:
            features[feat] = video_features.get(feat, 0)

        # Interaction features
        now = datetime.utcnow()
        features["interaction_time_hour"] = interaction_context.get("hour", now.hour)
        features["interaction_time_day"] = interaction_context.get("day", now.strftime("%A"))
        features["device"] = interaction_context.get("device", "unknown")

        # Derived features
        features["video_duration_log"] = np.log1p(features.get("video_duration", 0))
        features["view_count_log"] = np.log1p(features.get("view_count", 0))
        features["like_count_log"] = np.log1p(features.get("like_count", 0))

        # Channel tier
        subs = features.get("channel_subscriber_count", 0)
        if subs < 10000:
            features["channel_tier"] = "micro"
        elif subs < 100000:
            features["channel_tier"] = "small"
        elif subs < 1000000:
            features["channel_tier"] = "mid"
        else:
            features["channel_tier"] = "macro"

        # Engagement ratios
        views = features.get("view_count", 1)
        features["like_view_ratio"] = features.get("like_count", 0) / max(views, 1)
        features["comment_view_ratio"] = features.get("comment_count", 0) / max(views, 1)

        return features

    def _blend_scores(
        self,
        retrieval_score: float,
        ranker_score: float,
        retrieval_weight: float = 0.3,
    ) -> float:
        """Blend retrieval and ranker scores.

        Args:
            retrieval_score: Score from retrieval stage.
            ranker_score: Score from ranker model.
            retrieval_weight: Weight for retrieval score.

        Returns:
            Blended final score.
        """
        ranker_weight = 1.0 - retrieval_weight

        # Normalize scores to [0, 1] if needed
        if self.config.score_normalization:
            retrieval_score = min(max(retrieval_score, 0), 1)
            ranker_score = min(max(ranker_score, 0), 1)

        return retrieval_weight * retrieval_score + ranker_weight * ranker_score

    def warmup(self, sample_features: Optional[Dict[str, Any]] = None) -> None:
        """Warm up the ranker model.

        Args:
            sample_features: Sample features for warmup inference.
        """
        if not self._is_loaded:
            return

        # Create sample input
        if sample_features is None:
            sample_features = {
                "user_id": 0,
                "country": "US",
                "user_language": "en",
                "age": 25,
                "video_id": 0,
                "category": "Entertainment",
                "video_language": "en",
                "video_duration": 300,
                "popularity": "medium",
                "view_count": 1000,
                "like_count": 100,
                "comment_count": 10,
                "channel_subscriber_count": 10000,
                "interaction_time_hour": 12,
                "interaction_time_day": "Monday",
                "device": "mobile",
            }

        try:
            # Run warmup prediction
            self._model.predict_proba([sample_features])
            logger.info("Ranker model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")


class SageMakerRankerClient:
    """Client for calling SageMaker ranker endpoint.

    Used in production to call a deployed ranker model.

    Example:
        >>> client = SageMakerRankerClient(endpoint_name="ranker-endpoint")
        >>> scores = client.predict(features_batch)
    """

    def __init__(
        self,
        endpoint_name: str,
        aws_region: str = "us-east-1",
    ):
        """Initialize the SageMaker client.

        Args:
            endpoint_name: SageMaker endpoint name.
            aws_region: AWS region.
        """
        self.endpoint_name = endpoint_name
        self.aws_region = aws_region
        self._client = None
        self._available = False

        try:
            import boto3
            self._client = boto3.client(
                "sagemaker-runtime",
                region_name=aws_region,
            )
            self._available = True
        except ImportError:
            logger.warning("boto3 not available for SageMaker client")

    def predict(
        self,
        features_batch: List[Dict[str, Any]],
    ) -> List[float]:
        """Get predictions from SageMaker endpoint.

        Args:
            features_batch: List of feature dictionaries.

        Returns:
            List of prediction scores.
        """
        if not self._available:
            return [0.5] * len(features_batch)

        import json

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps({"instances": features_batch}),
            )

            result = json.loads(response["Body"].read().decode())
            predictions = result.get("predictions", [])

            # Extract scores (assuming probability of positive class)
            scores = []
            for pred in predictions:
                if isinstance(pred, list):
                    scores.append(pred[1] if len(pred) > 1 else pred[0])
                else:
                    scores.append(pred)

            return scores

        except Exception as e:
            logger.error(f"SageMaker prediction failed: {e}")
            return [0.5] * len(features_batch)

    def batch_predict(
        self,
        features_batches: List[List[Dict[str, Any]]],
    ) -> List[List[float]]:
        """Get predictions for multiple batches.

        Args:
            features_batches: List of feature batches.

        Returns:
            List of score lists.
        """
        return [self.predict(batch) for batch in features_batches]
