"""
Ranking service for scoring video candidates.

Uses the trained Ranker model (CatBoost) to score and rank
video candidates based on user-video feature combinations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from ..models.ranker import RankerModel
from ..models.model_config import RankerModelConfig
from ..feature_engineering.ranker_features import RankerFeatureTransformer
from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig
from .candidate_retrieval import VideoCandidate, RetrievalResult

logger = get_logger(__name__)


@dataclass
class RankedVideo:
    """A ranked video with score."""

    video_id: int
    retrieval_score: float
    ranker_score: float
    final_score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RankingResult:
    """Result of ranking candidates."""

    ranked_videos: List[RankedVideo]
    ranking_time_ms: float


class RankingService:
    """Service for ranking video candidates using the Ranker model.

    This service:
    1. Combines user features with candidate video features
    2. Generates ranking features
    3. Scores candidates using the CatBoost ranker
    4. Returns ranked candidates

    Example:
        >>> service = RankingService(config)
        >>> service.load_models()
        >>> result = service.rank(user_data, candidates, video_data)
    """

    def __init__(self, config: ServingConfig):
        """Initialize the ranking service.

        Args:
            config: Serving configuration.
        """
        self.config = config
        self.ranker_model: Optional[RankerModel] = None
        self.feature_transformer: Optional[RankerFeatureTransformer] = None
        self._is_loaded = False

    def load_models(
        self,
        model_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ) -> "RankingService":
        """Load the Ranker model and feature transformers.

        Args:
            model_path: Path to saved Ranker model.
            artifacts_path: Path to feature engineering artifacts.

        Returns:
            Self for method chaining.
        """
        model_path = model_path or self.config.ranker_model_path
        artifacts_path = artifacts_path or self.config.artifacts_path

        logger.info(f"Loading Ranker model from {model_path}")

        # Load model configuration
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            model_config = RankerModelConfig(**config_dict)
        else:
            model_config = RankerModelConfig()

        # Initialize and load ranker model
        self.ranker_model = RankerModel(
            model_config,
            cat_features=model_config.cat_features,
        )
        self.ranker_model.load(model_path)

        # Load feature transformer
        logger.info(f"Loading ranker feature transformer from {artifacts_path}")
        self.feature_transformer = RankerFeatureTransformer()
        self.feature_transformer.load(artifacts_path)

        self._is_loaded = True
        logger.info("Ranking service loaded successfully")

        return self

    def rank(
        self,
        user_data: Dict[str, Any],
        candidates: List[VideoCandidate],
        video_data: Dict[int, Dict[str, Any]],
        interaction_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """Rank video candidates for a user.

        Args:
            user_data: User features dictionary.
            candidates: List of video candidates from retrieval.
            video_data: Dictionary mapping video_id to video features.
            interaction_context: Context features (time, device, etc.).
            top_k: Number of top results to return.

        Returns:
            RankingResult with ranked videos.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        import time
        start_time = time.time()

        top_k = top_k or self.config.top_k_final

        if not candidates:
            return RankingResult(ranked_videos=[], ranking_time_ms=0)

        # Prepare features DataFrame
        features_df = self._prepare_ranking_features(
            user_data,
            candidates,
            video_data,
            interaction_context,
        )

        # Get ranker scores
        ranker_scores = self.ranker_model.predict_proba(features_df)

        # Combine with retrieval scores
        retrieval_scores = np.array([c.similarity_score for c in candidates])

        # Final score: combination of retrieval and ranker scores
        # Using weighted average (can be tuned)
        alpha = 0.3  # Weight for retrieval score
        final_scores = alpha * retrieval_scores + (1 - alpha) * ranker_scores

        # Sort by final score
        sorted_indices = np.argsort(final_scores)[::-1]

        # Create ranked videos
        ranked_videos = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            candidate = candidates[idx]
            ranked_video = RankedVideo(
                video_id=candidate.video_id,
                retrieval_score=float(retrieval_scores[idx]),
                ranker_score=float(ranker_scores[idx]),
                final_score=float(final_scores[idx]),
                rank=rank,
                metadata=candidate.metadata,
            )
            ranked_videos.append(ranked_video)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(f"Ranked {len(candidates)} candidates in {elapsed_ms:.2f}ms")

        return RankingResult(
            ranked_videos=ranked_videos,
            ranking_time_ms=elapsed_ms,
        )

    def _prepare_ranking_features(
        self,
        user_data: Dict[str, Any],
        candidates: List[VideoCandidate],
        video_data: Dict[int, Dict[str, Any]],
        interaction_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Prepare features for ranking.

        Args:
            user_data: User features.
            candidates: Video candidates.
            video_data: Video features by ID.
            interaction_context: Context features.

        Returns:
            DataFrame with ranking features.
        """
        rows = []

        for candidate in candidates:
            video_id = candidate.video_id
            video = video_data.get(video_id, {})

            # Combine user, video, and context features
            row = {
                # User features
                "country": user_data.get("country") or user_data.get("country_code", "US"),
                "user_language": user_data.get("user_language") or user_data.get("preferred_language", "en"),
                "age": user_data.get("age", 25),

                # Video features
                "category": video.get("category", "Unknown"),
                "child_categories": video.get("child_categories", "Unknown"),
                "video_language": video.get("language", "en"),
                "video_duration": video.get("duration", 300),
                "popularity": video.get("popularity", "low"),
                "view_count": video.get("view_count", 0),
                "like_count": video.get("like_count", 0),
                "comment_count": video.get("comment_count", 0),
                "channel_subscriber_count": video.get("channel_subscriber_count", 0),

                # Context features
                "device": interaction_context.get("device", "desktop") if interaction_context else "desktop",
                "interaction_time_hour": interaction_context.get("hour", 12) if interaction_context else 12,
                "interaction_time_day": interaction_context.get("day", "Monday") if interaction_context else "Monday",

                # Retrieval score as feature
                "retrieval_score": candidate.similarity_score,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Transform features using the ranker feature transformer
        # For simplicity, we'll use the raw features here
        # In production, you'd apply full transformations
        return self._apply_basic_transforms(df)

    def _apply_basic_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic feature transformations.

        Args:
            df: Raw features DataFrame.

        Returns:
            Transformed DataFrame.
        """
        result = df.copy()

        # Log transforms for count features
        for col in ["view_count", "like_count", "comment_count", "channel_subscriber_count"]:
            if col in result.columns:
                result[f"{col}_log"] = np.log1p(result[col])

        # Cyclical encoding for hour
        if "interaction_time_hour" in result.columns:
            hour = result["interaction_time_hour"]
            result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Video duration log
        if "video_duration" in result.columns:
            result["duration_log"] = np.log1p(result["video_duration"])

        # Engagement ratios
        if "channel_subscriber_count" in result.columns:
            subscribers = result["channel_subscriber_count"].clip(lower=1)
            if "like_count" in result.columns:
                result["like_subscriber_ratio"] = result["like_count"] / subscribers
            if "comment_count" in result.columns:
                result["comment_subscriber_ratio"] = result["comment_count"] / subscribers

        return result

    def rank_from_retrieval_result(
        self,
        retrieval_result: RetrievalResult,
        user_data: Dict[str, Any],
        video_data: Dict[int, Dict[str, Any]],
        interaction_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """Rank candidates from a RetrievalResult.

        Convenience method that takes a RetrievalResult directly.

        Args:
            retrieval_result: Result from candidate retrieval.
            user_data: User features dictionary.
            video_data: Video features by ID.
            interaction_context: Context features.
            top_k: Number of top results.

        Returns:
            RankingResult with ranked videos.
        """
        return self.rank(
            user_data=user_data,
            candidates=retrieval_result.candidates,
            video_data=video_data,
            interaction_context=interaction_context,
            top_k=top_k,
        )

    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Create dummy data
        dummy_df = pd.DataFrame([{
            "country": "US",
            "user_language": "en",
            "age": 25,
            "category": "Technology",
            "child_categories": "Programming",
            "video_language": "en",
            "video_duration": 300,
            "popularity": "medium",
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 10,
            "channel_subscriber_count": 10000,
            "device": "desktop",
            "interaction_time_hour": 12,
            "interaction_time_day": "Monday",
            "retrieval_score": 0.8,
        }])

        dummy_df = self._apply_basic_transforms(dummy_df)
        _ = self.ranker_model.predict_proba(dummy_df)

        logger.info("Ranking service warmed up")
