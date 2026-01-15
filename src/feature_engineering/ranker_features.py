"""
Ranker feature transformer for the ranking model.

Transforms features for the ranker model including both categorical
and numeric features with appropriate transformations.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from ..preprocessing.vocabulary_builder import StringLookupVocabulary
from ..preprocessing.normalizers import (
    StandardNormalizer,
    LogTransformer,
    BucketTransformer,
    CyclicalEncoder,
)
from ..preprocessing.artifacts import ArtifactManager
from ..config.feature_config import (
    FeatureConfig,
    DEFAULT_CONFIG,
    POPULARITY_LEVELS,
    DEVICE_TYPES,
    DAYS_OF_WEEK,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RankerFeatureTransformer:
    """Transform features for the ranker model.

    Handles both categorical and numeric features with appropriate
    transformations for gradient boosting (CatBoost) or neural network rankers.

    Categorical features (CatBoost handles directly):
    - country, user_language, category, child_categories
    - video_language, interaction_time_day, device

    Numeric features (with transformations):
    - age → normalize + bucket
    - video_duration → log + normalize + bucket
    - comment_count, like_count, view_count → log + ratio
    - channel_subscriber_count → log + tier bin
    - interaction_time_hour → direct + cyclical (sin/cos)

    Example:
        >>> transformer = RankerFeatureTransformer()
        >>> transformer.fit(ranker_df)
        >>> transformed_df = transformer.transform(ranker_df)
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_CONFIG,
        artifacts_dir: Optional[str] = None
    ):
        """Initialize the ranker feature transformer.

        Args:
            config: Feature configuration.
            artifacts_dir: Directory for saving/loading artifacts.
        """
        self.config = config
        self.artifacts_dir = artifacts_dir

        # Categorical vocabularies (for encoding, CatBoost can use directly)
        self.country_vocab = StringLookupVocabulary(name="ranker_country")
        self.user_language_vocab = StringLookupVocabulary(name="ranker_user_language")
        self.category_vocab = StringLookupVocabulary(name="ranker_category")
        self.child_categories_vocab = StringLookupVocabulary(name="ranker_child_categories")
        self.video_language_vocab = StringLookupVocabulary(name="ranker_video_language")
        self.device_vocab = StringLookupVocabulary(name="ranker_device")
        self.day_vocab = StringLookupVocabulary(name="ranker_day")
        self.popularity_vocab = StringLookupVocabulary(name="ranker_popularity")

        # Numeric transformers
        self.age_normalizer = StandardNormalizer(name="ranker_age")
        self.age_bucketer = BucketTransformer(
            name="ranker_age",
            boundaries=config.buckets.age_boundaries
        )

        self.duration_log = LogTransformer(name="ranker_duration")
        self.duration_normalizer = StandardNormalizer(name="ranker_duration")
        self.duration_bucketer = BucketTransformer(
            name="ranker_duration",
            boundaries=config.buckets.duration_boundaries
        )

        self.view_count_log = LogTransformer(name="view_count")
        self.like_count_log = LogTransformer(name="like_count")
        self.comment_count_log = LogTransformer(name="comment_count")

        self.subscriber_log = LogTransformer(name="subscriber_count")
        self.subscriber_bucketer = BucketTransformer(
            name="subscriber_tier",
            boundaries=config.buckets.subscriber_tier_boundaries
        )

        self.hour_encoder = CyclicalEncoder(name="hour", period=24)

        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "RankerFeatureTransformer":
        """Fit all transformers on ranker data.

        Args:
            df: DataFrame with ranker training data.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting ranker feature transformer...")

        # Fit categorical vocabularies
        if "country" in df.columns:
            self.country_vocab.build(df["country"].dropna())
        if "user_language" in df.columns:
            self.user_language_vocab.build(df["user_language"].dropna())
        if "category" in df.columns:
            self.category_vocab.build(df["category"].dropna())
        if "child_categories" in df.columns:
            # Handle pipe-separated child categories
            all_children = set()
            for cats in df["child_categories"].dropna():
                all_children.update(str(cats).split("|"))
            self.child_categories_vocab.build(list(all_children))
        if "video_language" in df.columns:
            self.video_language_vocab.build(df["video_language"].dropna())
        if "device" in df.columns:
            self.device_vocab.build(df["device"].dropna())
        if "interaction_time_day" in df.columns:
            self.day_vocab.build(df["interaction_time_day"].dropna())
        if "popularity" in df.columns:
            self.popularity_vocab.build(POPULARITY_LEVELS)

        # Fit numeric transformers
        if "age" in df.columns:
            self.age_normalizer.fit(df["age"].dropna())
            logger.info(f"Age stats: {self.age_normalizer.get_stats()}")

        if "video_duration" in df.columns:
            log_duration = self.duration_log.transform(df["video_duration"].dropna())
            self.duration_normalizer.fit(log_duration)
            logger.info(f"Duration stats (log): {self.duration_normalizer.get_stats()}")

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        for_catboost: bool = True,
        include_raw: bool = False
    ) -> pd.DataFrame:
        """Transform ranker features.

        Args:
            df: DataFrame with ranker data.
            for_catboost: If True, keep categorical as strings. If False, encode as indices.
            include_raw: Whether to include raw features.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = {}

        # === Categorical Features ===
        if for_catboost:
            # Keep as strings for CatBoost
            for col in ["country", "user_language", "category", "child_categories",
                        "video_language", "device", "interaction_time_day", "popularity"]:
                if col in df.columns:
                    result[col] = df[col].fillna("UNKNOWN").astype(str)
        else:
            # Encode as indices for neural network
            if "country" in df.columns:
                result["country_idx"] = self.country_vocab.lookup_batch(df["country"])
            if "user_language" in df.columns:
                result["user_language_idx"] = self.user_language_vocab.lookup_batch(df["user_language"])
            if "category" in df.columns:
                result["category_idx"] = self.category_vocab.lookup_batch(df["category"])
            if "video_language" in df.columns:
                result["video_language_idx"] = self.video_language_vocab.lookup_batch(df["video_language"])
            if "device" in df.columns:
                result["device_idx"] = self.device_vocab.lookup_batch(df["device"])
            if "interaction_time_day" in df.columns:
                result["day_idx"] = self.day_vocab.lookup_batch(df["interaction_time_day"])
            if "popularity" in df.columns:
                result["popularity_idx"] = self.popularity_vocab.lookup_batch(df["popularity"])

        # === Numeric Features ===

        # Age
        if "age" in df.columns:
            result["age_normalized"] = self.age_normalizer.transform(df["age"].fillna(30))
            result["age_bucket"] = self.age_bucketer.transform(df["age"].fillna(30))

        # Video duration
        if "video_duration" in df.columns:
            log_duration = self.duration_log.transform(df["video_duration"].fillna(300))
            result["duration_log_normalized"] = self.duration_normalizer.transform(log_duration)
            result["duration_bucket"] = self.duration_bucketer.transform(df["video_duration"].fillna(300))

        # View count
        if "view_count" in df.columns:
            result["view_count_log"] = self.view_count_log.transform(df["view_count"].fillna(0))

        # Like count
        if "like_count" in df.columns:
            result["like_count_log"] = self.like_count_log.transform(df["like_count"].fillna(0))
            # Like ratio (likes per view)
            if "view_count" in df.columns:
                views = df["view_count"].fillna(1).replace(0, 1)
                result["like_ratio"] = df["like_count"].fillna(0) / views

        # Comment count
        if "comment_count" in df.columns:
            result["comment_count_log"] = self.comment_count_log.transform(df["comment_count"].fillna(0))
            # Comment ratio (comments per view)
            if "view_count" in df.columns:
                views = df["view_count"].fillna(1).replace(0, 1)
                result["comment_ratio"] = df["comment_count"].fillna(0) / views

        # Channel subscriber count
        if "channel_subscriber_count" in df.columns:
            result["subscriber_count_log"] = self.subscriber_log.transform(
                df["channel_subscriber_count"].fillna(0)
            )
            result["subscriber_tier"] = self.subscriber_bucketer.transform(
                df["channel_subscriber_count"].fillna(0)
            )
            # Engagement rate (likes per subscriber)
            if "like_count" in df.columns:
                subs = df["channel_subscriber_count"].fillna(1).replace(0, 1)
                result["engagement_rate"] = df["like_count"].fillna(0) / subs

        # Interaction hour - direct and cyclical
        if "interaction_time_hour" in df.columns:
            hours = df["interaction_time_hour"].fillna(12)
            result["hour"] = hours
            cyclical = self.hour_encoder.transform(hours)
            result["hour_sin"] = cyclical[:, 0]
            result["hour_cos"] = cyclical[:, 1]

        # Include label if present
        if "label" in df.columns:
            result["label"] = df["label"].values

        # Include raw features if requested
        if include_raw:
            for col in df.columns:
                result[f"{col}_raw"] = df[col].values

        return pd.DataFrame(result)

    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names for CatBoost.

        Returns:
            List of categorical feature column names.
        """
        return [
            "country", "user_language", "category", "child_categories",
            "video_language", "device", "interaction_time_day", "popularity"
        ]

    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names.

        Returns:
            List of numeric feature column names.
        """
        return [
            "age_normalized", "age_bucket",
            "duration_log_normalized", "duration_bucket",
            "view_count_log", "like_count_log", "comment_count_log",
            "like_ratio", "comment_ratio",
            "subscriber_count_log", "subscriber_tier", "engagement_rate",
            "hour", "hour_sin", "hour_cos"
        ]

    def save(self, artifacts_dir: Optional[str] = None) -> None:
        """Save all artifacts.

        Args:
            artifacts_dir: Directory to save artifacts.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        save_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(save_dir)

        # Save vocabularies
        artifact_manager.save_vocabulary(self.country_vocab)
        artifact_manager.save_vocabulary(self.user_language_vocab)
        artifact_manager.save_vocabulary(self.category_vocab)
        artifact_manager.save_vocabulary(self.child_categories_vocab)
        artifact_manager.save_vocabulary(self.video_language_vocab)
        artifact_manager.save_vocabulary(self.device_vocab)
        artifact_manager.save_vocabulary(self.day_vocab)
        artifact_manager.save_vocabulary(self.popularity_vocab)

        # Save normalizer stats
        artifact_manager.save_normalizer_stats("ranker_age", self.age_normalizer.get_stats())
        artifact_manager.save_normalizer_stats("ranker_duration", self.duration_normalizer.get_stats())

        logger.info(f"Saved ranker feature artifacts to {save_dir}")

    def load(self, artifacts_dir: Optional[str] = None) -> "RankerFeatureTransformer":
        """Load all artifacts.

        Args:
            artifacts_dir: Directory to load artifacts from.

        Returns:
            Self for method chaining.
        """
        load_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(load_dir)

        # Load vocabularies
        self.country_vocab = artifact_manager.load_vocabulary("ranker_country", "string")
        self.user_language_vocab = artifact_manager.load_vocabulary("ranker_user_language", "string")
        self.category_vocab = artifact_manager.load_vocabulary("ranker_category", "string")
        self.child_categories_vocab = artifact_manager.load_vocabulary("ranker_child_categories", "string")
        self.video_language_vocab = artifact_manager.load_vocabulary("ranker_video_language", "string")
        self.device_vocab = artifact_manager.load_vocabulary("ranker_device", "string")
        self.day_vocab = artifact_manager.load_vocabulary("ranker_day", "string")
        self.popularity_vocab = artifact_manager.load_vocabulary("ranker_popularity", "string")

        # Load normalizer stats
        age_stats = artifact_manager.load_normalizer_stats("ranker_age")
        self.age_normalizer.mean = age_stats["mean"]
        self.age_normalizer.std = age_stats["std"]
        self.age_normalizer._is_fitted = True

        duration_stats = artifact_manager.load_normalizer_stats("ranker_duration")
        self.duration_normalizer.mean = duration_stats["mean"]
        self.duration_normalizer.std = duration_stats["std"]
        self.duration_normalizer._is_fitted = True

        self._is_fitted = True
        logger.info(f"Loaded ranker feature artifacts from {load_dir}")

        return self

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layers.

        Returns:
            Dictionary mapping feature names to vocab sizes.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        return {
            "country": self.country_vocab.vocab_size,
            "user_language": self.user_language_vocab.vocab_size,
            "category": self.category_vocab.vocab_size,
            "child_categories": self.child_categories_vocab.vocab_size,
            "video_language": self.video_language_vocab.vocab_size,
            "device": self.device_vocab.vocab_size,
            "day": self.day_vocab.vocab_size,
            "popularity": self.popularity_vocab.vocab_size,
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"RankerFeatureTransformer(fitted=True)"
        return "RankerFeatureTransformer(fitted=False)"
