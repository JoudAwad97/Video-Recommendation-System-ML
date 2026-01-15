"""
User feature transformer for Two-Tower model.

Transforms raw user features into model-ready format including
vocabulary lookups, normalization, and bucketing.
"""

from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from ..preprocessing.vocabulary_builder import StringLookupVocabulary, IntegerLookupVocabulary
from ..preprocessing.normalizers import StandardNormalizer, BucketTransformer
from ..preprocessing.artifacts import ArtifactManager
from ..config.feature_config import (
    FeatureConfig,
    DEFAULT_CONFIG,
    START_TOKEN,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class UserFeatureTransformer:
    """Transform user features for Two-Tower model.

    Handles the following transformations:
    - user_id → IntegerLookup index
    - country → StringLookup index
    - user_language → StringLookup index (shared with video_language)
    - age → Normalized value + bucket index
    - previously_watched_category → StringLookup index with [START] token

    Example:
        >>> transformer = UserFeatureTransformer()
        >>> transformer.fit(users_df)
        >>> transformed_df = transformer.transform(users_df)
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_CONFIG,
        artifacts_dir: Optional[str] = None
    ):
        """Initialize the user feature transformer.

        Args:
            config: Feature configuration.
            artifacts_dir: Directory for saving/loading artifacts.
        """
        self.config = config
        self.artifacts_dir = artifacts_dir

        # Initialize transformers
        self.user_id_vocab = IntegerLookupVocabulary(name="user_id")
        self.country_vocab = StringLookupVocabulary(name="country")
        self.language_vocab = StringLookupVocabulary(name="language")  # Shared
        self.category_vocab = StringLookupVocabulary(
            name="category",
            include_start=True,
            mask_value="-"
        )

        self.age_normalizer = StandardNormalizer(name="age")
        self.age_bucketer = BucketTransformer(
            name="age",
            boundaries=config.buckets.age_boundaries
        )

        self._is_fitted = False

    def fit(
        self,
        users_df: pd.DataFrame,
        videos_df: Optional[pd.DataFrame] = None,
        categories: Optional[list] = None
    ) -> "UserFeatureTransformer":
        """Fit all transformers on user data.

        Args:
            users_df: DataFrame with user data.
            videos_df: Optional video DataFrame for shared language vocab.
            categories: Optional list of categories for shared category vocab.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting user feature transformer...")

        # Fit user_id vocabulary
        self.user_id_vocab.build(users_df["id"])
        logger.info(f"User ID vocab size: {self.user_id_vocab.vocab_size}")

        # Fit country vocabulary
        self.country_vocab.build(users_df["country_code"])
        logger.info(f"Country vocab size: {self.country_vocab.vocab_size}")

        # Fit language vocabulary (combine user and video languages)
        languages = list(users_df["preferred_language"].unique())
        if videos_df is not None and "language" in videos_df.columns:
            languages.extend(videos_df["language"].unique())
        self.language_vocab.build(languages)
        logger.info(f"Language vocab size: {self.language_vocab.vocab_size}")

        # Fit category vocabulary
        if categories:
            self.category_vocab.build(categories)
        elif videos_df is not None and "category" in videos_df.columns:
            self.category_vocab.build(videos_df["category"])
        else:
            # Default categories
            default_categories = [
                "Technology", "Gaming", "Music", "Entertainment",
                "Education", "Sports", "News", "Cooking",
                "Travel", "Fitness", "Fashion", "Science"
            ]
            self.category_vocab.build(default_categories)
        logger.info(f"Category vocab size: {self.category_vocab.vocab_size}")

        # Fit age transformers
        self.age_normalizer.fit(users_df["age"])
        logger.info(f"Age stats: {self.age_normalizer.get_stats()}")

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_raw: bool = False
    ) -> pd.DataFrame:
        """Transform user features to model-ready format.

        Args:
            df: DataFrame with user features.
            include_raw: Whether to include raw features in output.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = {}

        # Transform user_id
        if "user_id" in df.columns:
            result["user_id_idx"] = self.user_id_vocab.lookup_batch(df["user_id"])
        elif "id" in df.columns:
            result["user_id_idx"] = self.user_id_vocab.lookup_batch(df["id"])

        # Transform country
        if "country" in df.columns:
            result["country_idx"] = self.country_vocab.lookup_batch(df["country"])
        elif "country_code" in df.columns:
            result["country_idx"] = self.country_vocab.lookup_batch(df["country_code"])

        # Transform user_language
        if "user_language" in df.columns:
            result["user_language_idx"] = self.language_vocab.lookup_batch(df["user_language"])
        elif "preferred_language" in df.columns:
            result["user_language_idx"] = self.language_vocab.lookup_batch(df["preferred_language"])

        # Transform age
        if "age" in df.columns:
            result["age_normalized"] = self.age_normalizer.transform(df["age"])
            result["age_bucket_idx"] = self.age_bucketer.transform(df["age"])

        # Transform previously_watched_category
        if "previously_watched_category" in df.columns:
            result["prev_category_idx"] = self.category_vocab.lookup_batch(
                df["previously_watched_category"]
            )

        # Include raw features if requested
        if include_raw:
            for col in ["user_id", "country", "user_language", "age", "previously_watched_category"]:
                if col in df.columns:
                    result[f"{col}_raw"] = df[col].values

        return pd.DataFrame(result)

    def transform_single(self, user_data: Dict) -> Dict:
        """Transform a single user's features.

        Args:
            user_data: Dictionary with user features.

        Returns:
            Dictionary with transformed features.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = {}

        # Transform user_id
        user_id = user_data.get("user_id") or user_data.get("id")
        if user_id is not None:
            result["user_id_idx"] = self.user_id_vocab.lookup(user_id)

        # Transform country
        country = user_data.get("country") or user_data.get("country_code")
        if country is not None:
            result["country_idx"] = self.country_vocab.lookup(country)

        # Transform user_language
        language = user_data.get("user_language") or user_data.get("preferred_language")
        if language is not None:
            result["user_language_idx"] = self.language_vocab.lookup(language)

        # Transform age
        age = user_data.get("age")
        if age is not None:
            result["age_normalized"] = float(self.age_normalizer.transform([age])[0])
            result["age_bucket_idx"] = int(self.age_bucketer.transform([age])[0])

        # Transform previously_watched_category
        prev_category = user_data.get("previously_watched_category")
        if prev_category is not None:
            result["prev_category_idx"] = self.category_vocab.lookup(prev_category)

        return result

    def save(self, artifacts_dir: Optional[str] = None) -> None:
        """Save all artifacts.

        Args:
            artifacts_dir: Directory to save artifacts (uses default if None).
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        save_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(save_dir)

        # Save vocabularies
        artifact_manager.save_vocabulary(self.user_id_vocab)
        artifact_manager.save_vocabulary(self.country_vocab)
        artifact_manager.save_vocabulary(self.language_vocab)
        artifact_manager.save_vocabulary(self.category_vocab)

        # Save normalizer stats
        artifact_manager.save_normalizer_stats("age", self.age_normalizer.get_stats())

        # Save bucket boundaries
        artifact_manager.save_bucket_boundaries("age", self.config.buckets.age_boundaries)

        logger.info(f"Saved user feature artifacts to {save_dir}")

    def load(self, artifacts_dir: Optional[str] = None) -> "UserFeatureTransformer":
        """Load all artifacts.

        Args:
            artifacts_dir: Directory to load artifacts from.

        Returns:
            Self for method chaining.
        """
        load_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(load_dir)

        # Load vocabularies
        self.user_id_vocab = artifact_manager.load_vocabulary("user_id", "integer")
        self.country_vocab = artifact_manager.load_vocabulary("country", "string")
        self.language_vocab = artifact_manager.load_vocabulary("language", "string")
        self.category_vocab = artifact_manager.load_vocabulary("category", "string")

        # Load normalizer stats
        age_stats = artifact_manager.load_normalizer_stats("age")
        self.age_normalizer.mean = age_stats["mean"]
        self.age_normalizer.std = age_stats["std"]
        self.age_normalizer._is_fitted = True

        # Load bucket boundaries
        age_boundaries = artifact_manager.load_bucket_boundaries("age")
        self.age_bucketer.boundaries = age_boundaries
        self.age_bucketer._is_fitted = True

        self._is_fitted = True
        logger.info(f"Loaded user feature artifacts from {load_dir}")

        return self

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layer initialization.

        Returns:
            Dictionary mapping feature names to vocab sizes.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        return {
            "user_id": self.user_id_vocab.vocab_size,
            "country": self.country_vocab.vocab_size,
            "language": self.language_vocab.vocab_size,
            "category": self.category_vocab.vocab_size,
            "age_bucket": self.age_bucketer.num_output_buckets,
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"UserFeatureTransformer(fitted=True, vocabs={self.get_vocab_sizes()})"
        return "UserFeatureTransformer(fitted=False)"
