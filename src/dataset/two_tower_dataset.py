"""
Two-Tower dataset generator.

Generates training dataset for the Two-Tower retrieval model
containing only positive user-video interaction pairs.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..feature_engineering.user_features import UserFeatureTransformer
from ..feature_engineering.video_features import VideoFeatureTransformer
from ..feature_engineering.interaction_features import InteractionFeatureProcessor
from ..config.feature_config import FeatureConfig, DEFAULT_CONFIG
from ..utils.io_utils import save_parquet, load_parquet, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TwoTowerDatasetGenerator:
    """Generate dataset for Two-Tower model training.

    Creates a dataset of positive user-video pairs with features
    for both user and video towers.

    The dataset includes:
    - User features: user_id, country, user_language, age, previously_watched_category
    - Video features: video_id, category, title, video_duration, popularity, video_language, tags

    Example:
        >>> generator = TwoTowerDatasetGenerator()
        >>> generator.fit(users_df, videos_df, interactions_df)
        >>> train_df, val_df, test_df = generator.generate_splits()
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_CONFIG,
        min_watch_ratio: float = 0.4
    ):
        """Initialize the dataset generator.

        Args:
            config: Feature configuration.
            min_watch_ratio: Minimum watch ratio for positive watch interaction.
        """
        self.config = config
        self.min_watch_ratio = min_watch_ratio

        # Initialize processors
        self.interaction_processor = InteractionFeatureProcessor(
            min_watch_ratio=min_watch_ratio
        )
        self.user_transformer = UserFeatureTransformer(config=config)
        self.video_transformer = VideoFeatureTransformer(config=config)

        self._users_df: Optional[pd.DataFrame] = None
        self._videos_df: Optional[pd.DataFrame] = None
        self._channels_df: Optional[pd.DataFrame] = None
        self._positive_interactions: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        channels_df: Optional[pd.DataFrame] = None,
        compute_embeddings: bool = True
    ) -> "TwoTowerDatasetGenerator":
        """Fit the generator on data.

        Args:
            users_df: DataFrame with user data.
            videos_df: DataFrame with video data.
            interactions_df: DataFrame with interaction data.
            channels_df: Optional DataFrame with channel data.
            compute_embeddings: Whether to pre-compute text embeddings.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting Two-Tower dataset generator...")

        self._users_df = users_df.copy()
        self._videos_df = videos_df.copy()
        self._channels_df = channels_df

        # Process interactions to get positive pairs
        processed = self.interaction_processor.process(interactions_df, videos_df)
        self._positive_interactions = processed[processed["is_positive"]].copy()

        logger.info(f"Found {len(self._positive_interactions)} positive interactions")

        # Fit user transformer
        self.user_transformer.fit(
            users_df=users_df,
            videos_df=videos_df,
            categories=videos_df["category"].unique().tolist()
        )

        # Fit video transformer (share language vocab)
        self.video_transformer.language_vocab = self.user_transformer.language_vocab
        self.video_transformer.fit(videos_df, compute_embeddings=compute_embeddings)

        self._is_fitted = True
        return self

    def generate(self) -> pd.DataFrame:
        """Generate the full Two-Tower training dataset.

        Returns:
            DataFrame with user-video positive pairs and all features.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        logger.info("Generating Two-Tower dataset...")

        # Start with positive interactions
        df = self._positive_interactions.copy()

        # Join user features
        user_cols = ["id", "country_code", "preferred_language", "age"]
        user_rename = {
            "id": "user_id",
            "country_code": "country",
            "preferred_language": "user_language"
        }
        df = df.merge(
            self._users_df[user_cols].rename(columns=user_rename),
            on="user_id",
            how="left",
            suffixes=("", "_user")
        )

        # Join video features
        video_cols = ["id", "category", "title", "duration", "popularity", "language", "manual_tags"]
        video_rename = {
            "id": "video_id",
            "duration": "video_duration",
            "language": "video_language",
            "manual_tags": "tags"
        }
        df = df.merge(
            self._videos_df[video_cols].rename(columns=video_rename),
            on="video_id",
            how="left",
            suffixes=("", "_video")
        )

        # Select final columns for Two-Tower
        final_cols = [
            # User features
            "user_id", "country", "user_language", "age", "previously_watched_category",
            # Video features
            "video_id", "category", "title", "video_duration", "popularity", "video_language", "tags"
        ]

        # Filter to only columns that exist
        final_cols = [c for c in final_cols if c in df.columns]
        df = df[final_cols].copy()

        # Drop duplicates (same user-video pair)
        df = df.drop_duplicates(subset=["user_id", "video_id"])

        logger.info(f"Generated dataset with {len(df)} unique user-video pairs")

        return df

    def generate_transformed(
        self,
        include_embeddings: bool = True
    ) -> pd.DataFrame:
        """Generate dataset with transformed features.

        Args:
            include_embeddings: Whether to include title/tag embeddings.

        Returns:
            DataFrame with transformed features ready for model training.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        # Get base dataset
        df = self.generate()

        # Transform user features
        user_transformed = self.user_transformer.transform(df)

        # Transform video features
        video_transformed = self.video_transformer.transform(
            df,
            include_embeddings=include_embeddings
        )

        # Combine
        result = pd.concat([user_transformed, video_transformed], axis=1)

        return result

    def generate_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        stratify_by: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate train/val/test splits.

        Args:
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            random_state: Random seed for reproducibility.
            stratify_by: Optional column to stratify by.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        df = self.generate()

        # First split: train + temp
        stratify = df[stratify_by] if stratify_by and stratify_by in df.columns else None
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=stratify
        )

        # Second split: val + test
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify = temp_df[stratify_by] if stratify_by and stratify_by in temp_df.columns else None
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=random_state,
            stratify=stratify
        )

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def generate_and_save(
        self,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        transform: bool = True
    ) -> Dict[str, Path]:
        """Generate splits and save to files.

        Args:
            output_dir: Directory to save datasets.
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            transform: Whether to save transformed features.

        Returns:
            Dictionary mapping split names to file paths.
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)

        train_df, val_df, test_df = self.generate_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        if transform:
            # Transform each split using fitted transformers
            train_df = self._transform_split(train_df)
            val_df = self._transform_split(val_df)
            test_df = self._transform_split(test_df)

        paths = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            filepath = output_path / f"{name}.parquet"
            save_parquet(df, filepath)
            paths[name] = filepath
            logger.info(f"Saved {name} set to {filepath}")

        return paths

    def _transform_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a single split using fitted transformers.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        # Transform user features
        user_transformed = self.user_transformer.transform(df)

        # Transform video features (without re-computing embeddings)
        video_transformed = self.video_transformer.transform(
            df,
            include_embeddings=True
        )

        # Combine and return
        return pd.concat([user_transformed, video_transformed], axis=1)

    def save_artifacts(self, artifacts_dir: str) -> None:
        """Save all artifacts (vocabularies, normalizers, etc.).

        Args:
            artifacts_dir: Directory to save artifacts.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        self.user_transformer.save(artifacts_dir)
        self.video_transformer.save(artifacts_dir)
        logger.info(f"Saved artifacts to {artifacts_dir}")

    def load_artifacts(self, artifacts_dir: str) -> "TwoTowerDatasetGenerator":
        """Load artifacts from directory.

        Args:
            artifacts_dir: Directory to load artifacts from.

        Returns:
            Self for method chaining.
        """
        self.user_transformer.load(artifacts_dir)
        self.video_transformer.load(artifacts_dir)
        self._is_fitted = True
        logger.info(f"Loaded artifacts from {artifacts_dir}")
        return self

    def get_feature_specs(self) -> Dict[str, Dict]:
        """Get feature specifications for model building.

        Returns:
            Dictionary with feature specs for user and video towers.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        return {
            "user_tower": {
                "vocab_sizes": self.user_transformer.get_vocab_sizes(),
                "embedding_dims": {
                    "user_id": self.config.embedding.user_id_dim,
                    "country": self.config.embedding.country_dim,
                    "language": self.config.embedding.language_dim,
                    "category": self.config.embedding.category_dim,
                    "age_bucket": self.config.embedding.age_bucket_dim,
                },
            },
            "video_tower": {
                "vocab_sizes": self.video_transformer.get_vocab_sizes(),
                "embedding_dims": {
                    "video_id": self.config.embedding.video_id_dim,
                    "category": self.config.embedding.category_dim,
                    "popularity": self.config.embedding.popularity_dim,
                    "duration_bucket": self.config.embedding.duration_bucket_dim,
                },
                "title_embedding_dim": self.config.embedding.title_embedding_dim,
                "tags_embedding_dim": self.config.embedding.tags_embedding_dim,
            },
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            n_pairs = len(self._positive_interactions) if self._positive_interactions is not None else 0
            return f"TwoTowerDatasetGenerator(fitted=True, positive_pairs={n_pairs})"
        return "TwoTowerDatasetGenerator(fitted=False)"
