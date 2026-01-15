"""
Ranker dataset generator.

Generates training dataset for the ranking model with both
positive and negative samples.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..feature_engineering.interaction_features import InteractionFeatureProcessor
from ..feature_engineering.ranker_features import RankerFeatureTransformer
from ..config.feature_config import FeatureConfig, DEFAULT_CONFIG
from ..utils.io_utils import save_parquet, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RankerDatasetGenerator:
    """Generate dataset for ranker model training.

    Creates a dataset with positive and negative samples for training
    a ranking model (XGBoost, CatBoost, or neural network).

    Features include:
    - User features: user_id, country, user_language, age
    - Video features: category, child_categories, video_duration, popularity, etc.
    - Context features: interaction_time_day, interaction_time_hour, device
    - Engagement features: view_count, like_count, comment_count, channel_subscriber_count
    - Label: 1 for positive, 0 for negative

    Example:
        >>> generator = RankerDatasetGenerator(negative_ratio=3)
        >>> generator.fit(users_df, videos_df, interactions_df, channels_df)
        >>> train_df, val_df, test_df = generator.generate_splits()
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_CONFIG,
        negative_ratio: int = 3,
        min_watch_ratio: float = 0.4
    ):
        """Initialize the dataset generator.

        Args:
            config: Feature configuration.
            negative_ratio: Number of negative samples per positive sample.
            min_watch_ratio: Minimum watch ratio for positive watch interaction.
        """
        self.config = config
        self.negative_ratio = negative_ratio
        self.min_watch_ratio = min_watch_ratio

        # Initialize processors
        self.interaction_processor = InteractionFeatureProcessor(
            min_watch_ratio=min_watch_ratio
        )
        self.feature_transformer = RankerFeatureTransformer(config=config)

        self._users_df: Optional[pd.DataFrame] = None
        self._videos_df: Optional[pd.DataFrame] = None
        self._channels_df: Optional[pd.DataFrame] = None
        self._dataset: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        channels_df: Optional[pd.DataFrame] = None
    ) -> "RankerDatasetGenerator":
        """Fit the generator on data.

        Args:
            users_df: DataFrame with user data.
            videos_df: DataFrame with video data.
            interactions_df: DataFrame with interaction data.
            channels_df: Optional DataFrame with channel data.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting Ranker dataset generator...")

        self._users_df = users_df.copy()
        self._videos_df = videos_df.copy()
        self._channels_df = channels_df.copy() if channels_df is not None else None

        # Generate complete dataset with positive and negative samples
        self._dataset = self.interaction_processor.create_ranker_dataset(
            interactions_df=interactions_df,
            videos_df=videos_df,
            users_df=users_df,
            channels_df=channels_df if channels_df is not None else pd.DataFrame({"id": [], "subscriber_count": []}),
            negative_ratio=self.negative_ratio
        )

        logger.info(f"Generated dataset with {len(self._dataset)} samples")
        logger.info(f"Positive: {(self._dataset['label'] == 1).sum()}, Negative: {(self._dataset['label'] == 0).sum()}")

        # Fit feature transformer
        self.feature_transformer.fit(self._dataset)

        self._is_fitted = True
        return self

    def generate(self, transform: bool = False, for_catboost: bool = True) -> pd.DataFrame:
        """Generate the full ranker training dataset.

        Args:
            transform: Whether to apply feature transformations.
            for_catboost: If transform=True, whether to keep categoricals as strings.

        Returns:
            DataFrame with all ranker features and labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        if transform:
            return self.feature_transformer.transform(
                self._dataset,
                for_catboost=for_catboost
            )

        return self._dataset.copy()

    def generate_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
        transform: bool = False,
        for_catboost: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate train/val/test splits.

        Args:
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            random_state: Random seed for reproducibility.
            stratify: Whether to stratify by label.
            transform: Whether to apply feature transformations.
            for_catboost: If transform=True, whether to keep categoricals as strings.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        df = self.generate(transform=False)

        # Stratify by label for balanced splits
        stratify_col = df["label"] if stratify else None

        # First split: train + temp
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=stratify_col
        )

        # Second split: val + test
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify_col = temp_df["label"] if stratify else None
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=random_state,
            stratify=stratify_col
        )

        # Apply transformations if requested
        if transform:
            train_df = self.feature_transformer.transform(train_df, for_catboost=for_catboost)
            val_df = self.feature_transformer.transform(val_df, for_catboost=for_catboost)
            test_df = self.feature_transformer.transform(test_df, for_catboost=for_catboost)

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Log label distribution
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if "label" in split_df.columns:
                pos = (split_df["label"] == 1).sum()
                neg = (split_df["label"] == 0).sum()
                logger.info(f"  {name} - Positive: {pos}, Negative: {neg}")

        return train_df, val_df, test_df

    def generate_and_save(
        self,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        transform: bool = True,
        for_catboost: bool = True
    ) -> Dict[str, Path]:
        """Generate splits and save to files.

        Args:
            output_dir: Directory to save datasets.
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            transform: Whether to apply feature transformations.
            for_catboost: Whether to keep categoricals as strings.

        Returns:
            Dictionary mapping split names to file paths.
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)

        train_df, val_df, test_df = self.generate_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            transform=transform,
            for_catboost=for_catboost
        )

        paths = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            filepath = output_path / f"{name}.parquet"
            save_parquet(df, filepath)
            paths[name] = filepath
            logger.info(f"Saved {name} set to {filepath}")

        return paths

    def get_feature_columns(self, for_catboost: bool = True) -> Dict[str, List[str]]:
        """Get feature column names by type.

        Args:
            for_catboost: Whether to return columns for CatBoost.

        Returns:
            Dictionary with categorical and numeric feature lists.
        """
        if for_catboost:
            return {
                "categorical": self.feature_transformer.get_categorical_features(),
                "numeric": self.feature_transformer.get_numeric_features(),
                "label": ["label"],
            }
        else:
            # For neural network, all features are numeric after encoding
            return {
                "features": (
                    self.feature_transformer.get_categorical_features() +
                    self.feature_transformer.get_numeric_features()
                ),
                "label": ["label"],
            }

    def get_catboost_pools(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, "Pool"]:
        """Generate CatBoost Pool objects for training.

        Args:
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.

        Returns:
            Dictionary with train, val, test Pools.

        Raises:
            ImportError: If catboost is not installed.
        """
        try:
            from catboost import Pool
        except ImportError:
            raise ImportError("catboost is required for this method. Install with: pip install catboost")

        train_df, val_df, test_df = self.generate_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            transform=True,
            for_catboost=True
        )

        cat_features = self.feature_transformer.get_categorical_features()
        cat_features = [c for c in cat_features if c in train_df.columns]

        feature_cols = [c for c in train_df.columns if c != "label"]

        pools = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            pools[name] = Pool(
                data=df[feature_cols],
                label=df["label"],
                cat_features=cat_features
            )

        return pools

    def save_artifacts(self, artifacts_dir: str) -> None:
        """Save all artifacts.

        Args:
            artifacts_dir: Directory to save artifacts.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        self.feature_transformer.save(artifacts_dir)
        logger.info(f"Saved ranker artifacts to {artifacts_dir}")

    def load_artifacts(self, artifacts_dir: str) -> "RankerDatasetGenerator":
        """Load artifacts from directory.

        Args:
            artifacts_dir: Directory to load artifacts from.

        Returns:
            Self for method chaining.
        """
        self.feature_transformer.load(artifacts_dir)
        self._is_fitted = True
        logger.info(f"Loaded ranker artifacts from {artifacts_dir}")
        return self

    def get_label_distribution(self) -> Dict[str, int]:
        """Get label distribution in the dataset.

        Returns:
            Dictionary with positive and negative counts.
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        return {
            "positive": int((self._dataset["label"] == 1).sum()),
            "negative": int((self._dataset["label"] == 0).sum()),
            "total": len(self._dataset),
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            dist = self.get_label_distribution()
            return f"RankerDatasetGenerator(fitted=True, samples={dist['total']}, pos/neg={dist['positive']}/{dist['negative']})"
        return "RankerDatasetGenerator(fitted=False)"
