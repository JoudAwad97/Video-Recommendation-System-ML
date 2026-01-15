"""
Training utilities for Two-Tower and Ranker models.

This module provides trainer classes that handle:
- Data loading from processed parquet files
- Feature preparation for model input
- Training loop orchestration
- Model checkpointing and evaluation
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

from src.models.two_tower import TwoTowerModel
from src.models.ranker import RankerModel
from src.models.model_config import (
    TwoTowerModelConfig,
    RankerModelConfig,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)
from src.models.metrics import (
    PrecisionAtK,
    DiversityMetric,
    compute_precision_at_k,
    compute_diversity,
)
from src.feature_engineering.user_features import UserFeatureTransformer
from src.feature_engineering.video_features import VideoFeatureTransformer
from src.config.feature_config import FeatureConfig, DEFAULT_CONFIG


class TwoTowerTrainer:
    """
    Trainer for the Two-Tower recommendation model.

    Handles data loading, preprocessing, training, and evaluation.
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        artifacts_dir: str = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            artifacts_dir: Directory containing vocabulary and normalizer artifacts
        """
        self.config = config or DEFAULT_TRAINING_CONFIG
        self.artifacts_dir = artifacts_dir or self.config.artifacts_dir
        self.model = None
        self.vocab_sizes = {}
        self.training_history = {}

        # Feature transformers (loaded lazily)
        self.user_transformer = None
        self.video_transformer = None

        # Load vocabulary sizes
        self._load_vocab_sizes()

    def _load_vocab_sizes(self):
        """Load vocabulary sizes from artifacts."""
        vocab_dir = os.path.join(self.artifacts_dir, "vocabularies")

        vocab_mappings = {
            "user_id": "user_id_vocab.json",
            "video_id": "video_id_vocab.json",
            "country": "country_vocab.json",
            "language": "language_vocab.json",
            "category": "category_vocab.json",
            "age_bucket": None,  # Computed from bucket config
            "duration_bucket": None,  # Computed from bucket config
        }

        for key, filename in vocab_mappings.items():
            if filename is None:
                continue
            filepath = os.path.join(vocab_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    vocab_data = json.load(f)
                    self.vocab_sizes[key] = len(vocab_data.get("vocab", {})) + 2  # +2 for PAD, UNK

        # Load bucket sizes
        buckets_dir = os.path.join(self.artifacts_dir, "buckets")
        if os.path.exists(os.path.join(buckets_dir, "age_boundaries.json")):
            with open(os.path.join(buckets_dir, "age_boundaries.json"), "r") as f:
                boundaries = json.load(f)
                self.vocab_sizes["age_bucket"] = len(boundaries.get("boundaries", [])) + 1

        if os.path.exists(os.path.join(buckets_dir, "duration_boundaries.json")):
            with open(os.path.join(buckets_dir, "duration_boundaries.json"), "r") as f:
                boundaries = json.load(f)
                self.vocab_sizes["duration_bucket"] = len(boundaries.get("boundaries", [])) + 1

        # Set defaults for missing vocabs
        self.vocab_sizes.setdefault("user_id", 10000)
        self.vocab_sizes.setdefault("video_id", 100000)
        self.vocab_sizes.setdefault("country", 200)
        self.vocab_sizes.setdefault("language", 50)
        self.vocab_sizes.setdefault("category", 50)
        self.vocab_sizes.setdefault("age_bucket", 10)
        self.vocab_sizes.setdefault("duration_bucket", 10)

    def load_data(
        self,
        data_dir: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load processed Two-Tower datasets.

        Args:
            data_dir: Directory containing train/val/test parquet files

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data_dir = data_dir or os.path.join(
            self.config.processed_data_dir, "two_tower"
        )

        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        return train_df, val_df, test_df

    def _load_transformers(self):
        """Load feature transformers from artifacts."""
        if self.user_transformer is None:
            self.user_transformer = UserFeatureTransformer(artifacts_dir=self.artifacts_dir)
            self.user_transformer.load(self.artifacts_dir)

        if self.video_transformer is None:
            self.video_transformer = VideoFeatureTransformer(artifacts_dir=self.artifacts_dir)
            self.video_transformer.load(self.artifacts_dir)

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_embeddings: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare features for model input.

        Handles both raw data (applies transformations) and pre-transformed data.

        Args:
            df: DataFrame with features (raw or pre-transformed)
            include_embeddings: Whether to include pre-computed embeddings

        Returns:
            Dictionary of feature tensors
        """
        features = {}
        n_samples = len(df)

        # Check if data is already transformed (has *_idx columns)
        is_transformed = "user_id_idx" in df.columns

        if not is_transformed:
            # Load transformers and apply transformations
            self._load_transformers()

            # Transform user features
            user_transformed = self.user_transformer.transform(df)
            video_transformed = self.video_transformer.transform(df)

            # User features
            features["user_id_idx"] = user_transformed["user_id_idx"].values.astype(np.int32)
            features["country_idx"] = user_transformed["country_idx"].values.astype(np.int32)
            features["user_language_idx"] = user_transformed["user_language_idx"].values.astype(np.int32)
            features["age_normalized"] = user_transformed["age_normalized"].values.astype(np.float32)
            features["age_bucket_idx"] = user_transformed["age_bucket_idx"].values.astype(np.int32)
            features["previously_watched_category_idx"] = user_transformed.get(
                "prev_category_idx",
                pd.Series(np.ones(n_samples, dtype=np.int32))
            ).values.astype(np.int32)

            # Video features
            features["video_id_idx"] = video_transformed["video_id_idx"].values.astype(np.int32)
            features["category_idx"] = video_transformed["category_idx"].values.astype(np.int32)
            features["video_duration_normalized"] = video_transformed["duration_log_normalized"].values.astype(np.float32)
            features["duration_bucket_idx"] = video_transformed["duration_bucket_idx"].values.astype(np.int32)
            features["video_language_idx"] = video_transformed["video_language_idx"].values.astype(np.int32)

            # Popularity one-hot (build from individual columns)
            popularity_cols = ["popularity_low", "popularity_medium", "popularity_high", "popularity_viral"]
            if all(col in video_transformed.columns for col in popularity_cols):
                features["popularity_onehot"] = video_transformed[popularity_cols].values.astype(np.float32)
            else:
                features["popularity_onehot"] = np.zeros((n_samples, 4), dtype=np.float32)
                features["popularity_onehot"][:, 0] = 1.0
        else:
            # Data is already transformed
            features["user_id_idx"] = df["user_id_idx"].values.astype(np.int32)
            features["country_idx"] = df["country_idx"].values.astype(np.int32)
            features["user_language_idx"] = df["user_language_idx"].values.astype(np.int32)
            features["age_normalized"] = df["age_normalized"].values.astype(np.float32)
            features["age_bucket_idx"] = df["age_bucket_idx"].values.astype(np.int32)
            features["previously_watched_category_idx"] = df["previously_watched_category_idx"].values.astype(np.int32)

            features["video_id_idx"] = df["video_id_idx"].values.astype(np.int32)
            features["category_idx"] = df["category_idx"].values.astype(np.int32)
            features["video_duration_normalized"] = df["video_duration_normalized"].values.astype(np.float32)
            features["duration_bucket_idx"] = df["duration_bucket_idx"].values.astype(np.int32)
            features["video_language_idx"] = df["video_language_idx"].values.astype(np.int32)

            if "popularity_onehot" in df.columns:
                features["popularity_onehot"] = np.stack(
                    df["popularity_onehot"].values
                ).astype(np.float32)
            else:
                features["popularity_onehot"] = np.zeros((n_samples, 4), dtype=np.float32)
                features["popularity_onehot"][:, 0] = 1.0

        # Pre-computed embeddings
        if include_embeddings:
            if "title_embedding" in df.columns:
                features["title_embedding"] = np.stack(
                    df["title_embedding"].values
                ).astype(np.float32)
            else:
                features["title_embedding"] = np.zeros((n_samples, 512), dtype=np.float32)

            if "tags_embedding" in df.columns:
                features["tags_embedding"] = np.stack(
                    df["tags_embedding"].values
                ).astype(np.float32)
            else:
                features["tags_embedding"] = np.zeros((n_samples, 100), dtype=np.float32)
        else:
            features["title_embedding"] = np.zeros((n_samples, 512), dtype=np.float32)
            features["tags_embedding"] = np.zeros((n_samples, 100), dtype=np.float32)

        return features

    def create_dataset(
        self,
        features: Dict[str, np.ndarray],
        batch_size: int = None,
        shuffle: bool = True,
        buffer_size: int = 10000,
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from features.

        Args:
            features: Dictionary of feature arrays
            batch_size: Batch size
            shuffle: Whether to shuffle
            buffer_size: Shuffle buffer size

        Returns:
            tf.data.Dataset
        """
        batch_size = batch_size or self.config.two_tower.batch_size

        dataset = tf.data.Dataset.from_tensor_slices(features)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_model(self) -> TwoTowerModel:
        """Build and compile the Two-Tower model."""
        self.model = TwoTowerModel(
            config=self.config.two_tower,
            vocab_sizes=self.vocab_sizes,
        )

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.two_tower.learning_rate
            ),
        )

        return self.model

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the Two-Tower model.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            include_embeddings: Whether to use pre-computed embeddings

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Prepare features
        train_features = self.prepare_features(train_df, include_embeddings)
        val_features = self.prepare_features(val_df, include_embeddings)

        # Create datasets
        train_dataset = self.create_dataset(train_features, shuffle=True)
        val_dataset = self.create_dataset(val_features, shuffle=False)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.two_tower.early_stopping_patience,
                min_delta=self.config.two_tower.early_stopping_min_delta,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ]

        # Create checkpoint directory
        checkpoint_dir = self.config.two_tower.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "model_best.weights.h5"),
                save_best_only=self.config.two_tower.save_best_only,
                save_weights_only=True,
                monitor="val_loss",
            )
        )

        # Train
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.two_tower.epochs,
            callbacks=callbacks,
        )

        self.training_history = history.history

        return self.training_history

    def evaluate(
        self,
        test_df: pd.DataFrame,
        include_embeddings: bool = True,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_df: Test DataFrame
            include_embeddings: Whether to use pre-computed embeddings
            k_values: List of k values for precision@k

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        test_features = self.prepare_features(test_df, include_embeddings)
        test_dataset = self.create_dataset(test_features, shuffle=False)

        # Basic evaluation
        results = self.model.evaluate(test_dataset, return_dict=True)

        return results

    def get_all_video_embeddings(
        self,
        video_df: pd.DataFrame,
        include_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for all videos.

        Args:
            video_df: DataFrame with video features
            include_embeddings: Whether to use pre-computed embeddings

        Returns:
            Video embeddings array (num_videos, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # Prepare video features
        features = self.prepare_features(video_df, include_embeddings)

        # Extract only video features
        video_features = {
            "video_id_idx": features["video_id_idx"],
            "category_idx": features["category_idx"],
            "title_embedding": features["title_embedding"],
            "video_duration_normalized": features["video_duration_normalized"],
            "duration_bucket_idx": features["duration_bucket_idx"],
            "popularity_onehot": features["popularity_onehot"],
            "video_language_idx": features["video_language_idx"],
            "tags_embedding": features["tags_embedding"],
        }

        # Get embeddings
        video_embeddings = self.model.get_video_embeddings(video_features)

        return video_embeddings.numpy()

    def save_model(self, save_dir: str = None):
        """Save the trained model."""
        save_dir = save_dir or self.config.two_tower.checkpoint_dir

        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save_towers(save_dir)

        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Model saved to {save_dir}")

    def load_model(self, save_dir: str = None):
        """Load a trained model."""
        save_dir = save_dir or self.config.two_tower.checkpoint_dir

        if self.model is None:
            self.build_model()

        self.model.load_towers(save_dir)

        # Load training history if available
        history_path = os.path.join(save_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.training_history = json.load(f)

        print(f"Model loaded from {save_dir}")


class RankerTrainer:
    """
    Trainer for the CatBoost Ranker model.

    Handles data loading, training, and evaluation.
    """

    def __init__(
        self,
        config: TrainingConfig = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config or DEFAULT_TRAINING_CONFIG
        self.model = None
        self.training_history = {}

    def load_data(
        self,
        data_dir: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load processed Ranker datasets.

        Args:
            data_dir: Directory containing train/val/test parquet files

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data_dir = data_dir or os.path.join(
            self.config.processed_data_dir, "ranker"
        )

        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        return train_df, val_df, test_df

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "label"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training.

        Args:
            df: DataFrame with all features and labels
            target_col: Name of target column

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Separate features and labels
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Handle any list-type columns (convert to string for CatBoost)
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if it's a list
                first_val = X[col].iloc[0]
                if isinstance(first_val, (list, np.ndarray)):
                    X[col] = X[col].apply(lambda x: str(x) if x is not None else "")

        return X, y

    def build_model(self) -> RankerModel:
        """Build the Ranker model."""
        self.model = RankerModel(
            config=self.config.ranker,
            cat_features=self.config.ranker.cat_features,
        )

        return self.model

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "label"
    ) -> Dict[str, Any]:
        """
        Train the Ranker model.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_col: Name of target column

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Prepare data
        X_train, y_train = self.prepare_features(train_df, target_col)
        X_val, y_val = self.prepare_features(val_df, target_col)

        # Train
        self.training_history = self.model.fit(
            X_train, y_train,
            X_val, y_val
        )

        return self.training_history

    def evaluate(
        self,
        test_df: pd.DataFrame,
        target_col: str = "label",
        user_id_col: str = "user_id",
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_df: Test DataFrame
            target_col: Name of target column
            user_id_col: Name of user ID column
            k_values: List of k values for ranking metrics

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare data
        X_test, y_test = self.prepare_features(test_df, target_col)

        # Basic metrics
        metrics = self.model._compute_metrics(X_test, y_test, "test")

        # Ranking metrics (if user_id is available)
        if user_id_col in test_df.columns:
            ranking_metrics = self.model.evaluate_ranking(
                X_test, y_test,
                test_df[user_id_col],
                k_values
            )
            metrics.update(ranking_metrics)

        return metrics

    def save_model(self, save_dir: str = None):
        """Save the trained model."""
        save_dir = save_dir or self.config.ranker.model_dir

        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(save_dir)

    def load_model(self, save_dir: str = None):
        """Load a trained model."""
        save_dir = save_dir or self.config.ranker.model_dir

        if self.model is None:
            self.build_model()

        self.model.load(save_dir)
