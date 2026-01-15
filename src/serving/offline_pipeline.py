"""
Offline inference pipeline for batch video embedding generation.

This pipeline processes all videos through the Video Tower to generate
embeddings that are then stored in a vector database for efficient
similarity search during online inference.
"""

from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from ..models.two_tower import TwoTowerModel
from ..models.model_config import TwoTowerModelConfig
from ..feature_engineering.video_features import VideoFeatureTransformer
from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig

logger = get_logger(__name__)


class OfflineInferencePipeline:
    """Pipeline for batch processing videos to generate embeddings.

    This pipeline:
    1. Loads the trained Video Tower model
    2. Processes videos in batches through feature transformation
    3. Generates embeddings for each video
    4. Returns embeddings with video IDs for vector database indexing

    Example:
        >>> pipeline = OfflineInferencePipeline(config)
        >>> pipeline.load_models()
        >>> embeddings_df = pipeline.process_videos(videos_df)
        >>> pipeline.save_embeddings(embeddings_df, "embeddings.parquet")
    """

    def __init__(self, config: ServingConfig):
        """Initialize the offline pipeline.

        Args:
            config: Serving configuration.
        """
        self.config = config
        self.model: Optional[TwoTowerModel] = None
        self.video_transformer: Optional[VideoFeatureTransformer] = None
        self._is_loaded = False

    def load_models(
        self,
        model_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ) -> "OfflineInferencePipeline":
        """Load the Two-Tower model and feature transformers.

        Args:
            model_path: Path to saved Two-Tower model.
            artifacts_path: Path to feature engineering artifacts.

        Returns:
            Self for method chaining.
        """
        model_path = model_path or self.config.two_tower_model_path
        artifacts_path = artifacts_path or self.config.artifacts_path

        logger.info(f"Loading Two-Tower model from {model_path}")

        # Load model configuration
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            model_config = TwoTowerModelConfig(**config_dict)
        else:
            model_config = TwoTowerModelConfig(
                embedding_dim=self.config.vector_db.embedding_dim
            )

        # Load vocab sizes
        vocab_path = Path(model_path) / "vocab_sizes.json"
        if vocab_path.exists():
            import json
            with open(vocab_path) as f:
                vocab_sizes = json.load(f)
        else:
            # Default vocab sizes - should match training
            vocab_sizes = {
                "user_id": 1000,
                "video_id": 5000,
                "country": 50,
                "language": 20,
                "category": 20,
                "age_bucket": 10,
                "duration_bucket": 10,
            }

        # Initialize model
        self.model = TwoTowerModel(model_config, vocab_sizes)

        # Build model by creating dummy inputs
        dummy_batch = self._create_dummy_batch(vocab_sizes)
        self.model(dummy_batch, training=False)

        # Load weights
        self.model.load_towers(model_path)

        # Load video feature transformer
        logger.info(f"Loading video feature transformer from {artifacts_path}")
        self.video_transformer = VideoFeatureTransformer()
        self.video_transformer.load(artifacts_path)

        self._is_loaded = True
        logger.info("Models loaded successfully")

        return self

    def _create_dummy_batch(self, vocab_sizes: Dict[str, int]) -> Dict[str, tf.Tensor]:
        """Create dummy batch for model building."""
        batch_size = 1
        return {
            # User features
            "user_id_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "country_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "user_language_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "age_normalized": tf.zeros((batch_size,), dtype=tf.float32),
            "age_bucket_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "previously_watched_category_idx": tf.zeros((batch_size,), dtype=tf.int32),
            # Video features
            "video_id_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "category_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "title_embedding": tf.zeros((batch_size, 384), dtype=tf.float32),
            "video_duration_normalized": tf.zeros((batch_size,), dtype=tf.float32),
            "duration_bucket_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "popularity_onehot": tf.zeros((batch_size, 4), dtype=tf.float32),
            "video_language_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "tags_embedding": tf.zeros((batch_size, 100), dtype=tf.float32),
        }

    def process_videos(
        self,
        videos_df: pd.DataFrame,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Process all videos to generate embeddings.

        Args:
            videos_df: DataFrame with video data.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress.

        Returns:
            DataFrame with video_id and embedding columns.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        batch_size = batch_size or self.config.batch_size
        n_videos = len(videos_df)
        logger.info(f"Processing {n_videos} videos in batches of {batch_size}")

        all_embeddings = []
        all_video_ids = []

        for batch_start in range(0, n_videos, batch_size):
            batch_end = min(batch_start + batch_size, n_videos)
            batch_df = videos_df.iloc[batch_start:batch_end]

            if show_progress and batch_start % (batch_size * 10) == 0:
                logger.info(f"Processing batch {batch_start // batch_size + 1}")

            # Transform features
            transformed = self.video_transformer.transform(batch_df)

            # Prepare inputs for video tower
            video_inputs = self._prepare_video_inputs(transformed, batch_df)

            # Generate embeddings
            embeddings = self.model.get_video_embeddings(video_inputs)

            all_embeddings.append(embeddings.numpy())
            all_video_ids.extend(batch_df["id"].tolist())

        # Combine results
        embeddings_array = np.vstack(all_embeddings)

        result_df = pd.DataFrame({
            "video_id": all_video_ids,
            "embedding": list(embeddings_array),
        })

        logger.info(f"Generated embeddings for {len(result_df)} videos")
        return result_df

    def _prepare_video_inputs(
        self,
        transformed_df: pd.DataFrame,
        raw_df: pd.DataFrame,
    ) -> Dict[str, tf.Tensor]:
        """Prepare video inputs for the model.

        Args:
            transformed_df: Transformed features DataFrame.
            raw_df: Raw video DataFrame (for additional features).

        Returns:
            Dictionary of tensors for video tower.
        """
        batch_size = len(transformed_df)

        # Build popularity one-hot from individual columns
        popularity_cols = [
            "popularity_low", "popularity_medium",
            "popularity_high", "popularity_viral"
        ]
        if all(col in transformed_df.columns for col in popularity_cols):
            popularity_onehot = np.stack([
                transformed_df[col].values for col in popularity_cols
            ], axis=1).astype(np.float32)
        else:
            popularity_onehot = np.zeros((batch_size, 4), dtype=np.float32)
            popularity_onehot[:, 0] = 1  # Default to "low"

        inputs = {
            "video_id_idx": tf.constant(
                transformed_df["video_id_idx"].values, dtype=tf.int32
            ),
            "category_idx": tf.constant(
                transformed_df["category_idx"].values, dtype=tf.int32
            ),
            "video_duration_normalized": tf.constant(
                transformed_df["duration_log_normalized"].values, dtype=tf.float32
            ),
            "duration_bucket_idx": tf.constant(
                transformed_df["duration_bucket_idx"].values, dtype=tf.int32
            ),
            "popularity_onehot": tf.constant(popularity_onehot, dtype=tf.float32),
            "video_language_idx": tf.constant(
                transformed_df["video_language_idx"].values, dtype=tf.int32
            ),
        }

        # Handle title embedding
        if "title_embedding" in transformed_df.columns:
            title_emb = np.stack(transformed_df["title_embedding"].values)
            inputs["title_embedding"] = tf.constant(title_emb, dtype=tf.float32)
        else:
            inputs["title_embedding"] = tf.zeros((batch_size, 384), dtype=tf.float32)

        # Handle tags embedding
        if "tags_embedding" in transformed_df.columns:
            tags_emb = np.stack(transformed_df["tags_embedding"].values)
            inputs["tags_embedding"] = tf.constant(tags_emb, dtype=tf.float32)
        else:
            inputs["tags_embedding"] = tf.zeros((batch_size, 100), dtype=tf.float32)

        return inputs

    def save_embeddings(
        self,
        embeddings_df: pd.DataFrame,
        output_path: str,
        format: str = "parquet",
    ) -> None:
        """Save embeddings to file.

        Args:
            embeddings_df: DataFrame with video_id and embedding columns.
            output_path: Path to save embeddings.
            format: Output format ("parquet", "numpy", "json").
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            # Convert embeddings to list for parquet serialization
            df_to_save = embeddings_df.copy()
            df_to_save["embedding"] = df_to_save["embedding"].apply(list)
            df_to_save.to_parquet(output_path, index=False)

        elif format == "numpy":
            # Save as numpy arrays
            video_ids = np.array(embeddings_df["video_id"].tolist())
            embeddings = np.stack(embeddings_df["embedding"].values)
            np.savez(
                output_path,
                video_ids=video_ids,
                embeddings=embeddings,
            )

        elif format == "json":
            import json
            data = {
                "video_ids": embeddings_df["video_id"].tolist(),
                "embeddings": [emb.tolist() for emb in embeddings_df["embedding"]],
            }
            with open(output_path, "w") as f:
                json.dump(data, f)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved embeddings to {output_path}")

    def load_embeddings(
        self,
        input_path: str,
        format: str = "parquet",
    ) -> pd.DataFrame:
        """Load embeddings from file.

        Args:
            input_path: Path to load embeddings from.
            format: Input format ("parquet", "numpy", "json").

        Returns:
            DataFrame with video_id and embedding columns.
        """
        input_path = Path(input_path)

        if format == "parquet":
            df = pd.read_parquet(input_path)
            df["embedding"] = df["embedding"].apply(np.array)
            return df

        elif format == "numpy":
            data = np.load(input_path)
            return pd.DataFrame({
                "video_id": data["video_ids"].tolist(),
                "embedding": list(data["embeddings"]),
            })

        elif format == "json":
            import json
            with open(input_path) as f:
                data = json.load(f)
            return pd.DataFrame({
                "video_id": data["video_ids"],
                "embedding": [np.array(emb) for emb in data["embeddings"]],
            })

        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_embeddings_iterator(
        self,
        videos_df: pd.DataFrame,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[List[int], np.ndarray]]:
        """Generate embeddings as an iterator for large datasets.

        Args:
            videos_df: DataFrame with video data.
            batch_size: Batch size for processing.

        Yields:
            Tuples of (video_ids, embeddings_array).
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        batch_size = batch_size or self.config.batch_size

        for batch_start in range(0, len(videos_df), batch_size):
            batch_end = min(batch_start + batch_size, len(videos_df))
            batch_df = videos_df.iloc[batch_start:batch_end]

            # Transform and generate embeddings
            transformed = self.video_transformer.transform(batch_df)
            video_inputs = self._prepare_video_inputs(transformed, batch_df)
            embeddings = self.model.get_video_embeddings(video_inputs)

            yield batch_df["id"].tolist(), embeddings.numpy()
