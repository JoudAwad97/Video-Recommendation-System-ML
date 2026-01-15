"""
Query encoder service for real-time user embedding generation.

Encodes user features through the User Tower to generate embeddings
for similarity search against the video embedding index.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import tensorflow as tf

from ..models.two_tower import TwoTowerModel
from ..models.model_config import TwoTowerModelConfig
from ..feature_engineering.user_features import UserFeatureTransformer
from ..utils.logging_utils import get_logger
from .serving_config import ServingConfig

logger = get_logger(__name__)


class QueryEncoderService:
    """Service for encoding user queries to embeddings.

    This service:
    1. Loads the trained User Tower model
    2. Transforms raw user features using the feature transformer
    3. Generates user embeddings for similarity search

    The service is designed for low-latency inference and supports
    both single-user and batch encoding.

    Example:
        >>> service = QueryEncoderService(config)
        >>> service.load_models()
        >>> embedding = service.encode_user(user_data)
        >>> embeddings = service.encode_users_batch(users_data)
    """

    def __init__(self, config: ServingConfig):
        """Initialize the query encoder service.

        Args:
            config: Serving configuration.
        """
        self.config = config
        self.model: Optional[TwoTowerModel] = None
        self.user_transformer: Optional[UserFeatureTransformer] = None
        self._is_loaded = False

    def load_models(
        self,
        model_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ) -> "QueryEncoderService":
        """Load the User Tower model and feature transformers.

        Args:
            model_path: Path to saved Two-Tower model.
            artifacts_path: Path to feature engineering artifacts.

        Returns:
            Self for method chaining.
        """
        model_path = model_path or self.config.two_tower_model_path
        artifacts_path = artifacts_path or self.config.artifacts_path

        logger.info(f"Loading User Tower model from {model_path}")

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

        # Build model
        dummy_batch = self._create_dummy_batch()
        self.model(dummy_batch, training=False)

        # Load weights
        self.model.load_towers(model_path)

        # Load user feature transformer
        logger.info(f"Loading user feature transformer from {artifacts_path}")
        self.user_transformer = UserFeatureTransformer()
        self.user_transformer.load(artifacts_path)

        self._is_loaded = True
        logger.info("Query encoder service loaded successfully")

        return self

    def _create_dummy_batch(self) -> Dict[str, tf.Tensor]:
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
            # Video features (needed for full model)
            "video_id_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "category_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "title_embedding": tf.zeros((batch_size, 384), dtype=tf.float32),
            "video_duration_normalized": tf.zeros((batch_size,), dtype=tf.float32),
            "duration_bucket_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "popularity_onehot": tf.zeros((batch_size, 4), dtype=tf.float32),
            "video_language_idx": tf.zeros((batch_size,), dtype=tf.int32),
            "tags_embedding": tf.zeros((batch_size, 100), dtype=tf.float32),
        }

    def encode_user(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Encode a single user to an embedding vector.

        Args:
            user_data: Dictionary with user features. Expected keys:
                - user_id or id: User identifier
                - country or country_code: User's country
                - user_language or preferred_language: User's language
                - age: User's age
                - previously_watched_category: Last watched category

        Returns:
            User embedding as numpy array (embedding_dim,).
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Transform user features
        transformed = self.user_transformer.transform_single(user_data)

        # Prepare inputs
        user_inputs = self._prepare_user_inputs(transformed)

        # Generate embedding
        embedding = self.model.get_user_embeddings(user_inputs)

        return embedding.numpy()[0]

    def encode_users_batch(
        self,
        users_data: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Encode multiple users to embeddings.

        Args:
            users_data: List of user data dictionaries.

        Returns:
            User embeddings as numpy array (n_users, embedding_dim).
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        import pandas as pd

        # Convert to DataFrame for batch transformation
        users_df = pd.DataFrame(users_data)

        # Transform features
        transformed_df = self.user_transformer.transform(users_df)

        # Prepare inputs
        user_inputs = self._prepare_batch_user_inputs(transformed_df)

        # Generate embeddings
        embeddings = self.model.get_user_embeddings(user_inputs)

        return embeddings.numpy()

    def _prepare_user_inputs(
        self,
        transformed: Dict[str, Any],
    ) -> Dict[str, tf.Tensor]:
        """Prepare single user inputs for the model.

        Args:
            transformed: Transformed user features dictionary.

        Returns:
            Dictionary of tensors for user tower.
        """
        return {
            "user_id_idx": tf.constant(
                [transformed.get("user_id_idx", 0)], dtype=tf.int32
            ),
            "country_idx": tf.constant(
                [transformed.get("country_idx", 0)], dtype=tf.int32
            ),
            "user_language_idx": tf.constant(
                [transformed.get("user_language_idx", 0)], dtype=tf.int32
            ),
            "age_normalized": tf.constant(
                [transformed.get("age_normalized", 0.0)], dtype=tf.float32
            ),
            "age_bucket_idx": tf.constant(
                [transformed.get("age_bucket_idx", 0)], dtype=tf.int32
            ),
            "previously_watched_category_idx": tf.constant(
                [transformed.get("prev_category_idx", 0)], dtype=tf.int32
            ),
        }

    def _prepare_batch_user_inputs(
        self,
        transformed_df,
    ) -> Dict[str, tf.Tensor]:
        """Prepare batch user inputs for the model.

        Args:
            transformed_df: Transformed user features DataFrame.

        Returns:
            Dictionary of tensors for user tower.
        """
        return {
            "user_id_idx": tf.constant(
                transformed_df["user_id_idx"].values, dtype=tf.int32
            ),
            "country_idx": tf.constant(
                transformed_df["country_idx"].values, dtype=tf.int32
            ),
            "user_language_idx": tf.constant(
                transformed_df["user_language_idx"].values, dtype=tf.int32
            ),
            "age_normalized": tf.constant(
                transformed_df["age_normalized"].values, dtype=tf.float32
            ),
            "age_bucket_idx": tf.constant(
                transformed_df["age_bucket_idx"].values, dtype=tf.int32
            ),
            "previously_watched_category_idx": tf.constant(
                transformed_df["prev_category_idx"].values, dtype=tf.int32
            ),
        }

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension.
        """
        if not self._is_loaded:
            return self.config.vector_db.embedding_dim
        return self.model.config.embedding_dim

    def warmup(self) -> None:
        """Warm up the model with a dummy inference.

        Call this after loading to ensure the model is ready for inference.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Run dummy inference
        dummy_user = {
            "id": 0,
            "country_code": "US",
            "preferred_language": "en",
            "age": 25,
            "previously_watched_category": "Technology",
        }

        _ = self.encode_user(dummy_user)
        logger.info("Query encoder warmed up")


class QueryEncoderServiceTFServing:
    """Query encoder service designed for TensorFlow Serving deployment.

    This version exports only the User Tower for deployment as a
    standalone TensorFlow SavedModel.
    """

    def __init__(self, model: TwoTowerModel):
        """Initialize with a trained model.

        Args:
            model: Trained TwoTowerModel.
        """
        self.model = model

    def export_user_tower(self, export_path: str) -> None:
        """Export the User Tower as a SavedModel.

        Args:
            export_path: Path to export the model.
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Create a concrete function for the user tower
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="user_id_idx"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="country_idx"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="user_language_idx"),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name="age_normalized"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="age_bucket_idx"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="prev_category_idx"),
        ])
        def serve_user_tower(
            user_id_idx,
            country_idx,
            user_language_idx,
            age_normalized,
            age_bucket_idx,
            prev_category_idx,
        ):
            inputs = {
                "user_id_idx": user_id_idx,
                "country_idx": country_idx,
                "user_language_idx": user_language_idx,
                "age_normalized": age_normalized,
                "age_bucket_idx": age_bucket_idx,
                "previously_watched_category_idx": prev_category_idx,
            }
            return self.model.get_user_embeddings(inputs)

        # Save with signatures
        tf.saved_model.save(
            self.model.user_tower,
            str(export_path),
            signatures={"serving_default": serve_user_tower},
        )

        logger.info(f"Exported User Tower to {export_path}")
