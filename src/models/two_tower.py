"""
Two-Tower Model for video recommendations.

This module implements the Two-Tower neural network architecture consisting of:
- User Tower (Query Tower): Encodes user features into embeddings
- Video Tower (Candidate Tower): Encodes video features into embeddings

Both towers produce embeddings in the same low-dimensional space, and similarity
is computed using dot product.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional, Tuple, Any
import json
import os

from src.models.model_config import TwoTowerModelConfig, DEFAULT_TWO_TOWER_CONFIG


class UserTower(keras.Model):
    """
    User Tower (Query Tower) of the Two-Tower model.

    Encodes user features:
    - user_id: Integer lookup -> Embedding
    - country: String lookup -> Embedding
    - user_language: String lookup -> Embedding (shared vocab)
    - age: Normalized value + Bucket index -> Concatenated
    - previously_watched_category: String lookup -> Embedding with [START] token
    """

    def __init__(
        self,
        config: TwoTowerModelConfig,
        vocab_sizes: Dict[str, int],
        name: str = "user_tower",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.vocab_sizes = vocab_sizes

        # Embedding layers for categorical features
        self.user_id_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("user_id", 10000),
            output_dim=config.user_id_embedding_dim,
            name="user_id_embedding",
        )

        self.country_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("country", 200),
            output_dim=config.country_embedding_dim,
            name="country_embedding",
        )

        self.language_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("language", 50),
            output_dim=config.language_embedding_dim,
            name="user_language_embedding",
        )

        self.category_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("category", 50),
            output_dim=config.category_embedding_dim,
            name="previously_watched_category_embedding",
        )

        self.age_bucket_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("age_bucket", 10),
            output_dim=config.age_bucket_embedding_dim,
            name="age_bucket_embedding",
        )

        # Dense layers for the tower
        self.dense_layers = []
        for i, units in enumerate(config.user_tower_hidden_dims):
            self.dense_layers.append(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(config.l2_regularization),
                    name=f"user_dense_{i}",
                )
            )
            self.dense_layers.append(
                keras.layers.Dropout(config.dropout_rate, name=f"user_dropout_{i}")
            )

        # Final projection layer to embedding dimension
        self.projection = keras.layers.Dense(
            config.embedding_dim,
            activation=None,  # No activation for final embedding
            name="user_projection",
        )

        # L2 normalization layer
        self.l2_normalize = keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name="user_l2_normalize",
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the User Tower.

        Args:
            inputs: Dictionary containing:
                - user_id_idx: User ID index tensor
                - country_idx: Country index tensor
                - user_language_idx: User language index tensor
                - age_normalized: Normalized age tensor
                - age_bucket_idx: Age bucket index tensor
                - previously_watched_category_idx: Category index tensor
            training: Whether in training mode

        Returns:
            User embedding tensor of shape (batch_size, embedding_dim)
        """
        # Get embeddings for categorical features
        user_id_emb = self.user_id_embedding(inputs["user_id_idx"])
        country_emb = self.country_embedding(inputs["country_idx"])
        language_emb = self.language_embedding(inputs["user_language_idx"])
        category_emb = self.category_embedding(inputs["previously_watched_category_idx"])
        age_bucket_emb = self.age_bucket_embedding(inputs["age_bucket_idx"])

        # Get normalized age
        age_normalized = tf.expand_dims(inputs["age_normalized"], axis=-1)

        # Concatenate all features
        concatenated = tf.concat([
            user_id_emb,
            country_emb,
            language_emb,
            category_emb,
            age_bucket_emb,
            age_normalized,
        ], axis=-1)

        # Pass through dense layers
        x = concatenated
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # Project to final embedding dimension
        x = self.projection(x)

        # L2 normalize the embedding
        return self.l2_normalize(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "vocab_sizes": self.vocab_sizes,
        }


class VideoTower(keras.Model):
    """
    Video Tower (Candidate Tower) of the Two-Tower model.

    Encodes video features:
    - video_id: Integer lookup -> Embedding
    - category: String lookup -> Embedding
    - title: Pre-computed BERT embedding (768 dim)
    - video_duration: Log transform + Normalize + Bucket
    - popularity: One-hot encoding (4 levels)
    - video_language: String lookup -> Embedding (shared vocab)
    - tags: Pre-computed CBOW embedding (100 dim)
    """

    def __init__(
        self,
        config: TwoTowerModelConfig,
        vocab_sizes: Dict[str, int],
        name: str = "video_tower",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.vocab_sizes = vocab_sizes

        # Embedding layers for categorical features
        self.video_id_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("video_id", 100000),
            output_dim=config.video_id_embedding_dim,
            name="video_id_embedding",
        )

        self.category_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("category", 50),
            output_dim=config.category_embedding_dim,
            name="video_category_embedding",
        )

        self.language_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("language", 50),
            output_dim=config.language_embedding_dim,
            name="video_language_embedding",
        )

        self.duration_bucket_embedding = keras.layers.Embedding(
            input_dim=vocab_sizes.get("duration_bucket", 10),
            output_dim=config.duration_bucket_embedding_dim,
            name="duration_bucket_embedding",
        )

        # Projection layer for title embedding (reduce dimensionality)
        self.title_projection = keras.layers.Dense(
            32,  # Project 512 -> 32
            activation="relu",
            name="title_projection",
        )

        # Projection layer for tags embedding
        self.tags_projection = keras.layers.Dense(
            16,  # Project 100 -> 16
            activation="relu",
            name="tags_projection",
        )

        # Dense layers for the tower
        self.dense_layers = []
        for i, units in enumerate(config.video_tower_hidden_dims):
            self.dense_layers.append(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(config.l2_regularization),
                    name=f"video_dense_{i}",
                )
            )
            self.dense_layers.append(
                keras.layers.Dropout(config.dropout_rate, name=f"video_dropout_{i}")
            )

        # Final projection layer to embedding dimension
        self.projection = keras.layers.Dense(
            config.embedding_dim,
            activation=None,
            name="video_projection",
        )

        # L2 normalization layer
        self.l2_normalize = keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name="video_l2_normalize",
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the Video Tower.

        Args:
            inputs: Dictionary containing:
                - video_id_idx: Video ID index tensor
                - category_idx: Category index tensor
                - title_embedding: Pre-computed title embedding (batch, 512)
                - video_duration_normalized: Normalized log duration
                - duration_bucket_idx: Duration bucket index tensor
                - popularity_onehot: One-hot popularity (batch, 4)
                - video_language_idx: Video language index tensor
                - tags_embedding: Pre-computed tags embedding (batch, 100)
            training: Whether in training mode

        Returns:
            Video embedding tensor of shape (batch_size, embedding_dim)
        """
        # Get embeddings for categorical features
        video_id_emb = self.video_id_embedding(inputs["video_id_idx"])
        category_emb = self.category_embedding(inputs["category_idx"])
        language_emb = self.language_embedding(inputs["video_language_idx"])
        duration_bucket_emb = self.duration_bucket_embedding(inputs["duration_bucket_idx"])

        # Project pre-computed embeddings
        title_proj = self.title_projection(inputs["title_embedding"])
        tags_proj = self.tags_projection(inputs["tags_embedding"])

        # Get other features
        duration_normalized = tf.expand_dims(inputs["video_duration_normalized"], axis=-1)
        popularity_onehot = inputs["popularity_onehot"]

        # Concatenate all features
        concatenated = tf.concat([
            video_id_emb,
            category_emb,
            language_emb,
            duration_bucket_emb,
            title_proj,
            tags_proj,
            duration_normalized,
            popularity_onehot,
        ], axis=-1)

        # Pass through dense layers
        x = concatenated
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # Project to final embedding dimension
        x = self.projection(x)

        # L2 normalize the embedding
        return self.l2_normalize(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "vocab_sizes": self.vocab_sizes,
        }


class TwoTowerModel(keras.Model):
    """
    Complete Two-Tower Model for video recommendations.

    Combines User Tower and Video Tower, computing similarity using dot product.
    Uses in-batch negatives for efficient training.
    """

    def __init__(
        self,
        config: TwoTowerModelConfig = None,
        vocab_sizes: Dict[str, int] = None,
        name: str = "two_tower_model",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config or DEFAULT_TWO_TOWER_CONFIG
        self.vocab_sizes = vocab_sizes or {}

        # Initialize towers
        self.user_tower = UserTower(self.config, self.vocab_sizes)
        self.video_tower = VideoTower(self.config, self.vocab_sizes)

        # Temperature for softmax
        self.temperature = tf.Variable(
            self.config.temperature,
            trainable=False,
            name="temperature",
        )

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.Mean(name="accuracy")

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the Two-Tower model.

        Args:
            inputs: Dictionary containing both user and video features
            training: Whether in training mode

        Returns:
            Tuple of (user_embeddings, video_embeddings)
        """
        # Extract user features
        user_inputs = {
            "user_id_idx": inputs["user_id_idx"],
            "country_idx": inputs["country_idx"],
            "user_language_idx": inputs["user_language_idx"],
            "age_normalized": inputs["age_normalized"],
            "age_bucket_idx": inputs["age_bucket_idx"],
            "previously_watched_category_idx": inputs["previously_watched_category_idx"],
        }

        # Extract video features
        video_inputs = {
            "video_id_idx": inputs["video_id_idx"],
            "category_idx": inputs["category_idx"],
            "title_embedding": inputs["title_embedding"],
            "video_duration_normalized": inputs["video_duration_normalized"],
            "duration_bucket_idx": inputs["duration_bucket_idx"],
            "popularity_onehot": inputs["popularity_onehot"],
            "video_language_idx": inputs["video_language_idx"],
            "tags_embedding": inputs["tags_embedding"],
        }

        # Get embeddings from both towers
        user_embeddings = self.user_tower(user_inputs, training=training)
        video_embeddings = self.video_tower(video_inputs, training=training)

        return user_embeddings, video_embeddings

    def compute_similarity(
        self,
        user_embeddings: tf.Tensor,
        video_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute similarity scores using dot product.

        Args:
            user_embeddings: (batch_size, embedding_dim)
            video_embeddings: (batch_size, embedding_dim)

        Returns:
            Similarity matrix of shape (batch_size, batch_size)
        """
        # Compute dot product: user_embeddings @ video_embeddings.T
        similarity = tf.matmul(
            user_embeddings,
            video_embeddings,
            transpose_b=True
        )
        return similarity / self.temperature

    def compute_in_batch_negatives_loss(
        self,
        user_embeddings: tf.Tensor,
        video_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute loss using in-batch negatives.

        For each (user, video) positive pair in the batch, all other videos
        in the batch serve as negative examples.

        Args:
            user_embeddings: (batch_size, embedding_dim)
            video_embeddings: (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        batch_size = tf.shape(user_embeddings)[0]

        # Compute similarity matrix: (batch_size, batch_size)
        similarity_matrix = self.compute_similarity(user_embeddings, video_embeddings)

        # Labels: diagonal elements are positive pairs (index i matches user i with video i)
        labels = tf.range(batch_size)

        # Softmax cross-entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=similarity_matrix,
        )

        return tf.reduce_mean(loss)

    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step using in-batch negatives."""
        with tf.GradientTape() as tape:
            # Forward pass
            user_embeddings, video_embeddings = self(data, training=True)

            # Compute loss
            loss = self.compute_in_batch_negatives_loss(user_embeddings, video_embeddings)

            # Add regularization losses
            if self.losses:
                loss = loss + tf.add_n(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute accuracy (top-1 accuracy for in-batch retrieval)
        similarity_matrix = self.compute_similarity(user_embeddings, video_embeddings)
        predictions = tf.argmax(similarity_matrix, axis=1)
        labels = tf.range(tf.shape(user_embeddings)[0])
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(labels, predictions.dtype)), tf.float32)
        )

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom test step."""
        # Forward pass
        user_embeddings, video_embeddings = self(data, training=False)

        # Compute loss
        loss = self.compute_in_batch_negatives_loss(user_embeddings, video_embeddings)

        # Compute accuracy
        similarity_matrix = self.compute_similarity(user_embeddings, video_embeddings)
        predictions = tf.argmax(similarity_matrix, axis=1)
        labels = tf.range(tf.shape(user_embeddings)[0])
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(labels, predictions.dtype)), tf.float32)
        )

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def get_user_embeddings(
        self,
        user_inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """Get user embeddings for inference."""
        return self.user_tower(user_inputs, training=False)

    def get_video_embeddings(
        self,
        video_inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """Get video embeddings for inference."""
        return self.video_tower(video_inputs, training=False)

    def save_towers(self, save_dir: str):
        """Save both towers separately for inference."""
        os.makedirs(save_dir, exist_ok=True)

        # Save user tower
        user_tower_path = os.path.join(save_dir, "user_tower")
        os.makedirs(user_tower_path, exist_ok=True)
        self.user_tower.save_weights(os.path.join(user_tower_path, "weights.weights.h5"))

        # Save video tower
        video_tower_path = os.path.join(save_dir, "video_tower")
        os.makedirs(video_tower_path, exist_ok=True)
        self.video_tower.save_weights(os.path.join(video_tower_path, "weights.weights.h5"))

        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "vocab_sizes": self.vocab_sizes,
            }, f, indent=2)

    def load_towers(self, save_dir: str):
        """Load both towers from saved weights."""
        # Load user tower
        user_tower_path = os.path.join(save_dir, "user_tower")
        self.user_tower.load_weights(os.path.join(user_tower_path, "weights.weights.h5"))

        # Load video tower
        video_tower_path = os.path.join(save_dir, "video_tower")
        self.video_tower.load_weights(os.path.join(video_tower_path, "weights.weights.h5"))

    def get_config(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "vocab_sizes": self.vocab_sizes,
        }
