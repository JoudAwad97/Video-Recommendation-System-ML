"""
Unit tests for Two-Tower model.
"""

import pytest
import numpy as np
import tensorflow as tf
import tempfile
import os

from src.models.two_tower import TwoTowerModel, UserTower, VideoTower
from src.models.model_config import TwoTowerModelConfig


class TestUserTower:
    """Tests for UserTower model."""

    @pytest.fixture
    def config(self):
        return TwoTowerModelConfig(
            embedding_dim=16,
            user_tower_hidden_dims=[32, 16],
        )

    @pytest.fixture
    def vocab_sizes(self):
        return {
            "user_id": 100,
            "country": 20,
            "language": 10,
            "category": 15,
            "age_bucket": 8,
        }

    @pytest.fixture
    def sample_inputs(self):
        batch_size = 4
        return {
            "user_id_idx": tf.constant([1, 2, 3, 4], dtype=tf.int32),
            "country_idx": tf.constant([0, 1, 2, 3], dtype=tf.int32),
            "user_language_idx": tf.constant([1, 1, 2, 2], dtype=tf.int32),
            "age_normalized": tf.constant([0.5, -0.3, 1.2, 0.0], dtype=tf.float32),
            "age_bucket_idx": tf.constant([2, 3, 4, 1], dtype=tf.int32),
            "previously_watched_category_idx": tf.constant([5, 6, 7, 8], dtype=tf.int32),
        }

    def test_user_tower_output_shape(self, config, vocab_sizes, sample_inputs):
        """Test that user tower outputs correct shape."""
        tower = UserTower(config, vocab_sizes)
        output = tower(sample_inputs, training=False)

        assert output.shape == (4, config.embedding_dim)

    def test_user_tower_l2_normalized(self, config, vocab_sizes, sample_inputs):
        """Test that output embeddings are L2 normalized."""
        tower = UserTower(config, vocab_sizes)
        output = tower(sample_inputs, training=False)

        # Check L2 norm is approximately 1
        norms = tf.norm(output, axis=1)
        np.testing.assert_array_almost_equal(norms.numpy(), np.ones(4), decimal=5)

    def test_user_tower_different_outputs(self, config, vocab_sizes, sample_inputs):
        """Test that different inputs produce different outputs."""
        tower = UserTower(config, vocab_sizes)
        output = tower(sample_inputs, training=False)

        # Check that each user has a different embedding
        for i in range(3):
            for j in range(i + 1, 4):
                assert not np.allclose(output[i].numpy(), output[j].numpy())


class TestVideoTower:
    """Tests for VideoTower model."""

    @pytest.fixture
    def config(self):
        return TwoTowerModelConfig(
            embedding_dim=16,
            video_tower_hidden_dims=[64, 32],
        )

    @pytest.fixture
    def vocab_sizes(self):
        return {
            "video_id": 500,
            "category": 15,
            "language": 10,
            "duration_bucket": 8,
        }

    @pytest.fixture
    def sample_inputs(self):
        batch_size = 4
        return {
            "video_id_idx": tf.constant([10, 20, 30, 40], dtype=tf.int32),
            "category_idx": tf.constant([1, 2, 3, 4], dtype=tf.int32),
            "title_embedding": tf.random.normal((4, 512)),
            "video_duration_normalized": tf.constant([0.5, 1.0, -0.5, 0.0], dtype=tf.float32),
            "duration_bucket_idx": tf.constant([2, 3, 1, 4], dtype=tf.int32),
            "popularity_onehot": tf.constant([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=tf.float32),
            "video_language_idx": tf.constant([1, 2, 3, 4], dtype=tf.int32),
            "tags_embedding": tf.random.normal((4, 100)),
        }

    def test_video_tower_output_shape(self, config, vocab_sizes, sample_inputs):
        """Test that video tower outputs correct shape."""
        tower = VideoTower(config, vocab_sizes)
        output = tower(sample_inputs, training=False)

        assert output.shape == (4, config.embedding_dim)

    def test_video_tower_l2_normalized(self, config, vocab_sizes, sample_inputs):
        """Test that output embeddings are L2 normalized."""
        tower = VideoTower(config, vocab_sizes)
        output = tower(sample_inputs, training=False)

        norms = tf.norm(output, axis=1)
        np.testing.assert_array_almost_equal(norms.numpy(), np.ones(4), decimal=5)


class TestTwoTowerModel:
    """Tests for complete TwoTowerModel."""

    @pytest.fixture
    def config(self):
        return TwoTowerModelConfig(
            embedding_dim=16,
            user_tower_hidden_dims=[32, 16],
            video_tower_hidden_dims=[64, 32],
            temperature=0.05,
        )

    @pytest.fixture
    def vocab_sizes(self):
        return {
            "user_id": 100,
            "video_id": 500,
            "country": 20,
            "language": 10,
            "category": 15,
            "age_bucket": 8,
            "duration_bucket": 8,
        }

    @pytest.fixture
    def sample_batch(self):
        batch_size = 8
        return {
            # User features
            "user_id_idx": tf.constant(list(range(batch_size)), dtype=tf.int32),
            "country_idx": tf.constant([i % 5 for i in range(batch_size)], dtype=tf.int32),
            "user_language_idx": tf.constant([i % 3 for i in range(batch_size)], dtype=tf.int32),
            "age_normalized": tf.random.normal((batch_size,)),
            "age_bucket_idx": tf.constant([i % 7 for i in range(batch_size)], dtype=tf.int32),
            "previously_watched_category_idx": tf.constant([i % 10 for i in range(batch_size)], dtype=tf.int32),
            # Video features
            "video_id_idx": tf.constant(list(range(batch_size)), dtype=tf.int32),
            "category_idx": tf.constant([i % 10 for i in range(batch_size)], dtype=tf.int32),
            "title_embedding": tf.random.normal((batch_size, 512)),
            "video_duration_normalized": tf.random.normal((batch_size,)),
            "duration_bucket_idx": tf.constant([i % 7 for i in range(batch_size)], dtype=tf.int32),
            "popularity_onehot": tf.one_hot([i % 4 for i in range(batch_size)], 4),
            "video_language_idx": tf.constant([i % 3 for i in range(batch_size)], dtype=tf.int32),
            "tags_embedding": tf.random.normal((batch_size, 100)),
        }

    def test_model_forward_pass(self, config, vocab_sizes, sample_batch):
        """Test forward pass returns correct shapes."""
        model = TwoTowerModel(config, vocab_sizes)
        user_emb, video_emb = model(sample_batch, training=False)

        assert user_emb.shape == (8, config.embedding_dim)
        assert video_emb.shape == (8, config.embedding_dim)

    def test_similarity_computation(self, config, vocab_sizes, sample_batch):
        """Test similarity matrix computation."""
        model = TwoTowerModel(config, vocab_sizes)
        user_emb, video_emb = model(sample_batch, training=False)

        similarity = model.compute_similarity(user_emb, video_emb)

        assert similarity.shape == (8, 8)

    def test_in_batch_negatives_loss(self, config, vocab_sizes, sample_batch):
        """Test in-batch negatives loss computation."""
        model = TwoTowerModel(config, vocab_sizes)
        user_emb, video_emb = model(sample_batch, training=False)

        loss = model.compute_in_batch_negatives_loss(user_emb, video_emb)

        # Loss should be a scalar
        assert loss.shape == ()
        # Loss should be positive
        assert loss.numpy() > 0

    def test_model_training_step(self, config, vocab_sizes, sample_batch):
        """Test training step."""
        model = TwoTowerModel(config, vocab_sizes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        # Run training step
        metrics = model.train_step(sample_batch)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_model_save_load(self, config, vocab_sizes, sample_batch):
        """Test model save and load."""
        model = TwoTowerModel(config, vocab_sizes)

        # Build model by running forward pass
        user_emb_before, video_emb_before = model(sample_batch, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model.save_towers(tmpdir)

            # Create new model and load
            model2 = TwoTowerModel(config, vocab_sizes)
            model2(sample_batch, training=False)  # Build model
            model2.load_towers(tmpdir)

            # Check outputs match
            user_emb_after, video_emb_after = model2(sample_batch, training=False)

            np.testing.assert_array_almost_equal(
                user_emb_before.numpy(),
                user_emb_after.numpy(),
                decimal=5
            )
            np.testing.assert_array_almost_equal(
                video_emb_before.numpy(),
                video_emb_after.numpy(),
                decimal=5
            )

    def test_get_embeddings_separately(self, config, vocab_sizes, sample_batch):
        """Test getting user and video embeddings separately."""
        model = TwoTowerModel(config, vocab_sizes)

        # Get user embeddings
        user_inputs = {
            "user_id_idx": sample_batch["user_id_idx"],
            "country_idx": sample_batch["country_idx"],
            "user_language_idx": sample_batch["user_language_idx"],
            "age_normalized": sample_batch["age_normalized"],
            "age_bucket_idx": sample_batch["age_bucket_idx"],
            "previously_watched_category_idx": sample_batch["previously_watched_category_idx"],
        }
        user_emb = model.get_user_embeddings(user_inputs)

        # Get video embeddings
        video_inputs = {
            "video_id_idx": sample_batch["video_id_idx"],
            "category_idx": sample_batch["category_idx"],
            "title_embedding": sample_batch["title_embedding"],
            "video_duration_normalized": sample_batch["video_duration_normalized"],
            "duration_bucket_idx": sample_batch["duration_bucket_idx"],
            "popularity_onehot": sample_batch["popularity_onehot"],
            "video_language_idx": sample_batch["video_language_idx"],
            "tags_embedding": sample_batch["tags_embedding"],
        }
        video_emb = model.get_video_embeddings(video_inputs)

        assert user_emb.shape == (8, config.embedding_dim)
        assert video_emb.shape == (8, config.embedding_dim)
