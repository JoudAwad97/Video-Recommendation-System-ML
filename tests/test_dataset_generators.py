"""Tests for dataset generators."""

import pytest
import pandas as pd
import numpy as np

from src.data.synthetic_generator import SyntheticDataGenerator
from src.dataset.two_tower_dataset import TwoTowerDatasetGenerator
from src.dataset.ranker_dataset import RankerDatasetGenerator


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""

    def test_generate_users(self):
        """Test user generation."""
        generator = SyntheticDataGenerator(seed=42)
        users = generator.generate_users(100)

        assert len(users) == 100
        assert "id" in users.columns
        assert "age" in users.columns
        assert "country_code" in users.columns
        assert users["id"].is_unique

    def test_generate_videos(self):
        """Test video generation."""
        generator = SyntheticDataGenerator(seed=42)
        channels = generator.generate_channels(10)
        videos = generator.generate_videos(50, channels)

        assert len(videos) == 50
        assert "id" in videos.columns
        assert "category" in videos.columns
        assert "duration" in videos.columns
        assert videos["id"].is_unique

    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = SyntheticDataGenerator(seed=42)
        users = generator.generate_users(100)
        channels = generator.generate_channels(10)
        videos = generator.generate_videos(50, channels)
        interactions = generator.generate_interactions(500, users, videos)

        assert len(interactions) == 500
        assert "user_id" in interactions.columns
        assert "video_id" in interactions.columns
        assert "interaction_type" in interactions.columns

    def test_generate_all(self):
        """Test generating all datasets."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_all(
            num_users=100,
            num_channels=10,
            num_videos=50,
            num_interactions=500
        )

        assert "users" in data
        assert "channels" in data
        assert "videos" in data
        assert "interactions" in data

    def test_reproducibility(self):
        """Test that same seed produces consistent structure."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        users1 = gen1.generate_users(10)
        users2 = gen2.generate_users(10)

        # Check structure is the same
        assert list(users1.columns) == list(users2.columns)
        assert len(users1) == len(users2)
        # IDs should be consistent
        pd.testing.assert_series_equal(users1["id"], users2["id"])


class TestTwoTowerDatasetGenerator:
    """Tests for TwoTowerDatasetGenerator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_all(
            num_users=100,
            num_channels=10,
            num_videos=50,
            num_interactions=500
        )

    def test_fit(self, sample_data):
        """Test fitting the generator."""
        gen = TwoTowerDatasetGenerator()
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            compute_embeddings=False
        )

        assert gen._is_fitted

    def test_generate(self, sample_data):
        """Test generating the dataset."""
        gen = TwoTowerDatasetGenerator()
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            compute_embeddings=False
        )

        dataset = gen.generate()

        assert len(dataset) > 0
        assert "user_id" in dataset.columns
        assert "video_id" in dataset.columns
        assert "category" in dataset.columns

    def test_generate_splits(self, sample_data):
        """Test generating train/val/test splits."""
        gen = TwoTowerDatasetGenerator()
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            compute_embeddings=False
        )

        train, val, test = gen.generate_splits(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )

        total = len(train) + len(val) + len(test)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_only_positive_pairs(self, sample_data):
        """Test that dataset contains only positive pairs."""
        gen = TwoTowerDatasetGenerator()
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            compute_embeddings=False
        )

        # Check that we have positive interactions
        assert gen._positive_interactions is not None
        assert len(gen._positive_interactions) > 0


class TestRankerDatasetGenerator:
    """Tests for RankerDatasetGenerator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_all(
            num_users=100,
            num_channels=10,
            num_videos=50,
            num_interactions=500
        )

    def test_fit(self, sample_data):
        """Test fitting the generator."""
        gen = RankerDatasetGenerator(negative_ratio=2)
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            channels_df=sample_data["channels"]
        )

        assert gen._is_fitted

    def test_generate_with_negatives(self, sample_data):
        """Test that dataset contains positive and negative samples."""
        gen = RankerDatasetGenerator(negative_ratio=2)
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            channels_df=sample_data["channels"]
        )

        dataset = gen.generate()

        assert "label" in dataset.columns
        pos_count = (dataset["label"] == 1).sum()
        neg_count = (dataset["label"] == 0).sum()

        assert pos_count > 0
        assert neg_count > 0

    def test_negative_ratio(self, sample_data):
        """Test that negative ratio is approximately correct."""
        gen = RankerDatasetGenerator(negative_ratio=3)
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            channels_df=sample_data["channels"]
        )

        dist = gen.get_label_distribution()
        # Ratio should be approximately 1:3 (allow some variance)
        ratio = dist["negative"] / dist["positive"]
        assert 2 <= ratio <= 4  # Allow some variance

    def test_generate_splits_stratified(self, sample_data):
        """Test that splits are stratified by label."""
        gen = RankerDatasetGenerator(negative_ratio=2)
        gen.fit(
            users_df=sample_data["users"],
            videos_df=sample_data["videos"],
            interactions_df=sample_data["interactions"],
            channels_df=sample_data["channels"]
        )

        train, val, test = gen.generate_splits(stratify=True)

        # Each split should have both positive and negative samples
        for split in [train, val, test]:
            if "label" in split.columns:
                assert (split["label"] == 1).sum() > 0
                assert (split["label"] == 0).sum() > 0
