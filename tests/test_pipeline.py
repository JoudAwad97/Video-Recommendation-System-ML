"""Tests for the processing pipeline."""

import pytest
import tempfile
from pathlib import Path

from src.pipeline.processing_pipeline import ProcessingPipeline
from src.pipeline.pipeline_config import PipelineConfig


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    def test_run_synthetic_pipeline(self):
        """Test running the full pipeline with synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                raw_data_dir=f"{tmpdir}/raw",
                artifacts_dir=f"{tmpdir}/artifacts",
                processed_data_dir=f"{tmpdir}/processed",
                negative_ratio=2,
                compute_embeddings=False,  # Skip for faster testing
            )

            pipeline = ProcessingPipeline(pipeline_config=config)
            results = pipeline.run_synthetic(
                num_users=50,
                num_channels=5,
                num_videos=25,
                num_interactions=200,
                save_outputs=True
            )

            # Check all expected outputs
            assert "two_tower_train" in results
            assert "two_tower_val" in results
            assert "two_tower_test" in results
            assert "ranker_train" in results
            assert "ranker_val" in results
            assert "ranker_test" in results

            # Check outputs were saved
            assert (Path(config.two_tower_output_dir) / "train.parquet").exists()
            assert (Path(config.ranker_output_dir) / "train.parquet").exists()

    def test_pipeline_metadata(self):
        """Test that pipeline records metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                raw_data_dir=f"{tmpdir}/raw",
                artifacts_dir=f"{tmpdir}/artifacts",
                processed_data_dir=f"{tmpdir}/processed",
                compute_embeddings=False,
            )

            pipeline = ProcessingPipeline(pipeline_config=config)
            pipeline.run_synthetic(
                num_users=30,
                num_channels=3,
                num_videos=15,
                num_interactions=100,
                save_outputs=False
            )

            metadata = pipeline.get_metadata()

            assert "start_time" in metadata
            assert "end_time" in metadata
            assert "duration_seconds" in metadata
            assert "num_users" in metadata
            assert metadata["num_users"] == 30

    def test_get_feature_specs(self):
        """Test getting feature specifications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                raw_data_dir=f"{tmpdir}/raw",
                artifacts_dir=f"{tmpdir}/artifacts",
                processed_data_dir=f"{tmpdir}/processed",
                compute_embeddings=False,
            )

            pipeline = ProcessingPipeline(pipeline_config=config)
            pipeline.run_synthetic(
                num_users=30,
                num_channels=3,
                num_videos=15,
                num_interactions=100,
                save_outputs=False
            )

            specs = pipeline.get_feature_specs()

            assert "two_tower" in specs
            assert "ranker" in specs
            assert "user_tower" in specs["two_tower"]
            assert "video_tower" in specs["two_tower"]


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.negative_ratio == 3
        assert config.min_watch_ratio == 0.4

    def test_output_dirs(self):
        """Test output directory properties."""
        config = PipelineConfig(processed_data_dir="my_output")

        assert "my_output" in config.two_tower_output_dir
        assert "my_output" in config.ranker_output_dir
        assert "two_tower" in config.two_tower_output_dir
        assert "ranker" in config.ranker_output_dir
