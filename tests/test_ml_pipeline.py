"""
Unit tests for ML pipeline components.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.ml_pipeline.pipeline_config import (
    MLPipelineConfig,
    PipelineStep,
    PipelineExecutionMode,
    ModelType,
    PreprocessingConfig,
    DataSplittingConfig,
    TrainingConfig,
    EvaluationConfig,
    DeploymentTargetConfig,
)
from src.ml_pipeline.preprocessing_job import (
    PreprocessingJob,
    PreprocessingResult,
)
from src.ml_pipeline.data_versioning import (
    DataVersionManager,
    DataVersion,
)
from src.ml_pipeline.training_orchestrator import (
    TrainingOrchestrator,
    TrainingJobResult,
    TrainingJobStatus,
)
from src.ml_pipeline.model_registry import (
    ModelRegistry,
    RegisteredModel,
    ModelMetrics,
    ModelStage,
    ModelStatus,
)
from src.ml_pipeline.deployment_pipeline import (
    DeploymentPipeline,
    DeploymentResult,
    DeploymentStrategy,
    DeploymentStatus,
)
from src.ml_pipeline.ml_pipeline import (
    MLPipeline,
    PipelineRunResult,
    PipelineRunStatus,
)


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_ml_pipeline_config(self):
        """Test default MLPipelineConfig."""
        config = MLPipelineConfig()

        assert config.pipeline_name == "video-recommendation-pipeline"
        assert config.execution_mode == PipelineExecutionMode.FULL_TRAINING
        assert config.model_type == ModelType.TWO_TOWER
        assert len(config.steps_to_run) == 6

    def test_pipeline_step_enum(self):
        """Test PipelineStep enum values."""
        assert PipelineStep.PREPROCESSING.value == "preprocessing"
        assert PipelineStep.TRAINING.value == "training"
        assert PipelineStep.DEPLOYMENT.value == "deployment"

    def test_execution_mode_enum(self):
        """Test PipelineExecutionMode enum values."""
        assert PipelineExecutionMode.FULL_TRAINING.value == "full_training"
        assert PipelineExecutionMode.CONTINUOUS_TRAINING.value == "continuous_training"
        assert PipelineExecutionMode.INFERENCE_ONLY.value == "inference_only"

    def test_should_run_step(self):
        """Test should_run_step method."""
        config = MLPipelineConfig()

        assert config.should_run_step(PipelineStep.PREPROCESSING) is True
        assert config.should_run_step(PipelineStep.TRAINING) is True
        assert config.should_run_step(PipelineStep.MONITORING_SETUP) is False

    def test_get_s3_path(self):
        """Test get_s3_path method."""
        config = MLPipelineConfig(s3_bucket="my-bucket")

        path = config.get_s3_path("data", "processed")
        assert path == "s3://my-bucket/data/processed"

    def test_training_config(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()

        assert config.model_type == "two_tower"
        assert config.embedding_dim == 16
        assert config.batch_size == 256
        assert config.epochs == 10

    def test_deployment_config(self):
        """Test DeploymentTargetConfig."""
        config = DeploymentTargetConfig()

        assert config.use_serverless is True
        assert config.serverless_memory_mb == 2048
        assert config.enable_canary_deployment is True


class TestPreprocessingJob:
    """Tests for PreprocessingJob."""

    @pytest.fixture
    def config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PreprocessingConfig(
                raw_interactions_path=f"{tmpdir}/interactions",
                raw_users_path=f"{tmpdir}/users",
                raw_videos_path=f"{tmpdir}/videos",
                processed_data_path=f"{tmpdir}/processed",
            )

    def test_create_preprocessing_job(self, config):
        """Test creating a preprocessing job."""
        job = PreprocessingJob(config)
        assert job.config == config

    def test_run_preprocessing_empty_data(self, config):
        """Test preprocessing with no data."""
        job = PreprocessingJob(config)
        result = job.run()

        assert result.job_id is not None
        assert result.status in ["completed", "failed"]

    def test_build_vocabulary(self, config):
        """Test vocabulary building."""
        job = PreprocessingJob(config)
        vocab = job._build_vocabulary(["a", "b", "c"])

        assert "[PAD]" in vocab
        assert "[OOV]" in vocab
        assert "a" in vocab
        assert vocab["[PAD]"] == 0
        assert vocab["[OOV]"] == 1

    def test_compute_feature_stats(self, config):
        """Test computing feature statistics."""
        job = PreprocessingJob(config)
        series = pd.Series([1, 2, 3, 4, 5])
        stats = job._compute_feature_stats(series)

        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"] == 3.0

    def test_compute_feature_stats_log_transform(self, config):
        """Test log-transformed statistics."""
        job = PreprocessingJob(config)
        series = pd.Series([10, 100, 1000])
        stats = job._compute_feature_stats(series, log_transform=True)

        assert "log_mean" in stats
        assert "log_std" in stats


class TestDataVersionManager:
    """Tests for DataVersionManager."""

    @pytest.fixture
    def config(self):
        return DataSplittingConfig(
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "user_id": list(range(100)),
            "video_id": list(range(100)),
            "label": [i % 2 for i in range(100)],
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="H"),
        })

    def test_create_version(self, config, sample_df):
        """Test creating a data version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataVersionManager(config, base_path=tmpdir)
            version = manager.create_version(sample_df)

            assert version.version_id == "v1"
            assert version.total_records == 100
            assert version.train_records == 80
            assert version.validation_records == 10
            assert version.test_records == 10

    def test_load_version(self, config, sample_df):
        """Test loading a data version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataVersionManager(config, base_path=tmpdir)
            version = manager.create_version(sample_df)

            result = manager.load_version(version.version_id)

            assert len(result.train_df) == 80
            assert len(result.validation_df) == 10
            assert len(result.test_df) == 10

    def test_stratified_split(self, config, sample_df):
        """Test stratified splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataVersionManager(config, base_path=tmpdir)
            version = manager.create_version(sample_df, stratify=True)

            result = manager.load_version(version.version_id)

            # Check label distribution is preserved
            train_ratio = result.train_df["label"].mean()
            assert 0.4 <= train_ratio <= 0.6  # Should be around 0.5

    def test_version_history(self, config, sample_df):
        """Test version history tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataVersionManager(config, base_path=tmpdir)

            v1 = manager.create_version(sample_df)
            v2 = manager.create_version(sample_df)

            versions = manager.list_versions()
            assert len(versions) == 2
            assert versions[0].version_number == 2  # Latest first

    def test_compare_versions(self, config, sample_df):
        """Test version comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataVersionManager(config, base_path=tmpdir)

            v1 = manager.create_version(sample_df)

            # Create different data
            sample_df2 = sample_df.copy()
            sample_df2 = pd.concat([sample_df2, sample_df2.head(20)], ignore_index=True)
            v2 = manager.create_version(sample_df2)

            comparison = manager.compare_versions(v1.version_id, v2.version_id)

            assert comparison["record_diff"] == 20


class TestTrainingOrchestrator:
    """Tests for TrainingOrchestrator."""

    @pytest.fixture
    def config(self):
        return TrainingConfig(
            model_type="two_tower",
            epochs=3,  # Reduced for testing
        )

    def test_create_orchestrator(self, config):
        """Test creating training orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = TrainingOrchestrator(config, output_path=tmpdir)
            assert orchestrator.config == config

    def test_train_two_tower(self, config):
        """Test training Two-Tower model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = TrainingOrchestrator(config, output_path=tmpdir)

            result = orchestrator.train(
                train_data_path="",
                val_data_path="",
                model_type="two_tower",
            )

            assert result.status == TrainingJobStatus.COMPLETED.value
            assert result.model_type == "two_tower"
            assert len(result.metrics_history) == 3  # 3 epochs

    def test_train_ranker(self, config):
        """Test training Ranker model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.model_type = "ranker"
            orchestrator = TrainingOrchestrator(config, output_path=tmpdir)

            result = orchestrator.train(
                train_data_path="",
                val_data_path="",
                model_type="ranker",
            )

            assert result.status == TrainingJobStatus.COMPLETED.value
            assert result.model_type == "ranker"

    def test_training_metrics_recorded(self, config):
        """Test that training metrics are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = TrainingOrchestrator(config, output_path=tmpdir)

            result = orchestrator.train(
                train_data_path="",
                val_data_path="",
            )

            assert result.final_train_loss > 0
            assert result.final_val_loss > 0
            assert result.best_epoch > 0

    def test_list_jobs(self, config):
        """Test listing training jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = TrainingOrchestrator(config, output_path=tmpdir)

            orchestrator.train(train_data_path="", val_data_path="")
            orchestrator.train(train_data_path="", val_data_path="")

            jobs = orchestrator.list_jobs()
            assert len(jobs) == 2


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelRegistry(base_path=tmpdir)

    def test_register_model(self, registry):
        """Test registering a model."""
        model = registry.register_model(
            name="two_tower",
            model_type="two_tower",
            artifact_path="",
            description="Test model",
        )

        assert model.model_id == "two_tower_v1"
        assert model.version == 1
        assert model.stage == ModelStage.DEVELOPMENT.value

    def test_get_model(self, registry):
        """Test getting a model by ID."""
        registered = registry.register_model(
            name="two_tower",
            model_type="two_tower",
            artifact_path="",
        )

        model = registry.get_model(registered.model_id)
        assert model is not None
        assert model.model_id == registered.model_id

    def test_version_increment(self, registry):
        """Test version incrementing."""
        v1 = registry.register_model(name="my_model", model_type="two_tower", artifact_path="")
        v2 = registry.register_model(name="my_model", model_type="two_tower", artifact_path="")

        assert v1.version == 1
        assert v2.version == 2
        assert v2.model_id == "my_model_v2"

    def test_promote_to_staging(self, registry):
        """Test promoting model to staging."""
        model = registry.register_model(
            name="test",
            model_type="two_tower",
            artifact_path="",
        )

        success = registry.promote_to_staging(model.model_id)
        assert success is True

        updated = registry.get_model(model.model_id)
        assert updated.stage == ModelStage.STAGING.value

    def test_promote_to_production(self, registry):
        """Test promoting model to production."""
        model = registry.register_model(
            name="test",
            model_type="two_tower",
            artifact_path="",
        )

        success = registry.promote_to_production(model.model_id)
        assert success is True

        prod = registry.get_production_model("two_tower")
        assert prod is not None
        assert prod.model_id == model.model_id

    def test_update_metrics(self, registry):
        """Test updating model metrics."""
        model = registry.register_model(
            name="test",
            model_type="two_tower",
            artifact_path="",
        )

        metrics = ModelMetrics(
            ndcg=0.45,
            mrr=0.35,
            precision_at_k={10: 0.15},
        )

        success = registry.update_metrics(model.model_id, metrics)
        assert success is True

        updated = registry.get_model(model.model_id)
        assert updated.metrics["ndcg"] == 0.45

    def test_compare_models(self, registry):
        """Test comparing models."""
        m1 = registry.register_model(name="test", model_type="two_tower", artifact_path="")
        registry.update_metrics(m1.model_id, ModelMetrics(ndcg=0.35))

        m2 = registry.register_model(name="test", model_type="two_tower", artifact_path="")
        registry.update_metrics(m2.model_id, ModelMetrics(ndcg=0.45))

        comparison = registry.compare_models(m1.model_id, m2.model_id)

        assert "ndcg" in comparison["metrics_comparison"]
        assert comparison["metrics_comparison"]["ndcg"]["difference"] == pytest.approx(0.1)

    def test_find_best_model(self, registry):
        """Test finding best model by metric."""
        m1 = registry.register_model(name="test1", model_type="two_tower", artifact_path="")
        registry.update_metrics(m1.model_id, ModelMetrics(ndcg=0.35))

        m2 = registry.register_model(name="test2", model_type="two_tower", artifact_path="")
        registry.update_metrics(m2.model_id, ModelMetrics(ndcg=0.45))

        best = registry.find_best_model("two_tower", metric="ndcg")
        assert best.model_id == m2.model_id

    def test_list_models_with_filters(self, registry):
        """Test listing models with filters."""
        registry.register_model(name="tt", model_type="two_tower", artifact_path="")
        registry.register_model(name="rk", model_type="ranker", artifact_path="")

        two_tower_models = registry.list_models(model_type="two_tower")
        assert len(two_tower_models) == 1

        ranker_models = registry.list_models(model_type="ranker")
        assert len(ranker_models) == 1


class TestDeploymentPipeline:
    """Tests for DeploymentPipeline."""

    @pytest.fixture
    def config(self):
        return DeploymentTargetConfig(
            endpoint_name_prefix="test-rec",
            enable_canary_deployment=False,  # Disable for faster tests
        )

    @pytest.fixture
    def pipeline(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DeploymentPipeline(
                config,
                registry=None,
                output_path=tmpdir,
            )

    def test_deploy_all_at_once(self, pipeline):
        """Test all-at-once deployment."""
        result = pipeline.deploy(
            model_id="test_model_v1",
            strategy=DeploymentStrategy.ALL_AT_ONCE,
        )

        assert result.status == DeploymentStatus.COMPLETED.value
        assert result.model_id == "test_model_v1"

    def test_deploy_canary(self, config):
        """Test canary deployment."""
        config.enable_canary_deployment = True
        config.canary_duration_minutes = 0  # Instant for testing

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = DeploymentPipeline(config, output_path=tmpdir)

            result = pipeline.deploy(
                model_id="test_model_v1",
                strategy=DeploymentStrategy.CANARY,
                canary_duration_minutes=0,
                wait_for_completion=False,
            )

            assert result.status in [
                DeploymentStatus.COMPLETED.value,
                DeploymentStatus.CANARY_RUNNING.value,
            ]

    def test_get_active_model(self, pipeline):
        """Test getting active model for endpoint."""
        pipeline.deploy(
            model_id="model_v1",
            endpoint_name="my-endpoint",
            strategy=DeploymentStrategy.ALL_AT_ONCE,
        )

        active = pipeline.get_active_model("my-endpoint")
        assert active == "model_v1"

    def test_rollback_deployment(self, pipeline):
        """Test rolling back a deployment."""
        # Deploy first model
        result1 = pipeline.deploy(
            model_id="model_v1",
            endpoint_name="my-endpoint",
            strategy=DeploymentStrategy.ALL_AT_ONCE,
        )

        # Deploy second model
        result2 = pipeline.deploy(
            model_id="model_v2",
            endpoint_name="my-endpoint",
            strategy=DeploymentStrategy.ALL_AT_ONCE,
        )

        # Rollback
        success = pipeline.rollback_deployment(
            result2.deployment_id,
            reason="Test rollback",
        )

        assert success is True

        updated = pipeline.get_deployment(result2.deployment_id)
        assert updated.status == DeploymentStatus.ROLLED_BACK.value

    def test_list_deployments(self, pipeline):
        """Test listing deployments."""
        pipeline.deploy(model_id="m1", strategy=DeploymentStrategy.ALL_AT_ONCE)
        pipeline.deploy(model_id="m2", strategy=DeploymentStrategy.ALL_AT_ONCE)

        deployments = pipeline.list_deployments()
        assert len(deployments) == 2


class TestMLPipeline:
    """Tests for the main MLPipeline orchestrator."""

    @pytest.fixture
    def config(self):
        return MLPipelineConfig(
            steps_to_run=[
                PipelineStep.DATA_SPLITTING,
                PipelineStep.TRAINING,
                PipelineStep.MODEL_REGISTRATION,
            ],
        )

    def test_create_pipeline(self, config):
        """Test creating ML pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            assert pipeline.config == config

    def test_run_pipeline(self, config):
        """Test running the ML pipeline."""
        config.training.epochs = 2  # Reduced for testing

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            assert result.status == PipelineRunStatus.COMPLETED.value
            assert result.data_version is not None
            assert result.model_id is not None

    def test_pipeline_step_results(self, config):
        """Test that step results are recorded."""
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            assert PipelineStep.DATA_SPLITTING.value in result.step_results
            assert PipelineStep.TRAINING.value in result.step_results
            assert PipelineStep.MODEL_REGISTRATION.value in result.step_results

    def test_pipeline_with_preprocessing(self, config):
        """Test pipeline with preprocessing step."""
        config.steps_to_run = [
            PipelineStep.PREPROCESSING,
            PipelineStep.DATA_SPLITTING,
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            assert PipelineStep.PREPROCESSING.value in result.step_results

    def test_get_run(self, config):
        """Test getting a run by ID."""
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            retrieved = pipeline.get_run(result.run_id)
            assert retrieved is not None
            assert retrieved.run_id == result.run_id

    def test_list_runs(self, config):
        """Test listing runs."""
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)

            pipeline.run()
            pipeline.run()

            runs = pipeline.list_runs()
            assert len(runs) == 2

    def test_pipeline_duration_tracking(self, config):
        """Test that pipeline duration is tracked."""
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            assert result.total_duration_seconds > 0


class TestIntegration:
    """Integration tests for the full ML pipeline flow."""

    def test_full_pipeline_flow(self):
        """Test complete pipeline from data to deployment."""
        config = MLPipelineConfig(
            steps_to_run=[
                PipelineStep.DATA_SPLITTING,
                PipelineStep.TRAINING,
                PipelineStep.EVALUATION,
                PipelineStep.MODEL_REGISTRATION,
            ],
        )
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            # Verify pipeline completed
            assert result.status == PipelineRunStatus.COMPLETED.value

            # Verify data version was created
            assert result.data_version is not None
            version = pipeline.data_version_manager.get_version(result.data_version)
            assert version is not None

            # Verify model was registered
            assert result.model_id is not None
            model = pipeline.model_registry.get_model(result.model_id)
            assert model is not None

    def test_continuous_training_mode(self):
        """Test continuous training execution mode."""
        config = MLPipelineConfig(
            execution_mode=PipelineExecutionMode.CONTINUOUS_TRAINING,
            steps_to_run=[
                PipelineStep.DATA_SPLITTING,
                PipelineStep.TRAINING,
            ],
        )
        config.training.epochs = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(config, base_path=tmpdir)
            result = pipeline.run()

            assert result.execution_mode == "continuous_training"
            assert result.status == PipelineRunStatus.COMPLETED.value
