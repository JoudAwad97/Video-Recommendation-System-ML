"""
Training job orchestrator for ML pipeline.

Handles training job management for Two-Tower and Ranker models
with support for distributed training and continuous learning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import time
import numpy as np

from ..utils.logging_utils import get_logger
from .pipeline_config import TrainingConfig, ModelType

logger = get_logger(__name__)


class TrainingJobStatus(Enum):
    """Training job status."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingMetrics:
    """Training metrics for a model."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0

    # Two-Tower specific
    train_recall_at_k: float = 0.0
    val_recall_at_k: float = 0.0

    # Ranker specific
    train_auc: float = 0.0
    val_auc: float = 0.0
    train_ndcg: float = 0.0
    val_ndcg: float = 0.0

    # Timing
    epoch_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_recall_at_k": self.train_recall_at_k,
            "val_recall_at_k": self.val_recall_at_k,
            "train_auc": self.train_auc,
            "val_auc": self.val_auc,
            "train_ndcg": self.train_ndcg,
            "val_ndcg": self.val_ndcg,
            "epoch_duration_seconds": self.epoch_duration_seconds,
        }


@dataclass
class TrainingJobResult:
    """Result of a training job."""

    job_id: str
    model_type: str
    status: str
    started_at: str
    completed_at: str = ""

    # Paths
    model_artifact_path: str = ""
    checkpoint_path: str = ""
    logs_path: str = ""

    # Final metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_epoch: int = 0

    # Training history
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    # Hyperparameters used
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Resource usage
    training_duration_seconds: float = 0.0
    instance_type: str = ""
    instance_count: int = 1

    # Errors
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_artifact_path": self.model_artifact_path,
            "checkpoint_path": self.checkpoint_path,
            "logs_path": self.logs_path,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_epoch": self.best_epoch,
            "metrics_history": self.metrics_history,
            "hyperparameters": self.hyperparameters,
            "training_duration_seconds": self.training_duration_seconds,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "error_message": self.error_message,
        }


class TrainingOrchestrator:
    """Orchestrator for training jobs.

    Manages the full training lifecycle:
    1. Job submission and tracking
    2. Hyperparameter configuration
    3. Distributed training setup
    4. Checkpointing and recovery
    5. Metrics collection

    Example:
        >>> orchestrator = TrainingOrchestrator(config)
        >>> result = orchestrator.train(
        ...     train_data_path="data/train.parquet",
        ...     val_data_path="data/val.parquet",
        ... )
    """

    def __init__(
        self,
        config: TrainingConfig,
        output_path: str = "models",
    ):
        """Initialize the training orchestrator.

        Args:
            config: Training configuration.
            output_path: Path for model outputs.
        """
        self.config = config
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Job tracking
        self._jobs: Dict[str, TrainingJobResult] = {}
        self._job_counter = 0
        self._active_job: Optional[str] = None

        # Callbacks
        self._progress_callbacks: List[Callable[[str, TrainingMetrics], None]] = []
        self._completion_callbacks: List[Callable[[TrainingJobResult], None]] = []

    def train(
        self,
        train_data_path: str,
        val_data_path: str,
        model_type: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        base_model_path: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> TrainingJobResult:
        """Run a training job.

        Args:
            train_data_path: Path to training data.
            val_data_path: Path to validation data.
            model_type: Model type (two_tower or ranker).
            hyperparameters: Optional hyperparameter overrides.
            base_model_path: Path to base model for continuous learning.
            job_name: Optional job name.

        Returns:
            TrainingJobResult.
        """
        self._job_counter += 1
        job_id = job_name or f"train_{self._job_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow().isoformat()

        model_type = model_type or self.config.model_type

        # Merge hyperparameters
        hparams = self._get_hyperparameters(model_type, hyperparameters)

        result = TrainingJobResult(
            job_id=job_id,
            model_type=model_type,
            status=TrainingJobStatus.STARTING.value,
            started_at=started_at,
            hyperparameters=hparams,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
        )

        self._jobs[job_id] = result
        self._active_job = job_id

        try:
            # Create job directory
            job_path = self.output_path / job_id
            job_path.mkdir(parents=True, exist_ok=True)

            result.checkpoint_path = str(job_path / "checkpoints")
            result.logs_path = str(job_path / "logs")
            Path(result.checkpoint_path).mkdir(exist_ok=True)
            Path(result.logs_path).mkdir(exist_ok=True)

            result.status = TrainingJobStatus.RUNNING.value

            # Run training based on model type
            if model_type == "two_tower":
                self._train_two_tower(
                    result, train_data_path, val_data_path, hparams, base_model_path
                )
            elif model_type == "ranker":
                self._train_ranker(
                    result, train_data_path, val_data_path, hparams, base_model_path
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Save model artifact
            result.model_artifact_path = str(job_path / "model")
            Path(result.model_artifact_path).mkdir(exist_ok=True)

            # Mark completed
            result.status = TrainingJobStatus.COMPLETED.value
            result.completed_at = datetime.utcnow().isoformat()

            # Calculate duration
            start_time = datetime.fromisoformat(started_at)
            end_time = datetime.fromisoformat(result.completed_at)
            result.training_duration_seconds = (end_time - start_time).total_seconds()

            logger.info(
                f"Training job {job_id} completed: "
                f"val_loss={result.final_val_loss:.4f}, "
                f"duration={result.training_duration_seconds:.1f}s"
            )

        except Exception as e:
            result.status = TrainingJobStatus.FAILED.value
            result.error_message = str(e)
            result.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Training job {job_id} failed: {e}")

        # Save job metadata
        self._save_job_metadata(result)

        # Trigger completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Completion callback error: {e}")

        self._active_job = None
        return result

    def _get_hyperparameters(
        self,
        model_type: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hyperparameters with optional overrides.

        Args:
            model_type: Model type.
            overrides: Parameter overrides.

        Returns:
            Combined hyperparameters.
        """
        if model_type == "two_tower":
            hparams = {
                "embedding_dim": self.config.embedding_dim,
                "user_tower_hidden_dims": self.config.user_tower_hidden_dims,
                "video_tower_hidden_dims": self.config.video_tower_hidden_dims,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
            }
        elif model_type == "ranker":
            hparams = {
                "iterations": self.config.ranker_iterations,
                "depth": self.config.ranker_depth,
                "learning_rate": self.config.ranker_learning_rate,
            }
        else:
            hparams = {}

        if overrides:
            hparams.update(overrides)

        return hparams

    def _train_two_tower(
        self,
        result: TrainingJobResult,
        train_path: str,
        val_path: str,
        hparams: Dict[str, Any],
        base_model_path: Optional[str] = None,
    ) -> None:
        """Train Two-Tower model.

        Args:
            result: Job result to update.
            train_path: Training data path.
            val_path: Validation data path.
            hparams: Hyperparameters.
            base_model_path: Base model for continuous learning.
        """
        epochs = hparams.get("epochs", 10)
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(epochs):
            epoch_start = time.time()

            # Simulate training (in production, would use actual TensorFlow training)
            train_loss = 1.0 / (epoch + 1) + 0.1 * (0.5 - np.random.random())
            val_loss = 1.2 / (epoch + 1) + 0.15 * (0.5 - np.random.random())
            recall_at_k = min(0.95, 0.3 + 0.07 * epoch + 0.05 * np.random.random())

            epoch_duration = time.time() - epoch_start

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_recall_at_k=recall_at_k,
                val_recall_at_k=recall_at_k * 0.95,
                epoch_duration_seconds=epoch_duration,
            )

            result.metrics_history.append(metrics.to_dict())

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

            # Report progress
            self._report_progress(result.job_id, metrics)

            logger.debug(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        result.final_train_loss = result.metrics_history[-1]["train_loss"]
        result.final_val_loss = result.metrics_history[-1]["val_loss"]
        result.best_epoch = best_epoch

    def _train_ranker(
        self,
        result: TrainingJobResult,
        train_path: str,
        val_path: str,
        hparams: Dict[str, Any],
        base_model_path: Optional[str] = None,
    ) -> None:
        """Train Ranker model.

        Args:
            result: Job result to update.
            train_path: Training data path.
            val_path: Validation data path.
            hparams: Hyperparameters.
            base_model_path: Base model for continuous learning.
        """
        iterations = hparams.get("iterations", 1000)

        # CatBoost training would happen here
        # Simulate training with periodic metrics

        checkpoint_iterations = [100, 250, 500, 750, iterations]
        best_val_loss = float("inf")
        best_iter = 0

        for i, iter_num in enumerate(checkpoint_iterations):
            epoch_start = time.time()

            # Simulate metrics
            train_loss = 0.5 / (i + 1) + 0.05 * np.random.random()
            val_loss = 0.6 / (i + 1) + 0.08 * np.random.random()
            train_auc = min(0.99, 0.7 + 0.05 * i + 0.02 * np.random.random())
            val_auc = train_auc * 0.98
            train_ndcg = min(0.95, 0.5 + 0.08 * i + 0.03 * np.random.random())
            val_ndcg = train_ndcg * 0.96

            epoch_duration = time.time() - epoch_start

            metrics = TrainingMetrics(
                epoch=iter_num,
                train_loss=train_loss,
                val_loss=val_loss,
                train_auc=train_auc,
                val_auc=val_auc,
                train_ndcg=train_ndcg,
                val_ndcg=val_ndcg,
                epoch_duration_seconds=epoch_duration,
            )

            result.metrics_history.append(metrics.to_dict())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iter = iter_num

            self._report_progress(result.job_id, metrics)

            logger.debug(
                f"Iteration {iter_num}: "
                f"train_loss={train_loss:.4f}, val_auc={val_auc:.4f}"
            )

        result.final_train_loss = result.metrics_history[-1]["train_loss"]
        result.final_val_loss = result.metrics_history[-1]["val_loss"]
        result.best_epoch = best_iter

    def _report_progress(self, job_id: str, metrics: TrainingMetrics) -> None:
        """Report training progress.

        Args:
            job_id: Job identifier.
            metrics: Current metrics.
        """
        for callback in self._progress_callbacks:
            try:
                callback(job_id, metrics)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _save_job_metadata(self, result: TrainingJobResult) -> None:
        """Save job metadata to file.

        Args:
            result: Job result to save.
        """
        job_path = self.output_path / result.job_id
        metadata_path = job_path / "job_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_job(self, job_id: str) -> Optional[TrainingJobResult]:
        """Get job result.

        Args:
            job_id: Job identifier.

        Returns:
            TrainingJobResult or None.
        """
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJobResult]:
        """List all jobs.

        Returns:
            List of job results.
        """
        return list(self._jobs.values())

    def stop_job(self, job_id: str) -> bool:
        """Stop a running job.

        Args:
            job_id: Job identifier.

        Returns:
            True if job was stopped.
        """
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        if job.status == TrainingJobStatus.RUNNING.value:
            job.status = TrainingJobStatus.STOPPED.value
            job.completed_at = datetime.utcnow().isoformat()
            logger.info(f"Stopped job {job_id}")
            return True

        return False

    def register_progress_callback(
        self,
        callback: Callable[[str, TrainingMetrics], None],
    ) -> None:
        """Register progress callback.

        Args:
            callback: Function called with (job_id, metrics).
        """
        self._progress_callbacks.append(callback)

    def register_completion_callback(
        self,
        callback: Callable[[TrainingJobResult], None],
    ) -> None:
        """Register completion callback.

        Args:
            callback: Function called with job result.
        """
        self._completion_callbacks.append(callback)


class SageMakerTrainingJob:
    """SageMaker training job wrapper.

    Runs training as a SageMaker Training job for
    scalable, managed execution with GPU support.
    """

    def __init__(
        self,
        config: TrainingConfig,
        role_arn: str,
        s3_bucket: str,
    ):
        """Initialize SageMaker training job.

        Args:
            config: Training configuration.
            role_arn: IAM role ARN.
            s3_bucket: S3 bucket for data.
        """
        self.config = config
        self.role_arn = role_arn
        self.s3_bucket = s3_bucket
        self._boto3_available = False

        try:
            import boto3
            self._boto3_available = True
            self._sagemaker_client = boto3.client("sagemaker")
        except ImportError:
            logger.warning("boto3 not available. SageMaker jobs will not work.")

    def submit(
        self,
        train_s3_uri: str,
        val_s3_uri: str,
        output_s3_uri: str,
        job_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit training job.

        Args:
            train_s3_uri: S3 URI for training data.
            val_s3_uri: S3 URI for validation data.
            output_s3_uri: S3 URI for output.
            job_name: Optional job name.
            hyperparameters: Hyperparameters.

        Returns:
            Training job ARN.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker jobs")

        job_name = job_name or f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        # Prepare hyperparameters
        hparams = hyperparameters or {}
        hparams_str = {k: str(v) for k, v in hparams.items()}

        response = self._sagemaker_client.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": self._get_training_image_uri(),
                "TrainingInputMode": "File",
            },
            RoleArn=self.role_arn,
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": val_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            OutputDataConfig={
                "S3OutputPath": output_s3_uri,
            },
            ResourceConfig={
                "InstanceType": self.config.instance_type,
                "InstanceCount": self.config.instance_count,
                "VolumeSizeInGB": 50,
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": self.config.max_wait_seconds,
            },
            HyperParameters=hparams_str,
            EnableManagedSpotTraining=self.config.use_spot_instances,
        )

        job_arn = response["TrainingJobArn"]
        logger.info(f"Submitted SageMaker Training job: {job_name}")
        return job_arn

    def _get_training_image_uri(self) -> str:
        """Get training container image URI.

        Returns:
            Container image URI.
        """
        # Use TensorFlow training container
        return f"763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13.0-gpu-py310"

    def get_status(self, job_name: str) -> Dict[str, Any]:
        """Get job status.

        Args:
            job_name: Training job name.

        Returns:
            Job status information.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker jobs")

        response = self._sagemaker_client.describe_training_job(
            TrainingJobName=job_name
        )

        return {
            "job_name": job_name,
            "status": response["TrainingJobStatus"],
            "secondary_status": response.get("SecondaryStatus"),
            "creation_time": response.get("CreationTime"),
            "training_end_time": response.get("TrainingEndTime"),
            "failure_reason": response.get("FailureReason"),
            "final_metric_data": response.get("FinalMetricDataList"),
        }
