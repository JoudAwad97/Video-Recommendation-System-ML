"""
Main ML pipeline orchestrator.

Orchestrates the full ML pipeline workflow:
1. Preprocessing
2. Data versioning
3. Training
4. Model registration
5. Deployment

Similar to AWS Step Functions orchestration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

from ..utils.logging_utils import get_logger
from .pipeline_config import (
    MLPipelineConfig,
    PipelineStep,
    PipelineExecutionMode,
    ModelType,
)
from .preprocessing_job import PreprocessingJob, PreprocessingResult
from .data_versioning import DataVersionManager, DataVersion, SplitResult
from .training_orchestrator import TrainingOrchestrator, TrainingJobResult
from .model_registry import ModelRegistry, RegisteredModel, ModelMetrics, ModelStage
from .deployment_pipeline import DeploymentPipeline, DeploymentResult, DeploymentStrategy

logger = get_logger(__name__)


class PipelineRunStatus(Enum):
    """Pipeline run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class StepResult:
    """Result of a pipeline step."""

    step: str
    status: str
    started_at: str
    completed_at: str = ""
    output: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class PipelineRunResult:
    """Result of a pipeline run."""

    run_id: str
    pipeline_name: str
    execution_mode: str
    status: str

    # Timestamps
    started_at: str
    completed_at: str = ""

    # Step results
    step_results: Dict[str, StepResult] = field(default_factory=dict)

    # Output artifacts
    data_version: str = ""
    model_id: str = ""
    deployment_id: str = ""

    # Metrics
    total_duration_seconds: float = 0.0

    # Errors
    failed_step: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "data_version": self.data_version,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "total_duration_seconds": self.total_duration_seconds,
            "failed_step": self.failed_step,
            "error_message": self.error_message,
        }


class MLPipeline:
    """Main ML pipeline orchestrator.

    Orchestrates the full ML pipeline workflow similar to AWS Step Functions:
    1. Preprocessing - Data preparation and feature engineering
    2. Data Splitting - Create versioned train/val/test splits
    3. Training - Train Two-Tower or Ranker models
    4. Evaluation - Evaluate model performance
    5. Model Registration - Register model to registry
    6. Deployment - Deploy model to endpoint

    Example:
        >>> config = MLPipelineConfig()
        >>> pipeline = MLPipeline(config)
        >>> result = pipeline.run()
    """

    def __init__(
        self,
        config: MLPipelineConfig,
        base_path: str = "pipeline_runs",
    ):
        """Initialize the ML pipeline.

        Args:
            config: Pipeline configuration.
            base_path: Base path for pipeline outputs.
        """
        self.config = config
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.preprocessing_job = PreprocessingJob(config.preprocessing)
        self.data_version_manager = DataVersionManager(
            config.data_splitting,
            base_path=str(self.base_path / "data_versions"),
        )
        self.training_orchestrator = TrainingOrchestrator(
            config.training,
            output_path=str(self.base_path / "training"),
        )
        self.model_registry = ModelRegistry(
            base_path=str(self.base_path / "model_registry"),
        )
        self.deployment_pipeline = DeploymentPipeline(
            config.deployment,
            registry=self.model_registry,
            output_path=str(self.base_path / "deployments"),
        )

        # Run tracking
        self._runs: Dict[str, PipelineRunResult] = {}
        self._run_counter = 0
        self._active_run: Optional[str] = None

        # Callbacks
        self._step_callbacks: List[Callable[[str, StepResult], None]] = []
        self._completion_callbacks: List[Callable[[PipelineRunResult], None]] = []

    def run(
        self,
        execution_mode: Optional[PipelineExecutionMode] = None,
        model_type: Optional[ModelType] = None,
        steps_to_run: Optional[List[PipelineStep]] = None,
        input_data_path: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> PipelineRunResult:
        """Run the ML pipeline.

        Args:
            execution_mode: Override execution mode.
            model_type: Override model type.
            steps_to_run: Override steps to run.
            input_data_path: Path to input data.
            run_name: Optional run name.

        Returns:
            PipelineRunResult.
        """
        self._run_counter += 1
        run_id = run_name or f"run_{self._run_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow().isoformat()

        execution_mode = execution_mode or self.config.execution_mode
        model_type = model_type or self.config.model_type
        steps_to_run = steps_to_run or self.config.steps_to_run

        result = PipelineRunResult(
            run_id=run_id,
            pipeline_name=self.config.pipeline_name,
            execution_mode=execution_mode.value,
            status=PipelineRunStatus.RUNNING.value,
            started_at=started_at,
        )

        self._runs[run_id] = result
        self._active_run = run_id

        # Create run directory
        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Track intermediate artifacts
        processed_data_path = None
        data_version = None
        train_result = None
        model = None

        try:
            # Step 1: Preprocessing
            if PipelineStep.PREPROCESSING in steps_to_run:
                step_result = self._run_preprocessing(
                    run_path, input_data_path
                )
                result.step_results[PipelineStep.PREPROCESSING.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Preprocessing failed: {step_result.error}")

                processed_data_path = step_result.output.get("processed_data_path")

            # Step 2: Data Splitting
            if PipelineStep.DATA_SPLITTING in steps_to_run:
                step_result = self._run_data_splitting(
                    run_path, processed_data_path
                )
                result.step_results[PipelineStep.DATA_SPLITTING.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Data splitting failed: {step_result.error}")

                data_version = step_result.output.get("version_id")
                result.data_version = data_version

            # Step 3: Training
            if PipelineStep.TRAINING in steps_to_run:
                step_result = self._run_training(
                    run_path, data_version, model_type.value
                )
                result.step_results[PipelineStep.TRAINING.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Training failed: {step_result.error}")

                train_result = step_result.output

            # Step 4: Evaluation
            if PipelineStep.EVALUATION in steps_to_run:
                step_result = self._run_evaluation(
                    run_path, train_result, data_version
                )
                result.step_results[PipelineStep.EVALUATION.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Evaluation failed: {step_result.error}")

            # Step 5: Model Registration
            if PipelineStep.MODEL_REGISTRATION in steps_to_run:
                step_result = self._run_model_registration(
                    run_path, train_result, model_type.value, data_version
                )
                result.step_results[PipelineStep.MODEL_REGISTRATION.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Model registration failed: {step_result.error}")

                model = step_result.output.get("model_id")
                result.model_id = model

            # Step 6: Deployment
            if PipelineStep.DEPLOYMENT in steps_to_run:
                step_result = self._run_deployment(run_path, model)
                result.step_results[PipelineStep.DEPLOYMENT.value] = step_result

                if step_result.status == "failed":
                    raise Exception(f"Deployment failed: {step_result.error}")

                result.deployment_id = step_result.output.get("deployment_id", "")

            result.status = PipelineRunStatus.COMPLETED.value
            logger.info(f"Pipeline run {run_id} completed successfully")

        except Exception as e:
            result.status = PipelineRunStatus.FAILED.value
            result.error_message = str(e)

            # Find failed step
            for step_name, step_result in result.step_results.items():
                if step_result.status == "failed":
                    result.failed_step = step_name
                    break

            logger.error(f"Pipeline run {run_id} failed: {e}")

        result.completed_at = datetime.utcnow().isoformat()

        # Calculate duration
        start_time = datetime.fromisoformat(started_at)
        end_time = datetime.fromisoformat(result.completed_at)
        result.total_duration_seconds = (end_time - start_time).total_seconds()

        # Save run metadata
        self._save_run_metadata(result, run_path)

        # Trigger completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Completion callback error: {e}")

        self._active_run = None
        return result

    def _run_preprocessing(
        self,
        run_path: Path,
        input_data_path: Optional[str],
    ) -> StepResult:
        """Run preprocessing step.

        Args:
            run_path: Run directory.
            input_data_path: Input data path.

        Returns:
            StepResult.
        """
        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.PREPROCESSING.value,
            status="running",
            started_at=started_at,
        )

        try:
            output_path = str(run_path / "processed_data")
            preprocess_result = self.preprocessing_job.run(
                output_path=output_path,
            )

            step_result.status = "completed" if preprocess_result.status == "completed" else "failed"
            step_result.output = {
                "job_id": preprocess_result.job_id,
                "records_processed": preprocess_result.records_processed,
                "processed_data_path": output_path,
                "vocabularies_built": preprocess_result.vocabularies_built,
            }

            if preprocess_result.errors:
                step_result.error = "; ".join(preprocess_result.errors)

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _run_data_splitting(
        self,
        run_path: Path,
        processed_data_path: Optional[str],
    ) -> StepResult:
        """Run data splitting step.

        Args:
            run_path: Run directory.
            processed_data_path: Path to processed data.

        Returns:
            StepResult.
        """
        import pandas as pd

        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.DATA_SPLITTING.value,
            status="running",
            started_at=started_at,
        )

        try:
            # Load processed data
            if processed_data_path:
                data_path = Path(processed_data_path) / "processed_data.parquet"
                if data_path.exists():
                    df = pd.read_parquet(data_path)
                else:
                    # Create sample data for testing
                    df = self._create_sample_data()
            else:
                df = self._create_sample_data()

            # Create version
            version = self.data_version_manager.create_version(df)

            step_result.status = "completed"
            step_result.output = {
                "version_id": version.version_id,
                "total_records": version.total_records,
                "train_records": version.train_records,
                "validation_records": version.validation_records,
                "test_records": version.test_records,
            }

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _create_sample_data(self) -> "pd.DataFrame":
        """Create sample data for testing.

        Returns:
            Sample DataFrame.
        """
        import pandas as pd
        import numpy as np

        n_samples = 1000
        return pd.DataFrame({
            "user_id": np.random.randint(1, 100, n_samples),
            "video_id": np.random.randint(1, 500, n_samples),
            "label": np.random.randint(0, 2, n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
        })

    def _run_training(
        self,
        run_path: Path,
        data_version: Optional[str],
        model_type: str,
    ) -> StepResult:
        """Run training step.

        Args:
            run_path: Run directory.
            data_version: Data version ID.
            model_type: Model type.

        Returns:
            StepResult.
        """
        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.TRAINING.value,
            status="running",
            started_at=started_at,
        )

        try:
            # Get data paths
            train_path = ""
            val_path = ""

            if data_version:
                version = self.data_version_manager.get_version(data_version)
                if version:
                    train_path = version.train_path
                    val_path = version.validation_path

            # Run training
            train_result = self.training_orchestrator.train(
                train_data_path=train_path,
                val_data_path=val_path,
                model_type=model_type,
            )

            step_result.status = "completed" if train_result.status == "completed" else "failed"
            step_result.output = {
                "job_id": train_result.job_id,
                "model_artifact_path": train_result.model_artifact_path,
                "final_val_loss": train_result.final_val_loss,
                "best_epoch": train_result.best_epoch,
                "training_duration_seconds": train_result.training_duration_seconds,
            }

            if train_result.error_message:
                step_result.error = train_result.error_message

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _run_evaluation(
        self,
        run_path: Path,
        train_result: Dict[str, Any],
        data_version: Optional[str],
    ) -> StepResult:
        """Run evaluation step.

        Args:
            run_path: Run directory.
            train_result: Training result.
            data_version: Data version ID.

        Returns:
            StepResult.
        """
        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.EVALUATION.value,
            status="running",
            started_at=started_at,
        )

        try:
            # Simulate evaluation metrics
            import numpy as np

            metrics = {
                "precision_at_10": 0.15 + np.random.uniform(-0.02, 0.02),
                "recall_at_10": 0.25 + np.random.uniform(-0.03, 0.03),
                "ndcg": 0.35 + np.random.uniform(-0.03, 0.03),
                "mrr": 0.30 + np.random.uniform(-0.02, 0.02),
                "diversity": 0.70 + np.random.uniform(-0.05, 0.05),
            }

            # Check against thresholds
            passes_thresholds = (
                metrics["precision_at_10"] >= self.config.evaluation.min_precision_at_10
                and metrics["ndcg"] >= self.config.evaluation.min_ndcg
            )

            step_result.status = "completed"
            step_result.output = {
                "metrics": metrics,
                "passes_thresholds": passes_thresholds,
            }

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _run_model_registration(
        self,
        run_path: Path,
        train_result: Dict[str, Any],
        model_type: str,
        data_version: Optional[str],
    ) -> StepResult:
        """Run model registration step.

        Args:
            run_path: Run directory.
            train_result: Training result.
            model_type: Model type.
            data_version: Data version ID.

        Returns:
            StepResult.
        """
        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.MODEL_REGISTRATION.value,
            status="running",
            started_at=started_at,
        )

        try:
            # Get artifact path
            artifact_path = train_result.get("model_artifact_path", "") if train_result else ""

            # Register model
            model = self.model_registry.register_model(
                name=f"{model_type}_model",
                model_type=model_type,
                artifact_path=artifact_path,
                training_job_id=train_result.get("job_id", "") if train_result else "",
                data_version=data_version or "",
                hyperparameters=self.training_orchestrator.config.__dict__,
            )

            step_result.status = "completed"
            step_result.output = {
                "model_id": model.model_id,
                "version": model.version,
            }

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _run_deployment(
        self,
        run_path: Path,
        model_id: Optional[str],
    ) -> StepResult:
        """Run deployment step.

        Args:
            run_path: Run directory.
            model_id: Model ID to deploy.

        Returns:
            StepResult.
        """
        started_at = datetime.utcnow().isoformat()
        step_result = StepResult(
            step=PipelineStep.DEPLOYMENT.value,
            status="running",
            started_at=started_at,
        )

        try:
            if not model_id:
                raise ValueError("No model ID provided for deployment")

            # Deploy model
            strategy = (
                DeploymentStrategy.CANARY
                if self.config.deployment.enable_canary_deployment
                else DeploymentStrategy.ALL_AT_ONCE
            )

            deploy_result = self.deployment_pipeline.deploy(
                model_id=model_id,
                strategy=strategy,
                wait_for_completion=False,  # Don't wait for canary
            )

            step_result.status = "completed" if deploy_result.status in ["completed", "canary_running"] else "failed"
            step_result.output = {
                "deployment_id": deploy_result.deployment_id,
                "endpoint_name": deploy_result.endpoint_name,
                "status": deploy_result.status,
            }

            if deploy_result.error_message:
                step_result.error = deploy_result.error_message

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)

        step_result.completed_at = datetime.utcnow().isoformat()
        self._report_step(step_result)
        return step_result

    def _report_step(self, step_result: StepResult) -> None:
        """Report step completion.

        Args:
            step_result: Step result.
        """
        for callback in self._step_callbacks:
            try:
                callback(self._active_run or "", step_result)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")

    def _save_run_metadata(
        self,
        result: PipelineRunResult,
        run_path: Path,
    ) -> None:
        """Save run metadata to file.

        Args:
            result: Run result.
            run_path: Run directory.
        """
        metadata_path = run_path / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_run(self, run_id: str) -> Optional[PipelineRunResult]:
        """Get run result.

        Args:
            run_id: Run identifier.

        Returns:
            PipelineRunResult or None.
        """
        return self._runs.get(run_id)

    def list_runs(self) -> List[PipelineRunResult]:
        """List all runs.

        Returns:
            List of run results.
        """
        return list(self._runs.values())

    def stop_run(self, run_id: str) -> bool:
        """Stop a running pipeline.

        Args:
            run_id: Run identifier.

        Returns:
            True if stopped.
        """
        if run_id not in self._runs:
            return False

        run = self._runs[run_id]
        if run.status == PipelineRunStatus.RUNNING.value:
            run.status = PipelineRunStatus.STOPPED.value
            run.completed_at = datetime.utcnow().isoformat()
            logger.info(f"Stopped pipeline run {run_id}")
            return True

        return False

    def register_step_callback(
        self,
        callback: Callable[[str, StepResult], None],
    ) -> None:
        """Register step callback.

        Args:
            callback: Function called with (run_id, step_result).
        """
        self._step_callbacks.append(callback)

    def register_completion_callback(
        self,
        callback: Callable[[PipelineRunResult], None],
    ) -> None:
        """Register completion callback.

        Args:
            callback: Function called with run result.
        """
        self._completion_callbacks.append(callback)
