"""
ML Pipeline module for video recommendation system.

This module provides components for orchestrating the full ML workflow:
- Pipeline configuration and step definitions
- Preprocessing jobs (feature engineering, vocabularies)
- Data versioning and splitting
- Training job orchestration
- Model registry and versioning
- Deployment pipeline
"""

from .pipeline_config import (
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
from .preprocessing_job import (
    PreprocessingJob,
    PreprocessingResult,
    SageMakerPreprocessingJob,
)
from .data_versioning import (
    DataVersionManager,
    DataVersion,
    SplitResult,
)
from .training_orchestrator import (
    TrainingOrchestrator,
    TrainingJobResult,
    TrainingMetrics,
    TrainingJobStatus,
    SageMakerTrainingJob,
)
from .model_registry import (
    ModelRegistry,
    RegisteredModel,
    ModelMetrics,
    ModelStage,
    ModelStatus,
)
from .deployment_pipeline import (
    DeploymentPipeline,
    DeploymentResult,
    DeploymentStrategy,
    DeploymentStatus,
    EndpointConfig,
    SageMakerDeployment,
)
from .ml_pipeline import (
    MLPipeline,
    PipelineRunResult,
    PipelineRunStatus,
    StepResult,
)

__all__ = [
    # Config
    "MLPipelineConfig",
    "PipelineStep",
    "PipelineExecutionMode",
    "ModelType",
    "PreprocessingConfig",
    "DataSplittingConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DeploymentTargetConfig",
    # Preprocessing
    "PreprocessingJob",
    "PreprocessingResult",
    "SageMakerPreprocessingJob",
    # Data Versioning
    "DataVersionManager",
    "DataVersion",
    "SplitResult",
    # Training
    "TrainingOrchestrator",
    "TrainingJobResult",
    "TrainingMetrics",
    "TrainingJobStatus",
    "SageMakerTrainingJob",
    # Model Registry
    "ModelRegistry",
    "RegisteredModel",
    "ModelMetrics",
    "ModelStage",
    "ModelStatus",
    # Deployment
    "DeploymentPipeline",
    "DeploymentResult",
    "DeploymentStrategy",
    "DeploymentStatus",
    "EndpointConfig",
    "SageMakerDeployment",
    # Main Pipeline
    "MLPipeline",
    "PipelineRunResult",
    "PipelineRunStatus",
    "StepResult",
]
