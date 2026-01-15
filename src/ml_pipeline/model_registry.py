"""
Model registry for ML pipeline.

Handles model versioning, metadata management, and model lifecycle
with support for comparison, promotion, and rollback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import shutil

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model status."""

    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    RETIRED = "retired"


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    # Retrieval metrics (Two-Tower)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    mrr: float = 0.0

    # Ranking metrics (Ranker)
    auc: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0

    # Business metrics
    diversity_score: float = 0.0
    coverage_score: float = 0.0

    # Latency
    inference_latency_p50_ms: float = 0.0
    inference_latency_p99_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg": self.ndcg,
            "mrr": self.mrr,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "diversity_score": self.diversity_score,
            "coverage_score": self.coverage_score,
            "inference_latency_p50_ms": self.inference_latency_p50_ms,
            "inference_latency_p99_ms": self.inference_latency_p99_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        """Create from dictionary."""
        return cls(
            precision_at_k=data.get("precision_at_k", {}),
            recall_at_k=data.get("recall_at_k", {}),
            ndcg=data.get("ndcg", 0.0),
            mrr=data.get("mrr", 0.0),
            auc=data.get("auc", 0.0),
            log_loss=data.get("log_loss", 0.0),
            accuracy=data.get("accuracy", 0.0),
            diversity_score=data.get("diversity_score", 0.0),
            coverage_score=data.get("coverage_score", 0.0),
            inference_latency_p50_ms=data.get("inference_latency_p50_ms", 0.0),
            inference_latency_p99_ms=data.get("inference_latency_p99_ms", 0.0),
        )


@dataclass
class RegisteredModel:
    """A registered model in the registry."""

    model_id: str
    model_name: str
    model_type: str  # two_tower or ranker
    version: int

    # Timestamps
    created_at: str
    updated_at: str = ""

    # Paths
    artifact_path: str = ""
    config_path: str = ""

    # Status
    stage: str = ModelStage.DEVELOPMENT.value
    status: str = ModelStatus.PENDING_VALIDATION.value

    # Training info
    training_job_id: str = ""
    data_version: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_by: str = ""

    # Lineage
    parent_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifact_path": self.artifact_path,
            "config_path": self.config_path,
            "stage": self.stage,
            "status": self.status,
            "training_job_id": self.training_job_id,
            "data_version": self.data_version,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "description": self.description,
            "tags": self.tags,
            "created_by": self.created_by,
            "parent_model_id": self.parent_model_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Registry for managing ML models.

    Handles:
    1. Model registration and versioning
    2. Metadata management
    3. Model stage transitions
    4. Model comparison and selection
    5. Model lineage tracking

    Example:
        >>> registry = ModelRegistry(base_path="models/registry")
        >>> model = registry.register_model(
        ...     name="two_tower_v1",
        ...     model_type="two_tower",
        ...     artifact_path="models/train_123/model",
        ... )
        >>> registry.promote_to_staging(model.model_id)
    """

    def __init__(self, base_path: str = "models/registry"):
        """Initialize the model registry.

        Args:
            base_path: Base path for registry storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Model tracking
        self._models: Dict[str, RegisteredModel] = {}
        self._models_by_name: Dict[str, List[str]] = {}
        self._version_counters: Dict[str, int] = {}

        # Stage tracking
        self._production_models: Dict[str, str] = {}  # model_type -> model_id

        # Load existing models
        self._load_registry()

    def register_model(
        self,
        name: str,
        model_type: str,
        artifact_path: str,
        training_job_id: str = "",
        data_version: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[ModelMetrics] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "",
        parent_model_id: Optional[str] = None,
    ) -> RegisteredModel:
        """Register a new model.

        Args:
            name: Model name.
            model_type: Model type (two_tower or ranker).
            artifact_path: Path to model artifacts.
            training_job_id: Training job identifier.
            data_version: Data version used for training.
            hyperparameters: Training hyperparameters.
            metrics: Evaluation metrics.
            description: Model description.
            tags: Model tags.
            created_by: Creator identifier.
            parent_model_id: Parent model for lineage.

        Returns:
            Registered model.
        """
        # Get next version
        version = self._get_next_version(name)
        model_id = f"{name}_v{version}"
        created_at = datetime.utcnow().isoformat()

        # Create model directory
        model_path = self.base_path / model_id
        model_path.mkdir(parents=True, exist_ok=True)

        # Copy artifacts
        dest_artifact_path = model_path / "artifacts"
        dest_artifact_path.mkdir(exist_ok=True)
        if artifact_path and Path(artifact_path).exists():
            if Path(artifact_path).is_dir():
                shutil.copytree(artifact_path, dest_artifact_path, dirs_exist_ok=True)
            else:
                shutil.copy2(artifact_path, dest_artifact_path)

        model = RegisteredModel(
            model_id=model_id,
            model_name=name,
            model_type=model_type,
            version=version,
            created_at=created_at,
            updated_at=created_at,
            artifact_path=str(dest_artifact_path),
            config_path=str(model_path / "config.json"),
            training_job_id=training_job_id,
            data_version=data_version,
            hyperparameters=hyperparameters or {},
            metrics=metrics.to_dict() if metrics else {},
            description=description,
            tags=tags or [],
            created_by=created_by,
            parent_model_id=parent_model_id,
        )

        # Save to registry
        self._models[model_id] = model

        if name not in self._models_by_name:
            self._models_by_name[name] = []
        self._models_by_name[name].append(model_id)

        # Save metadata
        self._save_model_metadata(model)
        self._save_registry_index()

        logger.info(f"Registered model {model_id} (type={model_type})")
        return model

    def _get_next_version(self, name: str) -> int:
        """Get next version number for a model name.

        Args:
            name: Model name.

        Returns:
            Next version number.
        """
        if name not in self._version_counters:
            self._version_counters[name] = 0
        self._version_counters[name] += 1
        return self._version_counters[name]

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Get a model by ID.

        Args:
            model_id: Model identifier.

        Returns:
            RegisteredModel or None.
        """
        return self._models.get(model_id)

    def get_model_versions(self, name: str) -> List[RegisteredModel]:
        """Get all versions of a model.

        Args:
            name: Model name.

        Returns:
            List of model versions.
        """
        model_ids = self._models_by_name.get(name, [])
        return [self._models[mid] for mid in model_ids if mid in self._models]

    def get_latest_version(self, name: str) -> Optional[RegisteredModel]:
        """Get latest version of a model.

        Args:
            name: Model name.

        Returns:
            Latest RegisteredModel or None.
        """
        versions = self.get_model_versions(name)
        if not versions:
            return None
        return max(versions, key=lambda m: m.version)

    def get_production_model(self, model_type: str) -> Optional[RegisteredModel]:
        """Get the current production model.

        Args:
            model_type: Model type.

        Returns:
            Production model or None.
        """
        model_id = self._production_models.get(model_type)
        if model_id:
            return self._models.get(model_id)
        return None

    def update_metrics(
        self,
        model_id: str,
        metrics: ModelMetrics,
    ) -> bool:
        """Update model metrics.

        Args:
            model_id: Model identifier.
            metrics: New metrics.

        Returns:
            True if updated.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        model.metrics = metrics.to_dict()
        model.updated_at = datetime.utcnow().isoformat()

        self._save_model_metadata(model)
        logger.info(f"Updated metrics for model {model_id}")
        return True

    def update_status(
        self,
        model_id: str,
        status: ModelStatus,
    ) -> bool:
        """Update model status.

        Args:
            model_id: Model identifier.
            status: New status.

        Returns:
            True if updated.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        model.status = status.value
        model.updated_at = datetime.utcnow().isoformat()

        self._save_model_metadata(model)
        logger.info(f"Updated status for model {model_id}: {status.value}")
        return True

    def promote_to_staging(self, model_id: str) -> bool:
        """Promote model to staging.

        Args:
            model_id: Model identifier.

        Returns:
            True if promoted.
        """
        return self._transition_stage(model_id, ModelStage.STAGING)

    def promote_to_production(self, model_id: str) -> bool:
        """Promote model to production.

        Args:
            model_id: Model identifier.

        Returns:
            True if promoted.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]

        # Demote current production model
        current_prod = self._production_models.get(model.model_type)
        if current_prod and current_prod != model_id:
            self._transition_stage(current_prod, ModelStage.ARCHIVED)

        # Promote new model
        if self._transition_stage(model_id, ModelStage.PRODUCTION):
            self._production_models[model.model_type] = model_id
            self._save_registry_index()
            return True

        return False

    def archive_model(self, model_id: str) -> bool:
        """Archive a model.

        Args:
            model_id: Model identifier.

        Returns:
            True if archived.
        """
        return self._transition_stage(model_id, ModelStage.ARCHIVED)

    def _transition_stage(self, model_id: str, stage: ModelStage) -> bool:
        """Transition model to a new stage.

        Args:
            model_id: Model identifier.
            stage: Target stage.

        Returns:
            True if transitioned.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        model.stage = stage.value
        model.updated_at = datetime.utcnow().isoformat()

        self._save_model_metadata(model)
        logger.info(f"Transitioned model {model_id} to stage {stage.value}")
        return True

    def compare_models(
        self,
        model_id_1: str,
        model_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two models.

        Args:
            model_id_1: First model ID.
            model_id_2: Second model ID.

        Returns:
            Comparison results.
        """
        m1 = self._models.get(model_id_1)
        m2 = self._models.get(model_id_2)

        if not m1 or not m2:
            raise ValueError("One or both models not found")

        comparison = {
            "model_1": model_id_1,
            "model_2": model_id_2,
            "metrics_comparison": {},
            "hyperparameter_differences": {},
        }

        # Compare metrics
        for key in set(m1.metrics.keys()) | set(m2.metrics.keys()):
            v1 = m1.metrics.get(key, 0)
            v2 = m2.metrics.get(key, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                comparison["metrics_comparison"][key] = {
                    "model_1": v1,
                    "model_2": v2,
                    "difference": v2 - v1,
                    "improvement_pct": ((v2 - v1) / v1 * 100) if v1 != 0 else 0,
                }

        # Compare hyperparameters
        for key in set(m1.hyperparameters.keys()) | set(m2.hyperparameters.keys()):
            v1 = m1.hyperparameters.get(key)
            v2 = m2.hyperparameters.get(key)
            if v1 != v2:
                comparison["hyperparameter_differences"][key] = {
                    "model_1": v1,
                    "model_2": v2,
                }

        return comparison

    def find_best_model(
        self,
        model_type: str,
        metric: str = "ndcg",
        stage: Optional[ModelStage] = None,
    ) -> Optional[RegisteredModel]:
        """Find the best model by a metric.

        Args:
            model_type: Model type.
            metric: Metric to compare.
            stage: Optional stage filter.

        Returns:
            Best model or None.
        """
        candidates = [
            m for m in self._models.values()
            if m.model_type == model_type
            and (stage is None or m.stage == stage.value)
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda m: m.metrics.get(metric, 0),
        )

    def list_models(
        self,
        model_type: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[RegisteredModel]:
        """List models with optional filters.

        Args:
            model_type: Filter by model type.
            stage: Filter by stage.
            status: Filter by status.

        Returns:
            List of models.
        """
        models = list(self._models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if stage:
            models = [m for m in models if m.stage == stage.value]

        if status:
            models = [m for m in models if m.status == status.value]

        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model.

        Args:
            model_id: Model identifier.

        Returns:
            True if deleted.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]

        # Don't delete production models
        if model.stage == ModelStage.PRODUCTION.value:
            logger.warning(f"Cannot delete production model {model_id}")
            return False

        # Remove from production tracking if needed
        if self._production_models.get(model.model_type) == model_id:
            del self._production_models[model.model_type]

        # Remove files
        model_path = self.base_path / model_id
        if model_path.exists():
            shutil.rmtree(model_path)

        # Remove from tracking
        del self._models[model_id]
        if model.model_name in self._models_by_name:
            self._models_by_name[model.model_name] = [
                mid for mid in self._models_by_name[model.model_name]
                if mid != model_id
            ]

        self._save_registry_index()
        logger.info(f"Deleted model {model_id}")
        return True

    def _save_model_metadata(self, model: RegisteredModel) -> None:
        """Save model metadata to file.

        Args:
            model: Model to save.
        """
        model_path = self.base_path / model.model_id
        model_path.mkdir(parents=True, exist_ok=True)

        metadata_path = model_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(model.to_dict(), f, indent=2)

    def _save_registry_index(self) -> None:
        """Save registry index to file."""
        index_path = self.base_path / "registry_index.json"
        index = {
            "models": list(self._models.keys()),
            "models_by_name": self._models_by_name,
            "version_counters": self._version_counters,
            "production_models": self._production_models,
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry from files."""
        index_path = self.base_path / "registry_index.json"
        if not index_path.exists():
            return

        with open(index_path) as f:
            index = json.load(f)

        self._models_by_name = index.get("models_by_name", {})
        self._version_counters = index.get("version_counters", {})
        self._production_models = index.get("production_models", {})

        # Load individual models
        for model_id in index.get("models", []):
            metadata_path = self.base_path / model_id / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    model_data = json.load(f)
                self._models[model_id] = RegisteredModel.from_dict(model_data)

        logger.info(f"Loaded {len(self._models)} models from registry")
