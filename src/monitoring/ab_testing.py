"""
A/B testing framework for recommendation experiments.

Supports traffic splitting, experiment management, and statistical
analysis of experiment results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import numpy as np
import hashlib

from ..utils.logging_utils import get_logger
from .monitoring_config import ABTestConfig
from .online_metrics import UserInteraction, OnlineMetricsCollector

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""

    name: str
    model_version: str
    traffic_percentage: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_version": self.model_version,
            "traffic_percentage": self.traffic_percentage,
            "description": self.description,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    experiment_id: str
    name: str
    description: str

    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)

    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    # Configuration
    min_sample_size: int = 1000
    max_duration_days: int = 14

    # Metrics to track
    primary_metric: str = "click_through_rate"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["completion_rate", "watch_time"]
    )

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "min_sample_size": self.min_sample_size,
            "max_duration_days": self.max_duration_days,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        variants = [ExperimentVariant(**v) for v in data.pop("variants", [])]
        status = ExperimentStatus(data.pop("status", "draft"))
        return cls(variants=variants, status=status, **data)


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""

    variant_name: str
    sample_size: int = 0

    # Primary metrics
    impressions: int = 0
    clicks: int = 0
    completions: int = 0
    click_through_rate: float = 0.0
    completion_rate: float = 0.0
    avg_watch_time: float = 0.0

    # Statistical info
    ctr_std_error: float = 0.0
    ctr_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_name": self.variant_name,
            "sample_size": self.sample_size,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "completions": self.completions,
            "click_through_rate": self.click_through_rate,
            "completion_rate": self.completion_rate,
            "avg_watch_time": self.avg_watch_time,
            "ctr_std_error": self.ctr_std_error,
            "ctr_confidence_interval": self.ctr_confidence_interval,
        }


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""

    experiment_id: str
    experiment_name: str
    generated_at: str

    # Duration
    start_time: str = ""
    end_time: str = ""
    duration_hours: float = 0.0

    # Per-variant metrics
    variant_metrics: Dict[str, VariantMetrics] = field(default_factory=dict)

    # Statistical comparison
    winner: Optional[str] = None
    is_significant: bool = False
    p_value: float = 1.0
    relative_improvement: float = 0.0
    confidence_level: float = 0.95

    # Recommendation
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "generated_at": self.generated_at,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": self.duration_hours,
            "variant_metrics": {k: v.to_dict() for k, v in self.variant_metrics.items()},
            "winner": self.winner,
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "relative_improvement": self.relative_improvement,
            "confidence_level": self.confidence_level,
            "recommendation": self.recommendation,
        }


class ABTestManager:
    """Manager for A/B testing experiments.

    Handles:
    - Experiment creation and management
    - User-to-variant assignment (consistent hashing)
    - Metrics collection and analysis
    - Statistical significance testing

    Example:
        >>> manager = ABTestManager(config)
        >>> experiment = manager.create_experiment(
        ...     name="New Ranker Model",
        ...     variants=[
        ...         ("control", "v1", 0.5),
        ...         ("treatment", "v2", 0.5),
        ...     ]
        ... )
        >>> manager.start_experiment(experiment.experiment_id)
        >>> variant = manager.get_user_variant(user_id, experiment.experiment_id)
        >>> result = manager.analyze_experiment(experiment.experiment_id, interactions)
    """

    def __init__(self, config: ABTestConfig):
        """Initialize the A/B test manager.

        Args:
            config: A/B testing configuration.
        """
        self.config = config
        self._experiments: Dict[str, Experiment] = {}
        self._experiment_counter = 0

        # Storage
        self._storage_path = Path(config.experiment_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        variants: List[Tuple[str, str, float]],
        description: str = "",
        primary_metric: str = "click_through_rate",
        min_sample_size: int = 1000,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            variants: List of (name, model_version, traffic_percentage) tuples.
            description: Experiment description.
            primary_metric: Primary metric to optimize.
            min_sample_size: Minimum samples per variant.

        Returns:
            Created experiment.
        """
        self._experiment_counter += 1
        experiment_id = f"exp_{self._experiment_counter}"

        # Create variant objects
        variant_objects = [
            ExperimentVariant(
                name=name,
                model_version=model_version,
                traffic_percentage=traffic_pct,
            )
            for name, model_version, traffic_pct in variants
        ]

        # Validate traffic percentages sum to 1
        total_traffic = sum(v.traffic_percentage for v in variant_objects)
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 1.0, got {total_traffic}")

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            primary_metric=primary_metric,
            min_sample_size=min_sample_size,
            max_duration_days=self.config.max_test_duration_days,
        )

        self._experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {experiment_id} - {name}")

        return experiment

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment.

        Args:
            experiment_id: Experiment identifier.
        """
        experiment = self._get_experiment(experiment_id)
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in status: {experiment.status}")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow().isoformat()

        logger.info(f"Started experiment: {experiment_id}")

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause a running experiment.

        Args:
            experiment_id: Experiment identifier.
        """
        experiment = self._get_experiment(experiment_id)
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in status: {experiment.status}")

        experiment.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment: {experiment_id}")

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as completed.

        Args:
            experiment_id: Experiment identifier.
        """
        experiment = self._get_experiment(experiment_id)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.utcnow().isoformat()

        logger.info(f"Completed experiment: {experiment_id}")

    def get_user_variant(
        self,
        user_id: int,
        experiment_id: str,
    ) -> Optional[ExperimentVariant]:
        """Get the variant assigned to a user.

        Uses consistent hashing to ensure users always get the same variant.

        Args:
            user_id: User identifier.
            experiment_id: Experiment identifier.

        Returns:
            Assigned variant, or None if experiment not running.
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        # Consistent hashing for user assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        # Find variant based on bucket
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    def _get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Experiment object.

        Raises:
            KeyError: If experiment not found.
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]

    def analyze_experiment(
        self,
        experiment_id: str,
        interactions: List[UserInteraction],
    ) -> ExperimentResult:
        """Analyze experiment results.

        Args:
            experiment_id: Experiment identifier.
            interactions: User interactions to analyze.

        Returns:
            Experiment result with statistical analysis.
        """
        experiment = self._get_experiment(experiment_id)

        # Filter interactions for this experiment
        exp_interactions = [
            i for i in interactions
            if i.experiment_id == experiment_id
        ]

        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            generated_at=datetime.utcnow().isoformat(),
            confidence_level=self.config.confidence_level,
        )

        if not exp_interactions:
            result.recommendation = "Insufficient data for analysis"
            return result

        # Set time range
        timestamps = [datetime.fromisoformat(i.timestamp) for i in exp_interactions]
        result.start_time = min(timestamps).isoformat()
        result.end_time = max(timestamps).isoformat()
        result.duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600

        # Compute metrics per variant
        for variant in experiment.variants:
            variant_interactions = [
                i for i in exp_interactions
                if i.model_version == variant.model_version
            ]

            metrics = self._compute_variant_metrics(variant.name, variant_interactions)
            result.variant_metrics[variant.name] = metrics

        # Statistical comparison (if 2 variants)
        if len(result.variant_metrics) == 2:
            self._compute_statistical_significance(result, experiment.primary_metric)

        # Generate recommendation
        result.recommendation = self._generate_recommendation(result, experiment)

        return result

    def _compute_variant_metrics(
        self,
        variant_name: str,
        interactions: List[UserInteraction],
    ) -> VariantMetrics:
        """Compute metrics for a variant.

        Args:
            variant_name: Name of the variant.
            interactions: Interactions for this variant.

        Returns:
            Variant metrics.
        """
        metrics = VariantMetrics(variant_name=variant_name)

        if not interactions:
            return metrics

        # Count interactions
        impressions = [i for i in interactions if i.interaction_type == "impression"]
        clicks = [i for i in interactions if i.interaction_type == "click"]
        completions = [i for i in interactions if i.interaction_type == "complete"]
        watches = [i for i in interactions if i.interaction_type in ["watch", "complete"]]

        metrics.sample_size = len(set(i.user_id for i in interactions))
        metrics.impressions = len(impressions)
        metrics.clicks = len(clicks)
        metrics.completions = len(completions)

        # CTR
        if metrics.impressions > 0:
            metrics.click_through_rate = metrics.clicks / metrics.impressions

            # Standard error for binomial proportion
            p = metrics.click_through_rate
            n = metrics.impressions
            metrics.ctr_std_error = np.sqrt(p * (1 - p) / n) if n > 0 else 0

            # 95% confidence interval
            z = 1.96
            margin = z * metrics.ctr_std_error
            metrics.ctr_confidence_interval = (
                max(0, p - margin),
                min(1, p + margin),
            )

        # Completion rate
        if clicks:
            metrics.completion_rate = len(completions) / len(clicks)

        # Watch time
        if watches:
            metrics.avg_watch_time = np.mean([i.watch_time_seconds for i in watches])

        return metrics

    def _compute_statistical_significance(
        self,
        result: ExperimentResult,
        primary_metric: str,
    ) -> None:
        """Compute statistical significance between two variants.

        Args:
            result: Experiment result to update.
            primary_metric: Primary metric for comparison.
        """
        variants = list(result.variant_metrics.values())
        if len(variants) != 2:
            return

        control = variants[0]
        treatment = variants[1]

        if primary_metric == "click_through_rate":
            # Two-proportion z-test
            p1 = control.click_through_rate
            p2 = treatment.click_through_rate
            n1 = control.impressions
            n2 = treatment.impressions

            if n1 == 0 or n2 == 0:
                return

            # Pooled proportion
            p_pool = (control.clicks + treatment.clicks) / (n1 + n2)

            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

            if se > 0:
                # Z-score
                z = (p2 - p1) / se

                # Two-tailed p-value
                from scipy import stats
                result.p_value = 2 * (1 - stats.norm.cdf(abs(z)))

                # Relative improvement
                if p1 > 0:
                    result.relative_improvement = (p2 - p1) / p1

                # Significance
                alpha = 1 - self.config.confidence_level
                result.is_significant = result.p_value < alpha

                # Winner
                if result.is_significant:
                    result.winner = treatment.variant_name if p2 > p1 else control.variant_name

    def _generate_recommendation(
        self,
        result: ExperimentResult,
        experiment: Experiment,
    ) -> str:
        """Generate recommendation based on results.

        Args:
            result: Experiment result.
            experiment: Experiment configuration.

        Returns:
            Recommendation string.
        """
        # Check sample size
        min_samples = experiment.min_sample_size
        for variant_name, metrics in result.variant_metrics.items():
            if metrics.sample_size < min_samples:
                return (
                    f"Continue experiment: {variant_name} has only "
                    f"{metrics.sample_size}/{min_samples} required samples"
                )

        # Check significance
        if not result.is_significant:
            return (
                f"No significant difference detected (p={result.p_value:.3f}). "
                f"Consider running longer or accepting null hypothesis."
            )

        # Winner recommendation
        improvement_pct = result.relative_improvement * 100
        return (
            f"Recommend deploying '{result.winner}' "
            f"(+{improvement_pct:.1f}% {experiment.primary_metric}, "
            f"p={result.p_value:.4f})"
        )

    def save_experiment(self, experiment_id: str) -> str:
        """Save experiment to file.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Path to saved file.
        """
        experiment = self._get_experiment(experiment_id)
        output_path = self._storage_path / f"{experiment_id}.json"

        with open(output_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

        return str(output_path)

    def load_experiment(self, input_path: str) -> Experiment:
        """Load experiment from file.

        Args:
            input_path: Path to experiment file.

        Returns:
            Loaded experiment.
        """
        with open(input_path) as f:
            data = json.load(f)

        experiment = Experiment.from_dict(data)
        self._experiments[experiment.experiment_id] = experiment

        return experiment

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """List experiments.

        Args:
            status: Filter by status.

        Returns:
            List of experiments.
        """
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Experiment or None.
        """
        return self._experiments.get(experiment_id)
