"""
Deployment pipeline for ML models.

Handles model deployment to SageMaker endpoints with support for
canary deployments, traffic shifting, and rollback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import time

from ..utils.logging_utils import get_logger
from .pipeline_config import DeploymentTargetConfig
from .model_registry import RegisteredModel, ModelRegistry

logger = get_logger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    LINEAR = "linear"
    ALL_AT_ONCE = "all_at_once"


class DeploymentStatus(Enum):
    """Deployment status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CANARY_RUNNING = "canary_running"
    SHIFTING_TRAFFIC = "shifting_traffic"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class EndpointConfig:
    """Configuration for a SageMaker endpoint."""

    endpoint_name: str
    model_name: str
    variant_name: str = "primary"

    # Instance configuration
    instance_type: str = "ml.m5.large"
    instance_count: int = 1

    # Serverless configuration
    use_serverless: bool = False
    serverless_memory_mb: int = 2048
    serverless_max_concurrency: int = 10

    # Traffic
    initial_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "endpoint_name": self.endpoint_name,
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "use_serverless": self.use_serverless,
            "serverless_memory_mb": self.serverless_memory_mb,
            "serverless_max_concurrency": self.serverless_max_concurrency,
            "initial_weight": self.initial_weight,
        }


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    deployment_id: str
    model_id: str
    endpoint_name: str
    status: str
    strategy: str

    # Timestamps
    started_at: str
    completed_at: str = ""

    # Endpoint details
    endpoint_arn: str = ""
    endpoint_config_name: str = ""

    # Canary details
    canary_traffic_percentage: float = 0.0
    canary_metrics: Dict[str, Any] = field(default_factory=dict)

    # Rollback info
    previous_model_id: Optional[str] = None
    rollback_reason: str = ""

    # Errors
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "endpoint_name": self.endpoint_name,
            "status": self.status,
            "strategy": self.strategy,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "endpoint_arn": self.endpoint_arn,
            "endpoint_config_name": self.endpoint_config_name,
            "canary_traffic_percentage": self.canary_traffic_percentage,
            "canary_metrics": self.canary_metrics,
            "previous_model_id": self.previous_model_id,
            "rollback_reason": self.rollback_reason,
            "error_message": self.error_message,
        }


class DeploymentPipeline:
    """Pipeline for deploying models to production.

    Handles:
    1. Creating SageMaker endpoints
    2. Canary deployments
    3. Traffic shifting
    4. Automatic rollback on failure
    5. Deployment history tracking

    Example:
        >>> pipeline = DeploymentPipeline(config, registry)
        >>> result = pipeline.deploy(
        ...     model_id="two_tower_v3",
        ...     strategy=DeploymentStrategy.CANARY,
        ... )
    """

    def __init__(
        self,
        config: DeploymentTargetConfig,
        registry: Optional[ModelRegistry] = None,
        output_path: str = "deployments",
    ):
        """Initialize the deployment pipeline.

        Args:
            config: Deployment configuration.
            registry: Model registry.
            output_path: Path for deployment logs.
        """
        self.config = config
        self.registry = registry
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Tracking
        self._deployments: Dict[str, DeploymentResult] = {}
        self._deployment_counter = 0
        self._active_endpoints: Dict[str, str] = {}  # endpoint_name -> model_id

        # Callbacks
        self._deployment_callbacks: List[Callable[[DeploymentResult], None]] = []

    def deploy(
        self,
        model_id: str,
        endpoint_name: Optional[str] = None,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        canary_traffic_percentage: Optional[float] = None,
        canary_duration_minutes: Optional[int] = None,
        wait_for_completion: bool = True,
    ) -> DeploymentResult:
        """Deploy a model to an endpoint.

        Args:
            model_id: Model identifier.
            endpoint_name: Target endpoint name.
            strategy: Deployment strategy.
            canary_traffic_percentage: Canary traffic percentage.
            canary_duration_minutes: Canary duration.
            wait_for_completion: Wait for deployment to complete.

        Returns:
            DeploymentResult.
        """
        self._deployment_counter += 1
        deployment_id = f"deploy_{self._deployment_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow().isoformat()

        # Get model info
        model = None
        if self.registry:
            model = self.registry.get_model(model_id)

        # Determine endpoint name
        if not endpoint_name:
            model_type = model.model_type if model else "model"
            endpoint_name = f"{self.config.endpoint_name_prefix}-{model_type}"

        # Get previous model
        previous_model_id = self._active_endpoints.get(endpoint_name)

        result = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            endpoint_name=endpoint_name,
            status=DeploymentStatus.PENDING.value,
            strategy=strategy.value,
            started_at=started_at,
            canary_traffic_percentage=canary_traffic_percentage or self.config.canary_traffic_percentage,
            previous_model_id=previous_model_id,
        )

        self._deployments[deployment_id] = result

        try:
            result.status = DeploymentStatus.IN_PROGRESS.value

            if strategy == DeploymentStrategy.CANARY:
                self._deploy_canary(
                    result,
                    model,
                    canary_traffic_percentage or self.config.canary_traffic_percentage,
                    canary_duration_minutes or self.config.canary_duration_minutes,
                    wait_for_completion,
                )
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                self._deploy_blue_green(result, model, wait_for_completion)
            elif strategy == DeploymentStrategy.ALL_AT_ONCE:
                self._deploy_all_at_once(result, model)
            else:
                self._deploy_linear(result, model, wait_for_completion)

            # Update tracking
            if result.status == DeploymentStatus.COMPLETED.value:
                self._active_endpoints[endpoint_name] = model_id

        except Exception as e:
            result.status = DeploymentStatus.FAILED.value
            result.error_message = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")

        result.completed_at = datetime.utcnow().isoformat()

        # Save deployment metadata
        self._save_deployment_metadata(result)

        # Trigger callbacks
        for callback in self._deployment_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Deployment callback error: {e}")

        return result

    def _deploy_canary(
        self,
        result: DeploymentResult,
        model: Optional[RegisteredModel],
        canary_percentage: float,
        canary_duration_minutes: int,
        wait: bool,
    ) -> None:
        """Deploy with canary strategy.

        Args:
            result: Deployment result to update.
            model: Model to deploy.
            canary_percentage: Canary traffic percentage.
            canary_duration_minutes: Canary duration.
            wait: Wait for canary completion.
        """
        result.status = DeploymentStatus.CANARY_RUNNING.value
        logger.info(
            f"Starting canary deployment for {result.model_id} "
            f"with {canary_percentage * 100}% traffic"
        )

        # Simulate canary deployment
        # In production, this would create a new variant and shift traffic

        if wait:
            # Simulate canary monitoring
            canary_seconds = canary_duration_minutes * 60
            check_interval = min(30, canary_seconds // 3)

            for elapsed in range(0, canary_seconds, check_interval):
                # Check canary metrics (simulated)
                canary_metrics = self._collect_canary_metrics(result.endpoint_name)
                result.canary_metrics = canary_metrics

                # Check for issues
                if self._should_rollback(canary_metrics):
                    self._rollback(result, "Canary metrics degraded")
                    return

                logger.debug(
                    f"Canary check {elapsed}/{canary_seconds}s: metrics OK"
                )

            # Canary passed, shift all traffic
            result.status = DeploymentStatus.SHIFTING_TRAFFIC.value
            logger.info(f"Canary passed, shifting 100% traffic to {result.model_id}")

        result.status = DeploymentStatus.COMPLETED.value
        logger.info(f"Canary deployment completed for {result.model_id}")

    def _deploy_blue_green(
        self,
        result: DeploymentResult,
        model: Optional[RegisteredModel],
        wait: bool,
    ) -> None:
        """Deploy with blue-green strategy.

        Args:
            result: Deployment result to update.
            model: Model to deploy.
            wait: Wait for deployment completion.
        """
        logger.info(f"Starting blue-green deployment for {result.model_id}")

        # In production, this would:
        # 1. Create new endpoint configuration
        # 2. Update endpoint to use new configuration
        # 3. Wait for deployment
        # 4. Delete old configuration

        result.status = DeploymentStatus.COMPLETED.value
        logger.info(f"Blue-green deployment completed for {result.model_id}")

    def _deploy_all_at_once(
        self,
        result: DeploymentResult,
        model: Optional[RegisteredModel],
    ) -> None:
        """Deploy all at once.

        Args:
            result: Deployment result to update.
            model: Model to deploy.
        """
        logger.info(f"Starting all-at-once deployment for {result.model_id}")

        # Direct replacement
        result.status = DeploymentStatus.COMPLETED.value
        logger.info(f"All-at-once deployment completed for {result.model_id}")

    def _deploy_linear(
        self,
        result: DeploymentResult,
        model: Optional[RegisteredModel],
        wait: bool,
    ) -> None:
        """Deploy with linear traffic shift.

        Args:
            result: Deployment result to update.
            model: Model to deploy.
            wait: Wait for deployment completion.
        """
        logger.info(f"Starting linear deployment for {result.model_id}")

        # Gradually shift traffic in steps
        steps = [0.1, 0.25, 0.5, 0.75, 1.0]

        if wait:
            for step in steps:
                logger.debug(f"Shifting {step * 100}% traffic to {result.model_id}")
                result.canary_traffic_percentage = step

                # Check metrics at each step
                metrics = self._collect_canary_metrics(result.endpoint_name)
                if self._should_rollback(metrics):
                    self._rollback(result, f"Metrics degraded at {step * 100}% traffic")
                    return

        result.status = DeploymentStatus.COMPLETED.value
        logger.info(f"Linear deployment completed for {result.model_id}")

    def _collect_canary_metrics(self, endpoint_name: str) -> Dict[str, Any]:
        """Collect metrics for canary evaluation.

        Args:
            endpoint_name: Endpoint name.

        Returns:
            Metrics dictionary.
        """
        # In production, would query CloudWatch/monitoring system
        import random

        return {
            "latency_p50_ms": 20 + random.uniform(-5, 5),
            "latency_p99_ms": 100 + random.uniform(-20, 20),
            "error_rate": random.uniform(0, 0.01),
            "invocations_per_minute": 100 + random.uniform(-20, 20),
        }

    def _should_rollback(self, metrics: Dict[str, Any]) -> bool:
        """Determine if deployment should be rolled back.

        Args:
            metrics: Current metrics.

        Returns:
            True if rollback is needed.
        """
        # Check error rate threshold
        if metrics.get("error_rate", 0) > 0.05:  # 5% error rate
            return True

        # Check latency threshold
        if metrics.get("latency_p99_ms", 0) > 500:  # 500ms p99
            return True

        return False

    def _rollback(self, result: DeploymentResult, reason: str) -> None:
        """Rollback deployment.

        Args:
            result: Deployment result.
            reason: Rollback reason.
        """
        result.status = DeploymentStatus.ROLLED_BACK.value
        result.rollback_reason = reason
        logger.warning(f"Rolling back deployment {result.deployment_id}: {reason}")

        # In production, would shift traffic back to previous model

    def rollback_deployment(
        self,
        deployment_id: str,
        reason: str = "Manual rollback",
    ) -> bool:
        """Manually rollback a deployment.

        Args:
            deployment_id: Deployment identifier.
            reason: Rollback reason.

        Returns:
            True if rolled back.
        """
        if deployment_id not in self._deployments:
            return False

        result = self._deployments[deployment_id]
        if result.status != DeploymentStatus.COMPLETED.value:
            logger.warning(f"Cannot rollback deployment in status {result.status}")
            return False

        # Restore previous model
        if result.previous_model_id:
            self._active_endpoints[result.endpoint_name] = result.previous_model_id

        result.status = DeploymentStatus.ROLLED_BACK.value
        result.rollback_reason = reason

        self._save_deployment_metadata(result)
        logger.info(f"Rolled back deployment {deployment_id}: {reason}")
        return True

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment result.

        Args:
            deployment_id: Deployment identifier.

        Returns:
            DeploymentResult or None.
        """
        return self._deployments.get(deployment_id)

    def list_deployments(
        self,
        endpoint_name: Optional[str] = None,
        status: Optional[DeploymentStatus] = None,
    ) -> List[DeploymentResult]:
        """List deployments with optional filters.

        Args:
            endpoint_name: Filter by endpoint.
            status: Filter by status.

        Returns:
            List of deployments.
        """
        deployments = list(self._deployments.values())

        if endpoint_name:
            deployments = [d for d in deployments if d.endpoint_name == endpoint_name]

        if status:
            deployments = [d for d in deployments if d.status == status.value]

        return deployments

    def get_active_model(self, endpoint_name: str) -> Optional[str]:
        """Get the currently active model for an endpoint.

        Args:
            endpoint_name: Endpoint name.

        Returns:
            Model ID or None.
        """
        return self._active_endpoints.get(endpoint_name)

    def _save_deployment_metadata(self, result: DeploymentResult) -> None:
        """Save deployment metadata to file.

        Args:
            result: Deployment result.
        """
        metadata_path = self.output_path / f"{result.deployment_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def register_deployment_callback(
        self,
        callback: Callable[[DeploymentResult], None],
    ) -> None:
        """Register deployment callback.

        Args:
            callback: Function called with deployment result.
        """
        self._deployment_callbacks.append(callback)


class SageMakerDeployment:
    """SageMaker deployment wrapper.

    Handles actual SageMaker endpoint creation and management.
    """

    def __init__(
        self,
        config: DeploymentTargetConfig,
        role_arn: str,
        s3_bucket: str,
    ):
        """Initialize SageMaker deployment.

        Args:
            config: Deployment configuration.
            role_arn: IAM role ARN.
            s3_bucket: S3 bucket.
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
            logger.warning("boto3 not available. SageMaker deployments will not work.")

    def create_model(
        self,
        model_name: str,
        model_data_url: str,
        image_uri: str,
    ) -> str:
        """Create SageMaker model.

        Args:
            model_name: Model name.
            model_data_url: S3 URL of model artifacts.
            image_uri: Container image URI.

        Returns:
            Model ARN.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        response = self._sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
            },
            ExecutionRoleArn=self.role_arn,
        )

        return response["ModelArn"]

    def create_endpoint_config(
        self,
        config_name: str,
        model_name: str,
        variant_name: str = "primary",
    ) -> str:
        """Create endpoint configuration.

        Args:
            config_name: Configuration name.
            model_name: Model name.
            variant_name: Variant name.

        Returns:
            Config ARN.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        if self.config.use_serverless:
            production_variants = [{
                "VariantName": variant_name,
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": self.config.serverless_memory_mb,
                    "MaxConcurrency": self.config.serverless_max_concurrency,
                },
            }]
        else:
            production_variants = [{
                "VariantName": variant_name,
                "ModelName": model_name,
                "InstanceType": self.config.instance_type,
                "InitialInstanceCount": self.config.initial_instance_count,
                "InitialVariantWeight": 1.0,
            }]

        response = self._sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=production_variants,
        )

        return response["EndpointConfigArn"]

    def create_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
    ) -> str:
        """Create endpoint.

        Args:
            endpoint_name: Endpoint name.
            config_name: Configuration name.

        Returns:
            Endpoint ARN.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        response = self._sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

        return response["EndpointArn"]

    def update_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
    ) -> None:
        """Update endpoint configuration.

        Args:
            endpoint_name: Endpoint name.
            config_name: New configuration name.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        self._sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get endpoint status.

        Args:
            endpoint_name: Endpoint name.

        Returns:
            Status information.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        response = self._sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        return {
            "endpoint_name": endpoint_name,
            "status": response["EndpointStatus"],
            "creation_time": response.get("CreationTime"),
            "last_modified_time": response.get("LastModifiedTime"),
            "failure_reason": response.get("FailureReason"),
        }

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete endpoint.

        Args:
            endpoint_name: Endpoint name.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployments")

        self._sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deleted endpoint {endpoint_name}")
