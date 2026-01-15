"""
SageMaker deployment utilities for video recommendation models.

Provides utilities for deploying models to AWS SageMaker including:
- Model packaging and uploading to S3
- Endpoint creation (serverless and provisioned)
- Auto-scaling configuration
- Inference handlers
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import tarfile
import tempfile
import os

from ..utils.logging_utils import get_logger
from .serving_config import SageMakerConfig

logger = get_logger(__name__)


@dataclass
class EndpointInfo:
    """Information about a deployed endpoint."""

    endpoint_name: str
    endpoint_arn: str
    status: str
    creation_time: str
    model_name: str
    instance_type: Optional[str] = None
    is_serverless: bool = False


class SageMakerDeployer:
    """Utility class for deploying models to SageMaker.

    Supports both serverless and provisioned endpoints.

    Example:
        >>> deployer = SageMakerDeployer(config)
        >>> model_uri = deployer.package_and_upload_model(model_path)
        >>> endpoint_info = deployer.deploy_serverless_endpoint()
    """

    def __init__(self, config: SageMakerConfig):
        """Initialize the deployer.

        Args:
            config: SageMaker configuration.
        """
        self.config = config
        self._boto3_available = False
        self._sagemaker_client = None
        self._s3_client = None

        try:
            import boto3
            self._boto3_available = True
        except ImportError:
            logger.warning(
                "boto3 not available. SageMaker deployment will not work. "
                "Install with: pip install boto3"
            )

    def _get_clients(self):
        """Initialize AWS clients lazily."""
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker deployment")

        if self._sagemaker_client is None:
            import boto3
            self._sagemaker_client = boto3.client("sagemaker")
            self._s3_client = boto3.client("s3")

        return self._sagemaker_client, self._s3_client

    def package_model(
        self,
        model_path: str,
        output_path: str,
        include_artifacts: bool = True,
        artifacts_path: Optional[str] = None,
    ) -> str:
        """Package model files into a tar.gz archive.

        Args:
            model_path: Path to model directory.
            output_path: Path for output archive.
            include_artifacts: Whether to include preprocessing artifacts.
            artifacts_path: Path to artifacts directory.

        Returns:
            Path to created archive.
        """
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output_path, "w:gz") as tar:
            # Add model files
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_path.parent)
                    tar.add(file_path, arcname=arcname)

            # Add artifacts if requested
            if include_artifacts and artifacts_path:
                artifacts_path = Path(artifacts_path)
                for file_path in artifacts_path.rglob("*"):
                    if file_path.is_file():
                        arcname = Path("artifacts") / file_path.relative_to(artifacts_path)
                        tar.add(file_path, arcname=arcname)

            # Add inference script
            inference_script = self._create_inference_script()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(inference_script)
                f.flush()
                tar.add(f.name, arcname="code/inference.py")
                os.unlink(f.name)

        logger.info(f"Packaged model to {output_path}")
        return str(output_path)

    def _create_inference_script(self) -> str:
        """Create the inference script for SageMaker.

        Returns:
            Python code as string.
        """
        return '''"""
SageMaker inference script for video recommendations.
"""

import json
import os
import numpy as np
import tensorflow as tf

# Global model references
model = None
user_transformer = None
video_transformer = None


def model_fn(model_dir):
    """Load model from the model directory."""
    global model, user_transformer, video_transformer

    # Load Two-Tower model
    from src.models.two_tower import TwoTowerModel
    from src.models.model_config import TwoTowerModelConfig
    from src.feature_engineering.user_features import UserFeatureTransformer

    # Load config
    config_path = os.path.join(model_dir, "two_tower", "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    vocab_path = os.path.join(model_dir, "two_tower", "vocab_sizes.json")
    with open(vocab_path) as f:
        vocab_sizes = json.load(f)

    config = TwoTowerModelConfig(**config_dict)
    model = TwoTowerModel(config, vocab_sizes)

    # Build model
    dummy_batch = _create_dummy_batch()
    model(dummy_batch, training=False)

    # Load weights
    model.load_towers(os.path.join(model_dir, "two_tower"))

    # Load transformers
    artifacts_dir = os.path.join(model_dir, "artifacts")
    user_transformer = UserFeatureTransformer()
    user_transformer.load(artifacts_dir)

    return model


def _create_dummy_batch():
    """Create dummy batch for model building."""
    batch_size = 1
    return {
        "user_id_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "country_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "user_language_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "age_normalized": tf.zeros((batch_size,), dtype=tf.float32),
        "age_bucket_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "previously_watched_category_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "video_id_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "category_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "title_embedding": tf.zeros((batch_size, 384), dtype=tf.float32),
        "video_duration_normalized": tf.zeros((batch_size,), dtype=tf.float32),
        "duration_bucket_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "popularity_onehot": tf.zeros((batch_size, 4), dtype=tf.float32),
        "video_language_idx": tf.zeros((batch_size,), dtype=tf.int32),
        "tags_embedding": tf.zeros((batch_size, 100), dtype=tf.float32),
    }


def input_fn(request_body, content_type):
    """Deserialize input data."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Generate predictions."""
    global user_transformer

    # Transform user features
    transformed = user_transformer.transform_single(input_data)

    # Prepare inputs
    user_inputs = {
        "user_id_idx": tf.constant([transformed.get("user_id_idx", 0)], dtype=tf.int32),
        "country_idx": tf.constant([transformed.get("country_idx", 0)], dtype=tf.int32),
        "user_language_idx": tf.constant([transformed.get("user_language_idx", 0)], dtype=tf.int32),
        "age_normalized": tf.constant([transformed.get("age_normalized", 0.0)], dtype=tf.float32),
        "age_bucket_idx": tf.constant([transformed.get("age_bucket_idx", 0)], dtype=tf.int32),
        "previously_watched_category_idx": tf.constant([transformed.get("prev_category_idx", 0)], dtype=tf.int32),
    }

    # Generate embedding
    embedding = model.get_user_embeddings(user_inputs)

    return embedding.numpy()[0].tolist()


def output_fn(prediction, accept):
    """Serialize output."""
    if accept == "application/json":
        return json.dumps({"embedding": prediction})
    raise ValueError(f"Unsupported accept type: {accept}")
'''

    def upload_to_s3(
        self,
        local_path: str,
        s3_bucket: str,
        s3_key: str,
    ) -> str:
        """Upload file to S3.

        Args:
            local_path: Local file path.
            s3_bucket: S3 bucket name.
            s3_key: S3 object key.

        Returns:
            S3 URI of uploaded file.
        """
        _, s3_client = self._get_clients()

        s3_client.upload_file(local_path, s3_bucket, s3_key)
        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        logger.info(f"Uploaded {local_path} to {s3_uri}")
        return s3_uri

    def create_model(
        self,
        model_data_url: str,
        model_name: Optional[str] = None,
    ) -> str:
        """Create a SageMaker model.

        Args:
            model_data_url: S3 URI of model archive.
            model_name: Name for the model.

        Returns:
            Model ARN.
        """
        sagemaker_client, _ = self._get_clients()

        model_name = model_name or self.config.model_name

        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": self.config.image_uri,
                "ModelDataUrl": model_data_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url,
                },
            },
            ExecutionRoleArn=self.config.role_arn,
            Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
        )

        logger.info(f"Created model: {model_name}")
        return response["ModelArn"]

    def deploy_serverless_endpoint(
        self,
        model_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> EndpointInfo:
        """Deploy a serverless inference endpoint.

        Args:
            model_name: Name of the SageMaker model.
            endpoint_name: Name for the endpoint.

        Returns:
            EndpointInfo with deployment details.
        """
        sagemaker_client, _ = self._get_clients()

        model_name = model_name or self.config.model_name
        endpoint_name = endpoint_name or self.config.endpoint_name
        config_name = f"{endpoint_name}-config"

        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": self.config.memory_size_mb,
                        "MaxConcurrency": self.config.max_concurrency,
                    },
                },
            ],
            Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
        )

        # Create endpoint
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
        )

        logger.info(f"Created serverless endpoint: {endpoint_name}")

        return EndpointInfo(
            endpoint_name=endpoint_name,
            endpoint_arn=response["EndpointArn"],
            status="Creating",
            creation_time=str(response.get("CreationTime", "")),
            model_name=model_name,
            is_serverless=True,
        )

    def deploy_provisioned_endpoint(
        self,
        model_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        initial_instance_count: int = 1,
    ) -> EndpointInfo:
        """Deploy a provisioned inference endpoint.

        Args:
            model_name: Name of the SageMaker model.
            endpoint_name: Name for the endpoint.
            instance_type: Instance type (e.g., ml.m5.large).
            initial_instance_count: Number of initial instances.

        Returns:
            EndpointInfo with deployment details.
        """
        sagemaker_client, _ = self._get_clients()

        model_name = model_name or self.config.model_name
        endpoint_name = endpoint_name or self.config.endpoint_name
        instance_type = instance_type or self.config.instance_type
        config_name = f"{endpoint_name}-config"

        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": initial_instance_count,
                },
            ],
            Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
        )

        # Create endpoint
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
        )

        logger.info(f"Created provisioned endpoint: {endpoint_name}")

        return EndpointInfo(
            endpoint_name=endpoint_name,
            endpoint_arn=response["EndpointArn"],
            status="Creating",
            creation_time=str(response.get("CreationTime", "")),
            model_name=model_name,
            instance_type=instance_type,
            is_serverless=False,
        )

    def configure_auto_scaling(
        self,
        endpoint_name: Optional[str] = None,
        min_capacity: Optional[int] = None,
        max_capacity: Optional[int] = None,
    ) -> None:
        """Configure auto-scaling for a provisioned endpoint.

        Args:
            endpoint_name: Name of the endpoint.
            min_capacity: Minimum instance count.
            max_capacity: Maximum instance count.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for auto-scaling")

        import boto3
        autoscaling_client = boto3.client("application-autoscaling")

        endpoint_name = endpoint_name or self.config.endpoint_name
        min_capacity = min_capacity or self.config.min_instances
        max_capacity = max_capacity or self.config.max_instances

        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

        # Register scalable target
        autoscaling_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

        # Configure scaling policy
        autoscaling_client.put_scaling_policy(
            PolicyName=f"{endpoint_name}-scaling-policy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": self.config.target_invocations_per_instance,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                },
                "ScaleInCooldown": self.config.scale_in_cooldown,
                "ScaleOutCooldown": self.config.scale_out_cooldown,
            },
        )

        logger.info(f"Configured auto-scaling for {endpoint_name}")

    def get_endpoint_status(
        self,
        endpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get endpoint status.

        Args:
            endpoint_name: Name of the endpoint.

        Returns:
            Dictionary with endpoint status.
        """
        sagemaker_client, _ = self._get_clients()

        endpoint_name = endpoint_name or self.config.endpoint_name

        response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        return {
            "endpoint_name": response["EndpointName"],
            "endpoint_arn": response["EndpointArn"],
            "status": response["EndpointStatus"],
            "creation_time": str(response["CreationTime"]),
            "last_modified_time": str(response["LastModifiedTime"]),
        }

    def delete_endpoint(
        self,
        endpoint_name: Optional[str] = None,
        delete_config: bool = True,
        delete_model: bool = False,
    ) -> None:
        """Delete a SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint.
            delete_config: Whether to delete endpoint config.
            delete_model: Whether to delete the model.
        """
        sagemaker_client, _ = self._get_clients()

        endpoint_name = endpoint_name or self.config.endpoint_name
        config_name = f"{endpoint_name}-config"

        # Delete endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deleted endpoint: {endpoint_name}")

        # Delete endpoint configuration
        if delete_config:
            try:
                sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=config_name
                )
                logger.info(f"Deleted endpoint config: {config_name}")
            except Exception as e:
                logger.warning(f"Could not delete endpoint config: {e}")

        # Delete model
        if delete_model:
            try:
                sagemaker_client.delete_model(
                    ModelName=self.config.model_name
                )
                logger.info(f"Deleted model: {self.config.model_name}")
            except Exception as e:
                logger.warning(f"Could not delete model: {e}")


class SageMakerInferenceClient:
    """Client for invoking SageMaker endpoints."""

    def __init__(self, endpoint_name: str, region: Optional[str] = None):
        """Initialize the client.

        Args:
            endpoint_name: Name of the SageMaker endpoint.
            region: AWS region.
        """
        self.endpoint_name = endpoint_name

        try:
            import boto3
            self._runtime_client = boto3.client(
                "sagemaker-runtime",
                region_name=region,
            )
        except ImportError:
            raise ImportError("boto3 is required for SageMaker inference")

    def invoke(
        self,
        user_data: Dict[str, Any],
    ) -> np.ndarray:
        """Invoke the endpoint to get user embedding.

        Args:
            user_data: User features dictionary.

        Returns:
            User embedding as numpy array.
        """
        response = self._runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(user_data),
        )

        result = json.loads(response["Body"].read().decode())
        return np.array(result["embedding"])

    def invoke_batch(
        self,
        users_data: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """Invoke endpoint for multiple users.

        Args:
            users_data: List of user data dictionaries.

        Returns:
            List of user embeddings.
        """
        # For batch, invoke sequentially (could be optimized)
        return [self.invoke(user_data) for user_data in users_data]
