"""
Lambda handler for model deployment.

This module provides the entry point for the deployment Lambda function
that deploys trained models to production as part of the ML pipeline.

Handler mapping for CDK: src.lambdas.deployment.handler
"""

from __future__ import annotations

import json
import os
import traceback
import tempfile
from datetime import datetime
from typing import Any, Dict

import numpy as np

# Lazy import logger
_logger = None


def _get_logger():
    """Get logger with lazy import."""
    global _logger
    if _logger is None:
        from ..utils.logging_utils import get_logger
        _logger = get_logger(__name__)
    return _logger


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for model deployment.

    This handler deploys trained models to production:
    - Copies embeddings to production S3 location
    - Updates DynamoDB with video features and embeddings
    - Updates model registry with new versions

    Args:
        event: Step Functions event containing:
            - evaluation: Evaluation results with deployment decision
            - training_results: Training results with model artifacts
            - preprocessing: Preprocessing results
        context: Lambda context

    Returns:
        Dictionary with deployment results
    """
    logger = _get_logger()
    logger.info(f"Starting model deployment with event: {json.dumps(event, default=str)}")

    start_time = datetime.utcnow()

    try:
        # Check if deployment was approved
        evaluation = event.get("evaluation", {})
        should_deploy = evaluation.get("should_deploy", False)

        if not should_deploy:
            logger.info("Deployment skipped - evaluation did not pass thresholds")
            return {
                "status": "skipped",
                "reason": "Evaluation did not pass thresholds",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            }

        # Get model artifacts and preprocessing result
        model_versions = evaluation.get("model_versions", {})
        training_results = event.get("training_results", [])
        preprocessing = event.get("preprocessing", {})

        # Extract configuration
        model_bucket = os.environ.get("MODEL_BUCKET")
        artifacts_bucket = os.environ.get("ARTIFACTS_BUCKET")

        # Deploy models
        deployment_results = {}

        # Deploy Two-Tower embeddings and update video features in DynamoDB
        two_tower_deployed = _deploy_two_tower_embeddings(
            training_results,
            preprocessing,
            model_bucket,
            artifacts_bucket,
        )
        deployment_results["two_tower"] = two_tower_deployed

        # Deploy Ranker model to S3
        ranker_deployed = _deploy_ranker_model(
            training_results,
            model_bucket,
        )
        deployment_results["ranker"] = ranker_deployed

        # Update model registry
        _update_model_registry(model_versions, deployment_results)

        duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "status": "success",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "deployments": deployment_results,
            "model_versions": model_versions,
        }

        logger.info(f"Deployment completed successfully in {duration_seconds:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Deployment failed: {traceback.format_exc()}")

        return {
            "status": "failed",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            }
        }


def _deploy_two_tower_embeddings(
    training_results: list,
    preprocessing: Dict[str, Any],
    model_bucket: str,
    artifacts_bucket: str,
) -> Dict[str, Any]:
    """Deploy Two-Tower model embeddings and update video features in DynamoDB."""
    logger = _get_logger()

    # Find Two-Tower result
    two_tower_result = None
    for result in training_results:
        if isinstance(result, dict) and result.get("model_type") == "two_tower":
            two_tower_result = result
            break

    if not two_tower_result or two_tower_result.get("status") == "failed":
        return {
            "status": "skipped",
            "reason": "No valid Two-Tower model to deploy",
        }

    try:
        import boto3

        s3 = boto3.client("s3")
        dynamodb = boto3.resource("dynamodb")

        # Get source artifacts
        source_artifacts = two_tower_result.get("model_artifacts", {})
        embeddings_key = source_artifacts.get("embeddings_key")
        version = source_artifacts.get("version")

        # Get preprocessing job_id for video catalog
        preprocess_job_id = preprocessing.get("job_id")

        # Download embeddings from S3
        with tempfile.TemporaryDirectory() as tmpdir:
            if embeddings_key:
                embeddings_path = os.path.join(tmpdir, "embeddings.npz")
                try:
                    s3.download_file(model_bucket, embeddings_key, embeddings_path)
                    data = np.load(embeddings_path, allow_pickle=True)
                    video_embeddings = data["embeddings"]
                    video_ids = data["video_ids"]
                    embedding_dim = int(data["embedding_dim"])
                    logger.info(f"Loaded embeddings: {len(video_ids)} videos, dim={embedding_dim}")
                except Exception as e:
                    logger.warning(f"Could not load embeddings: {e}")
                    video_embeddings = None
                    video_ids = None

            # Load video catalog for feature updates
            video_catalog = None
            if preprocess_job_id:
                catalog_key = f"preprocessing/{preprocess_job_id}/video_catalog.parquet"
                catalog_path = os.path.join(tmpdir, "catalog.parquet")
                try:
                    s3.download_file(artifacts_bucket or model_bucket, catalog_key, catalog_path)
                    import pandas as pd
                    video_catalog = pd.read_parquet(catalog_path)
                    logger.info(f"Loaded video catalog: {len(video_catalog)} videos")
                except Exception as e:
                    logger.warning(f"Could not load video catalog: {e}")

            # Update video features in DynamoDB
            video_features_table = os.environ.get("VIDEO_FEATURES_TABLE")
            videos_updated = 0

            if video_features_table and video_catalog is not None:
                table = dynamodb.Table(video_features_table)

                # Create video ID to embedding index mapping
                video_id_to_idx = {}
                if video_ids is not None:
                    for idx, vid in enumerate(video_ids):
                        if str(vid) not in ["[PAD]", "[OOV]"]:
                            video_id_to_idx[str(vid)] = idx

                # Update video features with embeddings
                with table.batch_writer() as batch:
                    for _, row in video_catalog.iterrows():
                        video_id = str(row.get("video_id", row.get("id", "")))
                        if not video_id:
                            continue

                        item = {
                            "video_id": video_id,
                            "category": str(row.get("category", "")),
                            "title": str(row.get("title", "")),
                            "video_duration": int(row.get("video_duration", row.get("duration", 0))),
                            "view_count": int(row.get("view_count", 0)),
                            "like_count": int(row.get("like_count", 0)),
                            "popularity": str(row.get("popularity", "low")),
                            "video_language": str(row.get("video_language", row.get("language", "English"))),
                            "channel_id": str(row.get("channel_id", "")),
                            "tags": str(row.get("manual_tags", row.get("augmented_tags", ""))),
                            "updated_at": datetime.utcnow().isoformat(),
                        }

                        # Add embedding if available
                        if video_id in video_id_to_idx and video_embeddings is not None:
                            idx = video_id_to_idx[video_id]
                            embedding = video_embeddings[idx].tolist()
                            item["embedding"] = json.dumps(embedding)
                            item["embedding_version"] = version

                        batch.put_item(Item=item)
                        videos_updated += 1

                logger.info(f"Updated {videos_updated} video features in DynamoDB")

            # Copy embeddings to production location
            prod_key = "vector_store/video_embeddings.npz"
            if embeddings_key:
                try:
                    s3.copy_object(
                        Bucket=model_bucket,
                        CopySource={"Bucket": model_bucket, "Key": embeddings_key},
                        Key=prod_key,
                    )
                    logger.info(f"Deployed embeddings to s3://{model_bucket}/{prod_key}")
                except Exception as e:
                    logger.warning(f"Could not copy embeddings to production: {e}")

        return {
            "status": "success",
            "artifact_location": f"s3://{model_bucket}/{prod_key}",
            "version": version,
            "videos_updated": videos_updated,
        }

    except Exception as e:
        logger.error(f"Failed to deploy Two-Tower embeddings: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


def _deploy_ranker_model(
    training_results: list,
    model_bucket: str,
) -> Dict[str, Any]:
    """Deploy Ranker model to S3."""
    logger = _get_logger()

    # Find Ranker result
    ranker_result = None
    for result in training_results:
        if isinstance(result, dict) and result.get("model_type") == "ranker":
            ranker_result = result
            break

    if not ranker_result or ranker_result.get("status") == "failed":
        return {
            "status": "skipped",
            "reason": "No valid Ranker model to deploy",
        }

    try:
        import boto3

        s3 = boto3.client("s3")

        # Get source and destination paths
        source_artifacts = ranker_result.get("model_artifacts", {})
        model_key = source_artifacts.get("model_key")
        version = source_artifacts.get("version")

        if not model_key:
            return {
                "status": "skipped",
                "reason": "No model artifact found",
            }

        # Copy to production location
        prod_key = "models/ranker/model.cbm"
        try:
            s3.copy_object(
                Bucket=model_bucket,
                CopySource={"Bucket": model_bucket, "Key": model_key},
                Key=prod_key,
            )
            logger.info(f"Deployed Ranker model to s3://{model_bucket}/{prod_key}")
        except Exception as e:
            # Model may already be in production location
            logger.warning(f"Could not copy ranker model: {e}")

        return {
            "status": "success",
            "artifact_location": f"s3://{model_bucket}/{prod_key}",
            "version": version,
        }

    except Exception as e:
        logger.error(f"Failed to deploy Ranker model: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


def _update_model_registry(
    model_versions: Dict[str, str],
    deployment_results: Dict[str, Any]
) -> None:
    """Update model registry with new versions."""
    logger = _get_logger()

    try:
        import boto3

        dynamodb = boto3.resource("dynamodb")
        table_name = os.environ.get("MODEL_REGISTRY_TABLE")

        if not table_name:
            logger.info("No MODEL_REGISTRY_TABLE configured, skipping registry update")
            return

        table = dynamodb.Table(table_name)
        timestamp = datetime.utcnow().isoformat()

        for model_type, result in deployment_results.items():
            version = result.get("version")
            if version and result.get("status") == "success":
                table.put_item(Item={
                    "model_type": model_type,
                    "version": version,
                    "deployed_at": timestamp,
                    "status": "active",
                    "artifact_location": result.get("artifact_location"),
                    "videos_updated": result.get("videos_updated", 0),
                })
                logger.info(f"Updated registry: {model_type} v{version}")

    except Exception as e:
        logger.warning(f"Failed to update model registry: {e}")
