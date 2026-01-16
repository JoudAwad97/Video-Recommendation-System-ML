"""
Lambda handlers for model training orchestration.

This module provides entry points for training Lambda functions
that orchestrate SageMaker training jobs as part of the ML pipeline.

Handler mappings for CDK:
- Two-Tower: src.lambdas.training.two_tower_handler
- Ranker: src.lambdas.training.ranker_handler
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict

# Lazy import logger
_logger = None


def _get_logger():
    """Get logger with lazy import."""
    global _logger
    if _logger is None:
        from ..utils.logging_utils import get_logger
        _logger = get_logger(__name__)
    return _logger


def two_tower_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for Two-Tower model training.

    This handler orchestrates SageMaker training jobs for the Two-Tower
    (candidate generation) model.

    Args:
        event: Step Functions event containing:
            - preprocessing: Results from preprocessing step
            - model_bucket: S3 bucket for model artifacts
            - config: Training configuration
        context: Lambda context

    Returns:
        Dictionary with training job details and model location
    """
    logger = _get_logger()
    logger.info(f"Starting Two-Tower training with event: {json.dumps(event, default=str)}")

    start_time = datetime.utcnow()

    try:
        # Extract configuration
        preprocessing_result = event.get("preprocessing", {})
        model_bucket = event.get("model_bucket") or os.environ.get("MODEL_BUCKET")
        artifacts_bucket = event.get("artifacts_bucket") or os.environ.get("ARTIFACTS_BUCKET")
        sagemaker_role = os.environ.get("SAGEMAKER_ROLE_ARN")

        if not model_bucket:
            raise ValueError("model_bucket is required")

        # Import training orchestrator
        from ..ml_pipeline.training_orchestrator import start_two_tower_training

        # Start training job
        result = start_two_tower_training(
            model_bucket=model_bucket,
            artifacts_bucket=artifacts_bucket,
            preprocessing_result=preprocessing_result,
            sagemaker_role=sagemaker_role,
            config=event.get("config", {}),
        )

        duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "status": "success",
            "model_type": "two_tower",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "training_job": result.get("training_job", {}),
            "model_artifacts": result.get("model_artifacts", {}),
            "metrics": result.get("metrics", {}),
        }

        logger.info(f"Two-Tower training completed in {duration_seconds:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Two-Tower training failed: {traceback.format_exc()}")

        return {
            "status": "failed",
            "model_type": "two_tower",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            }
        }


def ranker_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for Ranker model training.

    This handler orchestrates training for the CatBoost Ranker model.
    Unlike Two-Tower, the Ranker can be trained directly in Lambda
    for smaller datasets, or via SageMaker for larger ones.

    Args:
        event: Step Functions event containing:
            - preprocessing: Results from preprocessing step
            - model_bucket: S3 bucket for model artifacts
            - config: Training configuration
        context: Lambda context

    Returns:
        Dictionary with training results and model location
    """
    logger = _get_logger()
    logger.info(f"Starting Ranker training with event: {json.dumps(event, default=str)}")

    start_time = datetime.utcnow()

    try:
        # Extract configuration
        preprocessing_result = event.get("preprocessing", {})
        model_bucket = event.get("model_bucket") or os.environ.get("MODEL_BUCKET")
        artifacts_bucket = event.get("artifacts_bucket") or os.environ.get("ARTIFACTS_BUCKET")

        if not model_bucket:
            raise ValueError("model_bucket is required")

        # Import training orchestrator
        from ..ml_pipeline.training_orchestrator import start_ranker_training

        # Start training
        result = start_ranker_training(
            model_bucket=model_bucket,
            artifacts_bucket=artifacts_bucket,
            preprocessing_result=preprocessing_result,
            config=event.get("config", {}),
        )

        duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "status": "success",
            "model_type": "ranker",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "model_artifacts": result.get("model_artifacts", {}),
            "metrics": result.get("metrics", {}),
            "feature_importance": result.get("feature_importance", [])[:20],  # Top 20
        }

        logger.info(f"Ranker training completed in {duration_seconds:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Ranker training failed: {traceback.format_exc()}")

        return {
            "status": "failed",
            "model_type": "ranker",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            }
        }
