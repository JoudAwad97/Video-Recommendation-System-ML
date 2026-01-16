"""
Lambda handler for data preprocessing pipeline.

This module provides the entry point for the preprocessing Lambda function
that runs as part of the ML training pipeline (Step Functions).

Handler mapping for CDK: src.lambdas.preprocessing.handler
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


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for preprocessing job.

    This handler is triggered by Step Functions as part of the ML pipeline.
    It performs:
    - Data loading from S3
    - Feature engineering
    - Vocabulary building
    - Dataset generation for Two-Tower and Ranker models
    - Artifact saving to S3

    Args:
        event: Step Functions event containing:
            - data_bucket: S3 bucket for input data
            - artifacts_bucket: S3 bucket for output artifacts
            - config: Processing configuration
        context: Lambda context

    Returns:
        Dictionary with preprocessing results and artifact locations
    """
    logger = _get_logger()
    logger.info(f"Starting preprocessing job with event: {json.dumps(event, default=str)}")

    start_time = datetime.utcnow()

    try:
        # Extract configuration from event
        data_bucket = event.get("data_bucket") or os.environ.get("DATA_BUCKET")
        artifacts_bucket = event.get("artifacts_bucket") or os.environ.get("ARTIFACTS_BUCKET")
        model_bucket = event.get("model_bucket") or os.environ.get("MODEL_BUCKET")

        if not data_bucket or not artifacts_bucket:
            raise ValueError("data_bucket and artifacts_bucket are required")

        # Import preprocessing components
        from ..ml_pipeline.preprocessing_job import run_preprocessing

        # Run preprocessing
        result = run_preprocessing(
            data_bucket=data_bucket,
            artifacts_bucket=artifacts_bucket,
            model_bucket=model_bucket,
            config=event.get("config", {}),
        )

        # Calculate duration
        duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "status": "success",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "artifacts": result.get("artifacts", {}),
            "datasets": result.get("datasets", {}),
            "statistics": result.get("statistics", {}),
        }

        logger.info(f"Preprocessing completed successfully in {duration_seconds:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Preprocessing failed: {traceback.format_exc()}")

        return {
            "status": "failed",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
        }
