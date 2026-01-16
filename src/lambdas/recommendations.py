"""
Lambda handler for real-time recommendation serving.

This module provides the entry point for the recommendation API Lambda function.
It wraps the core recommendation service with Lambda-specific handling:
- Request/response formatting
- Error handling and status codes
- Health checks
- Warmup optimization

Handler mapping for CDK: src.lambdas.recommendations.handler
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, Optional

# Lazy imports for cold start optimization
_service = None
_initialized = False
_init_lock = None
_config = None


def _get_init_lock():
    """Get or create initialization lock."""
    global _init_lock
    if _init_lock is None:
        import threading
        _init_lock = threading.Lock()
    return _init_lock


def _get_config():
    """Get Lambda configuration."""
    global _config
    if _config is None:
        from ..config.settings import get_settings
        _config = get_settings()
    return _config


def _get_logger():
    """Get logger with lazy import."""
    from ..utils.logging_utils import get_logger, request_context
    return get_logger(__name__), request_context


def _initialize_service():
    """Initialize the recommendation service."""
    global _service, _initialized

    with _get_init_lock():
        if _initialized:
            return

        logger, _ = _get_logger()
        config = _get_config()

        logger.info("Initializing recommendation service...")
        start_time = time.time()

        try:
            # Import service components
            from ..serving.recommendation_service import (
                RecommendationService,
                RecommendationServiceConfig,
            )
            from ..serving.serving_config import ServingConfig, VectorDBConfig

            # Create service configuration using RecommendationServiceConfig
            serving_config = ServingConfig(
                num_candidates=config.serving.default_num_recommendations * config.serving.candidate_multiplier,
                top_k_final=config.serving.default_num_recommendations,
                vector_db=VectorDBConfig(embedding_dim=config.vector_store.embedding_dim),
            )

            service_config = RecommendationServiceConfig(
                serving=serving_config,
                default_num_recommendations=config.serving.default_num_recommendations,
                max_num_recommendations=config.serving.max_num_recommendations,
            )

            # Initialize service
            _service = RecommendationService(service_config)

            # Initialize components (required before warmup/get_recommendations)
            _service.initialize(
                use_sagemaker_feature_store=False,
                use_redis=config.redis.enabled,
            )

            # Load embeddings from S3 if configured
            if config.vector_store.s3_bucket:
                _load_embeddings_from_s3(config)

            # Warm up the service
            _service.warmup()

            _initialized = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Service initialized in {elapsed:.2f}ms")

        except Exception as e:
            logger.error(f"Failed to initialize service: {traceback.format_exc()}")
            raise


def _load_embeddings_from_s3(config):
    """Load video embeddings and features from S3."""
    global _service
    logger, _ = _get_logger()

    try:
        import boto3
        import numpy as np
        import io

        s3 = boto3.client("s3")
        bucket = config.vector_store.s3_bucket
        key = config.vector_store.s3_key

        logger.info(f"Loading embeddings from s3://{bucket}/{key}")

        # Load embeddings
        response = s3.get_object(Bucket=bucket, Key=key)
        embeddings_data = np.load(io.BytesIO(response["Body"].read()))

        video_ids = embeddings_data["video_ids"].tolist()
        embeddings = embeddings_data["embeddings"]

        logger.info(f"Loaded {len(video_ids)} video embeddings")

        # Create embeddings dict
        video_embeddings = {
            int(vid): embeddings[i] for i, vid in enumerate(video_ids)
        }

        # Load video features
        video_metadata = {}
        try:
            response = s3.get_object(
                Bucket=bucket,
                Key=config.vector_store.video_features_s3_key
            )
            features_data = json.loads(response["Body"].read().decode("utf-8"))
            video_metadata = {int(k): v for k, v in features_data.items()}
            logger.info(f"Loaded {len(video_metadata)} video features")
        except Exception as e:
            logger.warning(f"Could not load video features: {e}")
            video_metadata = {vid: {"video_id": vid} for vid in video_ids}

        # Load into service
        _service.load_video_data(video_metadata, video_embeddings)

    except Exception as e:
        logger.warning(f"Could not load embeddings from S3: {e}")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for recommendation requests.

    Supports the following endpoints:
    - GET /health: Health check
    - POST /recommendations: Get recommendations
    - GET /recommendations/{user_id}: Get recommendations for user

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        API Gateway response
    """
    start_time = time.time()
    logger, request_context_fn = _get_logger()

    # Extract request info
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    request_id = event.get("requestContext", {}).get("requestId", "unknown")

    with request_context_fn(request_id=request_id, path=path, method=http_method):
        try:
            # Handle CORS preflight requests
            if http_method == "OPTIONS":
                return _create_response(200, {"message": "OK"})

            # Health check endpoint
            if path == "/health" or path.endswith("/health"):
                return _handle_health_check()

            # Initialize service if needed
            if not _initialized:
                _initialize_service()

            # Route request
            if "/recommendations" in path:
                if http_method == "POST":
                    return _handle_post_recommendations(event, start_time)
                elif http_method == "GET":
                    return _handle_get_recommendations(event, start_time)

            # Unknown endpoint
            return _create_response(404, {"error": "Not found"})

        except Exception as e:
            logger.error(f"Request failed: {traceback.format_exc()}")
            return _create_error_response(
                500,
                "InternalError",
                str(e),
            )


def _handle_health_check() -> Dict[str, Any]:
    """Handle health check request."""
    from ..utils.health import HealthChecker, HealthStatus

    checker = HealthChecker(version=os.environ.get("SERVICE_VERSION", "1.0.0"))

    # Add checks based on configuration
    config = _get_config()

    if config.redis.enabled:
        from ..utils.health import RedisHealthCheck
        checker.add_check(RedisHealthCheck(
            host=config.redis.host,
            port=config.redis.port,
        ))

    if config.aws.user_features_table:
        from ..utils.health import DynamoDBHealthCheck
        checker.add_check(DynamoDBHealthCheck(
            table_name=config.aws.user_features_table,
            region=config.aws.region,
        ))

    # Run health checks
    health = checker.run_all()

    status_code = 200 if health.status == HealthStatus.HEALTHY else 503
    return _create_response(status_code, health.to_dict())


def _handle_post_recommendations(event: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Handle POST /recommendations request."""
    logger, _ = _get_logger()

    # Parse request body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return _create_error_response(400, "InvalidRequest", "Invalid JSON body")

    user_id = body.get("user_id")
    if user_id is None:
        return _create_error_response(400, "InvalidRequest", "user_id is required")

    num_recommendations = body.get("num_recommendations", _get_config().serving.default_num_recommendations)
    num_recommendations = min(num_recommendations, _get_config().serving.max_num_recommendations)

    # Get recommendations
    result = _service.get_recommendations(
        user_id=int(user_id),
        num_recommendations=num_recommendations,
    )

    # Format response - result is a RecommendationServiceResponse dataclass
    elapsed_ms = (time.time() - start_time) * 1000
    recommendations = [rec.to_dict() for rec in result.recommendations]
    response_body = {
        "user_id": user_id,
        "recommendations": recommendations,
        "total_latency_ms": round(elapsed_ms, 2),
        "stage_latencies_ms": result.stage_latencies,
    }

    logger.info(f"Served {len(recommendations)} recommendations in {elapsed_ms:.2f}ms")
    return _create_response(200, response_body)


def _handle_get_recommendations(event: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Handle GET /recommendations/{user_id} request."""
    # Extract user_id from path
    path_params = event.get("pathParameters", {}) or {}
    user_id = path_params.get("user_id")

    if not user_id:
        # Try to extract from path
        path = event.get("path", "")
        parts = path.rstrip("/").split("/")
        if len(parts) > 0 and parts[-1].isdigit():
            user_id = parts[-1]

    if not user_id:
        return _create_error_response(400, "InvalidRequest", "user_id is required")

    # Get num_recommendations from query string
    query_params = event.get("queryStringParameters", {}) or {}
    num_recommendations = int(query_params.get("n", _get_config().serving.default_num_recommendations))
    num_recommendations = min(num_recommendations, _get_config().serving.max_num_recommendations)

    # Get recommendations
    result = _service.get_recommendations(
        user_id=int(user_id),
        num_recommendations=num_recommendations,
    )

    # Format response - result is a RecommendationServiceResponse dataclass
    elapsed_ms = (time.time() - start_time) * 1000
    recommendations = [rec.to_dict() for rec in result.recommendations]
    response_body = {
        "user_id": int(user_id),
        "recommendations": recommendations,
        "total_latency_ms": round(elapsed_ms, 2),
    }

    return _create_response(200, response_body)


def _create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }


def _create_error_response(
    status_code: int,
    error_type: str,
    message: str
) -> Dict[str, Any]:
    """Create error response."""
    return _create_response(
        status_code,
        {
            "error": {
                "type": error_type,
                "message": message,
            }
        },
    )


# For local testing
if __name__ == "__main__":
    test_event = {
        "httpMethod": "GET",
        "path": "/recommendations/123",
        "pathParameters": {"user_id": "123"},
        "queryStringParameters": {"n": "10"},
    }
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
