"""
AWS Lambda handler for the recommendation API.

Handles requests from API Gateway and returns recommendations.
Designed to run behind AWS API Gateway with Lambda proxy integration.
"""

import json
import os
import traceback
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Service initialization happens at module level for Lambda warm starts
# Use threading lock for thread safety during initialization
_service = None
_initialized = False
_init_lock = threading.Lock()


@dataclass
class LambdaConfig:
    """Configuration for Lambda handler."""

    # Feature store settings
    use_sagemaker_feature_store: bool = False
    feature_store_region: str = "us-east-1"
    user_feature_group: str = "user-features"
    video_feature_group: str = "video-features"

    # Cache settings
    use_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    elasticache_endpoint: str = ""

    # Model settings
    model_path: str = "/opt/ml/model"
    artifacts_path: str = "/opt/ml/artifacts"

    # Vector store settings
    vector_store_type: str = "in_memory"  # or "pinecone", "opensearch"
    vector_store_endpoint: str = ""

    # Default settings
    default_num_recommendations: int = 20
    max_num_recommendations: int = 100

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_environment(cls) -> "LambdaConfig":
        """Create config from environment variables.

        Returns:
            LambdaConfig instance.
        """
        return cls(
            use_sagemaker_feature_store=os.environ.get(
                "USE_SAGEMAKER_FEATURE_STORE", "false"
            ).lower() == "true",
            feature_store_region=os.environ.get("FEATURE_STORE_REGION", "us-east-1"),
            user_feature_group=os.environ.get("USER_FEATURE_GROUP", "user-features"),
            video_feature_group=os.environ.get("VIDEO_FEATURE_GROUP", "video-features"),
            use_redis=os.environ.get("USE_REDIS", "false").lower() == "true",
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            elasticache_endpoint=os.environ.get("ELASTICACHE_ENDPOINT", ""),
            model_path=os.environ.get("MODEL_PATH", "/opt/ml/model"),
            artifacts_path=os.environ.get("ARTIFACTS_PATH", "/opt/ml/artifacts"),
            vector_store_type=os.environ.get("VECTOR_STORE_TYPE", "in_memory"),
            vector_store_endpoint=os.environ.get("VECTOR_STORE_ENDPOINT", ""),
            default_num_recommendations=int(
                os.environ.get("DEFAULT_NUM_RECOMMENDATIONS", "20")
            ),
            max_num_recommendations=int(
                os.environ.get("MAX_NUM_RECOMMENDATIONS", "100")
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def _create_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create an API Gateway response.

    Args:
        status_code: HTTP status code.
        body: Response body.
        headers: Additional headers.

    Returns:
        API Gateway response dict.
    """
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }

    if headers:
        default_headers.update(headers)

    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body),
    }


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
) -> Dict[str, Any]:
    """Create an error response.

    Args:
        status_code: HTTP status code.
        error_type: Error type identifier.
        message: Error message.

    Returns:
        API Gateway response dict.
    """
    return _create_response(
        status_code,
        {
            "error": {
                "type": error_type,
                "message": message,
            }
        },
    )


def _initialize_service(config: LambdaConfig) -> None:
    """Initialize the recommendation service.

    Thread-safe initialization using a lock to prevent race conditions.

    Args:
        config: Lambda configuration.
    """
    global _service, _initialized

    # Early exit without lock for performance
    if _initialized:
        return

    # Use lock for thread-safe initialization
    with _init_lock:
        # Double-check after acquiring lock
        if _initialized:
            return

        try:
            # Import here to avoid issues during cold start if not available
            from .recommendation_service import (
                RecommendationService,
                RecommendationServiceConfig,
            )
            from .feature_store_client import FeatureStoreConfig
            from .redis_cache_client import RedisCacheConfig
            from .serving_config import ServingConfig

            # Build service config
            feature_store_config = FeatureStoreConfig(
                user_feature_group=config.user_feature_group,
                video_feature_group=config.video_feature_group,
                aws_region=config.feature_store_region,
            )

            cache_config = RedisCacheConfig(
                host=config.redis_host,
                port=config.redis_port,
                use_elasticache=bool(config.elasticache_endpoint),
                elasticache_endpoint=config.elasticache_endpoint,
            )

            serving_config = ServingConfig()

            service_config = RecommendationServiceConfig(
                serving=serving_config,
                feature_store=feature_store_config,
                cache=cache_config,
                default_num_recommendations=config.default_num_recommendations,
                max_num_recommendations=config.max_num_recommendations,
            )

            # Create and initialize service
            _service = RecommendationService(service_config)
            _service.initialize(
                use_sagemaker_feature_store=config.use_sagemaker_feature_store,
                use_redis=config.use_redis,
                model_path=config.model_path if os.path.exists(config.model_path) else None,
                artifacts_path=config.artifacts_path if os.path.exists(config.artifacts_path) else None,
            )

            # Warm up the service
            _service.warmup()

            _initialized = True
            logger.info("Recommendation service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {traceback.format_exc()}")
            raise


def _parse_request_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse request body from API Gateway event.

    Args:
        event: API Gateway event.

    Returns:
        Parsed request body.
    """
    body = event.get("body", "{}")

    if isinstance(body, str):
        return json.loads(body) if body else {}
    return body or {}


def _get_user_id(event: Dict[str, Any], body: Dict[str, Any]) -> Optional[int]:
    """Extract user ID from request.

    Args:
        event: API Gateway event.
        body: Parsed request body.

    Returns:
        User ID or None.

    Raises:
        ValueError: If user_id is present but not a valid integer.
    """
    # Check path parameters first
    path_params = event.get("pathParameters") or {}
    if "user_id" in path_params:
        try:
            return int(path_params["user_id"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid user_id in path: {path_params['user_id']}") from e

    # Check query parameters
    query_params = event.get("queryStringParameters") or {}
    if "user_id" in query_params:
        try:
            return int(query_params["user_id"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid user_id in query: {query_params['user_id']}") from e

    # Check body
    if "user_id" in body:
        try:
            return int(body["user_id"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid user_id in body: {body['user_id']}") from e

    return None


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for recommendation requests.

    Supports the following endpoints:
    - GET /recommendations/{user_id} - Get recommendations for a user
    - POST /recommendations - Get recommendations with full request body
    - POST /interactions - Record a user interaction
    - GET /health - Health check

    Args:
        event: API Gateway event.
        context: Lambda context.

    Returns:
        API Gateway response.
    """
    try:
        # Handle OPTIONS requests for CORS
        http_method = event.get("httpMethod", "GET")
        if http_method == "OPTIONS":
            return _create_response(200, {})

        # Get resource path
        resource = event.get("resource", "")
        path = event.get("path", "")

        # Health check
        if path == "/health" or resource == "/health":
            return _handle_health_check()

        # Initialize service (will be no-op on warm starts)
        config = LambdaConfig.from_environment()
        _initialize_service(config)

        # Route to appropriate handler
        if "/recommendations" in path or "/recommendations" in resource:
            if http_method == "GET":
                return _handle_get_recommendations(event)
            elif http_method == "POST":
                return _handle_post_recommendations(event)

        if "/interactions" in path or "/interactions" in resource:
            if http_method == "POST":
                return _handle_post_interaction(event)

        if "/cached" in path:
            if http_method == "GET":
                return _handle_get_cached(event)

        # Unknown endpoint
        return _error_response(404, "NOT_FOUND", f"Endpoint not found: {path}")

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in request: {e}")
        return _error_response(400, "INVALID_JSON", f"Invalid JSON in request: {e}")
    except ValueError as e:
        logger.warning(f"Invalid parameter: {e}")
        return _error_response(400, "INVALID_PARAMETER", str(e))
    except Exception as e:
        # Log the full traceback using structured logging
        logger.error(f"Error handling request: {traceback.format_exc()}")
        return _error_response(500, "INTERNAL_ERROR", "An internal error occurred")


def _handle_health_check() -> Dict[str, Any]:
    """Handle health check request.

    Returns:
        Health check response.
    """
    global _service, _initialized

    status = "healthy"
    details = {
        "initialized": _initialized,
    }

    if _initialized and _service:
        try:
            stats = _service.get_stats()
            details.update(stats)
        except Exception as e:
            status = "degraded"
            details["error"] = str(e)

    return _create_response(200, {"status": status, "details": details})


def _handle_get_recommendations(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GET /recommendations/{user_id} request.

    Args:
        event: API Gateway event.

    Returns:
        Recommendations response.
    """
    global _service

    if _service is None:
        return _error_response(503, "SERVICE_UNAVAILABLE", "Service not initialized")

    # Get user ID
    body = _parse_request_body(event)
    user_id = _get_user_id(event, body)

    if user_id is None:
        return _error_response(400, "MISSING_PARAMETER", "user_id is required")

    # Get query parameters
    query_params = event.get("queryStringParameters") or {}
    try:
        num_recommendations = int(query_params.get("n", "20"))
    except (ValueError, TypeError):
        return _error_response(400, "INVALID_PARAMETER", "Invalid value for 'n' parameter")

    exclude_watched = query_params.get("exclude_watched", "true").lower() == "true"

    # Build excluded set
    excluded_ids = None
    if exclude_watched and _service.cache_client is not None:
        excluded_ids = _service.cache_client.get_watched_videos(user_id)

    # Get recommendations
    response = _service.get_recommendations(
        user_id=user_id,
        num_recommendations=num_recommendations,
        excluded_video_ids=excluded_ids,
    )

    return _create_response(200, response.to_dict())


def _handle_post_recommendations(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle POST /recommendations request.

    Args:
        event: API Gateway event.

    Returns:
        Recommendations response.
    """
    global _service

    body = _parse_request_body(event)

    # Get user ID
    user_id = _get_user_id(event, body)
    if user_id is None:
        return _error_response(400, "MISSING_PARAMETER", "user_id is required")

    # Get parameters from body
    num_recommendations = body.get("num_recommendations", 20)
    user_features = body.get("user_features")
    user_preferences = body.get("user_preferences")
    excluded_video_ids = body.get("excluded_video_ids")
    interaction_context = body.get("interaction_context")

    # Convert excluded IDs to set
    if excluded_video_ids:
        excluded_video_ids = set(excluded_video_ids)

    # Get recommendations
    response = _service.get_recommendations(
        user_id=user_id,
        num_recommendations=num_recommendations,
        user_features=user_features,
        excluded_video_ids=excluded_video_ids,
        user_preferences=user_preferences,
        interaction_context=interaction_context,
    )

    return _create_response(200, response.to_dict())


def _handle_post_interaction(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle POST /interactions request.

    Records a user interaction for future recommendations.

    Args:
        event: API Gateway event.

    Returns:
        Success response.
    """
    global _service

    body = _parse_request_body(event)

    # Validate required fields
    required_fields = ["user_id", "video_id", "category"]
    for field in required_fields:
        if field not in body:
            return _error_response(
                400, "MISSING_PARAMETER", f"{field} is required"
            )

    user_id = int(body["user_id"])
    video_id = int(body["video_id"])
    category = body["category"]
    interaction_type = body.get("interaction_type", "watch")
    duration_watched = float(body.get("duration_watched", 0.0))

    # Record interaction
    _service.record_interaction(
        user_id=user_id,
        video_id=video_id,
        category=category,
        interaction_type=interaction_type,
        duration_watched=duration_watched,
    )

    return _create_response(200, {"status": "recorded"})


def _handle_get_cached(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GET /cached/{user_id} request.

    Gets cached recommendations for a user.

    Args:
        event: API Gateway event.

    Returns:
        Cached recommendations response.
    """
    global _service

    body = _parse_request_body(event)
    user_id = _get_user_id(event, body)

    if user_id is None:
        return _error_response(400, "MISSING_PARAMETER", "user_id is required")

    cached = _service.get_cached_recommendations(user_id)

    if cached is None:
        return _create_response(200, {"cached": False, "recommendations": []})

    return _create_response(200, {"cached": True, "recommendations": cached})


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "httpMethod": "GET",
        "path": "/recommendations/123",
        "pathParameters": {"user_id": "123"},
        "queryStringParameters": {"n": "10"},
    }

    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
