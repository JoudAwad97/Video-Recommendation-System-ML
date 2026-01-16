"""Utility modules for the recommendation system.

Provides:
- Logging utilities with JSON/text formatting
- Health check framework
- Circuit breaker pattern
- I/O utilities for JSON and Parquet files

Note: io_utils requires pandas/pyarrow and is lazy-loaded.
"""

from typing import TYPE_CHECKING

# Light imports (no heavy dependencies)
from .logging_utils import (
    get_logger,
    get_logger_with_context,
    request_context,
    log_extra,
    log_function_call,
    timed,
    log_block,
    RequestContext,
    JSONFormatter,
    TextFormatter,
)

from .health import (
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    HealthChecker,
    ServiceHealth,
    DynamoDBHealthCheck,
    RedisHealthCheck,
    S3HealthCheck,
    VectorStoreHealthCheck,
    ModelHealthCheck,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    DegradedModeHandler,
    get_degraded_handler,
    with_fallback,
)

# Type checking imports for lazy-loaded modules
if TYPE_CHECKING:
    from .io_utils import (
        save_json,
        load_json,
        save_parquet,
        load_parquet,
        ensure_dir,
    )


def __getattr__(name: str):
    """Lazy import for io_utils (requires pandas)."""
    if name in ("save_json", "load_json", "save_parquet", "load_parquet", "ensure_dir"):
        from . import io_utils
        return getattr(io_utils, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Logging
    "get_logger",
    "get_logger_with_context",
    "request_context",
    "log_extra",
    "log_function_call",
    "timed",
    "log_block",
    "RequestContext",
    "JSONFormatter",
    "TextFormatter",
    # Health checks
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "HealthChecker",
    "ServiceHealth",
    "DynamoDBHealthCheck",
    "RedisHealthCheck",
    "S3HealthCheck",
    "VectorStoreHealthCheck",
    "ModelHealthCheck",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    # Degradation
    "DegradedModeHandler",
    "get_degraded_handler",
    "with_fallback",
    # I/O (lazy loaded)
    "save_json",
    "load_json",
    "save_parquet",
    "load_parquet",
    "ensure_dir",
]
