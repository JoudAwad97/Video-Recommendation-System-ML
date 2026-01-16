"""
Health check and circuit breaker utilities.

Provides:
- Health check framework for service dependencies
- Circuit breaker pattern for fault tolerance
- Graceful degradation support
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .logging_utils import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Health Check Framework
# =============================================================================

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "checked_at": self.checked_at,
        }


class HealthCheck(ABC):
    """Base class for health checks."""

    def __init__(self, name: str, timeout_ms: float = 5000):
        self.name = name
        self.timeout_ms = timeout_ms

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass

    def run(self) -> HealthCheckResult:
        """Run health check with timing."""
        start_time = time.time()
        try:
            result = self.check()
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
            )


class DynamoDBHealthCheck(HealthCheck):
    """Health check for DynamoDB."""

    def __init__(self, table_name: str, region: str = "us-east-2"):
        super().__init__(f"dynamodb:{table_name}")
        self.table_name = table_name
        self.region = region

    def check(self) -> HealthCheckResult:
        """Check DynamoDB table accessibility."""
        try:
            import boto3
            dynamodb = boto3.resource("dynamodb", region_name=self.region)
            table = dynamodb.Table(self.table_name)

            # Describe table to verify it exists and is accessible
            status = table.table_status

            if status == "ACTIVE":
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Table is active",
                    details={"table_status": status},
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Table status: {status}",
                    details={"table_status": status},
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis."""

    def __init__(self, host: str, port: int = 6379, password: str = ""):
        super().__init__(f"redis:{host}:{port}")
        self.host = host
        self.port = port
        self.password = password

    def check(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            import redis
            client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password or None,
                socket_timeout=5.0,
            )

            # PING command
            response = client.ping()

            if response:
                info = client.info("server")
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Redis is responsive",
                    details={
                        "redis_version": info.get("redis_version", "unknown"),
                    },
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="PING failed",
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class S3HealthCheck(HealthCheck):
    """Health check for S3 bucket."""

    def __init__(self, bucket_name: str, region: str = "us-east-2"):
        super().__init__(f"s3:{bucket_name}")
        self.bucket_name = bucket_name
        self.region = region

    def check(self) -> HealthCheckResult:
        """Check S3 bucket accessibility."""
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self.region)

            # Head bucket to verify access
            s3.head_bucket(Bucket=self.bucket_name)

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Bucket is accessible",
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class VectorStoreHealthCheck(HealthCheck):
    """Health check for vector store."""

    def __init__(self, vector_store):
        super().__init__("vector_store")
        self.vector_store = vector_store

    def check(self) -> HealthCheckResult:
        """Check vector store health."""
        try:
            # Check if store has data
            count = len(self.vector_store) if hasattr(self.vector_store, "__len__") else 0

            if count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Vector store is ready",
                    details={"vector_count": count},
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Vector store is empty",
                    details={"vector_count": 0},
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class ModelHealthCheck(HealthCheck):
    """Health check for ML models."""

    def __init__(self, model_name: str, model):
        super().__init__(f"model:{model_name}")
        self.model = model

    def check(self) -> HealthCheckResult:
        """Check model readiness."""
        try:
            if self.model is None:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Model not loaded",
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Model is ready",
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


@dataclass
class ServiceHealth:
    """Overall service health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
        }


class HealthChecker:
    """Health checker that runs multiple health checks."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.checks: List[HealthCheck] = []

    def add_check(self, check: HealthCheck) -> "HealthChecker":
        """Add a health check."""
        self.checks.append(check)
        return self

    def run_all(self) -> ServiceHealth:
        """Run all health checks and return overall status."""
        results = [check.run() for check in self.checks]

        # Determine overall status
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return ServiceHealth(
            status=overall_status,
            checks=results,
            version=self.version,
        )


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 2           # Successes to close from half-open
    timeout_seconds: float = 30.0        # Time before trying half-open
    half_open_max_calls: int = 3         # Max calls in half-open state


class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Implements the circuit breaker pattern to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Usage:
        cb = CircuitBreaker("redis_cache")

        @cb.protect
        def call_redis():
            return redis_client.get(key)

        # Or manually:
        if cb.allow_request():
            try:
                result = call_redis()
                cb.record_success()
            except Exception as e:
                cb.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[..., Any]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._update_state()
            return self._state

    def _update_state(self) -> None:
        """Update state based on timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    logger.info(
                        f"Circuit {self.name}: OPEN -> HALF_OPEN after {elapsed:.1f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            self._update_state()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure)")
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit {self.name}: CLOSED -> OPEN "
                        f"(failures: {self._failure_count})"
                    )
                    self._state = CircuitState.OPEN

    def protect(self, func: F) -> F:
        """Decorator to protect a function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                if self.fallback:
                    logger.debug(f"Circuit {self.name} open, using fallback")
                    return self.fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {self.name} is open")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                if self.fallback and self._state == CircuitState.OPEN:
                    return self.fallback(*args, **kwargs)
                raise

        return wrapper

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            self._update_state()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Graceful Degradation
# =============================================================================

class DegradedModeHandler:
    """Handles graceful degradation when services are unavailable.

    Provides fallback behavior when dependencies fail.
    """

    def __init__(self):
        self._degraded_services: Dict[str, bool] = {}
        self._lock = threading.RLock()

    def mark_degraded(self, service: str) -> None:
        """Mark a service as degraded."""
        with self._lock:
            self._degraded_services[service] = True
            logger.warning(f"Service {service} marked as degraded")

    def mark_healthy(self, service: str) -> None:
        """Mark a service as healthy."""
        with self._lock:
            self._degraded_services[service] = False
            logger.info(f"Service {service} marked as healthy")

    def is_degraded(self, service: str) -> bool:
        """Check if service is degraded."""
        with self._lock:
            return self._degraded_services.get(service, False)

    def get_degraded_services(self) -> List[str]:
        """Get list of degraded services."""
        with self._lock:
            return [s for s, degraded in self._degraded_services.items() if degraded]


# Global degraded mode handler
_degraded_handler = DegradedModeHandler()


def get_degraded_handler() -> DegradedModeHandler:
    """Get global degraded mode handler."""
    return _degraded_handler


def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable[..., Any]] = None,
    log_error: bool = True,
) -> Callable[[F], F]:
    """Decorator for graceful degradation with fallback.

    Args:
        fallback_value: Static value to return on failure.
        fallback_func: Function to call on failure.
        log_error: Whether to log the error.

    Usage:
        @with_fallback(fallback_value=[])
        def get_recommendations(user_id):
            return recommendation_service.get(user_id)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Function {func.__name__} failed, using fallback: {e}"
                    )
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper
    return decorator
