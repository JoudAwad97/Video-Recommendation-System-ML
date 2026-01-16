"""
Logging utilities for the recommendation system.

Provides:
- Structured JSON logging for production
- Human-readable text logging for development
- Request context tracking
- Sensitive field masking
- Performance timing decorators
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar
import threading


# Type variable for generic function wrapper
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Configuration
# =============================================================================

# Fields that should be masked in logs
SENSITIVE_FIELDS = frozenset({
    "password",
    "api_key",
    "apikey",
    "secret",
    "token",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "secret_key",
})

# Maximum length for logged values
MAX_VALUE_LENGTH = 1000


# =============================================================================
# Request Context
# =============================================================================

@dataclass
class RequestContext:
    """Context for tracking request-specific information."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


# Thread-local storage for request context
_context_local = threading.local()


def get_request_context() -> Optional[RequestContext]:
    """Get current request context."""
    return getattr(_context_local, "context", None)


def set_request_context(context: RequestContext) -> None:
    """Set current request context."""
    _context_local.context = context


def clear_request_context() -> None:
    """Clear current request context."""
    _context_local.context = None


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None,
    **extra
):
    """Context manager for request tracking.

    Usage:
        with request_context(user_id="123", path="/recommendations"):
            logger.info("Processing request")
    """
    ctx = RequestContext(
        request_id=request_id or str(uuid.uuid4()),
        user_id=user_id,
        path=path,
        method=method,
        extra=extra,
    )
    set_request_context(ctx)
    try:
        yield ctx
    finally:
        clear_request_context()


# =============================================================================
# JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for production use."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_request_id: bool = True,
        mask_sensitive: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_request_id = include_request_id
        self.mask_sensitive = mask_sensitive
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add request context
        ctx = get_request_context()
        if ctx and self.include_request_id:
            log_data["request_id"] = ctx.request_id
            if ctx.user_id:
                log_data["user_id"] = ctx.user_id
            if ctx.path:
                log_data["path"] = ctx.path
            if ctx.method:
                log_data["method"] = ctx.method
            log_data["elapsed_ms"] = round(ctx.elapsed_ms(), 2)

        # Add extra fields from record
        if hasattr(record, "extra"):
            extra = record.extra
            if self.mask_sensitive:
                extra = _mask_sensitive_data(extra)
            log_data["extra"] = extra

        # Add static extra fields
        log_data.update(self.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }

        # Add source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as colored text."""
        # Color codes
        color = self.COLORS.get(record.levelname, "") if self.use_colors else ""
        reset = self.COLORS["RESET"] if self.use_colors else ""

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Request context
        ctx_str = ""
        ctx = get_request_context()
        if ctx:
            parts = [f"req={ctx.request_id[:8]}"]
            if ctx.user_id:
                parts.append(f"user={ctx.user_id}")
            ctx_str = f" [{', '.join(parts)}]"

        # Build message
        msg = f"{timestamp} {color}{record.levelname:8}{reset} {record.name}{ctx_str} - {record.getMessage()}"

        # Add extra data if present
        if hasattr(record, "extra") and record.extra:
            extra = record.extra
            if isinstance(extra, dict):
                extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
                msg += f" | {extra_str}"

        # Add exception info
        if record.exc_info:
            msg += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return msg


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(
    name: str,
    level: Optional[int] = None,
    format_type: Optional[str] = None,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger (usually __name__).
        level: Logging level. If None, uses LOG_LEVEL env var or INFO.
        format_type: "json" or "text". If None, uses LOG_FORMAT env var or "json".

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if logger.handlers:
        return logger

    # Determine level
    if level is None:
        level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)

    # Determine format
    if format_type is None:
        format_type = os.environ.get("LOG_FORMAT", "json").lower()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    if format_type == "json":
        formatter = JSONFormatter(
            include_timestamp=True,
            include_request_id=True,
            mask_sensitive=True,
            extra_fields={
                "service": os.environ.get("SERVICE_NAME", "video-recommendation-service"),
                "environment": os.environ.get("ENVIRONMENT", "dev"),
            },
        )
    else:
        formatter = TextFormatter(use_colors=True)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that supports extra context."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message to add extra context."""
        extra = kwargs.get("extra", {})
        if self.extra:
            extra = {**self.extra, **extra}
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger_with_context(
    name: str,
    **context
) -> LoggerAdapter:
    """Get a logger with default context.

    Args:
        name: Name of the logger.
        **context: Default context to include in all log messages.

    Returns:
        LoggerAdapter with context.
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


# =============================================================================
# Utility Functions
# =============================================================================

def _mask_sensitive_data(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """Recursively mask sensitive fields in data."""
    if depth > max_depth:
        return "...TRUNCATED..."

    if isinstance(data, dict):
        return {
            k: "***MASKED***" if _is_sensitive_field(k) else _mask_sensitive_data(v, depth + 1)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return [_mask_sensitive_data(item, depth + 1) for item in data]
    elif isinstance(data, str) and len(data) > MAX_VALUE_LENGTH:
        return data[:MAX_VALUE_LENGTH] + f"...({len(data)} chars)"
    else:
        return data


def _is_sensitive_field(field_name: str) -> bool:
    """Check if field name indicates sensitive data."""
    name_lower = field_name.lower()
    return any(sensitive in name_lower for sensitive in SENSITIVE_FIELDS)


def log_extra(logger: logging.Logger, level: int, msg: str, **extra) -> None:
    """Log a message with extra structured data.

    Args:
        logger: Logger instance.
        level: Logging level.
        msg: Log message.
        **extra: Extra structured data to include.
    """
    record = logger.makeRecord(
        logger.name,
        level,
        "(unknown)",
        0,
        msg,
        (),
        None,
    )
    record.extra = extra
    logger.handle(record)


# =============================================================================
# Decorators
# =============================================================================

def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
    log_time: bool = True,
) -> Callable[[F], F]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use. If None, creates one from function module.
        level: Logging level for the messages.
        log_args: Whether to log function arguments.
        log_result: Whether to log function result.
        log_time: Whether to log execution time.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            # Build extra context
            extra = {"function": func_name}
            if log_args:
                extra["args_count"] = len(args)
                extra["kwargs_keys"] = list(kwargs.keys())

            # Log entry
            log_extra(logger, level, f"Entering {func_name}", **extra)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000

                # Log success
                extra = {"function": func_name, "status": "success"}
                if log_time:
                    extra["elapsed_ms"] = round(elapsed, 2)
                if log_result and result is not None:
                    extra["result_type"] = type(result).__name__

                log_extra(logger, level, f"Completed {func_name}", **extra)
                return result

            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                extra = {
                    "function": func_name,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                if log_time:
                    extra["elapsed_ms"] = round(elapsed, 2)

                log_extra(logger, logging.ERROR, f"Failed {func_name}", **extra)
                raise

        return wrapper
    return decorator


def timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use.
        level: Logging level.
        name: Custom name for the operation.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = name or func.__qualname__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                log_extra(
                    logger,
                    level,
                    f"{op_name} completed",
                    operation=op_name,
                    elapsed_ms=round(elapsed, 2),
                    status="success",
                )
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                log_extra(
                    logger,
                    logging.ERROR,
                    f"{op_name} failed",
                    operation=op_name,
                    elapsed_ms=round(elapsed, 2),
                    status="error",
                    error=str(e),
                )
                raise

        return wrapper
    return decorator


@contextmanager
def log_block(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **extra
):
    """Context manager to log a block of code with timing.

    Usage:
        with log_block(logger, "data_processing", user_id=123):
            process_data()
    """
    start_time = time.time()
    log_extra(logger, level, f"Starting {operation}", operation=operation, **extra)

    try:
        yield
        elapsed = (time.time() - start_time) * 1000
        log_extra(
            logger,
            level,
            f"Completed {operation}",
            operation=operation,
            elapsed_ms=round(elapsed, 2),
            status="success",
            **extra,
        )
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        log_extra(
            logger,
            logging.ERROR,
            f"Failed {operation}",
            operation=operation,
            elapsed_ms=round(elapsed, 2),
            status="error",
            error=str(e),
            **extra,
        )
        raise
