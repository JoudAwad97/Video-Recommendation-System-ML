"""
Lambda function entry points for Video Recommendation System.

This module provides clean separation of Lambda handlers from core logic.
Each handler is designed for minimal cold start time with lazy imports.

Available handlers:
- recommendations: Real-time recommendation serving
- preprocessing: Data preprocessing pipeline
- training: Model training orchestration
- evaluation: Model evaluation
- deployment: Model deployment to production
- feature_ingestion: Feature store ingestion
- event_processing: User interaction event processing
- merge_job: Prediction-label merge jobs
"""

from typing import TYPE_CHECKING

# Lazy imports for Lambda cold start optimization
if TYPE_CHECKING:
    from .recommendations import handler as recommendations_handler
    from .preprocessing import handler as preprocessing_handler
    from .training import two_tower_handler, ranker_handler
    from .evaluation import handler as evaluation_handler
    from .deployment import handler as deployment_handler
    from .feature_ingestion import handler as feature_ingestion_handler
    from .event_processing import handler as event_processing_handler
    from .merge_job import handler as merge_job_handler


def __getattr__(name: str):
    """Lazy import Lambda handlers."""
    if name == "recommendations_handler":
        from .recommendations import handler
        return handler
    elif name == "preprocessing_handler":
        from .preprocessing import handler
        return handler
    elif name == "two_tower_handler":
        from .training import two_tower_handler
        return two_tower_handler
    elif name == "ranker_handler":
        from .training import ranker_handler
        return ranker_handler
    elif name == "evaluation_handler":
        from .evaluation import handler
        return handler
    elif name == "deployment_handler":
        from .deployment import handler
        return handler
    elif name == "feature_ingestion_handler":
        from .feature_ingestion import handler
        return handler
    elif name == "event_processing_handler":
        from .event_processing import handler
        return handler
    elif name == "merge_job_handler":
        from .merge_job import handler
        return handler

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "recommendations_handler",
    "preprocessing_handler",
    "two_tower_handler",
    "ranker_handler",
    "evaluation_handler",
    "deployment_handler",
    "feature_ingestion_handler",
    "event_processing_handler",
    "merge_job_handler",
]
