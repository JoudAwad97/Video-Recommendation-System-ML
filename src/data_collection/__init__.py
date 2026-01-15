"""
Data collection module for video recommendation system.

This module provides components for collecting predictions and ground truth labels:
- Inference ID tracking for linking predictions to feedback
- Ground truth label collection from user interactions
- Prediction-label merge jobs for training data generation
- Feedback loop pipeline for continuous model improvement
"""

from .collection_config import (
    DataCollectionConfig,
    LabelingRulesConfig,
    MergeJobConfig,
)
from .inference_tracker import (
    InferenceTracker,
    InferenceEvent,
    TrackedPrediction,
)
from .ground_truth_collector import (
    GroundTruthCollector,
    UserFeedback,
    LabeledInteraction,
)
from .merge_job import (
    PredictionLabelMerger,
    MergedRecord,
    MergeResult,
)
from .feedback_loop import (
    FeedbackLoopPipeline,
    RetrainingDataset,
)

__all__ = [
    # Config
    "DataCollectionConfig",
    "LabelingRulesConfig",
    "MergeJobConfig",
    # Inference Tracking
    "InferenceTracker",
    "InferenceEvent",
    "TrackedPrediction",
    # Ground Truth
    "GroundTruthCollector",
    "UserFeedback",
    "LabeledInteraction",
    # Merge Job
    "PredictionLabelMerger",
    "MergedRecord",
    "MergeResult",
    # Feedback Loop
    "FeedbackLoopPipeline",
    "RetrainingDataset",
]
