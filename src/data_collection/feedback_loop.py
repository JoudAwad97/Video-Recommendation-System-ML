"""
Feedback loop pipeline for continuous model improvement.

Orchestrates the full feedback loop:
1. Collect predictions with inference IDs
2. Gather ground truth from user interactions
3. Merge predictions with labels
4. Generate retraining datasets
5. Trigger model retraining
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np

from ..utils.logging_utils import get_logger
from .collection_config import DataCollectionConfig
from .inference_tracker import InferenceTracker, TrackedPrediction
from .ground_truth_collector import GroundTruthCollector, LabeledInteraction
from .merge_job import PredictionLabelMerger, MergedRecord, MergeResult

logger = get_logger(__name__)


@dataclass
class RetrainingDataset:
    """A dataset prepared for model retraining."""

    dataset_id: str
    created_at: str

    # Data paths
    train_path: str = ""
    validation_path: str = ""
    test_path: str = ""

    # Statistics
    total_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    unique_users: int = 0
    unique_videos: int = 0

    # Time range
    data_start_date: str = ""
    data_end_date: str = ""

    # Quality metrics
    avg_label_confidence: float = 0.0
    label_sources: Dict[str, int] = field(default_factory=dict)

    # Model info
    base_model_version: str = ""
    target_model_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "test_path": self.test_path,
            "total_samples": self.total_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "unique_users": self.unique_users,
            "unique_videos": self.unique_videos,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "avg_label_confidence": self.avg_label_confidence,
            "label_sources": self.label_sources,
            "base_model_version": self.base_model_version,
            "target_model_version": self.target_model_version,
        }


class FeedbackLoopPipeline:
    """Orchestrates the full feedback loop for model improvement.

    The pipeline:
    1. Tracks inferences as they occur
    2. Collects user feedback linked to inference IDs
    3. Merges predictions with labels
    4. Generates balanced retraining datasets
    5. Can trigger model retraining

    Example:
        >>> pipeline = FeedbackLoopPipeline(config)
        >>>
        >>> # Track inference
        >>> prediction = pipeline.track_inference(user_id, video_ids, scores)
        >>>
        >>> # Record feedback (in real system, from event stream)
        >>> pipeline.record_feedback(user_id, video_id, "click", prediction.inference_id)
        >>>
        >>> # Periodically, generate retraining data
        >>> dataset = pipeline.generate_retraining_dataset()
    """

    def __init__(self, config: DataCollectionConfig):
        """Initialize the feedback loop pipeline.

        Args:
            config: Data collection configuration.
        """
        self.config = config

        # Initialize components
        self.inference_tracker = InferenceTracker(config)
        self.ground_truth_collector = GroundTruthCollector(
            config, self.inference_tracker
        )
        self.merger = PredictionLabelMerger(config.merge_job)

        # Callbacks
        self._retraining_callbacks: List[Callable[[RetrainingDataset], None]] = []

        # State
        self._dataset_counter = 0

        # Storage
        self._output_path = Path(config.retraining_data_path)
        self._output_path.mkdir(parents=True, exist_ok=True)

    def track_inference(
        self,
        user_id: int,
        video_ids: List[int],
        scores: Optional[Dict[int, float]] = None,
        model_version: str = "",
        experiment_id: Optional[str] = None,
        user_features: Optional[Dict[str, Any]] = None,
        context_features: Optional[Dict[str, Any]] = None,
    ) -> TrackedPrediction:
        """Track a new inference.

        Args:
            user_id: User identifier.
            video_ids: Recommended video IDs.
            scores: Prediction scores.
            model_version: Model version.
            experiment_id: Experiment ID.
            user_features: User features.
            context_features: Context features.

        Returns:
            Tracked prediction with inference_id.
        """
        return self.inference_tracker.track_inference(
            user_id=user_id,
            video_ids=video_ids,
            scores=scores,
            model_version=model_version,
            experiment_id=experiment_id,
            user_features=user_features,
            context_features=context_features,
        )

    def record_feedback(
        self,
        user_id: int,
        video_id: int,
        interaction_type: str,
        inference_id: Optional[str] = None,
        watch_time_seconds: float = 0.0,
        video_duration_seconds: float = 0.0,
        **kwargs,
    ) -> None:
        """Record user feedback.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            interaction_type: Type of interaction.
            inference_id: Optional linked inference ID.
            watch_time_seconds: Watch time.
            video_duration_seconds: Video duration.
            **kwargs: Additional feedback attributes.
        """
        self.ground_truth_collector.record_feedback(
            user_id=user_id,
            video_id=video_id,
            interaction_type=interaction_type,
            inference_id=inference_id,
            watch_time_seconds=watch_time_seconds,
            video_duration_seconds=video_duration_seconds,
            **kwargs,
        )

    def process_labels(self) -> List[LabeledInteraction]:
        """Process feedback into labels.

        Returns:
            List of newly labeled interactions.
        """
        return self.ground_truth_collector.process_feedback_to_labels()

    def merge_predictions_and_labels(
        self,
        min_samples: Optional[int] = None,
    ) -> Optional[MergeResult]:
        """Merge predictions with labels.

        Args:
            min_samples: Minimum samples required.

        Returns:
            MergeResult or None if insufficient data.
        """
        # Get predictions and labels
        predictions = list(self.inference_tracker._predictions.values())
        labeled = self.ground_truth_collector.get_labeled_interactions()

        min_samples = min_samples or self.config.min_samples_for_retraining

        if len(labeled) < min_samples:
            logger.info(
                f"Insufficient data: {len(labeled)}/{min_samples} samples"
            )
            return None

        merged_records, result = self.merger.merge(predictions, labeled)
        return result

    def generate_retraining_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        balance_labels: bool = True,
        base_model_version: str = "",
    ) -> Optional[RetrainingDataset]:
        """Generate a retraining dataset.

        Args:
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            balance_labels: Whether to balance positive/negative labels.
            base_model_version: Current model version being replaced.

        Returns:
            RetrainingDataset or None if insufficient data.
        """
        # Process any pending feedback
        self.process_labels()

        # Get all data
        predictions = list(self.inference_tracker._predictions.values())
        labeled = self.ground_truth_collector.get_labeled_interactions()

        if len(labeled) < self.config.min_samples_for_retraining:
            logger.info(
                f"Insufficient data: {len(labeled)}/{self.config.min_samples_for_retraining}"
            )
            return None

        # Merge predictions with labels
        merged_records, merge_result = self.merger.merge(predictions, labeled)

        if not merged_records:
            logger.warning("No merged records after joining")
            return None

        # Convert to DataFrame
        df = self.merger.to_dataframe(merged_records)

        # Balance labels if requested
        if balance_labels and self.config.positive_negative_ratio > 0:
            df = self._balance_dataset(df, self.config.positive_negative_ratio)

        # Split into train/val/test
        train_df, val_df, test_df = self._split_dataset(
            df, train_ratio, val_ratio, test_ratio
        )

        # Create dataset
        self._dataset_counter += 1
        dataset_id = f"retrain_dataset_{self._dataset_counter}"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        train_path = str(self._output_path / f"{dataset_id}_train_{timestamp}.parquet")
        val_path = str(self._output_path / f"{dataset_id}_val_{timestamp}.parquet")
        test_path = str(self._output_path / f"{dataset_id}_test_{timestamp}.parquet")

        # Save datasets
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Compute statistics
        label_sources = df["label_source"].value_counts().to_dict()
        timestamps = pd.to_datetime(df["inference_timestamp"])

        dataset = RetrainingDataset(
            dataset_id=dataset_id,
            created_at=datetime.utcnow().isoformat(),
            train_path=train_path,
            validation_path=val_path,
            test_path=test_path,
            total_samples=len(df),
            positive_samples=int((df["label"] == 1).sum()),
            negative_samples=int((df["label"] == 0).sum()),
            unique_users=df["user_id"].nunique(),
            unique_videos=df["video_id"].nunique(),
            data_start_date=timestamps.min().isoformat() if len(timestamps) > 0 else "",
            data_end_date=timestamps.max().isoformat() if len(timestamps) > 0 else "",
            label_sources=label_sources,
            base_model_version=base_model_version,
        )

        # Save metadata
        metadata_path = self._output_path / f"{dataset_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        logger.info(
            f"Generated retraining dataset {dataset_id}: "
            f"{len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )

        # Trigger callbacks
        for callback in self._retraining_callbacks:
            try:
                callback(dataset)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return dataset

    def _balance_dataset(
        self,
        df: pd.DataFrame,
        ratio: float,
    ) -> pd.DataFrame:
        """Balance positive and negative samples.

        Args:
            df: Input DataFrame.
            ratio: Target positive/negative ratio.

        Returns:
            Balanced DataFrame.
        """
        positive_df = df[df["label"] == 1]
        negative_df = df[df["label"] == 0]

        n_positive = len(positive_df)
        n_negative = len(negative_df)

        if n_positive == 0 or n_negative == 0:
            return df

        target_negative = int(n_positive / ratio)

        if n_negative > target_negative:
            # Downsample negatives
            negative_df = negative_df.sample(n=target_negative, random_state=42)
        elif n_negative < target_negative:
            # Downsample positives to match
            target_positive = int(n_negative * ratio)
            if n_positive > target_positive:
                positive_df = positive_df.sample(n=target_positive, random_state=42)

        balanced_df = pd.concat([positive_df, negative_df], ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def _split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple:
        """Split dataset into train/val/test.

        Args:
            df: Input DataFrame.
            train_ratio: Training fraction.
            val_ratio: Validation fraction.
            test_ratio: Test fraction.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def register_retraining_callback(
        self,
        callback: Callable[[RetrainingDataset], None],
    ) -> None:
        """Register a callback to be called when retraining data is ready.

        Args:
            callback: Function to call with RetrainingDataset.
        """
        self._retraining_callbacks.append(callback)

    def cleanup(self) -> Dict[str, int]:
        """Clean up expired data.

        Returns:
            Dictionary with cleanup counts.
        """
        expired = self.inference_tracker.cleanup_expired()
        return {"expired_inferences": expired}

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with stats from all components.
        """
        return {
            "inference_tracker": self.inference_tracker.get_stats(),
            "ground_truth_collector": self.ground_truth_collector.get_stats(),
            "datasets_generated": self._dataset_counter,
        }

    def save_state(self, output_dir: str) -> None:
        """Save pipeline state for recovery.

        Args:
            output_dir: Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.inference_tracker.save_predictions(
            str(output_dir / "predictions.jsonl")
        )
        self.ground_truth_collector.save_feedback(
            str(output_dir / "feedback.jsonl")
        )
        self.ground_truth_collector.save_labeled_interactions(
            str(output_dir / "labeled.jsonl")
        )

        logger.info(f"Saved pipeline state to {output_dir}")

    def load_state(self, input_dir: str) -> None:
        """Load pipeline state from saved files.

        Args:
            input_dir: Input directory.
        """
        input_dir = Path(input_dir)

        predictions_path = input_dir / "predictions.jsonl"
        if predictions_path.exists():
            self.inference_tracker.load_predictions(str(predictions_path))

        feedback_path = input_dir / "feedback.jsonl"
        if feedback_path.exists():
            self.ground_truth_collector.load_feedback(str(feedback_path))

        logger.info(f"Loaded pipeline state from {input_dir}")
