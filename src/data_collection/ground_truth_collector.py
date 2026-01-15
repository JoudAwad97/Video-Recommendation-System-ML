"""
Ground truth label collector from user interactions.

Collects user feedback (watches, clicks, likes, etc.) and converts
them to labeled training examples based on business rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict

from ..utils.logging_utils import get_logger
from .collection_config import (
    DataCollectionConfig,
    InteractionType,
    LabelValue,
)
from .inference_tracker import InferenceTracker, TrackedPrediction

logger = get_logger(__name__)


@dataclass
class UserFeedback:
    """User feedback event."""

    # Identifiers
    feedback_id: str
    user_id: int
    video_id: int

    # Linked inference (if known)
    inference_id: Optional[str] = None

    # Interaction details
    interaction_type: str = "impression"
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Engagement metrics
    watch_time_seconds: float = 0.0
    video_duration_seconds: float = 0.0
    watch_percentage: float = 0.0

    # Context
    session_id: Optional[str] = None
    device: str = ""
    position_in_list: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "inference_id": self.inference_id,
            "interaction_type": self.interaction_type,
            "timestamp": self.timestamp,
            "watch_time_seconds": self.watch_time_seconds,
            "video_duration_seconds": self.video_duration_seconds,
            "watch_percentage": self.watch_percentage,
            "session_id": self.session_id,
            "device": self.device,
            "position_in_list": self.position_in_list,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFeedback":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LabeledInteraction:
    """A labeled user-video interaction for training."""

    # Identifiers
    interaction_id: str
    inference_id: str
    user_id: int
    video_id: int

    # Label
    label: int  # 1 = positive, 0 = negative
    label_source: str  # "watch", "like", "dislike", etc.
    label_confidence: float = 1.0

    # Timestamps
    inference_timestamp: str = ""
    feedback_timestamp: str = ""
    labeled_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Features (from inference)
    user_features: Dict[str, Any] = field(default_factory=dict)
    video_features: Dict[str, Any] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)

    # Prediction info
    predicted_score: float = 0.0
    position_shown: int = 0
    model_version: str = ""
    experiment_id: Optional[str] = None

    # Engagement metrics
    watch_time_seconds: float = 0.0
    watch_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "inference_id": self.inference_id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "label": self.label,
            "label_source": self.label_source,
            "label_confidence": self.label_confidence,
            "inference_timestamp": self.inference_timestamp,
            "feedback_timestamp": self.feedback_timestamp,
            "labeled_at": self.labeled_at,
            "user_features": self.user_features,
            "video_features": self.video_features,
            "context_features": self.context_features,
            "predicted_score": self.predicted_score,
            "position_shown": self.position_shown,
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
            "watch_time_seconds": self.watch_time_seconds,
            "watch_percentage": self.watch_percentage,
        }


class GroundTruthCollector:
    """Collector for ground truth labels from user feedback.

    Processes user interactions and converts them to labeled examples
    using configurable business rules.

    Example:
        >>> collector = GroundTruthCollector(config, inference_tracker)
        >>> collector.record_feedback(user_feedback)
        >>> labeled = collector.get_labeled_interactions()
    """

    def __init__(
        self,
        config: DataCollectionConfig,
        inference_tracker: Optional[InferenceTracker] = None,
    ):
        """Initialize the ground truth collector.

        Args:
            config: Data collection configuration.
            inference_tracker: Optional inference tracker for linking.
        """
        self.config = config
        self.inference_tracker = inference_tracker
        self._feedback: List[UserFeedback] = []
        self._labeled_interactions: List[LabeledInteraction] = []
        self._feedback_counter = 0
        self._interaction_counter = 0

        # Index for deduplication
        self._feedback_by_inference: Dict[str, List[UserFeedback]] = defaultdict(list)

        # Storage
        self._storage_path = Path(config.local_storage_path) / "ground_truth"
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def record_feedback(
        self,
        user_id: int,
        video_id: int,
        interaction_type: str,
        inference_id: Optional[str] = None,
        watch_time_seconds: float = 0.0,
        video_duration_seconds: float = 0.0,
        session_id: Optional[str] = None,
        device: str = "",
        position_in_list: int = 0,
    ) -> UserFeedback:
        """Record user feedback.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            interaction_type: Type of interaction.
            inference_id: Optional linked inference ID.
            watch_time_seconds: Time spent watching.
            video_duration_seconds: Total video duration.
            session_id: Session identifier.
            device: User device.
            position_in_list: Position video was shown.

        Returns:
            Created UserFeedback.
        """
        self._feedback_counter += 1
        feedback_id = f"fb_{self._feedback_counter}"

        # Calculate watch percentage
        watch_percentage = 0.0
        if video_duration_seconds > 0:
            watch_percentage = watch_time_seconds / video_duration_seconds

        # Try to find linked inference if not provided
        if not inference_id and self.inference_tracker:
            prediction = self.inference_tracker.find_inference_for_feedback(
                user_id=user_id,
                video_id=video_id,
            )
            if prediction:
                inference_id = prediction.inference_id

        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            video_id=video_id,
            inference_id=inference_id,
            interaction_type=interaction_type,
            watch_time_seconds=watch_time_seconds,
            video_duration_seconds=video_duration_seconds,
            watch_percentage=watch_percentage,
            session_id=session_id,
            device=device,
            position_in_list=position_in_list,
        )

        self._feedback.append(feedback)

        if inference_id:
            self._feedback_by_inference[inference_id].append(feedback)
            if self.inference_tracker:
                self.inference_tracker.mark_feedback_received(
                    inference_id, video_id, interaction_type
                )

        logger.debug(
            f"Recorded {interaction_type} feedback for user {user_id}, "
            f"video {video_id}, inference {inference_id}"
        )

        return feedback

    def process_feedback_to_labels(
        self,
        feedback_list: Optional[List[UserFeedback]] = None,
    ) -> List[LabeledInteraction]:
        """Process feedback into labeled interactions.

        Args:
            feedback_list: Feedback to process. Uses all unprocessed if None.

        Returns:
            List of labeled interactions.
        """
        feedback_list = feedback_list or self._feedback
        new_labeled = []

        for feedback in feedback_list:
            if not feedback.inference_id:
                continue

            # Get the linked prediction
            prediction = None
            if self.inference_tracker:
                prediction = self.inference_tracker.get_prediction(feedback.inference_id)

            # Determine label
            try:
                interaction_type = InteractionType(feedback.interaction_type)
            except ValueError:
                interaction_type = InteractionType.IMPRESSION

            label_value = self.config.labeling_rules.get_label_for_interaction(
                interaction_type=interaction_type,
                watch_percentage=feedback.watch_percentage,
                watch_seconds=feedback.watch_time_seconds,
            )

            # Skip unknown labels unless configured otherwise
            if label_value == LabelValue.UNKNOWN:
                continue

            # Create labeled interaction
            self._interaction_counter += 1
            labeled = LabeledInteraction(
                interaction_id=f"labeled_{self._interaction_counter}",
                inference_id=feedback.inference_id,
                user_id=feedback.user_id,
                video_id=feedback.video_id,
                label=label_value.value,
                label_source=feedback.interaction_type,
                feedback_timestamp=feedback.timestamp,
                watch_time_seconds=feedback.watch_time_seconds,
                watch_percentage=feedback.watch_percentage,
                position_shown=feedback.position_in_list,
            )

            # Add features from prediction
            if prediction:
                labeled.inference_timestamp = prediction.timestamp
                labeled.user_features = prediction.user_features
                labeled.context_features = prediction.context_features
                labeled.predicted_score = prediction.video_scores.get(feedback.video_id, 0)
                labeled.model_version = prediction.model_version
                labeled.experiment_id = prediction.experiment_id

            new_labeled.append(labeled)
            self._labeled_interactions.append(labeled)

        logger.info(f"Processed {len(new_labeled)} labeled interactions")
        return new_labeled

    def get_labeled_interactions(
        self,
        include_positive: bool = True,
        include_negative: bool = True,
        min_confidence: float = 0.0,
    ) -> List[LabeledInteraction]:
        """Get labeled interactions with optional filtering.

        Args:
            include_positive: Include positive labels.
            include_negative: Include negative labels.
            min_confidence: Minimum label confidence.

        Returns:
            Filtered list of labeled interactions.
        """
        result = []

        for labeled in self._labeled_interactions:
            if labeled.label_confidence < min_confidence:
                continue

            if labeled.label == 1 and not include_positive:
                continue

            if labeled.label == 0 and not include_negative:
                continue

            result.append(labeled)

        return result

    def get_feedback_for_inference(
        self,
        inference_id: str,
    ) -> List[UserFeedback]:
        """Get all feedback for an inference.

        Args:
            inference_id: Inference identifier.

        Returns:
            List of feedback for this inference.
        """
        return self._feedback_by_inference.get(inference_id, [])

    def aggregate_feedback_for_video(
        self,
        inference_id: str,
        video_id: int,
    ) -> Dict[str, Any]:
        """Aggregate all feedback for a video in an inference.

        Args:
            inference_id: Inference identifier.
            video_id: Video identifier.

        Returns:
            Aggregated metrics.
        """
        feedback_list = [
            f for f in self._feedback_by_inference.get(inference_id, [])
            if f.video_id == video_id
        ]

        if not feedback_list:
            return {}

        interactions = [f.interaction_type for f in feedback_list]
        watch_times = [f.watch_time_seconds for f in feedback_list if f.watch_time_seconds > 0]

        return {
            "total_interactions": len(feedback_list),
            "interaction_types": list(set(interactions)),
            "was_clicked": "click" in interactions,
            "was_completed": "complete" in interactions,
            "was_liked": "like" in interactions,
            "was_disliked": "dislike" in interactions,
            "max_watch_time": max(watch_times) if watch_times else 0,
            "total_watch_time": sum(watch_times) if watch_times else 0,
        }

    def save_feedback(self, output_path: Optional[str] = None) -> str:
        """Save feedback to file.

        Args:
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        output_path = output_path or str(
            self._storage_path / f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        with open(output_path, "w") as f:
            for fb in self._feedback:
                f.write(fb.to_json() + "\n")

        logger.info(f"Saved {len(self._feedback)} feedback records to {output_path}")
        return output_path

    def save_labeled_interactions(self, output_path: Optional[str] = None) -> str:
        """Save labeled interactions to file.

        Args:
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        output_path = output_path or str(
            self._storage_path / f"labeled_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        with open(output_path, "w") as f:
            for labeled in self._labeled_interactions:
                f.write(json.dumps(labeled.to_dict()) + "\n")

        logger.info(f"Saved {len(self._labeled_interactions)} labeled interactions to {output_path}")
        return output_path

    def load_feedback(self, input_path: str) -> int:
        """Load feedback from file.

        Args:
            input_path: Input file path.

        Returns:
            Number of records loaded.
        """
        loaded = 0
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    fb = UserFeedback.from_dict(json.loads(line))
                    self._feedback.append(fb)
                    if fb.inference_id:
                        self._feedback_by_inference[fb.inference_id].append(fb)
                    loaded += 1

        logger.info(f"Loaded {loaded} feedback records from {input_path}")
        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics.

        Returns:
            Dictionary with stats.
        """
        positive_count = sum(1 for l in self._labeled_interactions if l.label == 1)
        negative_count = sum(1 for l in self._labeled_interactions if l.label == 0)

        interaction_types = defaultdict(int)
        for fb in self._feedback:
            interaction_types[fb.interaction_type] += 1

        return {
            "total_feedback": len(self._feedback),
            "total_labeled": len(self._labeled_interactions),
            "positive_labels": positive_count,
            "negative_labels": negative_count,
            "linked_to_inference": sum(1 for fb in self._feedback if fb.inference_id),
            "interaction_types": dict(interaction_types),
        }
