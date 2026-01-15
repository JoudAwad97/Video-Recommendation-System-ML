"""
Inference tracking service for linking predictions to user feedback.

Each inference gets a unique ID that's tracked through the entire
feedback loop to enable joining predictions with ground truth labels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from collections import defaultdict

from ..utils.logging_utils import get_logger
from .collection_config import DataCollectionConfig

logger = get_logger(__name__)


@dataclass
class TrackedPrediction:
    """A prediction with tracking information."""

    # Tracking IDs
    inference_id: str
    request_id: str

    # User and content
    user_id: int
    recommended_video_ids: List[int]

    # Timestamps
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    expires_at: Optional[str] = None

    # Model info
    model_version: str = ""
    experiment_id: Optional[str] = None

    # Scores for each video
    video_scores: Dict[int, float] = field(default_factory=dict)

    # Context
    user_features: Dict[str, Any] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inference_id": self.inference_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "recommended_video_ids": self.recommended_video_ids,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
            "video_scores": self.video_scores,
            "user_features": self.user_features,
            "context_features": self.context_features,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPrediction":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TrackedPrediction":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class InferenceEvent:
    """An event associated with an inference (for auditing)."""

    event_id: str
    inference_id: str
    event_type: str  # "created", "feedback_received", "labeled", "expired"
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceTracker:
    """Service for tracking inferences through the feedback loop.

    Each recommendation request gets a unique inference_id that:
    1. Is returned to the client application
    2. Is stored with the prediction details
    3. Is included in user feedback events
    4. Enables joining predictions with labels

    Example:
        >>> tracker = InferenceTracker(config)
        >>> prediction = tracker.track_inference(
        ...     user_id=123,
        ...     video_ids=[1, 2, 3],
        ...     scores={1: 0.9, 2: 0.8, 3: 0.7},
        ... )
        >>> # Client receives prediction.inference_id
        >>> # Later, when feedback arrives:
        >>> tracker.get_prediction(inference_id)
    """

    def __init__(self, config: DataCollectionConfig):
        """Initialize the inference tracker.

        Args:
            config: Data collection configuration.
        """
        self.config = config
        self._predictions: Dict[str, TrackedPrediction] = {}
        self._predictions_by_user: Dict[int, Set[str]] = defaultdict(set)
        self._events: List[InferenceEvent] = []
        self._event_counter = 0

        # Storage
        self._storage_path = Path(config.local_storage_path) / "inferences"
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def track_inference(
        self,
        user_id: int,
        video_ids: List[int],
        scores: Optional[Dict[int, float]] = None,
        request_id: Optional[str] = None,
        model_version: str = "",
        experiment_id: Optional[str] = None,
        user_features: Optional[Dict[str, Any]] = None,
        context_features: Optional[Dict[str, Any]] = None,
    ) -> TrackedPrediction:
        """Track a new inference.

        Args:
            user_id: User who received the recommendations.
            video_ids: List of recommended video IDs.
            scores: Optional scores for each video.
            request_id: Optional request identifier.
            model_version: Model version used.
            experiment_id: A/B test experiment ID.
            user_features: User features used for inference.
            context_features: Context features (device, time, etc.).

        Returns:
            TrackedPrediction with unique inference_id.
        """
        inference_id = self._generate_inference_id()
        request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"

        # Calculate expiry
        ttl = timedelta(hours=self.config.inference_ttl_hours)
        expires_at = (datetime.utcnow() + ttl).isoformat()

        prediction = TrackedPrediction(
            inference_id=inference_id,
            request_id=request_id,
            user_id=user_id,
            recommended_video_ids=video_ids,
            expires_at=expires_at,
            model_version=model_version,
            experiment_id=experiment_id,
            video_scores=scores or {},
            user_features=user_features or {},
            context_features=context_features or {},
        )

        # Store prediction
        self._predictions[inference_id] = prediction
        self._predictions_by_user[user_id].add(inference_id)

        # Log event
        self._log_event(inference_id, "created", {
            "user_id": user_id,
            "num_videos": len(video_ids),
        })

        logger.debug(f"Tracked inference {inference_id} for user {user_id}")

        return prediction

    def _generate_inference_id(self) -> str:
        """Generate a unique inference ID.

        Returns:
            Unique inference ID string.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = uuid.uuid4().hex[:8]
        return f"inf_{timestamp}_{random_part}"

    def get_prediction(self, inference_id: str) -> Optional[TrackedPrediction]:
        """Get a tracked prediction by inference ID.

        Args:
            inference_id: Inference identifier.

        Returns:
            TrackedPrediction or None if not found/expired.
        """
        prediction = self._predictions.get(inference_id)

        if prediction and prediction.expires_at:
            # Check expiry
            expires = datetime.fromisoformat(prediction.expires_at)
            if datetime.utcnow() > expires:
                self._log_event(inference_id, "expired")
                return None

        return prediction

    def get_predictions_for_user(
        self,
        user_id: int,
        limit: int = 100,
    ) -> List[TrackedPrediction]:
        """Get recent predictions for a user.

        Args:
            user_id: User identifier.
            limit: Maximum number of predictions.

        Returns:
            List of tracked predictions.
        """
        inference_ids = self._predictions_by_user.get(user_id, set())
        predictions = []

        for inf_id in inference_ids:
            pred = self.get_prediction(inf_id)
            if pred:
                predictions.append(pred)

        # Sort by timestamp, most recent first
        predictions.sort(key=lambda p: p.timestamp, reverse=True)

        return predictions[:limit]

    def find_inference_for_feedback(
        self,
        user_id: int,
        video_id: int,
        feedback_timestamp: Optional[datetime] = None,
    ) -> Optional[TrackedPrediction]:
        """Find the inference that led to a feedback event.

        Searches for the most recent prediction that included this video
        for this user, within the attribution window.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            feedback_timestamp: When the feedback occurred.

        Returns:
            TrackedPrediction or None if not found.
        """
        feedback_timestamp = feedback_timestamp or datetime.utcnow()
        predictions = self.get_predictions_for_user(user_id)

        for pred in predictions:
            # Check if video was in recommendations
            if video_id not in pred.recommended_video_ids:
                continue

            # Check time window
            pred_time = datetime.fromisoformat(pred.timestamp)
            time_diff = (feedback_timestamp - pred_time).total_seconds() / 3600

            max_delay = self.config.labeling_rules.max_label_delay_hours
            if time_diff <= max_delay:
                return pred

        return None

    def mark_feedback_received(
        self,
        inference_id: str,
        video_id: int,
        interaction_type: str,
    ) -> None:
        """Mark that feedback was received for an inference.

        Args:
            inference_id: Inference identifier.
            video_id: Video that received feedback.
            interaction_type: Type of interaction.
        """
        self._log_event(inference_id, "feedback_received", {
            "video_id": video_id,
            "interaction_type": interaction_type,
        })

    def _log_event(
        self,
        inference_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an event for an inference.

        Args:
            inference_id: Inference identifier.
            event_type: Type of event.
            metadata: Additional event data.
        """
        self._event_counter += 1
        event = InferenceEvent(
            event_id=f"evt_{self._event_counter}",
            inference_id=inference_id,
            event_type=event_type,
            metadata=metadata or {},
        )
        self._events.append(event)

    def cleanup_expired(self) -> int:
        """Remove expired inferences.

        Returns:
            Number of inferences removed.
        """
        now = datetime.utcnow()
        expired_ids = []

        for inf_id, pred in self._predictions.items():
            if pred.expires_at:
                expires = datetime.fromisoformat(pred.expires_at)
                if now > expires:
                    expired_ids.append(inf_id)

        for inf_id in expired_ids:
            pred = self._predictions.pop(inf_id)
            self._predictions_by_user[pred.user_id].discard(inf_id)
            self._log_event(inf_id, "expired")

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired inferences")

        return len(expired_ids)

    def save_predictions(self, output_path: Optional[str] = None) -> str:
        """Save predictions to file.

        Args:
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        output_path = output_path or str(
            self._storage_path / f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        with open(output_path, "w") as f:
            for pred in self._predictions.values():
                f.write(pred.to_json() + "\n")

        logger.info(f"Saved {len(self._predictions)} predictions to {output_path}")
        return output_path

    def load_predictions(self, input_path: str) -> int:
        """Load predictions from file.

        Args:
            input_path: Input file path.

        Returns:
            Number of predictions loaded.
        """
        loaded = 0
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    pred = TrackedPrediction.from_json(line)
                    self._predictions[pred.inference_id] = pred
                    self._predictions_by_user[pred.user_id].add(pred.inference_id)
                    loaded += 1

        logger.info(f"Loaded {loaded} predictions from {input_path}")
        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "total_predictions": len(self._predictions),
            "unique_users": len(self._predictions_by_user),
            "total_events": len(self._events),
        }
