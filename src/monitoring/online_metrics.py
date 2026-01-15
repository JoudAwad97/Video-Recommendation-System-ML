"""
Online metrics collection for business KPIs.

Collects and aggregates metrics like CTR, watch time, completions,
and explicit user feedback to measure recommendation quality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

from ..utils.logging_utils import get_logger
from .monitoring_config import OnlineMetricsConfig

logger = get_logger(__name__)


@dataclass
class UserInteraction:
    """A single user interaction with a recommended video."""

    # Identifiers
    interaction_id: str
    user_id: int
    video_id: int
    recommendation_request_id: str

    # Timestamps
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Interaction type
    interaction_type: str = "impression"  # impression, click, watch, complete, like, dislike

    # Engagement metrics
    watch_time_seconds: float = 0.0
    video_duration_seconds: float = 0.0
    completion_rate: float = 0.0

    # Context
    position_in_list: int = 0
    model_version: str = ""
    experiment_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "recommendation_request_id": self.recommendation_request_id,
            "timestamp": self.timestamp,
            "interaction_type": self.interaction_type,
            "watch_time_seconds": self.watch_time_seconds,
            "video_duration_seconds": self.video_duration_seconds,
            "completion_rate": self.completion_rate,
            "position_in_list": self.position_in_list,
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
        }


@dataclass
class OnlineMetricsReport:
    """Report of online metrics for a time window."""

    report_id: str
    generated_at: str
    window_start: str
    window_end: str

    # Volume metrics
    total_impressions: int = 0
    total_clicks: int = 0
    total_completions: int = 0
    unique_users: int = 0
    unique_videos: int = 0

    # Engagement metrics
    click_through_rate: float = 0.0
    completion_rate: float = 0.0
    avg_watch_time_seconds: float = 0.0
    total_watch_time_hours: float = 0.0

    # Position metrics
    avg_click_position: float = 0.0
    click_position_distribution: Dict[int, int] = field(default_factory=dict)

    # Explicit feedback
    total_likes: int = 0
    total_dislikes: int = 0
    like_rate: float = 0.0

    # Per-model metrics (for A/B testing)
    metrics_by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-experiment metrics
    metrics_by_experiment: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "total_impressions": self.total_impressions,
            "total_clicks": self.total_clicks,
            "total_completions": self.total_completions,
            "unique_users": self.unique_users,
            "unique_videos": self.unique_videos,
            "click_through_rate": self.click_through_rate,
            "completion_rate": self.completion_rate,
            "avg_watch_time_seconds": self.avg_watch_time_seconds,
            "total_watch_time_hours": self.total_watch_time_hours,
            "avg_click_position": self.avg_click_position,
            "like_rate": self.like_rate,
            "metrics_by_model": self.metrics_by_model,
            "metrics_by_experiment": self.metrics_by_experiment,
        }


class OnlineMetricsCollector:
    """Collector for online business metrics.

    Tracks user interactions with recommendations to measure:
    - Click-through rate (CTR)
    - Watch time and completion rate
    - Explicit feedback (likes/dislikes)
    - Position bias effects

    Example:
        >>> collector = OnlineMetricsCollector(config)
        >>> collector.record_impression(user_id, video_id, request_id, position)
        >>> collector.record_click(user_id, video_id, request_id)
        >>> collector.record_watch(user_id, video_id, watch_time, duration)
        >>> report = collector.generate_report(window_hours=24)
    """

    def __init__(self, config: OnlineMetricsConfig):
        """Initialize the metrics collector.

        Args:
            config: Online metrics configuration.
        """
        self.config = config
        self._interactions: List[UserInteraction] = []
        self._interaction_counter = 0

        # Index for quick lookups
        self._impressions_by_request: Dict[str, Set[int]] = defaultdict(set)
        self._clicks_by_request: Dict[str, Set[int]] = defaultdict(set)

        # Storage path
        self._storage_path = Path(config.metrics_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def record_impression(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        position: int,
        model_version: str = "",
        experiment_id: Optional[str] = None,
        video_duration: float = 0.0,
    ) -> str:
        """Record a video impression (shown to user).

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            position: Position in recommendation list (1-indexed).
            model_version: Model version that generated the recommendation.
            experiment_id: A/B test experiment ID.
            video_duration: Duration of the video in seconds.

        Returns:
            Interaction ID.
        """
        interaction = self._create_interaction(
            user_id=user_id,
            video_id=video_id,
            request_id=request_id,
            interaction_type="impression",
            position=position,
            model_version=model_version,
            experiment_id=experiment_id,
            video_duration=video_duration,
        )

        self._interactions.append(interaction)
        self._impressions_by_request[request_id].add(video_id)

        return interaction.interaction_id

    def record_click(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        position: Optional[int] = None,
    ) -> str:
        """Record a video click.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            position: Position in list (if known).

        Returns:
            Interaction ID.
        """
        interaction = self._create_interaction(
            user_id=user_id,
            video_id=video_id,
            request_id=request_id,
            interaction_type="click",
            position=position or 0,
        )

        self._interactions.append(interaction)
        self._clicks_by_request[request_id].add(video_id)

        return interaction.interaction_id

    def record_watch(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        watch_time_seconds: float,
        video_duration_seconds: float,
    ) -> str:
        """Record video watch time.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            watch_time_seconds: Time spent watching.
            video_duration_seconds: Total video duration.

        Returns:
            Interaction ID.
        """
        completion_rate = (
            watch_time_seconds / video_duration_seconds
            if video_duration_seconds > 0 else 0
        )

        interaction = self._create_interaction(
            user_id=user_id,
            video_id=video_id,
            request_id=request_id,
            interaction_type="watch",
            watch_time=watch_time_seconds,
            video_duration=video_duration_seconds,
            completion_rate=completion_rate,
        )

        self._interactions.append(interaction)

        return interaction.interaction_id

    def record_completion(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        video_duration_seconds: float,
    ) -> str:
        """Record video completion.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            video_duration_seconds: Total video duration.

        Returns:
            Interaction ID.
        """
        interaction = self._create_interaction(
            user_id=user_id,
            video_id=video_id,
            request_id=request_id,
            interaction_type="complete",
            watch_time=video_duration_seconds,
            video_duration=video_duration_seconds,
            completion_rate=1.0,
        )

        self._interactions.append(interaction)

        return interaction.interaction_id

    def record_feedback(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        feedback_type: str,  # "like" or "dislike"
    ) -> str:
        """Record explicit user feedback.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            feedback_type: Type of feedback ("like" or "dislike").

        Returns:
            Interaction ID.
        """
        interaction = self._create_interaction(
            user_id=user_id,
            video_id=video_id,
            request_id=request_id,
            interaction_type=feedback_type,
        )

        self._interactions.append(interaction)

        return interaction.interaction_id

    def _create_interaction(
        self,
        user_id: int,
        video_id: int,
        request_id: str,
        interaction_type: str,
        position: int = 0,
        model_version: str = "",
        experiment_id: Optional[str] = None,
        watch_time: float = 0.0,
        video_duration: float = 0.0,
        completion_rate: float = 0.0,
    ) -> UserInteraction:
        """Create a user interaction record.

        Args:
            user_id: User identifier.
            video_id: Video identifier.
            request_id: Recommendation request ID.
            interaction_type: Type of interaction.
            position: Position in recommendation list.
            model_version: Model version.
            experiment_id: Experiment ID.
            watch_time: Watch time in seconds.
            video_duration: Video duration in seconds.
            completion_rate: Completion rate (0-1).

        Returns:
            User interaction record.
        """
        self._interaction_counter += 1
        return UserInteraction(
            interaction_id=f"int_{self._interaction_counter}",
            user_id=user_id,
            video_id=video_id,
            recommendation_request_id=request_id,
            interaction_type=interaction_type,
            position_in_list=position,
            model_version=model_version,
            experiment_id=experiment_id,
            watch_time_seconds=watch_time,
            video_duration_seconds=video_duration,
            completion_rate=completion_rate,
        )

    def generate_report(
        self,
        window_hours: int = 24,
        end_time: Optional[datetime] = None,
    ) -> OnlineMetricsReport:
        """Generate online metrics report for a time window.

        Args:
            window_hours: Size of time window in hours.
            end_time: End of window (defaults to now).

        Returns:
            Online metrics report.
        """
        import uuid

        end_time = end_time or datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)

        # Filter interactions in window
        window_interactions = [
            i for i in self._interactions
            if start_time <= datetime.fromisoformat(i.timestamp) <= end_time
        ]

        report = OnlineMetricsReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow().isoformat(),
            window_start=start_time.isoformat(),
            window_end=end_time.isoformat(),
        )

        if not window_interactions:
            return report

        # Count by interaction type
        impressions = [i for i in window_interactions if i.interaction_type == "impression"]
        clicks = [i for i in window_interactions if i.interaction_type == "click"]
        watches = [i for i in window_interactions if i.interaction_type == "watch"]
        completions = [i for i in window_interactions if i.interaction_type == "complete"]
        likes = [i for i in window_interactions if i.interaction_type == "like"]
        dislikes = [i for i in window_interactions if i.interaction_type == "dislike"]

        # Volume metrics
        report.total_impressions = len(impressions)
        report.total_clicks = len(clicks)
        report.total_completions = len(completions)
        report.unique_users = len(set(i.user_id for i in window_interactions))
        report.unique_videos = len(set(i.video_id for i in window_interactions))

        # CTR
        if report.total_impressions > 0:
            report.click_through_rate = report.total_clicks / report.total_impressions

        # Watch metrics
        all_watches = watches + completions
        if all_watches:
            watch_times = [i.watch_time_seconds for i in all_watches]
            report.avg_watch_time_seconds = np.mean(watch_times)
            report.total_watch_time_hours = sum(watch_times) / 3600

            completion_rates = [i.completion_rate for i in all_watches if i.completion_rate > 0]
            if completion_rates:
                report.completion_rate = np.mean(completion_rates)

        # Position metrics
        if clicks:
            positions = [i.position_in_list for i in clicks if i.position_in_list > 0]
            if positions:
                report.avg_click_position = np.mean(positions)
                report.click_position_distribution = dict(
                    zip(*np.unique(positions, return_counts=True))
                )

        # Feedback metrics
        report.total_likes = len(likes)
        report.total_dislikes = len(dislikes)
        total_feedback = report.total_likes + report.total_dislikes
        if total_feedback > 0:
            report.like_rate = report.total_likes / total_feedback

        # Per-model metrics
        report.metrics_by_model = self._compute_metrics_by_group(
            window_interactions, "model_version"
        )

        # Per-experiment metrics
        report.metrics_by_experiment = self._compute_metrics_by_group(
            window_interactions, "experiment_id"
        )

        logger.info(
            f"Generated report: {report.total_impressions} impressions, "
            f"CTR={report.click_through_rate:.2%}"
        )

        return report

    def _compute_metrics_by_group(
        self,
        interactions: List[UserInteraction],
        group_field: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics grouped by a field.

        Args:
            interactions: List of interactions.
            group_field: Field to group by.

        Returns:
            Dictionary of metrics by group.
        """
        groups: Dict[str, List[UserInteraction]] = defaultdict(list)

        for interaction in interactions:
            group_value = getattr(interaction, group_field, None)
            if group_value:
                groups[group_value].append(interaction)

        metrics_by_group = {}
        for group_name, group_interactions in groups.items():
            impressions = sum(1 for i in group_interactions if i.interaction_type == "impression")
            clicks = sum(1 for i in group_interactions if i.interaction_type == "click")
            completions = sum(1 for i in group_interactions if i.interaction_type == "complete")

            metrics_by_group[group_name] = {
                "impressions": impressions,
                "clicks": clicks,
                "ctr": clicks / impressions if impressions > 0 else 0,
                "completions": completions,
            }

        return metrics_by_group

    def save_interactions(self, output_path: Optional[str] = None) -> str:
        """Save interactions to file.

        Args:
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        output_path = output_path or str(
            self._storage_path / f"interactions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        with open(output_path, "w") as f:
            for interaction in self._interactions:
                f.write(json.dumps(interaction.to_dict()) + "\n")

        logger.info(f"Saved {len(self._interactions)} interactions to {output_path}")
        return output_path

    def load_interactions(self, input_path: str) -> int:
        """Load interactions from file.

        Args:
            input_path: Input file path.

        Returns:
            Number of interactions loaded.
        """
        loaded = 0
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    interaction = UserInteraction(**data)
                    self._interactions.append(interaction)
                    loaded += 1

        logger.info(f"Loaded {loaded} interactions from {input_path}")
        return loaded

    def clear_interactions(self) -> None:
        """Clear all stored interactions."""
        self._interactions = []
        self._impressions_by_request.clear()
        self._clicks_by_request.clear()
        self._interaction_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "total_interactions": len(self._interactions),
            "interaction_types": dict(
                zip(*np.unique(
                    [i.interaction_type for i in self._interactions],
                    return_counts=True
                ))
            ) if self._interactions else {},
            "unique_requests": len(self._impressions_by_request),
        }
