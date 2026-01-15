"""
Configuration for data collection components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class InteractionType(Enum):
    """Types of user interactions."""

    IMPRESSION = "impression"
    CLICK = "click"
    WATCH = "watch"
    COMPLETE = "complete"
    LIKE = "like"
    DISLIKE = "dislike"
    SHARE = "share"
    SKIP = "skip"


class LabelValue(Enum):
    """Possible label values."""

    POSITIVE = 1
    NEGATIVE = 0
    UNKNOWN = -1


@dataclass
class LabelingRulesConfig:
    """Configuration for auto-labeling user interactions.

    Defines business rules for converting interactions to labels.
    """

    # Watch time thresholds
    min_watch_percentage_positive: float = 0.4  # >40% watch = positive
    min_watch_percentage_negative: float = 0.1  # <10% watch = negative
    min_watch_seconds_positive: float = 30.0  # At least 30s for positive

    # Interaction type weights
    click_is_positive: bool = True
    complete_is_positive: bool = True
    like_is_positive: bool = True
    dislike_is_negative: bool = True
    share_is_positive: bool = True
    skip_is_negative: bool = True

    # Time-based rules
    max_label_delay_hours: int = 24  # Only label within 24h of inference
    session_timeout_minutes: int = 30  # Session boundary for attribution

    def get_label_for_interaction(
        self,
        interaction_type: InteractionType,
        watch_percentage: float = 0.0,
        watch_seconds: float = 0.0,
    ) -> LabelValue:
        """Determine label based on interaction type and metrics.

        Args:
            interaction_type: Type of user interaction.
            watch_percentage: Percentage of video watched (0-1).
            watch_seconds: Total seconds watched.

        Returns:
            Label value (POSITIVE, NEGATIVE, or UNKNOWN).
        """
        if interaction_type == InteractionType.LIKE:
            return LabelValue.POSITIVE if self.like_is_positive else LabelValue.UNKNOWN

        if interaction_type == InteractionType.DISLIKE:
            return LabelValue.NEGATIVE if self.dislike_is_negative else LabelValue.UNKNOWN

        if interaction_type == InteractionType.COMPLETE:
            return LabelValue.POSITIVE if self.complete_is_positive else LabelValue.UNKNOWN

        if interaction_type == InteractionType.SHARE:
            return LabelValue.POSITIVE if self.share_is_positive else LabelValue.UNKNOWN

        if interaction_type == InteractionType.SKIP:
            return LabelValue.NEGATIVE if self.skip_is_negative else LabelValue.UNKNOWN

        if interaction_type == InteractionType.CLICK:
            return LabelValue.POSITIVE if self.click_is_positive else LabelValue.UNKNOWN

        if interaction_type == InteractionType.WATCH:
            # Apply watch percentage rules
            if (watch_percentage >= self.min_watch_percentage_positive and
                    watch_seconds >= self.min_watch_seconds_positive):
                return LabelValue.POSITIVE
            elif watch_percentage < self.min_watch_percentage_negative:
                return LabelValue.NEGATIVE
            return LabelValue.UNKNOWN

        return LabelValue.UNKNOWN


@dataclass
class MergeJobConfig:
    """Configuration for prediction-label merge jobs."""

    # Storage paths
    predictions_path: str = "data/predictions"
    ground_truth_path: str = "data/ground_truth"
    merged_output_path: str = "data/merged"

    # Join settings
    join_key: str = "inference_id"
    join_window_hours: int = 24  # How long to wait for labels

    # Output settings
    output_format: str = "parquet"  # parquet, jsonl, csv
    partition_by: List[str] = field(default_factory=lambda: ["date"])

    # Filtering
    min_interactions_per_inference: int = 1
    require_explicit_feedback: bool = False  # Require like/dislike

    # AWS settings (for Athena)
    athena_database: str = "video_recommendations"
    athena_workgroup: str = "primary"
    s3_output_location: str = ""


@dataclass
class DataCollectionConfig:
    """Main configuration for data collection."""

    # Sub-configurations
    labeling_rules: LabelingRulesConfig = field(default_factory=LabelingRulesConfig)
    merge_job: MergeJobConfig = field(default_factory=MergeJobConfig)

    # Inference tracking
    inference_ttl_hours: int = 48  # How long to keep inference records
    batch_size: int = 1000

    # Storage settings
    storage_type: str = "local"  # local, s3, dynamodb
    local_storage_path: str = "data/collection"
    s3_bucket: str = ""
    s3_prefix: str = "data-collection"
    dynamodb_table: str = "inference_tracking"

    # Retraining settings
    min_samples_for_retraining: int = 10000
    retraining_data_path: str = "data/retraining"
    include_negative_samples: bool = True
    positive_negative_ratio: float = 1.0  # 1:1 ratio

    # Quality filters
    min_label_confidence: float = 0.0
    exclude_bot_traffic: bool = True
    min_user_history_length: int = 0
