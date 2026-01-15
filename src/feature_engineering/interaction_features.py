"""
Interaction feature processor for deriving features from user-video interactions.

Processes interaction data to derive features like previously_watched_category
and determines positive/negative interaction labels.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from ..config.feature_config import (
    POSITIVE_INTERACTION_TYPES,
    DAYS_OF_WEEK,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class InteractionFeatureProcessor:
    """Process interaction data to derive features and labels.

    Handles:
    - Deriving previously_watched_category from interaction history
    - Determining positive/negative interaction labels
    - Extracting time-based features (day, hour)
    - Computing watch ratio for watch interactions

    Example:
        >>> processor = InteractionFeatureProcessor(min_watch_ratio=0.4)
        >>> processed_df = processor.process(interactions_df, videos_df)
    """

    def __init__(
        self,
        min_watch_ratio: float = 0.4,
        positive_interaction_types: Optional[List[str]] = None
    ):
        """Initialize the interaction processor.

        Args:
            min_watch_ratio: Minimum watch ratio for positive watch interaction.
            positive_interaction_types: List of interaction types considered positive.
        """
        self.min_watch_ratio = min_watch_ratio
        self.positive_interaction_types = positive_interaction_types or POSITIVE_INTERACTION_TYPES

    def process(
        self,
        interactions_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Process interactions and derive features.

        Args:
            interactions_df: DataFrame with interaction data.
            videos_df: DataFrame with video data (for category lookup).
            users_df: Optional DataFrame with user data.

        Returns:
            Processed DataFrame with derived features.
        """
        logger.info(f"Processing {len(interactions_df)} interactions...")

        df = interactions_df.copy()

        # Create video lookup for category
        video_category_map = dict(zip(videos_df["id"], videos_df["category"]))
        video_duration_map = dict(zip(videos_df["id"], videos_df["duration"]))

        # Derive previously_watched_category
        df["previously_watched_category"] = df["previously_watched_video"].apply(
            lambda x: video_category_map.get(x, "-") if pd.notna(x) else "-"
        )

        # Extract time features from timestamp
        df["interaction_datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["interaction_time_day"] = df["interaction_datetime"].dt.day_name().str.upper()
        df["interaction_time_hour"] = df["interaction_datetime"].dt.hour

        # Compute watch ratio for watch interactions
        df["watch_ratio"] = df.apply(
            lambda row: self._compute_watch_ratio(
                row["interaction_type"],
                row["interaction_value"],
                video_duration_map.get(row["video_id"], 300)
            ),
            axis=1
        )

        # Determine positive/negative label
        df["is_positive"] = df.apply(
            lambda row: self._is_positive_interaction(
                row["interaction_type"],
                row.get("watch_ratio", 0)
            ),
            axis=1
        )

        logger.info(f"Positive interactions: {df['is_positive'].sum()}")
        logger.info(f"Negative interactions: {(~df['is_positive']).sum()}")

        return df

    def _compute_watch_ratio(
        self,
        interaction_type: str,
        interaction_value: str,
        video_duration: int
    ) -> float:
        """Compute watch ratio from interaction value.

        Args:
            interaction_type: Type of interaction.
            interaction_value: Value string (e.g., "120 seconds").
            video_duration: Video duration in seconds.

        Returns:
            Watch ratio between 0 and 1.
        """
        if interaction_type != "watch" or video_duration == 0:
            return 0.0

        if pd.isna(interaction_value) or interaction_value == "-":
            return 0.0

        try:
            # Parse "X seconds" format
            watch_seconds = int(str(interaction_value).split()[0])
            return min(watch_seconds / video_duration, 1.0)
        except (ValueError, IndexError):
            return 0.0

    def _is_positive_interaction(
        self,
        interaction_type: str,
        watch_ratio: float
    ) -> bool:
        """Determine if interaction is positive.

        Args:
            interaction_type: Type of interaction.
            watch_ratio: Watch ratio (for watch interactions).

        Returns:
            True if interaction is positive.
        """
        # Explicit negative
        if interaction_type == "dislike":
            return False

        # Watch interaction requires minimum ratio
        if interaction_type == "watch":
            return watch_ratio >= self.min_watch_ratio

        # Impression is considered positive (light engagement)
        if interaction_type == "impression":
            return True

        # Other positive types
        return interaction_type in self.positive_interaction_types

    def get_positive_interactions(
        self,
        interactions_df: pd.DataFrame,
        videos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Get only positive interactions.

        Args:
            interactions_df: DataFrame with interaction data.
            videos_df: DataFrame with video data.

        Returns:
            DataFrame with only positive interactions.
        """
        processed = self.process(interactions_df, videos_df)
        return processed[processed["is_positive"]].copy()

    def get_negative_interactions(
        self,
        interactions_df: pd.DataFrame,
        videos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Get only negative interactions.

        Args:
            interactions_df: DataFrame with interaction data.
            videos_df: DataFrame with video data.

        Returns:
            DataFrame with only negative interactions.
        """
        processed = self.process(interactions_df, videos_df)
        return processed[~processed["is_positive"]].copy()

    def generate_negative_samples(
        self,
        positive_interactions: pd.DataFrame,
        all_video_ids: List[int],
        negative_ratio: int = 3
    ) -> pd.DataFrame:
        """Generate negative samples for ranker training.

        For each positive interaction, samples random videos that the user
        didn't interact with as negative samples.

        Args:
            positive_interactions: DataFrame with positive interactions.
            all_video_ids: List of all available video IDs.
            negative_ratio: Number of negatives per positive.

        Returns:
            DataFrame with negative samples.
        """
        logger.info(f"Generating negative samples with ratio 1:{negative_ratio}...")

        video_id_set = set(all_video_ids)
        negative_samples = []

        # Group by user to get their interaction history
        user_interactions = positive_interactions.groupby("user_id")["video_id"].apply(set).to_dict()

        for _, row in positive_interactions.iterrows():
            user_id = row["user_id"]
            interacted_videos = user_interactions.get(user_id, set())

            # Get videos user hasn't interacted with
            candidate_videos = list(video_id_set - interacted_videos)

            if len(candidate_videos) == 0:
                continue

            # Sample negative videos
            n_samples = min(negative_ratio, len(candidate_videos))
            sampled_videos = np.random.choice(candidate_videos, size=n_samples, replace=False)

            # Create negative samples with same context as positive
            for neg_video_id in sampled_videos:
                neg_sample = {
                    "user_id": user_id,
                    "video_id": neg_video_id,
                    "interaction_type": "negative_sample",
                    "interaction_value": "-",
                    "previously_watched_category": row.get("previously_watched_category", "-"),
                    "interaction_time_day": row.get("interaction_time_day", "MONDAY"),
                    "interaction_time_hour": row.get("interaction_time_hour", 12),
                    "device": row.get("device", "Mobile"),
                    "is_positive": False,
                }
                negative_samples.append(neg_sample)

        neg_df = pd.DataFrame(negative_samples)
        logger.info(f"Generated {len(neg_df)} negative samples")

        return neg_df

    def create_ranker_dataset(
        self,
        interactions_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        users_df: pd.DataFrame,
        channels_df: pd.DataFrame,
        negative_ratio: int = 3
    ) -> pd.DataFrame:
        """Create complete ranker training dataset with pos/neg samples.

        Args:
            interactions_df: DataFrame with interaction data.
            videos_df: DataFrame with video data.
            users_df: DataFrame with user data.
            channels_df: DataFrame with channel data.
            negative_ratio: Number of negatives per positive.

        Returns:
            Complete ranker training dataset.
        """
        # Process interactions
        processed = self.process(interactions_df, videos_df)

        # Get positive samples
        positive = processed[processed["is_positive"]].copy()
        positive["label"] = 1

        # Generate negative samples
        negative = self.generate_negative_samples(
            positive,
            all_video_ids=videos_df["id"].tolist(),
            negative_ratio=negative_ratio
        )
        negative["label"] = 0

        # Combine
        combined = pd.concat([positive, negative], ignore_index=True)

        # Join with user features
        user_cols = ["id", "country_code", "preferred_language", "age"]
        user_cols = [c for c in user_cols if c in users_df.columns]
        combined = combined.merge(
            users_df[user_cols].rename(columns={
                "id": "user_id",
                "country_code": "country",
                "preferred_language": "user_language"
            }),
            on="user_id",
            how="left"
        )

        # Join with video features
        video_cols = ["id", "category", "child_categories", "duration", "popularity",
                      "language", "view_count", "like_count", "comment_count", "channel_id"]
        video_cols = [c for c in video_cols if c in videos_df.columns]
        combined = combined.merge(
            videos_df[video_cols].rename(columns={
                "id": "video_id",
                "duration": "video_duration",
                "language": "video_language"
            }),
            on="video_id",
            how="left"
        )

        # Join with channel features
        if "channel_id" in combined.columns:
            channel_cols = ["id", "subscriber_count"]
            channel_cols = [c for c in channel_cols if c in channels_df.columns]
            combined = combined.merge(
                channels_df[channel_cols].rename(columns={
                    "id": "channel_id",
                    "subscriber_count": "channel_subscriber_count"
                }),
                on="channel_id",
                how="left"
            )

        logger.info(f"Created ranker dataset with {len(combined)} samples "
                    f"({positive.shape[0]} positive, {negative.shape[0]} negative)")

        return combined

    def __repr__(self) -> str:
        return f"InteractionFeatureProcessor(min_watch_ratio={self.min_watch_ratio})"
