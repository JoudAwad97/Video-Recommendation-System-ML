"""
Data schemas for the video recommendation system.

Defines dataclasses for User, Video, Channel, and Interaction entities
matching the database schemas from Phase 1.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class User:
    """User entity schema.

    Attributes:
        id: Primary key user ID.
        name: User's full name.
        email: User's email address.
        username: User's username.
        age: User's age in years.
        gender: User's gender.
        country_code: ISO country code.
        city: User's city.
        preferred_language: User's preferred language.
        is_active: Whether the user account is active.
        is_verified: Whether the user is verified.
        timezone: User's timezone.
    """

    id: int
    name: str
    email: str
    username: str
    age: int
    gender: str
    country_code: str
    city: str
    preferred_language: str
    is_active: bool = True
    is_verified: bool = False
    timezone: str = "UTC"


@dataclass
class Video:
    """Video entity schema.

    Attributes:
        id: Primary key video ID.
        channel_id: Foreign key to channel.
        duration: Video duration in seconds.
        manual_tags: User-added tags (pipe-separated).
        title: Video title.
        description: Video description.
        augmented_tags: Auto-generated tags (pipe-separated).
        category: Main category.
        child_categories: Sub-categories (pipe-separated).
        view_count: Total view count.
        like_count: Total like count.
        comment_count: Total comment count.
        language: Video language.
    """

    id: int
    channel_id: int
    duration: int  # in seconds
    manual_tags: str
    title: str
    description: str
    augmented_tags: str
    category: str
    child_categories: str
    view_count: int
    like_count: int
    comment_count: int
    language: str


@dataclass
class Channel:
    """Channel entity schema (video author).

    Attributes:
        id: Primary key channel ID.
        name: Channel name.
        subscriber_count: Total subscriber count.
        video_count: Number of videos uploaded.
        total_views: Cumulative view count across all videos.
        created_at: Channel creation timestamp.
    """

    id: int
    name: str
    subscriber_count: int
    video_count: int
    total_views: int
    created_at: datetime


@dataclass
class Interaction:
    """User-Video interaction entity schema.

    Attributes:
        user_id: Foreign key to user.
        video_id: Foreign key to video.
        interaction_type: Type of interaction (like, impression, watch, click, comment).
        interaction_value: Value associated with interaction (duration, comment text, etc.).
        location_lat: Latitude coordinate.
        location_long: Longitude coordinate.
        previously_watched_video: ID of previously watched video.
        session_id: Session UUID.
        timestamp: Unix timestamp of interaction.
        device: Device type (Mobile, Desktop, Laptop, Tablet).
    """

    user_id: int
    video_id: int
    interaction_type: str
    interaction_value: Optional[str]
    location_lat: float
    location_long: float
    previously_watched_video: Optional[int]
    session_id: str
    timestamp: int
    device: str


@dataclass
class TwoTowerSample:
    """Sample for Two-Tower model training (positive pairs only).

    Attributes:
        user_id: User ID.
        country: User's country.
        user_language: User's preferred language.
        age: User's age.
        previously_watched_category: Category of previously watched video.
        video_id: Video ID.
        category: Video category.
        title: Video title.
        video_duration: Video duration in seconds.
        popularity: Popularity level.
        video_language: Video language.
        tags: Video tags (pipe-separated).
    """

    # User features
    user_id: int
    country: str
    user_language: str
    age: int
    previously_watched_category: str

    # Video features
    video_id: int
    category: str
    title: str
    video_duration: int
    popularity: str
    video_language: str
    tags: str


@dataclass
class RankerSample:
    """Sample for Ranker model training (with labels).

    Attributes:
        All user and video features plus additional context features.
        label: 1 for positive interaction, 0 for negative.
    """

    # User features
    user_id: int
    country: str
    user_language: str
    age: int

    # Video features
    video_id: int
    category: str
    child_categories: str
    video_duration: int
    popularity: str
    tags: str
    comment_count: int
    like_count: int
    view_count: int
    video_language: str
    channel_subscriber_count: int

    # Context features
    interaction_time_day: str
    interaction_time_hour: int
    device: str

    # Label
    label: int
