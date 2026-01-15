"""
Synthetic data generator for testing the recommendation system pipeline.

Generates realistic fake data matching the schemas defined in Phase 1,
with proper distributions for categorical and numeric fields.
"""

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

from .schemas import User, Video, Channel, Interaction
from ..config.feature_config import (
    POPULARITY_LEVELS,
    DEVICE_TYPES,
    DAYS_OF_WEEK,
    INTERACTION_TYPES,
)


class SyntheticDataGenerator:
    """Generate synthetic data for testing the recommendation pipeline.

    Generates users, videos, channels, and interactions with realistic
    distributions matching the expected production data patterns.
    """

    # Country codes with relative population weights
    COUNTRIES = [
        ("US", 30), ("UK", 10), ("DE", 10), ("FR", 8), ("CA", 8),
        ("AU", 5), ("BR", 5), ("IN", 8), ("JP", 5), ("MX", 5),
        ("ES", 3), ("IT", 3)
    ]

    # Languages with relative usage weights
    LANGUAGES = [
        ("English", 50), ("German", 10), ("French", 8), ("Spanish", 10),
        ("Portuguese", 5), ("Japanese", 5), ("Hindi", 5), ("Italian", 3),
        ("Chinese", 4)
    ]

    # Video categories with relative frequency weights
    CATEGORIES = [
        ("Technology", 15), ("Gaming", 15), ("Music", 12), ("Entertainment", 12),
        ("Education", 10), ("Sports", 8), ("News", 5), ("Cooking", 5),
        ("Travel", 5), ("Fitness", 5), ("Fashion", 4), ("Science", 4)
    ]

    # Child categories mapping
    CHILD_CATEGORIES = {
        "Technology": ["AI", "Software", "Hardware", "Mobile", "Web", "Gadgets"],
        "Gaming": ["RPG", "FPS", "Strategy", "Sports", "Indie", "Mobile"],
        "Music": ["Pop", "Rock", "Jazz", "Classical", "Hip-Hop", "Electronic"],
        "Entertainment": ["Comedy", "Drama", "Reality", "Talk Show", "Documentary"],
        "Education": ["Tutorial", "Course", "Lecture", "How-To", "Language"],
        "Sports": ["Football", "Basketball", "Soccer", "Tennis", "Golf"],
        "News": ["World", "Politics", "Business", "Tech", "Entertainment"],
        "Cooking": ["Recipes", "Baking", "Healthy", "Quick Meals", "Cuisine"],
        "Travel": ["Adventure", "Budget", "Luxury", "City", "Nature"],
        "Fitness": ["Cardio", "Weight-Loss", "Yoga", "HIIT", "Strength"],
        "Fashion": ["Style", "Beauty", "Trends", "DIY", "Reviews"],
        "Science": ["Physics", "Biology", "Chemistry", "Space", "Environment"],
    }

    # Sample tags per category
    CATEGORY_TAGS = {
        "Technology": ["tech", "software", "programming", "AI", "machine learning", "coding", "developer", "app"],
        "Gaming": ["gaming", "gameplay", "walkthrough", "guide", "tips", "streamer", "esports", "RPG"],
        "Music": ["music", "song", "artist", "album", "concert", "cover", "remix", "instrumental"],
        "Entertainment": ["entertainment", "funny", "comedy", "vlog", "reaction", "challenge", "trends"],
        "Education": ["education", "tutorial", "learn", "course", "beginner", "advanced", "tips"],
        "Sports": ["sports", "highlights", "game", "player", "team", "championship", "training"],
        "News": ["news", "breaking", "update", "analysis", "report", "interview", "current events"],
        "Cooking": ["cooking", "recipe", "food", "chef", "kitchen", "meal prep", "healthy eating"],
        "Travel": ["travel", "adventure", "destination", "vlog", "tips", "budget", "explore"],
        "Fitness": ["fitness", "workout", "exercise", "health", "gym", "cardio", "strength"],
        "Fashion": ["fashion", "style", "outfit", "trends", "beauty", "makeup", "haul"],
        "Science": ["science", "experiment", "research", "discovery", "facts", "explain"],
    }

    # Gender options
    GENDERS = ["Male", "Female", "Other", "Prefer not to say"]

    # Cities by country
    CITIES = {
        "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "UK": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"],
        "DE": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
        "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"],
        "CA": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
        "AU": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
        "BR": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Fortaleza"],
        "IN": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"],
        "JP": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya"],
        "MX": ["Mexico City", "Guadalajara", "Monterrey", "Cancun", "Tijuana"],
        "ES": ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"],
        "IT": ["Rome", "Milan", "Naples", "Turin", "Florence"],
    }

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _weighted_choice(self, choices: List[Tuple[str, int]]) -> str:
        """Select from weighted choices.

        Args:
            choices: List of (value, weight) tuples.

        Returns:
            Selected value.
        """
        values, weights = zip(*choices)
        return random.choices(values, weights=weights, k=1)[0]

    def _generate_email(self, name: str) -> str:
        """Generate email from name."""
        domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "email.com"]
        username = name.lower().replace(" ", ".") + str(random.randint(1, 999))
        return f"{username}@{random.choice(domains)}"

    def _generate_username(self, name: str) -> str:
        """Generate username from name."""
        base = name.lower().replace(" ", "_")
        return f"{base}{random.randint(1, 9999)}"

    def _generate_video_title(self, category: str) -> str:
        """Generate realistic video title for category."""
        titles = {
            "Technology": [
                "Introduction to {topic}",
                "How to Build a {topic} App",
                "{topic} Tutorial for Beginners",
                "Advanced {topic} Techniques",
                "{topic} in 2024: Complete Guide",
            ],
            "Gaming": [
                "{topic} Boss Guide",
                "{topic} Walkthrough Part {num}",
                "Best {topic} Tips and Tricks",
                "{topic} Gameplay - No Commentary",
                "How to Beat {topic}",
            ],
            "Music": [
                "{topic} - Official Music Video",
                "{topic} Cover by {artist}",
                "{topic} Tutorial - Learn Piano",
                "Best {topic} Songs Playlist",
                "{topic} Live Performance",
            ],
            "Education": [
                "Learn {topic} in {num} Minutes",
                "{topic} Explained Simply",
                "Complete {topic} Course",
                "{topic} for Beginners",
                "Understanding {topic}",
            ],
        }

        topics = {
            "Technology": ["Machine Learning", "React", "Python", "AWS", "Docker", "Kubernetes"],
            "Gaming": ["Elden Ring", "Zelda", "Mario", "Final Fantasy", "Dark Souls"],
            "Music": ["Jazz Piano", "Guitar Solo", "Pop Hits", "Classical", "Rock Anthem"],
            "Education": ["Calculus", "Physics", "History", "Programming", "Data Science"],
        }

        category_titles = titles.get(category, ["Amazing {topic} Video", "{topic} Content", "Best {topic}"])
        category_topics = topics.get(category, ["General", "Tutorial", "Guide", "Review"])

        template = random.choice(category_titles)
        topic = random.choice(category_topics)

        title = template.format(
            topic=topic,
            num=random.randint(1, 50),
            artist=f"User{random.randint(1, 1000)}"
        )
        return title

    def _generate_tags(self, category: str, num_tags: int = 4) -> str:
        """Generate pipe-separated tags for a video."""
        base_tags = self.CATEGORY_TAGS.get(category, ["video", "content"])
        selected_tags = random.sample(base_tags, min(num_tags, len(base_tags)))
        return "|".join(selected_tags)

    def generate_users(self, num_users: int) -> pd.DataFrame:
        """Generate synthetic user data.

        Args:
            num_users: Number of users to generate.

        Returns:
            DataFrame with user data.
        """
        users = []
        first_names = ["John", "Jane", "Mike", "Sarah", "David", "Emma", "Chris", "Lisa",
                       "Tom", "Amy", "James", "Emily", "Robert", "Anna", "William", "Maria"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                      "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson"]

        for i in range(1, num_users + 1):
            country = self._weighted_choice(self.COUNTRIES)
            name = f"{random.choice(first_names)} {random.choice(last_names)}"

            user = {
                "id": i,
                "name": name,
                "email": self._generate_email(name),
                "username": self._generate_username(name),
                "age": int(np.random.normal(32, 12)),  # Normal distribution around 32
                "gender": random.choice(self.GENDERS),
                "country_code": country,
                "city": random.choice(self.CITIES.get(country, ["Unknown"])),
                "preferred_language": self._weighted_choice(self.LANGUAGES),
                "is_active": random.random() > 0.1,  # 90% active
                "is_verified": random.random() > 0.7,  # 30% verified
                "timezone": f"UTC{random.choice(['-8', '-5', '+0', '+1', '+5', '+9'])}",
            }
            # Clamp age to reasonable range
            user["age"] = max(13, min(80, user["age"]))
            users.append(user)

        return pd.DataFrame(users)

    def generate_channels(self, num_channels: int) -> pd.DataFrame:
        """Generate synthetic channel data.

        Args:
            num_channels: Number of channels to generate.

        Returns:
            DataFrame with channel data.
        """
        channels = []
        base_date = datetime(2015, 1, 1)

        for i in range(1, num_channels + 1):
            # Subscriber count follows power law distribution
            subscriber_count = int(np.random.pareto(1.5) * 1000) + 100

            channel = {
                "id": i,
                "name": f"Channel_{i}_{random.choice(['Official', 'Pro', 'Gaming', 'Tech', 'Music'])}",
                "subscriber_count": subscriber_count,
                "video_count": random.randint(10, 500),
                "total_views": subscriber_count * random.randint(50, 500),
                "created_at": base_date + timedelta(days=random.randint(0, 3000)),
            }
            channels.append(channel)

        return pd.DataFrame(channels)

    def generate_videos(
        self,
        num_videos: int,
        channels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate synthetic video data.

        Args:
            num_videos: Number of videos to generate.
            channels_df: DataFrame of channels (for foreign key).

        Returns:
            DataFrame with video data.
        """
        videos = []
        channel_ids = channels_df["id"].tolist()

        for i in range(1, num_videos + 1):
            category = self._weighted_choice(self.CATEGORIES)
            child_cats = self.CHILD_CATEGORIES.get(category, ["General"])

            # Duration follows log-normal distribution (most videos 3-15 min)
            duration = int(np.random.lognormal(mean=6, sigma=1))
            duration = max(30, min(7200, duration))  # 30 sec to 2 hours

            # View counts follow power law
            view_count = int(np.random.pareto(1.2) * 100) + 10
            like_ratio = random.uniform(0.02, 0.08)
            comment_ratio = random.uniform(0.001, 0.01)

            # Determine popularity based on view count
            if view_count > 1000000:
                popularity = "viral"
            elif view_count > 100000:
                popularity = "high"
            elif view_count > 10000:
                popularity = "medium"
            else:
                popularity = "low"

            video = {
                "id": i,
                "channel_id": random.choice(channel_ids),
                "duration": duration,
                "manual_tags": self._generate_tags(category, num_tags=3),
                "title": self._generate_video_title(category),
                "description": f"A {category.lower()} video about various topics. Watch and enjoy!",
                "augmented_tags": self._generate_tags(category, num_tags=4),
                "category": category,
                "child_categories": "|".join(random.sample(child_cats, min(2, len(child_cats)))),
                "view_count": view_count,
                "like_count": int(view_count * like_ratio),
                "comment_count": int(view_count * comment_ratio),
                "language": self._weighted_choice(self.LANGUAGES),
                "popularity": popularity,
            }
            videos.append(video)

        return pd.DataFrame(videos)

    def generate_interactions(
        self,
        num_interactions: int,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate synthetic user-video interactions.

        Args:
            num_interactions: Number of interactions to generate.
            users_df: DataFrame of users.
            videos_df: DataFrame of videos.

        Returns:
            DataFrame with interaction data.
        """
        interactions = []
        user_ids = users_df["id"].tolist()
        video_ids = videos_df["id"].tolist()
        video_durations = dict(zip(videos_df["id"], videos_df["duration"]))

        # Create interaction weights (watch and impression more common)
        interaction_weights = [
            ("like", 15),
            ("dislike", 3),
            ("watch", 40),
            ("click", 20),
            ("impression", 15),
            ("comment", 5),
            ("share", 2),
        ]

        base_timestamp = int(datetime(2024, 1, 1).timestamp())

        for _ in range(num_interactions):
            user_id = random.choice(user_ids)
            video_id = random.choice(video_ids)
            interaction_type = self._weighted_choice(interaction_weights)
            video_duration = video_durations.get(video_id, 300)

            # Generate interaction value based on type
            if interaction_type == "watch":
                # Watch duration as percentage of video
                watch_ratio = random.betavariate(2, 3)  # Skewed towards shorter watches
                watch_seconds = int(video_duration * watch_ratio)
                interaction_value = f"{watch_seconds} seconds"
            elif interaction_type == "impression":
                interaction_value = f"{random.randint(1, 10)} seconds"
            elif interaction_type == "comment":
                comments = ["Great video!", "Very helpful", "Thanks for sharing",
                            "Interesting content", "Good video", "Nice work"]
                interaction_value = random.choice(comments)
            else:
                interaction_value = "-"

            # Previously watched video (30% have previous)
            previously_watched = random.choice(video_ids) if random.random() > 0.3 else None

            interaction = {
                "user_id": user_id,
                "video_id": video_id,
                "interaction_type": interaction_type,
                "interaction_value": interaction_value,
                "location_lat": round(random.uniform(-90, 90), 4),
                "location_long": round(random.uniform(-180, 180), 4),
                "previously_watched_video": previously_watched,
                "session_id": str(uuid.uuid4()),
                "timestamp": base_timestamp + random.randint(0, 30 * 24 * 3600),  # Within 30 days
                "device": random.choice(DEVICE_TYPES),
            }
            interactions.append(interaction)

        return pd.DataFrame(interactions)

    def generate_all(
        self,
        num_users: int = 1000,
        num_channels: int = 100,
        num_videos: int = 500,
        num_interactions: int = 10000
    ) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets.

        Args:
            num_users: Number of users to generate.
            num_channels: Number of channels to generate.
            num_videos: Number of videos to generate.
            num_interactions: Number of interactions to generate.

        Returns:
            Dictionary with all DataFrames.
        """
        print(f"Generating {num_users} users...")
        users_df = self.generate_users(num_users)

        print(f"Generating {num_channels} channels...")
        channels_df = self.generate_channels(num_channels)

        print(f"Generating {num_videos} videos...")
        videos_df = self.generate_videos(num_videos, channels_df)

        print(f"Generating {num_interactions} interactions...")
        interactions_df = self.generate_interactions(num_interactions, users_df, videos_df)

        return {
            "users": users_df,
            "channels": channels_df,
            "videos": videos_df,
            "interactions": interactions_df,
        }
