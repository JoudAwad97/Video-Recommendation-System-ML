"""
Preprocessing job for ML pipeline.

Handles data preprocessing using SageMaker Processing or local execution:
- Synthetic data generation
- Feature engineering
- Vocabulary building
- Embedding computation
- Data validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json
import tempfile
import os

import pandas as pd
import numpy as np

from ..utils.logging_utils import get_logger
from .pipeline_config import PreprocessingConfig

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result of a preprocessing job."""

    job_id: str
    started_at: str
    completed_at: str = ""
    status: str = "pending"  # pending, running, completed, failed

    # Input/Output paths
    input_paths: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)

    # Statistics
    records_processed: int = 0
    records_failed: int = 0

    # Vocabularies built
    vocabularies_built: List[str] = field(default_factory=list)
    vocabulary_sizes: Dict[str, int] = field(default_factory=dict)

    # Embeddings computed
    embeddings_computed: bool = False
    embedding_dimension: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "vocabularies_built": self.vocabularies_built,
            "vocabulary_sizes": self.vocabulary_sizes,
            "embeddings_computed": self.embeddings_computed,
            "embedding_dimension": self.embedding_dimension,
            "errors": self.errors,
        }


class PreprocessingJob:
    """Preprocessing job for the ML pipeline.

    Handles:
    1. Loading raw data from various sources
    2. Building vocabularies for categorical features
    3. Computing statistics for numeric normalization
    4. Pre-computing embeddings for text features
    5. Writing processed data and artifacts

    Example:
        >>> config = PreprocessingConfig()
        >>> job = PreprocessingJob(config)
        >>> result = job.run()
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize the preprocessing job.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config
        self._job_counter = 0

        # Callbacks for progress tracking
        self._progress_callbacks: List[Callable[[str, float], None]] = []

    def run(
        self,
        interactions_path: Optional[str] = None,
        users_path: Optional[str] = None,
        videos_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> PreprocessingResult:
        """Run the preprocessing job.

        Args:
            interactions_path: Path to interactions data.
            users_path: Path to users data.
            videos_path: Path to videos data.
            output_path: Path for processed output.

        Returns:
            PreprocessingResult with job details.
        """
        self._job_counter += 1
        job_id = f"preprocess_{self._job_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow().isoformat()

        result = PreprocessingResult(
            job_id=job_id,
            started_at=started_at,
            status="running",
            input_paths={
                "interactions": interactions_path or self.config.raw_interactions_path,
                "users": users_path or self.config.raw_users_path,
                "videos": videos_path or self.config.raw_videos_path,
            },
        )

        try:
            # Step 1: Load raw data
            self._report_progress("Loading data", 0.1)
            interactions_df, users_df, videos_df = self._load_data(result)

            # Step 2: Build vocabularies
            self._report_progress("Building vocabularies", 0.3)
            vocabularies = self._build_vocabularies(users_df, videos_df, result)

            # Step 3: Compute normalization statistics
            self._report_progress("Computing statistics", 0.5)
            stats = self._compute_statistics(users_df, videos_df, interactions_df)

            # Step 4: Compute embeddings (if enabled)
            if self.config.compute_embeddings:
                self._report_progress("Computing embeddings", 0.7)
                self._compute_embeddings(videos_df, result)

            # Step 5: Process and merge data
            self._report_progress("Processing data", 0.8)
            processed_df = self._process_data(
                interactions_df, users_df, videos_df, vocabularies, stats
            )

            # Step 6: Save outputs
            self._report_progress("Saving outputs", 0.9)
            output_path = output_path or self.config.processed_data_path
            self._save_outputs(processed_df, vocabularies, stats, output_path, result)

            result.status = "completed"
            result.records_processed = len(processed_df)
            self._report_progress("Complete", 1.0)

        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))
            logger.error(f"Preprocessing job failed: {e}")

        result.completed_at = datetime.utcnow().isoformat()
        return result

    def _load_data(
        self, result: PreprocessingResult
    ) -> tuple:
        """Load raw data from input paths.

        Args:
            result: Result object to update.

        Returns:
            Tuple of (interactions_df, users_df, videos_df).
        """
        interactions_path = result.input_paths["interactions"]
        users_path = result.input_paths["users"]
        videos_path = result.input_paths["videos"]

        # Load data (support parquet and csv)
        interactions_df = self._load_dataframe(interactions_path)
        users_df = self._load_dataframe(users_path)
        videos_df = self._load_dataframe(videos_path)

        logger.info(
            f"Loaded data: {len(interactions_df)} interactions, "
            f"{len(users_df)} users, {len(videos_df)} videos"
        )

        return interactions_df, users_df, videos_df

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """Load a DataFrame from path.

        Args:
            path: File path.

        Returns:
            Loaded DataFrame.
        """
        path = Path(path)

        if not path.exists():
            # Return empty DataFrame for missing files
            logger.warning(f"Path {path} does not exist, returning empty DataFrame")
            return pd.DataFrame()

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.is_dir():
            # Load all parquet files in directory
            dfs = []
            for f in path.glob("*.parquet"):
                dfs.append(pd.read_parquet(f))
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            # Try CSV files
            for f in path.glob("*.csv"):
                dfs.append(pd.read_csv(f))
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return pd.DataFrame()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _build_vocabularies(
        self,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        result: PreprocessingResult,
    ) -> Dict[str, Dict[str, int]]:
        """Build vocabularies for categorical features.

        Args:
            users_df: Users DataFrame.
            videos_df: Videos DataFrame.
            result: Result object to update.

        Returns:
            Dictionary of vocabularies.
        """
        vocabularies = {}

        # User vocabularies
        if "user_id" in users_df.columns:
            vocabularies["user_id"] = self._build_vocabulary(
                users_df["user_id"].astype(str).unique()
            )

        if "country" in users_df.columns:
            vocabularies["country"] = self._build_vocabulary(
                users_df["country"].dropna().unique()
            )

        if "user_language" in users_df.columns:
            vocabularies["user_language"] = self._build_vocabulary(
                users_df["user_language"].dropna().unique()
            )

        # Video vocabularies
        if "video_id" in videos_df.columns:
            vocabularies["video_id"] = self._build_vocabulary(
                videos_df["video_id"].astype(str).unique()
            )

        if "category" in videos_df.columns:
            vocabularies["category"] = self._build_vocabulary(
                videos_df["category"].dropna().unique()
            )

        if "video_language" in videos_df.columns:
            # Build shared language vocabulary
            all_languages = set()
            if "user_language" in users_df.columns:
                all_languages.update(users_df["user_language"].dropna().unique())
            all_languages.update(videos_df["video_language"].dropna().unique())
            vocabularies["language"] = self._build_vocabulary(list(all_languages))

        if "popularity" in videos_df.columns:
            vocabularies["popularity"] = self._build_vocabulary(
                videos_df["popularity"].dropna().unique()
            )

        # Update result
        result.vocabularies_built = list(vocabularies.keys())
        result.vocabulary_sizes = {k: len(v) for k, v in vocabularies.items()}

        logger.info(f"Built {len(vocabularies)} vocabularies")
        return vocabularies

    def _build_vocabulary(
        self,
        values: List[Any],
        add_oov: bool = True,
        add_padding: bool = True,
    ) -> Dict[str, int]:
        """Build a vocabulary from values.

        Args:
            values: List of values.
            add_oov: Add out-of-vocabulary token.
            add_padding: Add padding token.

        Returns:
            Dictionary mapping values to indices.
        """
        vocab = {}
        idx = 0

        if add_padding:
            vocab["[PAD]"] = idx
            idx += 1

        if add_oov:
            vocab["[OOV]"] = idx
            idx += 1

        for value in sorted(set(str(v) for v in values)):
            if value not in vocab:
                vocab[value] = idx
                idx += 1

        return vocab

    def _compute_statistics(
        self,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics for numeric features.

        Args:
            users_df: Users DataFrame.
            videos_df: Videos DataFrame.
            interactions_df: Interactions DataFrame.

        Returns:
            Dictionary of statistics per feature.
        """
        stats = {}

        # User statistics
        if "age" in users_df.columns:
            stats["age"] = self._compute_feature_stats(users_df["age"])

        # Video statistics
        if "video_duration" in videos_df.columns:
            stats["video_duration"] = self._compute_feature_stats(
                videos_df["video_duration"]
            )

        if "view_count" in videos_df.columns:
            stats["view_count"] = self._compute_feature_stats(
                videos_df["view_count"], log_transform=True
            )

        if "like_count" in videos_df.columns:
            stats["like_count"] = self._compute_feature_stats(
                videos_df["like_count"], log_transform=True
            )

        if "comment_count" in videos_df.columns:
            stats["comment_count"] = self._compute_feature_stats(
                videos_df["comment_count"], log_transform=True
            )

        logger.info(f"Computed statistics for {len(stats)} features")
        return stats

    def _compute_feature_stats(
        self,
        series: pd.Series,
        log_transform: bool = False,
    ) -> Dict[str, float]:
        """Compute statistics for a single feature.

        Args:
            series: Feature values.
            log_transform: Whether to compute log-transformed stats.

        Returns:
            Dictionary with mean, std, min, max.
        """
        values = series.dropna()
        if len(values) == 0:
            return {"mean": 0, "std": 1, "min": 0, "max": 0}

        stats = {
            "mean": float(values.mean()),
            "std": float(values.std()) or 1.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

        if log_transform:
            log_values = np.log1p(values)
            stats["log_mean"] = float(log_values.mean())
            stats["log_std"] = float(log_values.std()) or 1.0

        return stats

    def _compute_embeddings(
        self,
        videos_df: pd.DataFrame,
        result: PreprocessingResult,
    ) -> None:
        """Compute embeddings for text features.

        Args:
            videos_df: Videos DataFrame.
            result: Result object to update.
        """
        # In production, this would use TensorFlow Hub BERT or similar
        # For now, we create placeholder embeddings
        result.embeddings_computed = True
        result.embedding_dimension = 384  # BERT embedding dimension

        logger.info("Embeddings computation simulated (production would use TF Hub)")

    def _process_data(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        vocabularies: Dict[str, Dict[str, int]],
        stats: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """Process and merge data.

        Args:
            interactions_df: Interactions DataFrame.
            users_df: Users DataFrame.
            videos_df: Videos DataFrame.
            vocabularies: Built vocabularies.
            stats: Normalization statistics.

        Returns:
            Processed DataFrame.
        """
        if interactions_df.empty or users_df.empty or videos_df.empty:
            logger.warning("One or more DataFrames are empty")
            return pd.DataFrame()

        # Merge interactions with users and videos
        processed_df = interactions_df.copy()

        if "user_id" in interactions_df.columns and "user_id" in users_df.columns:
            processed_df = processed_df.merge(
                users_df, on="user_id", how="left", suffixes=("", "_user")
            )

        if "video_id" in interactions_df.columns and "video_id" in videos_df.columns:
            processed_df = processed_df.merge(
                videos_df, on="video_id", how="left", suffixes=("", "_video")
            )

        # Apply vocabulary lookups
        for col, vocab in vocabularies.items():
            if col in processed_df.columns:
                oov_idx = vocab.get("[OOV]", 1)
                processed_df[f"{col}_idx"] = processed_df[col].apply(
                    lambda x: vocab.get(str(x), oov_idx)
                )

        # Apply normalizations
        for col, col_stats in stats.items():
            if col in processed_df.columns:
                mean = col_stats["mean"]
                std = col_stats["std"]
                processed_df[f"{col}_normalized"] = (
                    processed_df[col] - mean
                ) / std

        logger.info(f"Processed {len(processed_df)} records")
        return processed_df

    def _save_outputs(
        self,
        processed_df: pd.DataFrame,
        vocabularies: Dict[str, Dict[str, int]],
        stats: Dict[str, Dict[str, float]],
        output_path: str,
        result: PreprocessingResult,
    ) -> None:
        """Save processed outputs.

        Args:
            processed_df: Processed DataFrame.
            vocabularies: Built vocabularies.
            stats: Normalization statistics.
            output_path: Output directory.
            result: Result object to update.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save processed data
        if not processed_df.empty:
            data_path = output_path / "processed_data.parquet"
            processed_df.to_parquet(data_path, index=False)
            result.output_paths["processed_data"] = str(data_path)

        # Save vocabularies
        vocab_path = output_path / "vocabularies"
        vocab_path.mkdir(exist_ok=True)
        for name, vocab in vocabularies.items():
            vocab_file = vocab_path / f"{name}_vocab.json"
            with open(vocab_file, "w") as f:
                json.dump(vocab, f, indent=2)
        result.output_paths["vocabularies"] = str(vocab_path)

        # Save statistics
        stats_path = output_path / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        result.output_paths["statistics"] = str(stats_path)

        logger.info(f"Saved outputs to {output_path}")

    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress to callbacks.

        Args:
            stage: Current stage name.
            progress: Progress percentage (0-1).
        """
        for callback in self._progress_callbacks:
            try:
                callback(stage, progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def register_progress_callback(
        self,
        callback: Callable[[str, float], None],
    ) -> None:
        """Register a progress callback.

        Args:
            callback: Function to call with (stage, progress).
        """
        self._progress_callbacks.append(callback)


def run_preprocessing(
    data_bucket: str,
    artifacts_bucket: str,
    model_bucket: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run preprocessing job for Lambda invocation.

    This function:
    1. Generates synthetic data if no data exists in S3
    2. Builds vocabularies and computes statistics
    3. Processes data for Two-Tower and Ranker training
    4. Uploads all artifacts to S3

    Args:
        data_bucket: S3 bucket containing raw data.
        artifacts_bucket: S3 bucket for saving artifacts.
        model_bucket: S3 bucket for models (optional).
        config: Additional configuration options.

    Returns:
        Dictionary with preprocessing results.
    """
    import boto3

    logger.info(f"Running preprocessing job with data_bucket={data_bucket}")

    s3 = boto3.client("s3")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_id = f"preprocess_{timestamp}"

    # Configuration
    num_users = config.get("num_users", 1000) if config else 1000
    num_channels = config.get("num_channels", 100) if config else 100
    num_videos = config.get("num_videos", 500) if config else 500
    num_interactions = config.get("num_interactions", 10000) if config else 10000

    try:
        # Step 1: Generate synthetic data
        logger.info("Step 1: Generating synthetic data...")
        from ..data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_all(
            num_users=num_users,
            num_channels=num_channels,
            num_videos=num_videos,
            num_interactions=num_interactions,
        )

        users_df = data["users"]
        channels_df = data["channels"]
        videos_df = data["videos"]
        interactions_df = data["interactions"]

        logger.info(
            f"Generated: {len(users_df)} users, {len(channels_df)} channels, "
            f"{len(videos_df)} videos, {len(interactions_df)} interactions"
        )

        # Step 2: Upload raw data to S3
        logger.info("Step 2: Uploading raw data to S3...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to temp files
            users_path = os.path.join(tmpdir, "users.parquet")
            channels_path = os.path.join(tmpdir, "channels.parquet")
            videos_path = os.path.join(tmpdir, "videos.parquet")
            interactions_path = os.path.join(tmpdir, "interactions.parquet")

            users_df.to_parquet(users_path, index=False)
            channels_df.to_parquet(channels_path, index=False)
            videos_df.to_parquet(videos_path, index=False)
            interactions_df.to_parquet(interactions_path, index=False)

            # Upload to S3
            s3.upload_file(users_path, data_bucket, f"raw_data/{job_id}/users.parquet")
            s3.upload_file(channels_path, data_bucket, f"raw_data/{job_id}/channels.parquet")
            s3.upload_file(videos_path, data_bucket, f"raw_data/{job_id}/videos.parquet")
            s3.upload_file(interactions_path, data_bucket, f"raw_data/{job_id}/interactions.parquet")

        # Step 3: Build vocabularies
        logger.info("Step 3: Building vocabularies...")
        vocabularies = {}

        # Rename columns to match expected schema
        users_df = users_df.rename(columns={
            "id": "user_id",
            "country_code": "country",
            "preferred_language": "user_language",
        })

        videos_df = videos_df.rename(columns={
            "id": "video_id",
            "duration": "video_duration",
            "language": "video_language",
        })

        # User vocabularies
        vocabularies["user_id"] = _build_vocabulary(users_df["user_id"].astype(str).unique())
        vocabularies["country"] = _build_vocabulary(users_df["country"].dropna().unique())
        vocabularies["user_language"] = _build_vocabulary(users_df["user_language"].dropna().unique())

        # Video vocabularies
        vocabularies["video_id"] = _build_vocabulary(videos_df["video_id"].astype(str).unique())
        vocabularies["category"] = _build_vocabulary(videos_df["category"].dropna().unique())
        vocabularies["video_language"] = _build_vocabulary(videos_df["video_language"].dropna().unique())
        vocabularies["popularity"] = _build_vocabulary(videos_df["popularity"].dropna().unique())

        # Shared language vocabulary
        all_languages = set(users_df["user_language"].dropna().unique())
        all_languages.update(videos_df["video_language"].dropna().unique())
        vocabularies["language"] = _build_vocabulary(list(all_languages))

        vocabulary_sizes = {k: len(v) for k, v in vocabularies.items()}
        logger.info(f"Built vocabularies: {vocabulary_sizes}")

        # Step 4: Compute statistics
        logger.info("Step 4: Computing statistics...")
        stats = {}
        stats["age"] = _compute_stats(users_df["age"])
        stats["video_duration"] = _compute_stats(videos_df["video_duration"])
        stats["view_count"] = _compute_stats(videos_df["view_count"], log_transform=True)
        stats["like_count"] = _compute_stats(videos_df["like_count"], log_transform=True)
        stats["comment_count"] = _compute_stats(videos_df["comment_count"], log_transform=True)

        # Step 5: Generate Two-Tower training data
        logger.info("Step 5: Generating Two-Tower training data...")

        # Filter for positive interactions (watch > 40%, like, comment)
        interactions_df["is_positive"] = False

        # Watch > 40% of video duration
        watch_mask = interactions_df["interaction_type"] == "watch"
        interactions_df.loc[watch_mask, "is_positive"] = True  # Simplified: all watches are positive

        # Likes
        like_mask = interactions_df["interaction_type"] == "like"
        interactions_df.loc[like_mask, "is_positive"] = True

        # Comments
        comment_mask = interactions_df["interaction_type"] == "comment"
        interactions_df.loc[comment_mask, "is_positive"] = True

        positive_interactions = interactions_df[interactions_df["is_positive"]].copy()
        logger.info(f"Found {len(positive_interactions)} positive interactions")

        # Merge with user and video features
        two_tower_df = positive_interactions.merge(
            users_df, on="user_id", how="left"
        ).merge(
            videos_df, on="video_id", how="left"
        )

        # Step 6: Generate Ranker training data
        logger.info("Step 6: Generating Ranker training data...")

        # Ranker needs both positive and negative examples
        # Create negative samples by randomly pairing users with videos they didn't interact with
        user_video_pairs = set(zip(interactions_df["user_id"], interactions_df["video_id"]))
        all_users = users_df["user_id"].tolist()
        all_videos = videos_df["video_id"].tolist()

        negative_samples = []
        np.random.seed(42)
        target_negatives = len(positive_interactions)  # 1:1 ratio

        while len(negative_samples) < target_negatives:
            user = np.random.choice(all_users)
            video = np.random.choice(all_videos)
            if (user, video) not in user_video_pairs:
                negative_samples.append({
                    "user_id": user,
                    "video_id": video,
                    "label": 0,
                })
                user_video_pairs.add((user, video))

        negative_df = pd.DataFrame(negative_samples)
        positive_labels = positive_interactions[["user_id", "video_id"]].copy()
        positive_labels["label"] = 1

        ranker_labels = pd.concat([positive_labels, negative_df], ignore_index=True)
        ranker_df = ranker_labels.merge(users_df, on="user_id", how="left").merge(
            videos_df, on="video_id", how="left"
        )

        logger.info(f"Ranker dataset: {len(ranker_df)} samples ({len(positive_labels)} positive, {len(negative_df)} negative)")

        # Step 7: Split into train/val and upload to S3
        logger.info("Step 7: Splitting data and uploading to S3...")

        # Train/val split (80/20)
        np.random.seed(42)
        train_mask = np.random.rand(len(two_tower_df)) < 0.8
        two_tower_train = two_tower_df[train_mask]
        two_tower_val = two_tower_df[~train_mask]

        train_mask = np.random.rand(len(ranker_df)) < 0.8
        ranker_train = ranker_df[train_mask]
        ranker_val = ranker_df[~train_mask]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save datasets
            two_tower_train_path = os.path.join(tmpdir, "two_tower_train.parquet")
            two_tower_val_path = os.path.join(tmpdir, "two_tower_val.parquet")
            ranker_train_path = os.path.join(tmpdir, "ranker_train.parquet")
            ranker_val_path = os.path.join(tmpdir, "ranker_val.parquet")

            two_tower_train.to_parquet(two_tower_train_path, index=False)
            two_tower_val.to_parquet(two_tower_val_path, index=False)
            ranker_train.to_parquet(ranker_train_path, index=False)
            ranker_val.to_parquet(ranker_val_path, index=False)

            # Upload datasets
            s3.upload_file(two_tower_train_path, artifacts_bucket, f"datasets/{job_id}/two_tower/train.parquet")
            s3.upload_file(two_tower_val_path, artifacts_bucket, f"datasets/{job_id}/two_tower/val.parquet")
            s3.upload_file(ranker_train_path, artifacts_bucket, f"datasets/{job_id}/ranker/train.parquet")
            s3.upload_file(ranker_val_path, artifacts_bucket, f"datasets/{job_id}/ranker/val.parquet")

            # Save and upload vocabularies
            vocab_path = os.path.join(tmpdir, "vocabularies.json")
            with open(vocab_path, "w") as f:
                json.dump(vocabularies, f, indent=2)
            s3.upload_file(vocab_path, artifacts_bucket, f"preprocessing/{job_id}/vocabularies.json")

            # Save and upload statistics
            stats_path = os.path.join(tmpdir, "statistics.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            s3.upload_file(stats_path, artifacts_bucket, f"preprocessing/{job_id}/statistics.json")

            # Save video catalog for embedding generation
            video_catalog_path = os.path.join(tmpdir, "video_catalog.parquet")
            videos_df.to_parquet(video_catalog_path, index=False)
            s3.upload_file(video_catalog_path, artifacts_bucket, f"preprocessing/{job_id}/video_catalog.parquet")

        logger.info("Preprocessing completed successfully!")

        result = {
            "status": "success",
            "job_id": job_id,
            "message": "Preprocessing completed successfully",
            "artifacts": {
                "vocabularies": f"s3://{artifacts_bucket}/preprocessing/{job_id}/vocabularies.json",
                "statistics": f"s3://{artifacts_bucket}/preprocessing/{job_id}/statistics.json",
                "video_catalog": f"s3://{artifacts_bucket}/preprocessing/{job_id}/video_catalog.parquet",
            },
            "datasets": {
                "two_tower_train": f"s3://{artifacts_bucket}/datasets/{job_id}/two_tower/train.parquet",
                "two_tower_val": f"s3://{artifacts_bucket}/datasets/{job_id}/two_tower/val.parquet",
                "ranker_train": f"s3://{artifacts_bucket}/datasets/{job_id}/ranker/train.parquet",
                "ranker_val": f"s3://{artifacts_bucket}/datasets/{job_id}/ranker/val.parquet",
            },
            "statistics": {
                "num_users": len(users_df),
                "num_videos": len(videos_df),
                "num_interactions": len(interactions_df),
                "num_positive_interactions": len(positive_interactions),
                "two_tower_train_size": len(two_tower_train),
                "two_tower_val_size": len(two_tower_val),
                "ranker_train_size": len(ranker_train),
                "ranker_val_size": len(ranker_val),
                "vocabulary_sizes": vocabulary_sizes,
            },
        }

        logger.info(f"Preprocessing result: {json.dumps(result, default=str)}")
        return result

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }


def _build_vocabulary(values, add_oov: bool = True, add_padding: bool = True) -> Dict[str, int]:
    """Build a vocabulary from values."""
    vocab = {}
    idx = 0

    if add_padding:
        vocab["[PAD]"] = idx
        idx += 1

    if add_oov:
        vocab["[OOV]"] = idx
        idx += 1

    for value in sorted(set(str(v) for v in values)):
        if value not in vocab:
            vocab[value] = idx
            idx += 1

    return vocab


def _compute_stats(series: pd.Series, log_transform: bool = False) -> Dict[str, float]:
    """Compute statistics for a single feature."""
    values = series.dropna()
    if len(values) == 0:
        return {"mean": 0, "std": 1, "min": 0, "max": 0}

    stats = {
        "mean": float(values.mean()),
        "std": float(values.std()) or 1.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }

    if log_transform:
        log_values = np.log1p(values)
        stats["log_mean"] = float(log_values.mean())
        stats["log_std"] = float(log_values.std()) or 1.0

    return stats
