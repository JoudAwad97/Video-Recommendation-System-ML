"""
Preprocessing job for ML pipeline.

Handles data preprocessing using SageMaker Processing or local execution:
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


class SageMakerPreprocessingJob:
    """SageMaker Processing job wrapper.

    Runs preprocessing as a SageMaker Processing job for
    scalable, managed execution.
    """

    def __init__(self, config: PreprocessingConfig, role_arn: str, s3_bucket: str):
        """Initialize SageMaker preprocessing job.

        Args:
            config: Preprocessing configuration.
            role_arn: IAM role ARN for SageMaker.
            s3_bucket: S3 bucket for data.
        """
        self.config = config
        self.role_arn = role_arn
        self.s3_bucket = s3_bucket
        self._boto3_available = False

        try:
            import boto3
            self._boto3_available = True
            self._sagemaker_client = boto3.client("sagemaker")
        except ImportError:
            logger.warning("boto3 not available. SageMaker jobs will not work.")

    def submit(
        self,
        input_s3_uri: str,
        output_s3_uri: str,
        job_name: Optional[str] = None,
    ) -> str:
        """Submit the processing job.

        Args:
            input_s3_uri: S3 URI for input data.
            output_s3_uri: S3 URI for output.
            job_name: Optional job name.

        Returns:
            Processing job ARN.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker jobs")

        job_name = job_name or f"preprocess-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        response = self._sagemaker_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": self.config.instance_count,
                    "InstanceType": self.config.instance_type,
                    "VolumeSizeInGB": 50,
                }
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": self.config.max_runtime_seconds
            },
            AppSpecification={
                "ImageUri": self._get_processing_image_uri(),
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/preprocess.py"],
            },
            RoleArn=self.role_arn,
            ProcessingInputs=[
                {
                    "InputName": "input-data",
                    "S3Input": {
                        "S3Uri": input_s3_uri,
                        "LocalPath": "/opt/ml/processing/input/data",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "output-data",
                        "S3Output": {
                            "S3Uri": output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
        )

        job_arn = response["ProcessingJobArn"]
        logger.info(f"Submitted SageMaker Processing job: {job_name}")
        return job_arn

    def _get_processing_image_uri(self) -> str:
        """Get the processing container image URI.

        Returns:
            Container image URI.
        """
        # Use sklearn processing container as base
        return f"683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

    def get_status(self, job_name: str) -> Dict[str, Any]:
        """Get job status.

        Args:
            job_name: Processing job name.

        Returns:
            Job status information.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for SageMaker jobs")

        response = self._sagemaker_client.describe_processing_job(
            ProcessingJobName=job_name
        )

        return {
            "job_name": job_name,
            "status": response["ProcessingJobStatus"],
            "creation_time": response.get("CreationTime"),
            "processing_end_time": response.get("ProcessingEndTime"),
            "failure_reason": response.get("FailureReason"),
        }
