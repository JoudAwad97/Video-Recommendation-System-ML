"""
Data versioning and splitting for ML pipeline.

Handles versioned data storage and train/validation/test splitting
with support for incremental updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

from ..utils.logging_utils import get_logger
from .pipeline_config import DataSplittingConfig

logger = get_logger(__name__)


@dataclass
class DataVersion:
    """A versioned dataset snapshot."""

    version_id: str
    version_number: int
    created_at: str

    # Data paths
    base_path: str = ""
    train_path: str = ""
    validation_path: str = ""
    test_path: str = ""

    # Statistics
    total_records: int = 0
    train_records: int = 0
    validation_records: int = 0
    test_records: int = 0

    # Metadata
    parent_version: Optional[str] = None
    data_hash: str = ""
    split_ratios: Dict[str, float] = field(default_factory=dict)

    # Data range
    data_start_date: str = ""
    data_end_date: str = ""

    # Quality info
    positive_samples: int = 0
    negative_samples: int = 0
    unique_users: int = 0
    unique_videos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "created_at": self.created_at,
            "base_path": self.base_path,
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "test_path": self.test_path,
            "total_records": self.total_records,
            "train_records": self.train_records,
            "validation_records": self.validation_records,
            "test_records": self.test_records,
            "parent_version": self.parent_version,
            "data_hash": self.data_hash,
            "split_ratios": self.split_ratios,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "unique_users": self.unique_users,
            "unique_videos": self.unique_videos,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SplitResult:
    """Result of a data splitting operation."""

    version: DataVersion
    train_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class DataVersionManager:
    """Manager for versioned datasets.

    Handles:
    1. Creating new data versions
    2. Splitting data into train/val/test
    3. Tracking version history
    4. Supporting incremental updates

    Example:
        >>> manager = DataVersionManager(config, base_path="data/versioned")
        >>> version = manager.create_version(df)
        >>> train_df, val_df, test_df = manager.load_version(version.version_id)
    """

    def __init__(
        self,
        config: DataSplittingConfig,
        base_path: str = "data/versioned",
    ):
        """Initialize the version manager.

        Args:
            config: Data splitting configuration.
            base_path: Base path for versioned data.
        """
        self.config = config
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Version tracking
        self._versions: Dict[str, DataVersion] = {}
        self._latest_version: Optional[str] = None
        self._version_counter = 0

        # Load existing versions
        self._load_version_history()

    def create_version(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        timestamp_column: Optional[str] = None,
        user_column: str = "user_id",
        video_column: str = "video_id",
        stratify: bool = True,
        parent_version: Optional[str] = None,
    ) -> DataVersion:
        """Create a new data version.

        Args:
            df: Input DataFrame.
            label_column: Column containing labels.
            timestamp_column: Column for time-based splitting.
            user_column: Column containing user IDs.
            video_column: Column containing video IDs.
            stratify: Whether to stratify split by label.
            parent_version: Optional parent version ID.

        Returns:
            Created DataVersion.
        """
        self._version_counter += 1
        version_number = self._version_counter
        version_id = f"{self.config.version_prefix}{version_number}"
        created_at = datetime.utcnow().isoformat()

        # Create version directory
        version_path = self.base_path / version_id
        version_path.mkdir(parents=True, exist_ok=True)

        # Compute data hash for tracking changes
        data_hash = self._compute_hash(df)

        # Split data
        train_df, val_df, test_df = self._split_data(
            df,
            label_column=label_column,
            timestamp_column=timestamp_column,
            stratify=stratify,
        )

        # Save splits
        train_path = version_path / f"train.{self.config.output_format}"
        val_path = version_path / f"validation.{self.config.output_format}"
        test_path = version_path / f"test.{self.config.output_format}"

        self._save_dataframe(train_df, train_path)
        self._save_dataframe(val_df, val_path)
        self._save_dataframe(test_df, test_path)

        # Compute statistics
        positive_samples = 0
        negative_samples = 0
        if label_column in df.columns:
            positive_samples = int((df[label_column] == 1).sum())
            negative_samples = int((df[label_column] == 0).sum())

        unique_users = df[user_column].nunique() if user_column in df.columns else 0
        unique_videos = df[video_column].nunique() if video_column in df.columns else 0

        # Get date range
        data_start_date = ""
        data_end_date = ""
        if timestamp_column and timestamp_column in df.columns:
            timestamps = pd.to_datetime(df[timestamp_column])
            data_start_date = timestamps.min().isoformat()
            data_end_date = timestamps.max().isoformat()

        # Create version object
        version = DataVersion(
            version_id=version_id,
            version_number=version_number,
            created_at=created_at,
            base_path=str(version_path),
            train_path=str(train_path),
            validation_path=str(val_path),
            test_path=str(test_path),
            total_records=len(df),
            train_records=len(train_df),
            validation_records=len(val_df),
            test_records=len(test_df),
            parent_version=parent_version or self._latest_version,
            data_hash=data_hash,
            split_ratios={
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
                "test": self.config.test_ratio,
            },
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            unique_users=unique_users,
            unique_videos=unique_videos,
        )

        # Save version metadata
        self._save_version_metadata(version)

        # Update tracking
        self._versions[version_id] = version
        self._latest_version = version_id

        logger.info(
            f"Created data version {version_id}: "
            f"{len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )

        return version

    def _split_data(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        timestamp_column: Optional[str] = None,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.

        Args:
            df: Input DataFrame.
            label_column: Column containing labels.
            timestamp_column: Column for time-based splitting.
            stratify: Whether to stratify split by label.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Normalize ratios
        total = self.config.train_ratio + self.config.validation_ratio + self.config.test_ratio
        train_ratio = self.config.train_ratio / total
        val_ratio = self.config.validation_ratio / total

        if timestamp_column and timestamp_column in df.columns:
            # Time-based split
            df = df.sort_values(timestamp_column)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
        elif stratify and label_column in df.columns:
            # Stratified split
            train_df, val_df, test_df = self._stratified_split(
                df, label_column, train_ratio, val_ratio
            )
        else:
            # Random split
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def _stratified_split(
        self,
        df: pd.DataFrame,
        label_column: str,
        train_ratio: float,
        val_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform stratified split.

        Args:
            df: Input DataFrame.
            label_column: Column containing labels.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Split each label class separately
        for label in df[label_column].unique():
            label_df = df[df[label_column] == label].sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            n = len(label_df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_dfs.append(label_df.iloc[:train_end])
            val_dfs.append(label_df.iloc[train_end:val_end])
            test_dfs.append(label_df.iloc[val_end:])

        # Combine and shuffle
        train_df = pd.concat(train_dfs, ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        val_df = pd.concat(val_dfs, ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

        return train_df, val_df, test_df

    def _save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to file.

        Args:
            df: DataFrame to save.
            path: Output path.
        """
        if self.config.output_format == "parquet":
            df.to_parquet(path, index=False)
        elif self.config.output_format == "csv":
            df.to_csv(path, index=False)
        elif self.config.output_format == "tfrecord":
            # Would need TensorFlow for TFRecord
            logger.warning("TFRecord format not implemented, using parquet")
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        else:
            raise ValueError(f"Unknown format: {self.config.output_format}")

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for change detection.

        Args:
            df: Input DataFrame.

        Returns:
            Hash string.
        """
        # Hash based on shape and sample of data
        hash_input = f"{df.shape}_{df.columns.tolist()}"
        if len(df) > 0:
            sample_str = df.head(100).to_json()
            hash_input += sample_str
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _save_version_metadata(self, version: DataVersion) -> None:
        """Save version metadata to file.

        Args:
            version: Version to save.
        """
        metadata_path = Path(version.base_path) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        # Also update the global version history
        self._save_version_history()

    def _save_version_history(self) -> None:
        """Save version history to file."""
        history_path = self.base_path / "version_history.json"
        history = {
            "latest_version": self._latest_version,
            "version_count": self._version_counter,
            "versions": {
                vid: v.to_dict() for vid, v in self._versions.items()
            },
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def _load_version_history(self) -> None:
        """Load version history from file."""
        history_path = self.base_path / "version_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

            self._latest_version = history.get("latest_version")
            self._version_counter = history.get("version_count", 0)

            for vid, vdata in history.get("versions", {}).items():
                self._versions[vid] = DataVersion.from_dict(vdata)

            logger.info(f"Loaded {len(self._versions)} existing versions")

    def load_version(
        self,
        version_id: Optional[str] = None,
    ) -> SplitResult:
        """Load a specific version.

        Args:
            version_id: Version to load. Uses latest if None.

        Returns:
            SplitResult with loaded DataFrames.
        """
        version_id = version_id or self._latest_version
        if not version_id or version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        version = self._versions[version_id]

        # Load DataFrames
        train_df = self._load_dataframe(version.train_path)
        val_df = self._load_dataframe(version.validation_path)
        test_df = self._load_dataframe(version.test_path)

        logger.info(
            f"Loaded version {version_id}: "
            f"{len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )

        return SplitResult(
            version=version,
            train_df=train_df,
            validation_df=val_df,
            test_df=test_df,
        )

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """Load DataFrame from path.

        Args:
            path: File path.

        Returns:
            Loaded DataFrame.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Path {path} does not exist")
            return pd.DataFrame()

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unknown format: {path.suffix}")

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version metadata.

        Args:
            version_id: Version identifier.

        Returns:
            DataVersion or None.
        """
        return self._versions.get(version_id)

    def get_latest_version(self) -> Optional[DataVersion]:
        """Get the latest version.

        Returns:
            Latest DataVersion or None.
        """
        if self._latest_version:
            return self._versions.get(self._latest_version)
        return None

    def list_versions(self) -> List[DataVersion]:
        """List all versions.

        Returns:
            List of DataVersions sorted by version number.
        """
        return sorted(
            self._versions.values(),
            key=lambda v: v.version_number,
            reverse=True,
        )

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two versions.

        Args:
            version_id_1: First version ID.
            version_id_2: Second version ID.

        Returns:
            Comparison statistics.
        """
        v1 = self._versions.get(version_id_1)
        v2 = self._versions.get(version_id_2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        return {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "record_diff": v2.total_records - v1.total_records,
            "positive_diff": v2.positive_samples - v1.positive_samples,
            "negative_diff": v2.negative_samples - v1.negative_samples,
            "user_diff": v2.unique_users - v1.unique_users,
            "video_diff": v2.unique_videos - v1.unique_videos,
            "hash_changed": v1.data_hash != v2.data_hash,
        }

    def delete_version(self, version_id: str) -> bool:
        """Delete a version.

        Args:
            version_id: Version to delete.

        Returns:
            True if deleted.
        """
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]
        version_path = Path(version.base_path)

        # Remove files
        import shutil
        if version_path.exists():
            shutil.rmtree(version_path)

        # Update tracking
        del self._versions[version_id]

        if self._latest_version == version_id:
            # Update latest to previous version
            remaining = self.list_versions()
            self._latest_version = remaining[0].version_id if remaining else None

        self._save_version_history()
        logger.info(f"Deleted version {version_id}")
        return True
