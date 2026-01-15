"""
Data loader for loading raw data from various sources.

Supports loading from local files (CSV, Parquet, JSON) and provides
a unified interface for data access.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

from ..utils.io_utils import load_parquet, load_json


class DataLoader:
    """Load data from various sources.

    Provides a unified interface for loading user, video, channel,
    and interaction data from local files or other sources.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the data loader.

        Args:
            data_dir: Base directory containing data files.
        """
        self.data_dir = Path(data_dir)

    def load_users(self, filename: str = "users.parquet") -> pd.DataFrame:
        """Load user data.

        Args:
            filename: Name of the file to load.

        Returns:
            DataFrame with user data.
        """
        filepath = self.data_dir / filename
        return self._load_file(filepath)

    def load_videos(self, filename: str = "videos.parquet") -> pd.DataFrame:
        """Load video data.

        Args:
            filename: Name of the file to load.

        Returns:
            DataFrame with video data.
        """
        filepath = self.data_dir / filename
        return self._load_file(filepath)

    def load_channels(self, filename: str = "channels.parquet") -> pd.DataFrame:
        """Load channel data.

        Args:
            filename: Name of the file to load.

        Returns:
            DataFrame with channel data.
        """
        filepath = self.data_dir / filename
        return self._load_file(filepath)

    def load_interactions(self, filename: str = "interactions.parquet") -> pd.DataFrame:
        """Load interaction data.

        Args:
            filename: Name of the file to load.

        Returns:
            DataFrame with interaction data.
        """
        filepath = self.data_dir / filename
        return self._load_file(filepath)

    def load_all(
        self,
        users_file: str = "users.parquet",
        videos_file: str = "videos.parquet",
        channels_file: str = "channels.parquet",
        interactions_file: str = "interactions.parquet"
    ) -> Dict[str, pd.DataFrame]:
        """Load all datasets.

        Args:
            users_file: Filename for users data.
            videos_file: Filename for videos data.
            channels_file: Filename for channels data.
            interactions_file: Filename for interactions data.

        Returns:
            Dictionary with all DataFrames.
        """
        return {
            "users": self.load_users(users_file),
            "videos": self.load_videos(videos_file),
            "channels": self.load_channels(channels_file),
            "interactions": self.load_interactions(interactions_file),
        }

    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load file based on extension.

        Args:
            filepath: Path to the file.

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If file extension is not supported.
            FileNotFoundError: If file does not exist.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(filepath)
        elif suffix == ".csv":
            return pd.read_csv(filepath)
        elif suffix == ".json":
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_dataframes(data: Dict[str, pd.DataFrame]) -> "DataLoader":
        """Create a mock DataLoader from in-memory DataFrames.

        Useful for testing without file I/O.

        Args:
            data: Dictionary of DataFrames.

        Returns:
            DataLoader-like object that returns the provided DataFrames.
        """
        return InMemoryDataLoader(data)


class InMemoryDataLoader:
    """In-memory data loader for testing purposes.

    Provides the same interface as DataLoader but serves data from memory.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """Initialize with in-memory data.

        Args:
            data: Dictionary mapping dataset names to DataFrames.
        """
        self.data = data

    def load_users(self, filename: str = None) -> pd.DataFrame:
        """Load user data from memory."""
        return self.data.get("users", pd.DataFrame())

    def load_videos(self, filename: str = None) -> pd.DataFrame:
        """Load video data from memory."""
        return self.data.get("videos", pd.DataFrame())

    def load_channels(self, filename: str = None) -> pd.DataFrame:
        """Load channel data from memory."""
        return self.data.get("channels", pd.DataFrame())

    def load_interactions(self, filename: str = None) -> pd.DataFrame:
        """Load interaction data from memory."""
        return self.data.get("interactions", pd.DataFrame())

    def load_all(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load all datasets from memory."""
        return self.data
