"""
Pipeline configuration dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline.

    Attributes:
        raw_data_dir: Directory containing raw input data.
        artifacts_dir: Directory for saving preprocessing artifacts.
        processed_data_dir: Directory for saving processed datasets.
        train_ratio: Ratio of data for training set.
        val_ratio: Ratio of data for validation set.
        test_ratio: Ratio of data for test set.
        negative_ratio: Number of negative samples per positive for ranker.
        min_watch_ratio: Minimum watch ratio for positive interaction.
        compute_embeddings: Whether to pre-compute text embeddings.
        random_seed: Random seed for reproducibility.
    """

    # Directories
    raw_data_dir: str = "data/raw"
    artifacts_dir: str = "artifacts"
    processed_data_dir: str = "processed_data"

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Sampling configuration
    negative_ratio: int = 3
    min_watch_ratio: float = 0.4

    # Processing options
    compute_embeddings: bool = True
    random_seed: int = 42

    # Output subdirectories
    @property
    def two_tower_output_dir(self) -> str:
        return str(Path(self.processed_data_dir) / "two_tower")

    @property
    def ranker_output_dir(self) -> str:
        return str(Path(self.processed_data_dir) / "ranker")

    @property
    def tfrecords_dir(self) -> str:
        return str(Path(self.processed_data_dir) / "tfrecords")
