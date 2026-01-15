"""
Main processing pipeline orchestration.

Orchestrates the full data processing pipeline including:
1. Load raw data
2. Build vocabularies
3. Compute normalization statistics
4. Pre-compute embeddings (title, tags)
5. Generate Two-Tower dataset
6. Generate Ranker dataset
7. Save all artifacts and processed data
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd

from .pipeline_config import PipelineConfig
from ..data.data_loader import DataLoader, InMemoryDataLoader
from ..data.synthetic_generator import SyntheticDataGenerator
from ..dataset.two_tower_dataset import TwoTowerDatasetGenerator
from ..dataset.ranker_dataset import RankerDatasetGenerator
from ..preprocessing.artifacts import ArtifactManager
from ..config.feature_config import FeatureConfig, DEFAULT_CONFIG
from ..utils.io_utils import save_json, save_parquet, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProcessingPipeline:
    """Orchestrate the full data processing pipeline.

    Handles all steps from raw data to processed training datasets:
    1. Data loading (from files or synthetic generation)
    2. Vocabulary building for categorical features
    3. Normalization statistics computation
    4. Text embedding pre-computation
    5. Two-Tower dataset generation
    6. Ranker dataset generation
    7. Artifact saving

    Example:
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.run_synthetic(num_users=1000, num_videos=500)
        >>> # Or with real data:
        >>> pipeline.run(data_loader)
    """

    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """Initialize the processing pipeline.

        Args:
            pipeline_config: Pipeline configuration.
            feature_config: Feature configuration.
        """
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.feature_config = feature_config or DEFAULT_CONFIG

        # Initialize generators
        self.two_tower_generator = TwoTowerDatasetGenerator(
            config=self.feature_config,
            min_watch_ratio=self.pipeline_config.min_watch_ratio
        )
        self.ranker_generator = RankerDatasetGenerator(
            config=self.feature_config,
            negative_ratio=self.pipeline_config.negative_ratio,
            min_watch_ratio=self.pipeline_config.min_watch_ratio
        )

        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._run_metadata: Dict = {}

    def run(
        self,
        data_loader: DataLoader,
        save_outputs: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Run the full processing pipeline.

        Args:
            data_loader: DataLoader instance with raw data.
            save_outputs: Whether to save outputs to disk.

        Returns:
            Dictionary with all processed datasets.
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Processing Pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading data...")
        self._data = data_loader.load_all()
        self._log_data_summary()

        # Step 2: Fit Two-Tower generator
        logger.info("\n[Step 2/5] Processing Two-Tower dataset...")
        self.two_tower_generator.fit(
            users_df=self._data["users"],
            videos_df=self._data["videos"],
            interactions_df=self._data["interactions"],
            channels_df=self._data.get("channels"),
            compute_embeddings=self.pipeline_config.compute_embeddings
        )

        # Step 3: Generate Two-Tower splits
        logger.info("\n[Step 3/5] Generating Two-Tower train/val/test splits...")
        two_tower_train, two_tower_val, two_tower_test = self.two_tower_generator.generate_splits(
            train_ratio=self.pipeline_config.train_ratio,
            val_ratio=self.pipeline_config.val_ratio,
            test_ratio=self.pipeline_config.test_ratio,
            random_state=self.pipeline_config.random_seed
        )

        # Step 4: Fit Ranker generator
        logger.info("\n[Step 4/5] Processing Ranker dataset...")
        self.ranker_generator.fit(
            users_df=self._data["users"],
            videos_df=self._data["videos"],
            interactions_df=self._data["interactions"],
            channels_df=self._data.get("channels")
        )

        # Step 5: Generate Ranker splits
        logger.info("\n[Step 5/5] Generating Ranker train/val/test splits...")
        ranker_train, ranker_val, ranker_test = self.ranker_generator.generate_splits(
            train_ratio=self.pipeline_config.train_ratio,
            val_ratio=self.pipeline_config.val_ratio,
            test_ratio=self.pipeline_config.test_ratio,
            random_state=self.pipeline_config.random_seed,
            transform=True
        )

        # Collect results
        results = {
            "two_tower_train": two_tower_train,
            "two_tower_val": two_tower_val,
            "two_tower_test": two_tower_test,
            "ranker_train": ranker_train,
            "ranker_val": ranker_val,
            "ranker_test": ranker_test,
        }

        # Save outputs
        if save_outputs:
            self._save_outputs(results)

        # Record metadata
        end_time = datetime.now()
        self._run_metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "num_users": len(self._data["users"]),
            "num_videos": len(self._data["videos"]),
            "num_interactions": len(self._data["interactions"]),
            "two_tower_samples": len(two_tower_train) + len(two_tower_val) + len(two_tower_test),
            "ranker_samples": len(ranker_train) + len(ranker_val) + len(ranker_test),
        }

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"Duration: {self._run_metadata['duration_seconds']:.2f} seconds")
        logger.info("=" * 60)

        return results

    def run_synthetic(
        self,
        num_users: int = 1000,
        num_channels: int = 100,
        num_videos: int = 500,
        num_interactions: int = 10000,
        save_outputs: bool = True,
        save_synthetic: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Run pipeline with synthetic data.

        Args:
            num_users: Number of synthetic users.
            num_channels: Number of synthetic channels.
            num_videos: Number of synthetic videos.
            num_interactions: Number of synthetic interactions.
            save_outputs: Whether to save processed outputs.
            save_synthetic: Whether to save synthetic raw data.

        Returns:
            Dictionary with all processed datasets.
        """
        logger.info("Generating synthetic data...")

        # Generate synthetic data
        generator = SyntheticDataGenerator(seed=self.pipeline_config.random_seed)
        synthetic_data = generator.generate_all(
            num_users=num_users,
            num_channels=num_channels,
            num_videos=num_videos,
            num_interactions=num_interactions
        )

        # Save synthetic data if requested
        if save_synthetic:
            raw_dir = Path(self.pipeline_config.raw_data_dir)
            ensure_dir(raw_dir)
            for name, df in synthetic_data.items():
                save_parquet(df, raw_dir / f"{name}.parquet")
            logger.info(f"Saved synthetic data to {raw_dir}")

        # Create in-memory data loader
        data_loader = InMemoryDataLoader(synthetic_data)

        # Run pipeline
        return self.run(data_loader, save_outputs=save_outputs)

    def _log_data_summary(self) -> None:
        """Log summary of loaded data."""
        for name, df in self._data.items():
            logger.info(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

    def _save_outputs(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save all outputs to disk.

        Args:
            results: Dictionary of processed DataFrames.
        """
        logger.info("\nSaving outputs...")

        # Save Two-Tower datasets
        two_tower_dir = Path(self.pipeline_config.two_tower_output_dir)
        ensure_dir(two_tower_dir)
        save_parquet(results["two_tower_train"], two_tower_dir / "train.parquet")
        save_parquet(results["two_tower_val"], two_tower_dir / "val.parquet")
        save_parquet(results["two_tower_test"], two_tower_dir / "test.parquet")
        logger.info(f"  Saved Two-Tower datasets to {two_tower_dir}")

        # Save Ranker datasets
        ranker_dir = Path(self.pipeline_config.ranker_output_dir)
        ensure_dir(ranker_dir)
        save_parquet(results["ranker_train"], ranker_dir / "train.parquet")
        save_parquet(results["ranker_val"], ranker_dir / "val.parquet")
        save_parquet(results["ranker_test"], ranker_dir / "test.parquet")
        logger.info(f"  Saved Ranker datasets to {ranker_dir}")

        # Save artifacts
        artifacts_dir = Path(self.pipeline_config.artifacts_dir)
        self.two_tower_generator.save_artifacts(str(artifacts_dir))
        self.ranker_generator.save_artifacts(str(artifacts_dir))
        logger.info(f"  Saved artifacts to {artifacts_dir}")

        # Save pipeline metadata
        save_json(self._run_metadata, artifacts_dir / "pipeline_metadata.json")

    def get_feature_specs(self) -> Dict:
        """Get feature specifications for model building.

        Returns:
            Dictionary with Two-Tower and Ranker feature specs.
        """
        return {
            "two_tower": self.two_tower_generator.get_feature_specs(),
            "ranker": {
                "categorical_features": self.ranker_generator.feature_transformer.get_categorical_features(),
                "numeric_features": self.ranker_generator.feature_transformer.get_numeric_features(),
                "vocab_sizes": self.ranker_generator.feature_transformer.get_vocab_sizes(),
            },
        }

    def get_metadata(self) -> Dict:
        """Get pipeline run metadata.

        Returns:
            Dictionary with run metadata.
        """
        return self._run_metadata.copy()

    def __repr__(self) -> str:
        return f"ProcessingPipeline(config={self.pipeline_config})"
