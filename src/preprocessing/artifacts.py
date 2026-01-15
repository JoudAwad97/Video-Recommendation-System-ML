"""
Artifact management for preprocessing outputs.

Handles saving and loading of vocabularies, normalizers, and other
preprocessing artifacts in a structured way.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json

from ..utils.io_utils import save_json, load_json, ensure_dir
from .vocabulary_builder import (
    StringLookupVocabulary,
    IntegerLookupVocabulary,
    HashBucketVocabulary,
)


class ArtifactManager:
    """Manage preprocessing artifacts.

    Provides a unified interface for saving and loading all preprocessing
    artifacts including vocabularies, normalizers, and bucket configurations.
    """

    def __init__(self, artifacts_dir: Union[str, Path]):
        """Initialize the artifact manager.

        Args:
            artifacts_dir: Base directory for storing artifacts.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.vocabularies_dir = self.artifacts_dir / "vocabularies"
        self.normalizers_dir = self.artifacts_dir / "normalizers"
        self.buckets_dir = self.artifacts_dir / "buckets"
        self.embeddings_dir = self.artifacts_dir / "embeddings"

        # Ensure directories exist
        for dir_path in [
            self.vocabularies_dir,
            self.normalizers_dir,
            self.buckets_dir,
            self.embeddings_dir,
        ]:
            ensure_dir(dir_path)

    def save_vocabulary(
        self,
        vocab: Union[StringLookupVocabulary, IntegerLookupVocabulary, HashBucketVocabulary],
        name: Optional[str] = None
    ) -> Path:
        """Save a vocabulary to file.

        Args:
            vocab: Vocabulary to save.
            name: Optional name override (uses vocab.name if not provided).

        Returns:
            Path where vocabulary was saved.
        """
        vocab_name = name or vocab.name
        filepath = self.vocabularies_dir / f"{vocab_name}_vocab.json"
        vocab.save(filepath)
        return filepath

    def load_vocabulary(
        self,
        name: str,
        vocab_type: str = "string"
    ) -> Union[StringLookupVocabulary, IntegerLookupVocabulary, HashBucketVocabulary]:
        """Load a vocabulary from file.

        Args:
            name: Name of the vocabulary (without _vocab.json suffix).
            vocab_type: Type of vocabulary ("string", "integer", "hash").

        Returns:
            Loaded vocabulary.
        """
        filepath = self.vocabularies_dir / f"{name}_vocab.json"

        if vocab_type == "string":
            return StringLookupVocabulary(name=name).load(filepath)
        elif vocab_type == "integer":
            return IntegerLookupVocabulary(name=name).load(filepath)
        elif vocab_type == "hash":
            return HashBucketVocabulary.load(filepath)
        else:
            raise ValueError(f"Unknown vocabulary type: {vocab_type}")

    def save_normalizer_stats(
        self,
        name: str,
        stats: Dict[str, float]
    ) -> Path:
        """Save normalizer statistics to file.

        Args:
            name: Name of the feature.
            stats: Dictionary with statistics (mean, std, min, max, etc.).

        Returns:
            Path where stats were saved.
        """
        filepath = self.normalizers_dir / f"{name}_stats.json"
        save_json(stats, filepath)
        return filepath

    def load_normalizer_stats(self, name: str) -> Dict[str, float]:
        """Load normalizer statistics from file.

        Args:
            name: Name of the feature.

        Returns:
            Dictionary with statistics.
        """
        filepath = self.normalizers_dir / f"{name}_stats.json"
        return load_json(filepath)

    def save_bucket_boundaries(
        self,
        name: str,
        boundaries: list
    ) -> Path:
        """Save bucket boundaries to file.

        Args:
            name: Name of the feature.
            boundaries: List of bucket boundaries.

        Returns:
            Path where boundaries were saved.
        """
        filepath = self.buckets_dir / f"{name}_boundaries.json"
        save_json({"name": name, "boundaries": boundaries}, filepath)
        return filepath

    def load_bucket_boundaries(self, name: str) -> list:
        """Load bucket boundaries from file.

        Args:
            name: Name of the feature.

        Returns:
            List of bucket boundaries.
        """
        filepath = self.buckets_dir / f"{name}_boundaries.json"
        data = load_json(filepath)
        return data["boundaries"]

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save pipeline metadata.

        Args:
            metadata: Dictionary with metadata (version, timestamp, etc.).

        Returns:
            Path where metadata was saved.
        """
        filepath = self.artifacts_dir / "metadata.json"
        save_json(metadata, filepath)
        return filepath

    def load_metadata(self) -> Dict[str, Any]:
        """Load pipeline metadata.

        Returns:
            Dictionary with metadata.
        """
        filepath = self.artifacts_dir / "metadata.json"
        return load_json(filepath)

    def list_vocabularies(self) -> list:
        """List all saved vocabularies.

        Returns:
            List of vocabulary names.
        """
        vocab_files = self.vocabularies_dir.glob("*_vocab.json")
        return [f.stem.replace("_vocab", "") for f in vocab_files]

    def list_normalizers(self) -> list:
        """List all saved normalizer stats.

        Returns:
            List of normalizer names.
        """
        stats_files = self.normalizers_dir.glob("*_stats.json")
        return [f.stem.replace("_stats", "") for f in stats_files]

    def list_buckets(self) -> list:
        """List all saved bucket boundaries.

        Returns:
            List of bucket names.
        """
        boundary_files = self.buckets_dir.glob("*_boundaries.json")
        return [f.stem.replace("_boundaries", "") for f in boundary_files]

    def exists(self, artifact_type: str, name: str) -> bool:
        """Check if an artifact exists.

        Args:
            artifact_type: Type of artifact ("vocabulary", "normalizer", "bucket").
            name: Name of the artifact.

        Returns:
            True if artifact exists.
        """
        if artifact_type == "vocabulary":
            return (self.vocabularies_dir / f"{name}_vocab.json").exists()
        elif artifact_type == "normalizer":
            return (self.normalizers_dir / f"{name}_stats.json").exists()
        elif artifact_type == "bucket":
            return (self.buckets_dir / f"{name}_boundaries.json").exists()
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
