"""
Tag embedding using CBOW-style approach.

Provides tag embedding functionality for video tags using a simple
averaging approach over pre-trained word vectors or learned embeddings.
"""

from typing import List, Union, Optional, Dict, Set
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.io_utils import save_json, load_json, save_parquet, load_parquet, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TagEmbedder:
    """Embed video tags using CBOW-style averaging.

    Splits pipe-separated tags and computes an average embedding
    for each video's tag set.

    Example:
        >>> embedder = TagEmbedder(embedding_dim=100)
        >>> embedder.fit(["tech|ai|ml", "gaming|rpg", "music|pop|rock"])
        >>> embeddings = embedder.embed(["tech|ai", "gaming"])
        >>> embeddings.shape  # (2, 100)
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        separator: str = "|",
        min_tag_freq: int = 1,
        cache_dir: Optional[str] = None
    ):
        """Initialize the tag embedder.

        Args:
            embedding_dim: Dimension of tag embeddings.
            separator: Character used to separate tags.
            min_tag_freq: Minimum frequency for a tag to be included.
            cache_dir: Directory to cache embeddings.
        """
        self.embedding_dim = embedding_dim
        self.separator = separator
        self.min_tag_freq = min_tag_freq
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._tag_vocab: Dict[str, int] = {}
        self._tag_embeddings: Optional[np.ndarray] = None
        self._is_fitted = False

    def _parse_tags(self, tag_string: Union[str, None]) -> List[str]:
        """Parse pipe-separated tag string into list.

        Args:
            tag_string: Pipe-separated tags or None.

        Returns:
            List of individual tags.
        """
        if tag_string is None or (isinstance(tag_string, float) and np.isnan(tag_string)):
            return []

        tag_string = str(tag_string).strip()
        if not tag_string or tag_string == "-":
            return []

        # Split and clean tags
        tags = [t.strip().lower() for t in tag_string.split(self.separator)]
        return [t for t in tags if t]

    def fit(
        self,
        tag_strings: Union[List[str], np.ndarray, pd.Series],
        pretrained_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> "TagEmbedder":
        """Build tag vocabulary and initialize embeddings.

        Args:
            tag_strings: List of pipe-separated tag strings.
            pretrained_embeddings: Optional dict mapping tags to embeddings.

        Returns:
            Self for method chaining.
        """
        if isinstance(tag_strings, pd.Series):
            tag_strings = tag_strings.tolist()

        # Count tag frequencies
        tag_counts: Dict[str, int] = {}
        for tag_string in tag_strings:
            for tag in self._parse_tags(tag_string):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Filter by minimum frequency
        filtered_tags = [
            tag for tag, count in tag_counts.items()
            if count >= self.min_tag_freq
        ]

        # Sort for deterministic ordering
        filtered_tags = sorted(filtered_tags)

        # Build vocabulary (0 reserved for unknown)
        self._tag_vocab = {"[UNK]": 0}
        for idx, tag in enumerate(filtered_tags, start=1):
            self._tag_vocab[tag] = idx

        # Initialize embeddings
        vocab_size = len(self._tag_vocab)
        self._tag_embeddings = np.random.randn(vocab_size, self.embedding_dim).astype(np.float32)

        # Normalize initial embeddings
        norms = np.linalg.norm(self._tag_embeddings, axis=1, keepdims=True) + 1e-8
        self._tag_embeddings = self._tag_embeddings / norms

        # Use pretrained embeddings if provided
        if pretrained_embeddings:
            for tag, idx in self._tag_vocab.items():
                if tag in pretrained_embeddings:
                    emb = pretrained_embeddings[tag]
                    if len(emb) == self.embedding_dim:
                        self._tag_embeddings[idx] = emb

        self._is_fitted = True
        logger.info(f"Tag vocabulary built with {vocab_size} tags")

        return self

    def embed(self, tag_strings: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """Embed tag strings using CBOW-style averaging.

        Args:
            tag_strings: List of pipe-separated tag strings.

        Returns:
            Numpy array of shape (n_samples, embedding_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        if isinstance(tag_strings, pd.Series):
            tag_strings = tag_strings.tolist()
        elif isinstance(tag_strings, np.ndarray):
            tag_strings = tag_strings.tolist()

        embeddings = []
        for tag_string in tag_strings:
            tags = self._parse_tags(tag_string)

            if not tags:
                # Return zero embedding for empty tags
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            else:
                # Get indices for tags
                indices = [
                    self._tag_vocab.get(tag, 0)  # 0 is UNK
                    for tag in tags
                ]

                # Average embeddings (CBOW-style)
                tag_embs = self._tag_embeddings[indices]
                avg_emb = np.mean(tag_embs, axis=0)

                # Normalize
                norm = np.linalg.norm(avg_emb) + 1e-8
                avg_emb = avg_emb / norm

                embeddings.append(avg_emb)

        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, tag_string: str) -> np.ndarray:
        """Embed a single tag string.

        Args:
            tag_string: Pipe-separated tag string.

        Returns:
            Embedding vector.
        """
        return self.embed([tag_string])[0]

    def get_tag_embedding(self, tag: str) -> np.ndarray:
        """Get embedding for a single tag.

        Args:
            tag: Individual tag.

        Returns:
            Tag embedding vector.
        """
        if not self._is_fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        tag = tag.strip().lower()
        idx = self._tag_vocab.get(tag, 0)
        return self._tag_embeddings[idx]

    def embed_and_cache(
        self,
        tag_strings: Union[List[str], pd.Series],
        ids: Union[List[int], pd.Series],
        cache_name: str = "tag_embeddings"
    ) -> pd.DataFrame:
        """Embed tags and cache results.

        Args:
            tag_strings: List of tag strings to embed.
            ids: Corresponding IDs for each tag string.
            cache_name: Name for the cache file.

        Returns:
            DataFrame with id and embedding columns.
        """
        if isinstance(tag_strings, pd.Series):
            tag_strings = tag_strings.tolist()
        if isinstance(ids, pd.Series):
            ids = ids.tolist()

        # Check cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_name}.parquet"
            if cache_path.exists():
                logger.info(f"Loading cached tag embeddings from {cache_path}")
                return load_parquet(cache_path)

        # Compute embeddings
        logger.info(f"Computing tag embeddings for {len(tag_strings)} items...")
        embeddings = self.embed(tag_strings)

        # Create DataFrame
        embedding_cols = {f"tag_emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        df = pd.DataFrame({"id": ids, **embedding_cols})

        # Save cache
        if self.cache_dir:
            ensure_dir(self.cache_dir)
            save_parquet(df, cache_path)
            logger.info(f"Cached tag embeddings to {cache_path}")

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """Save embedder state to files.

        Args:
            filepath: Base path for saving (without extension).
        """
        if not self._is_fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        # Save vocabulary
        vocab_path = filepath.parent / f"{filepath.stem}_vocab.json"
        save_json({
            "vocab": self._tag_vocab,
            "embedding_dim": self.embedding_dim,
            "separator": self.separator,
            "min_tag_freq": self.min_tag_freq,
        }, vocab_path)

        # Save embeddings
        emb_path = filepath.parent / f"{filepath.stem}_embeddings.npy"
        np.save(emb_path, self._tag_embeddings)

        logger.info(f"Saved tag embedder to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "TagEmbedder":
        """Load embedder state from files.

        Args:
            filepath: Base path (without extension).

        Returns:
            Self for method chaining.
        """
        filepath = Path(filepath)

        # Load vocabulary
        vocab_path = filepath.parent / f"{filepath.stem}_vocab.json"
        vocab_data = load_json(vocab_path)

        self._tag_vocab = vocab_data["vocab"]
        self.embedding_dim = vocab_data["embedding_dim"]
        self.separator = vocab_data["separator"]
        self.min_tag_freq = vocab_data["min_tag_freq"]

        # Load embeddings
        emb_path = filepath.parent / f"{filepath.stem}_embeddings.npy"
        self._tag_embeddings = np.load(emb_path)

        self._is_fitted = True
        logger.info(f"Loaded tag embedder from {filepath}")

        return self

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._tag_vocab)

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Return tag vocabulary."""
        return self._tag_vocab.copy()

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"TagEmbedder(dim={self.embedding_dim}, vocab_size={self.vocab_size})"
        return f"TagEmbedder(dim={self.embedding_dim}, not fitted)"
