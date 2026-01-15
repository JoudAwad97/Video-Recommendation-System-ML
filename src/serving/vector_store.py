"""
Vector store implementations for efficient similarity search.

Supports FAISS for local/development and can be extended for
cloud-native solutions like Pinecone or Weaviate.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import json

from ..utils.logging_utils import get_logger
from .serving_config import VectorDBConfig

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: List[int],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add embeddings to the store.

        Args:
            ids: List of video IDs.
            embeddings: Array of embeddings (n_samples, embedding_dim).
            metadata: Optional metadata for each embedding.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[List[int], List[float]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            Tuple of (video_ids, scores).
        """
        pass

    @abstractmethod
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings in batch.

        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim).
            top_k: Number of results per query.

        Returns:
            Tuple of (ids_array, scores_array) with shapes (n_queries, top_k).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors in the store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search.

    Supports multiple index types:
    - Flat: Exact search (best quality, slower)
    - IVFFlat: Inverted file index (faster, approximate)
    - IVFPQ: Product quantization (fastest, most approximate)
    - HNSW: Hierarchical navigable small world (good balance)

    Example:
        >>> config = VectorDBConfig(embedding_dim=16, index_type="IVFFlat")
        >>> store = FAISSVectorStore(config)
        >>> store.add(video_ids, embeddings)
        >>> ids, scores = store.search(query_embedding, top_k=100)
    """

    def __init__(self, config: VectorDBConfig):
        """Initialize FAISS vector store.

        Args:
            config: Vector database configuration.
        """
        self.config = config
        self.index = None
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self._size = 0

        # Import faiss here to make it optional
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu or pip install faiss-gpu"
            )

    def _create_index(self, embedding_dim: int) -> "faiss.Index":
        """Create FAISS index based on configuration.

        Args:
            embedding_dim: Dimension of embeddings.

        Returns:
            FAISS index.
        """
        index_type = self.config.index_type

        if index_type == "Flat":
            # Exact L2 search
            index = self.faiss.IndexFlatIP(embedding_dim)

        elif index_type == "IVFFlat":
            # IVF with flat quantizer
            quantizer = self.faiss.IndexFlatIP(embedding_dim)
            index = self.faiss.IndexIVFFlat(
                quantizer,
                embedding_dim,
                self.config.nlist,
                self.faiss.METRIC_INNER_PRODUCT,
            )

        elif index_type == "IVFPQ":
            # IVF with product quantization
            quantizer = self.faiss.IndexFlatIP(embedding_dim)
            index = self.faiss.IndexIVFPQ(
                quantizer,
                embedding_dim,
                self.config.nlist,
                self.config.m,
                8,  # bits per sub-quantizer
            )

        elif index_type == "HNSW":
            # Hierarchical NSW
            index = self.faiss.IndexHNSWFlat(embedding_dim, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 64

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return index

    def add(
        self,
        ids: List[int],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add embeddings to the index.

        Args:
            ids: List of video IDs.
            embeddings: Array of embeddings (n_samples, embedding_dim).
            metadata: Optional metadata (not used in FAISS, stored separately).
        """
        if len(ids) != len(embeddings):
            raise ValueError("Number of IDs must match number of embeddings")

        embeddings = embeddings.astype(np.float32)

        # Normalize embeddings for inner product search (cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        # Create index if not exists
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.index = self._create_index(embedding_dim)

        # Train index if needed (IVF-based indexes)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            logger.info("Index training complete")

        # Add embeddings
        self.index.add(embeddings)

        # Update ID mappings
        for i, video_id in enumerate(ids):
            idx = self._size + i
            self.id_to_idx[video_id] = idx
            self.idx_to_id[idx] = video_id

        self._size += len(ids)
        logger.info(f"Added {len(ids)} vectors. Total: {self._size}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[List[int], List[float]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            Tuple of (video_ids, scores).
        """
        if self.index is None or self._size == 0:
            return [], []

        # Reshape and normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Set search parameters for IVF indexes
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.config.nprobe

        # Search
        scores, indices = self.index.search(query, min(top_k, self._size))

        # Convert indices to video IDs
        video_ids = []
        valid_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in self.idx_to_id:
                video_ids.append(self.idx_to_id[idx])
                valid_scores.append(float(score))

        return video_ids, valid_scores

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings in batch.

        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim).
            top_k: Number of results per query.

        Returns:
            Tuple of (ids_array, scores_array) with shapes (n_queries, top_k).
        """
        if self.index is None or self._size == 0:
            n_queries = len(query_embeddings)
            return np.zeros((n_queries, top_k), dtype=np.int64), np.zeros((n_queries, top_k))

        # Normalize queries
        queries = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-8)

        # Set search parameters
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.config.nprobe

        # Batch search
        scores, indices = self.index.search(queries, min(top_k, self._size))

        # Convert indices to video IDs
        video_ids = np.zeros_like(indices, dtype=np.int64)
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                idx = indices[i, j]
                if idx >= 0 and idx in self.idx_to_id:
                    video_ids[i, j] = self.idx_to_id[idx]

        return video_ids, scores

    def save(self, path: str) -> None:
        """Save index and mappings to disk.

        Args:
            path: Directory path to save index.
        """
        if self.index is None:
            raise RuntimeError("No index to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        self.faiss.write_index(self.index, str(index_path))

        # Save ID mappings
        mappings = {
            "id_to_idx": {str(k): v for k, v in self.id_to_idx.items()},
            "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()},
            "size": self._size,
        }
        mappings_path = path / "mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(mappings, f)

        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "index_type": self.config.index_type,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe,
                "embedding_dim": self.config.embedding_dim,
            }, f)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str) -> None:
        """Load index and mappings from disk.

        Args:
            path: Directory path to load index from.
        """
        path = Path(path)

        # Load FAISS index
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = self.faiss.read_index(str(index_path))

        # Load ID mappings
        mappings_path = path / "mappings.json"
        with open(mappings_path) as f:
            mappings = json.load(f)

        self.id_to_idx = {int(k): v for k, v in mappings["id_to_idx"].items()}
        self.idx_to_id = {int(k): v for k, v in mappings["idx_to_id"].items()}
        self._size = mappings["size"]

        logger.info(f"Loaded FAISS index with {self._size} vectors from {path}")

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self._size

    def remove(self, ids: List[int]) -> None:
        """Remove embeddings by ID.

        Note: FAISS doesn't support efficient removal. This marks IDs as invalid.

        Args:
            ids: List of video IDs to remove.
        """
        for video_id in ids:
            if video_id in self.id_to_idx:
                idx = self.id_to_idx[video_id]
                del self.idx_to_id[idx]
                del self.id_to_idx[video_id]

        logger.info(f"Marked {len(ids)} vectors as removed")

    def rebuild_index(
        self,
        ids: List[int],
        embeddings: np.ndarray,
    ) -> None:
        """Rebuild the index from scratch.

        Use this for periodic maintenance to remove deleted vectors.

        Args:
            ids: List of all valid video IDs.
            embeddings: Array of all valid embeddings.
        """
        self.index = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self._size = 0

        self.add(ids, embeddings)
        logger.info("Index rebuilt successfully")


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using numpy.

    Useful for testing and small datasets. Uses brute-force search.
    """

    def __init__(self, config: VectorDBConfig):
        """Initialize in-memory vector store."""
        self.config = config
        self.embeddings: Optional[np.ndarray] = None
        self.video_ids: List[int] = []

    def add(
        self,
        ids: List[int],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add embeddings to the store."""
        embeddings = embeddings.astype(np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.video_ids.extend(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[List[int], List[float]]:
        """Search using brute-force cosine similarity."""
        if self.embeddings is None or len(self.video_ids) == 0:
            return [], []

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Compute similarities
        similarities = np.dot(self.embeddings, query.T).flatten()

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        video_ids = [self.video_ids[i] for i in top_indices]
        scores = similarities[top_indices].tolist()

        return video_ids, scores

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search using brute-force."""
        if self.embeddings is None or len(self.video_ids) == 0:
            n_queries = len(query_embeddings)
            return np.zeros((n_queries, top_k), dtype=np.int64), np.zeros((n_queries, top_k))

        # Normalize queries
        queries = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-8)

        # Compute all similarities
        similarities = np.dot(queries, self.embeddings.T)

        # Get top-k for each query
        top_k = min(top_k, len(self.video_ids))
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]

        video_ids = np.array([[self.video_ids[i] for i in row] for row in top_indices])
        scores = np.take_along_axis(similarities, top_indices, axis=1)

        return video_ids, scores

    def save(self, path: str) -> None:
        """Save to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(
            path / "store.npz",
            embeddings=self.embeddings if self.embeddings is not None else np.array([]),
            video_ids=np.array(self.video_ids),
        )

    def load(self, path: str) -> None:
        """Load from disk."""
        data = np.load(Path(path) / "store.npz")
        self.embeddings = data["embeddings"] if len(data["embeddings"]) > 0 else None
        self.video_ids = data["video_ids"].tolist()

    @property
    def size(self) -> int:
        """Number of vectors in store."""
        return len(self.video_ids)
