"""
Text embedding using TensorFlow Hub BERT.

Provides text embedding functionality for video titles using pre-trained
BERT models from TensorFlow Hub.
"""

from typing import List, Union, Optional, Dict
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.io_utils import save_parquet, load_parquet, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TitleEmbedder:
    """Embed video titles using TensorFlow Hub BERT.

    Uses a pre-trained BERT model from TensorFlow Hub to generate
    dense vector representations of video titles.

    Example:
        >>> embedder = TitleEmbedder()
        >>> embeddings = embedder.embed(["How to code in Python", "Gaming tips"])
        >>> embeddings.shape  # (2, 768)
    """

    # Default TF Hub model URLs
    BERT_MODELS = {
        "small_bert": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "bert_base": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        "universal_sentence_encoder": "https://tfhub.dev/google/universal-sentence-encoder/4",
    }

    PREPROCESSOR_MODELS = {
        "small_bert": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "universal_sentence_encoder": None,  # USE doesn't need preprocessing
    }

    def __init__(
        self,
        model_name: str = "universal_sentence_encoder",
        embedding_dim: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """Initialize the title embedder.

        Args:
            model_name: Name of the pre-trained model to use.
            embedding_dim: Expected embedding dimension.
            batch_size: Batch size for embedding computation.
            cache_dir: Directory to cache embeddings.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._model = None
        self._preprocessor = None
        self._is_loaded = False

    def _load_model(self) -> None:
        """Load the TensorFlow Hub model."""
        if self._is_loaded:
            return

        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            logger.info(f"Loading text embedding model: {self.model_name}")

            model_url = self.BERT_MODELS.get(self.model_name)
            preprocessor_url = self.PREPROCESSOR_MODELS.get(self.model_name)

            if model_url is None:
                raise ValueError(f"Unknown model: {self.model_name}")

            if self.model_name == "universal_sentence_encoder":
                # USE is simpler - no preprocessor needed
                self._model = hub.load(model_url)
            else:
                # BERT models need preprocessor
                self._preprocessor = hub.load(preprocessor_url)
                self._model = hub.load(model_url)

            self._is_loaded = True
            logger.info(f"Model loaded successfully")

        except ImportError as e:
            logger.warning(f"TensorFlow Hub not available: {e}")
            logger.warning("Using fallback random embeddings for testing")
            self._is_loaded = False

    def embed(self, texts: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()

        # Handle empty or None texts
        texts = [str(t) if t is not None and pd.notna(t) else "" for t in texts]

        self._load_model()

        if not self._is_loaded:
            # Fallback to random embeddings for testing
            return self._fallback_embed(texts)

        try:
            import tensorflow as tf

            embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                if self.model_name == "universal_sentence_encoder":
                    batch_embeddings = self._model(batch).numpy()
                else:
                    # BERT-style models
                    preprocessed = self._preprocessor(batch)
                    outputs = self._model(preprocessed)
                    # Use pooled output for sentence embedding
                    batch_embeddings = outputs["pooled_output"].numpy()

                embeddings.append(batch_embeddings)

            return np.vstack(embeddings)

        except Exception as e:
            logger.warning(f"Error during embedding: {e}")
            return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Generate fallback embeddings for testing.

        Uses a deterministic hash-based approach to generate consistent
        embeddings for the same text.

        Args:
            texts: List of texts.

        Returns:
            Numpy array of pseudo-embeddings.
        """
        embeddings = []
        for text in texts:
            # Use hash for deterministic pseudo-random embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            embeddings.append(embedding)

        return np.array(embeddings)

    def embed_and_cache(
        self,
        texts: Union[List[str], pd.Series],
        ids: Union[List[int], pd.Series],
        cache_name: str = "title_embeddings"
    ) -> pd.DataFrame:
        """Embed texts and cache results.

        Args:
            texts: List of texts to embed.
            ids: Corresponding IDs for each text.
            cache_name: Name for the cache file.

        Returns:
            DataFrame with id and embedding columns.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(ids, pd.Series):
            ids = ids.tolist()

        # Check cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_name}.parquet"
            if cache_path.exists():
                logger.info(f"Loading cached embeddings from {cache_path}")
                return load_parquet(cache_path)

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embed(texts)

        # Create DataFrame
        embedding_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        df = pd.DataFrame({"id": ids, **embedding_cols})

        # Save cache
        if self.cache_dir:
            ensure_dir(self.cache_dir)
            save_parquet(df, cache_path)
            logger.info(f"Cached embeddings to {cache_path}")

        return df

    def get_embedding_from_cache(
        self,
        ids: Union[List[int], pd.Series],
        cache_df: pd.DataFrame
    ) -> np.ndarray:
        """Get embeddings from cached DataFrame.

        Args:
            ids: IDs to look up.
            cache_df: Cached embedding DataFrame.

        Returns:
            Numpy array of embeddings.
        """
        if isinstance(ids, pd.Series):
            ids = ids.tolist()

        # Get embedding columns
        emb_cols = [c for c in cache_df.columns if c.startswith("emb_")]

        # Look up embeddings
        id_to_idx = {id_val: idx for idx, id_val in enumerate(cache_df["id"])}
        embeddings = []

        for id_val in ids:
            if id_val in id_to_idx:
                idx = id_to_idx[id_val]
                emb = cache_df.iloc[idx][emb_cols].values
            else:
                # Return zero embedding for unknown IDs
                emb = np.zeros(len(emb_cols))
            embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def __repr__(self) -> str:
        return f"TitleEmbedder(model='{self.model_name}', dim={self.embedding_dim})"
