"""
Video feature transformer for Two-Tower model.

Transforms raw video features into model-ready format including
vocabulary lookups, normalization, bucketing, and embeddings.
"""

from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from ..preprocessing.vocabulary_builder import StringLookupVocabulary, IntegerLookupVocabulary
from ..preprocessing.normalizers import (
    StandardNormalizer,
    LogTransformer,
    BucketTransformer,
    CompositeTransformer,
)
from ..preprocessing.text_embedder import TitleEmbedder
from ..preprocessing.tag_embedder import TagEmbedder
from ..preprocessing.artifacts import ArtifactManager
from ..config.feature_config import (
    FeatureConfig,
    DEFAULT_CONFIG,
    POPULARITY_LEVELS,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoFeatureTransformer:
    """Transform video features for Two-Tower model.

    Handles the following transformations:
    - video_id → IntegerLookup index
    - category → StringLookup index
    - title → Pre-computed BERT embedding
    - video_duration → Log transform + normalize + bucket
    - popularity → One-hot encoding
    - video_language → StringLookup index (shared with user_language)
    - tags → Pre-computed CBOW embedding

    Example:
        >>> transformer = VideoFeatureTransformer()
        >>> transformer.fit(videos_df)
        >>> transformed_df = transformer.transform(videos_df)
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_CONFIG,
        artifacts_dir: Optional[str] = None,
        language_vocab: Optional[StringLookupVocabulary] = None
    ):
        """Initialize the video feature transformer.

        Args:
            config: Feature configuration.
            artifacts_dir: Directory for saving/loading artifacts.
            language_vocab: Shared language vocabulary (from user transformer).
        """
        self.config = config
        self.artifacts_dir = artifacts_dir

        # Initialize transformers
        self.video_id_vocab = IntegerLookupVocabulary(name="video_id")
        self.category_vocab = StringLookupVocabulary(name="video_category")
        self.language_vocab = language_vocab or StringLookupVocabulary(name="language")
        self.popularity_vocab = StringLookupVocabulary(name="popularity")

        # Duration transformers
        self.duration_log = LogTransformer(name="duration")
        self.duration_normalizer = StandardNormalizer(name="duration")
        self.duration_bucketer = BucketTransformer(
            name="duration",
            boundaries=config.buckets.duration_boundaries
        )

        # Text embedders
        self.title_embedder = TitleEmbedder(
            model_name="universal_sentence_encoder",
            embedding_dim=512,
            cache_dir=artifacts_dir
        )
        self.tag_embedder = TagEmbedder(
            embedding_dim=config.embedding.tags_embedding_dim,
            cache_dir=artifacts_dir
        )

        # Cached embeddings
        self._title_embeddings_cache: Optional[pd.DataFrame] = None
        self._tag_embeddings_cache: Optional[pd.DataFrame] = None

        self._is_fitted = False

    def fit(
        self,
        videos_df: pd.DataFrame,
        compute_embeddings: bool = True
    ) -> "VideoFeatureTransformer":
        """Fit all transformers on video data.

        Args:
            videos_df: DataFrame with video data.
            compute_embeddings: Whether to pre-compute title/tag embeddings.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting video feature transformer...")

        # Fit video_id vocabulary
        self.video_id_vocab.build(videos_df["id"])
        logger.info(f"Video ID vocab size: {self.video_id_vocab.vocab_size}")

        # Fit category vocabulary
        self.category_vocab.build(videos_df["category"])
        logger.info(f"Category vocab size: {self.category_vocab.vocab_size}")

        # Fit language vocabulary (if not shared)
        if "language" in videos_df.columns:
            all_languages = list(videos_df["language"].unique())
            # Combine with existing vocab if already built
            if hasattr(self.language_vocab, '_vocab') and self.language_vocab._is_built:
                existing = list(self.language_vocab._vocab.keys())
                all_languages = list(set(all_languages + existing))
            self.language_vocab.build(all_languages)
        logger.info(f"Language vocab size: {self.language_vocab.vocab_size}")

        # Fit popularity vocabulary
        self.popularity_vocab.build(POPULARITY_LEVELS)
        logger.info(f"Popularity vocab size: {self.popularity_vocab.vocab_size}")

        # Fit duration transformers
        durations = videos_df["duration"].values
        log_durations = self.duration_log.transform(durations)
        self.duration_normalizer.fit(log_durations)
        logger.info(f"Duration stats (log): {self.duration_normalizer.get_stats()}")

        # Fit tag embedder
        if "manual_tags" in videos_df.columns:
            all_tags = videos_df["manual_tags"].tolist()
            if "augmented_tags" in videos_df.columns:
                all_tags.extend(videos_df["augmented_tags"].tolist())
            self.tag_embedder.fit(all_tags)
            logger.info(f"Tag vocab size: {self.tag_embedder.vocab_size}")

        # Pre-compute embeddings
        if compute_embeddings:
            self._compute_embeddings(videos_df)

        self._is_fitted = True
        return self

    def _compute_embeddings(self, videos_df: pd.DataFrame) -> None:
        """Pre-compute title and tag embeddings.

        Args:
            videos_df: DataFrame with video data.
        """
        logger.info("Pre-computing title embeddings...")
        self._title_embeddings_cache = self.title_embedder.embed_and_cache(
            texts=videos_df["title"],
            ids=videos_df["id"],
            cache_name="title_embeddings"
        )

        logger.info("Pre-computing tag embeddings...")
        # Combine manual and augmented tags
        tags = videos_df["manual_tags"].fillna("") + "|" + videos_df.get("augmented_tags", "").fillna("")
        self._tag_embeddings_cache = self.tag_embedder.embed_and_cache(
            tag_strings=tags,
            ids=videos_df["id"],
            cache_name="tag_embeddings"
        )

    def transform(
        self,
        df: pd.DataFrame,
        include_embeddings: bool = True,
        include_raw: bool = False
    ) -> pd.DataFrame:
        """Transform video features to model-ready format.

        Args:
            df: DataFrame with video features.
            include_embeddings: Whether to include title/tag embeddings.
            include_raw: Whether to include raw features in output.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = {}

        # Transform video_id
        if "video_id" in df.columns:
            result["video_id_idx"] = self.video_id_vocab.lookup_batch(df["video_id"])
        elif "id" in df.columns:
            result["video_id_idx"] = self.video_id_vocab.lookup_batch(df["id"])

        # Transform category
        if "category" in df.columns:
            result["category_idx"] = self.category_vocab.lookup_batch(df["category"])

        # Transform video_language
        if "video_language" in df.columns:
            result["video_language_idx"] = self.language_vocab.lookup_batch(df["video_language"])
        elif "language" in df.columns:
            result["video_language_idx"] = self.language_vocab.lookup_batch(df["language"])

        # Transform popularity
        if "popularity" in df.columns:
            result["popularity_idx"] = self.popularity_vocab.lookup_batch(df["popularity"])
            # One-hot encoding
            one_hot = np.zeros((len(df), len(POPULARITY_LEVELS)))
            for i, pop in enumerate(df["popularity"]):
                idx = POPULARITY_LEVELS.index(pop) if pop in POPULARITY_LEVELS else 0
                one_hot[i, idx] = 1
            for j, level in enumerate(POPULARITY_LEVELS):
                result[f"popularity_{level}"] = one_hot[:, j]

        # Transform duration
        if "video_duration" in df.columns:
            duration_col = "video_duration"
        elif "duration" in df.columns:
            duration_col = "duration"
        else:
            duration_col = None

        if duration_col:
            log_duration = self.duration_log.transform(df[duration_col])
            result["duration_log_normalized"] = self.duration_normalizer.transform(log_duration)
            result["duration_bucket_idx"] = self.duration_bucketer.transform(df[duration_col])

        # Include embeddings
        if include_embeddings:
            video_ids = df.get("video_id", df.get("id"))
            if video_ids is not None and self._title_embeddings_cache is not None:
                title_embs = self.title_embedder.get_embedding_from_cache(
                    video_ids, self._title_embeddings_cache
                )
                for i in range(title_embs.shape[1]):
                    result[f"title_emb_{i}"] = title_embs[:, i]

            if video_ids is not None and self._tag_embeddings_cache is not None:
                tag_embs = self._get_tag_embeddings(video_ids)
                for i in range(tag_embs.shape[1]):
                    result[f"tag_emb_{i}"] = tag_embs[:, i]

        # Include raw features if requested
        if include_raw:
            for col in ["video_id", "category", "title", "video_duration", "popularity", "video_language", "tags"]:
                if col in df.columns:
                    result[f"{col}_raw"] = df[col].values

        return pd.DataFrame(result)

    def _get_tag_embeddings(self, video_ids: pd.Series) -> np.ndarray:
        """Get tag embeddings from cache.

        Args:
            video_ids: Video IDs to look up.

        Returns:
            Numpy array of tag embeddings.
        """
        if self._tag_embeddings_cache is None:
            return np.zeros((len(video_ids), self.config.embedding.tags_embedding_dim))

        tag_emb_cols = [c for c in self._tag_embeddings_cache.columns if c.startswith("tag_emb_")]
        id_to_idx = {id_val: idx for idx, id_val in enumerate(self._tag_embeddings_cache["id"])}

        embeddings = []
        for vid in video_ids:
            if vid in id_to_idx:
                idx = id_to_idx[vid]
                emb = self._tag_embeddings_cache.iloc[idx][tag_emb_cols].values
            else:
                emb = np.zeros(len(tag_emb_cols))
            embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def transform_single(self, video_data: Dict) -> Dict:
        """Transform a single video's features.

        Args:
            video_data: Dictionary with video features.

        Returns:
            Dictionary with transformed features.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = {}

        # Transform video_id
        video_id = video_data.get("video_id") or video_data.get("id")
        if video_id is not None:
            result["video_id_idx"] = self.video_id_vocab.lookup(video_id)

        # Transform category
        category = video_data.get("category")
        if category is not None:
            result["category_idx"] = self.category_vocab.lookup(category)

        # Transform video_language
        language = video_data.get("video_language") or video_data.get("language")
        if language is not None:
            result["video_language_idx"] = self.language_vocab.lookup(language)

        # Transform popularity
        popularity = video_data.get("popularity")
        if popularity is not None:
            result["popularity_idx"] = self.popularity_vocab.lookup(popularity)
            # One-hot
            for j, level in enumerate(POPULARITY_LEVELS):
                result[f"popularity_{level}"] = 1.0 if popularity == level else 0.0

        # Transform duration
        duration = video_data.get("video_duration") or video_data.get("duration")
        if duration is not None:
            log_duration = self.duration_log.transform([duration])[0]
            result["duration_log_normalized"] = float(self.duration_normalizer.transform([log_duration])[0])
            result["duration_bucket_idx"] = int(self.duration_bucketer.transform([duration])[0])

        return result

    def save(self, artifacts_dir: Optional[str] = None) -> None:
        """Save all artifacts.

        Args:
            artifacts_dir: Directory to save artifacts.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        save_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(save_dir)

        # Save vocabularies
        artifact_manager.save_vocabulary(self.video_id_vocab)
        artifact_manager.save_vocabulary(self.category_vocab, "video_category")
        artifact_manager.save_vocabulary(self.popularity_vocab)

        # Save normalizer stats
        artifact_manager.save_normalizer_stats("duration", self.duration_normalizer.get_stats())

        # Save bucket boundaries
        artifact_manager.save_bucket_boundaries("duration", self.config.buckets.duration_boundaries)

        # Save tag embedder
        tag_embedder_path = save_dir / "embeddings" / "tag_embedder"
        self.tag_embedder.save(tag_embedder_path)

        logger.info(f"Saved video feature artifacts to {save_dir}")

    def load(self, artifacts_dir: Optional[str] = None) -> "VideoFeatureTransformer":
        """Load all artifacts.

        Args:
            artifacts_dir: Directory to load artifacts from.

        Returns:
            Self for method chaining.
        """
        load_dir = Path(artifacts_dir or self.artifacts_dir or self.config.artifacts_dir)
        artifact_manager = ArtifactManager(load_dir)

        # Load vocabularies
        self.video_id_vocab = artifact_manager.load_vocabulary("video_id", "integer")
        self.category_vocab = artifact_manager.load_vocabulary("video_category", "string")
        self.popularity_vocab = artifact_manager.load_vocabulary("popularity", "string")

        # Load shared language vocabulary
        self.language_vocab = artifact_manager.load_vocabulary("language", "string")

        # Load normalizer stats
        duration_stats = artifact_manager.load_normalizer_stats("duration")
        self.duration_normalizer.mean = duration_stats["mean"]
        self.duration_normalizer.std = duration_stats["std"]
        self.duration_normalizer._is_fitted = True

        # Load bucket boundaries
        duration_boundaries = artifact_manager.load_bucket_boundaries("duration")
        self.duration_bucketer.boundaries = duration_boundaries
        self.duration_bucketer._is_fitted = True

        # Load tag embedder
        tag_embedder_path = load_dir / "embeddings" / "tag_embedder"
        if (tag_embedder_path.parent / f"{tag_embedder_path.stem}_vocab.json").exists():
            self.tag_embedder.load(tag_embedder_path)

        # Load cached embeddings if available
        title_cache_path = load_dir / "embeddings" / "title_embeddings.parquet"
        if title_cache_path.exists():
            self._title_embeddings_cache = pd.read_parquet(title_cache_path)

        tag_cache_path = load_dir / "embeddings" / "tag_embeddings.parquet"
        if tag_cache_path.exists():
            self._tag_embeddings_cache = pd.read_parquet(tag_cache_path)

        self._is_fitted = True
        logger.info(f"Loaded video feature artifacts from {load_dir}")

        return self

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layer initialization.

        Returns:
            Dictionary mapping feature names to vocab sizes.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        return {
            "video_id": self.video_id_vocab.vocab_size,
            "video_category": self.category_vocab.vocab_size,
            "popularity": self.popularity_vocab.vocab_size,
            "duration_bucket": self.duration_bucketer.num_output_buckets,
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"VideoFeatureTransformer(fitted=True, vocabs={self.get_vocab_sizes()})"
        return "VideoFeatureTransformer(fitted=False)"
