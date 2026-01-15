"""
Evaluation metrics for recommendation models.

Two-Tower Model Metrics:
- Precision@k: Proportion of relevant items in top-k recommendations
- Diversity: Average pairwise dissimilarity of recommended items

Ranker Model Metrics:
- MRR (Mean Reciprocal Rank): Focuses on rank of first relevant item
- nDCG (Normalized Discounted Cumulative Gain): Ranking quality with graded relevance
- mAP (Mean Average Precision): Ranking quality with binary relevance
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Optional, Union


class PrecisionAtK(keras.metrics.Metric):
    """
    Precision@k metric for recommendation systems.

    Measures the proportion of relevant videos among the top k recommended videos.
    """

    def __init__(self, k: int = 10, name: str = None, **kwargs):
        super().__init__(name=name or f"precision_at_{k}", **kwargs)
        self.k = k
        self.precision_sum = self.add_weight(
            name="precision_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        """
        Update metric state.

        Args:
            y_true: Binary labels indicating relevance (batch_size, num_items)
            y_pred: Predicted scores/similarities (batch_size, num_items)
            sample_weight: Optional sample weights
        """
        # Get top-k indices
        _, top_k_indices = tf.math.top_k(y_pred, k=self.k)

        # Gather relevance labels for top-k predictions
        batch_size = tf.shape(y_true)[0]
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), axis=1),
            self.k,
            axis=1
        )
        indices = tf.stack([batch_indices, top_k_indices], axis=2)
        top_k_relevance = tf.gather_nd(y_true, indices)

        # Calculate precision for each sample
        precision = tf.reduce_sum(tf.cast(top_k_relevance, tf.float32), axis=1) / self.k

        if sample_weight is not None:
            precision = precision * sample_weight

        self.precision_sum.assign_add(tf.reduce_sum(precision))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self) -> tf.Tensor:
        return self.precision_sum / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.precision_sum.assign(0.0)
        self.count.assign(0.0)


class DiversityMetric(keras.metrics.Metric):
    """
    Diversity metric for recommendation systems.

    Measures how dissimilar recommended videos are to each other.
    Lower average pairwise similarity = higher diversity.
    """

    def __init__(self, k: int = 10, name: str = "diversity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.diversity_sum = self.add_weight(
            name="diversity_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )

    def update_state(
        self,
        embeddings: tf.Tensor,
        predictions: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        """
        Update diversity metric.

        Args:
            embeddings: Video embeddings (num_videos, embedding_dim)
            predictions: Predicted scores (batch_size, num_videos)
            sample_weight: Optional sample weights
        """
        batch_size = tf.shape(predictions)[0]

        # Get top-k video indices for each user
        _, top_k_indices = tf.math.top_k(predictions, k=self.k)

        # Gather embeddings for top-k videos
        top_k_embeddings = tf.gather(embeddings, top_k_indices)

        # Compute pairwise cosine similarity within each user's recommendations
        # Normalize embeddings
        top_k_normalized = tf.nn.l2_normalize(top_k_embeddings, axis=-1)

        # Compute similarity matrix for each user
        similarity = tf.matmul(
            top_k_normalized,
            top_k_normalized,
            transpose_b=True
        )

        # Create mask to exclude diagonal (self-similarity)
        mask = 1.0 - tf.eye(self.k, batch_shape=[batch_size])

        # Calculate average pairwise similarity (excluding diagonal)
        masked_similarity = similarity * mask
        avg_similarity = tf.reduce_sum(masked_similarity, axis=[1, 2]) / (self.k * (self.k - 1))

        # Diversity = 1 - average_similarity
        diversity = 1.0 - avg_similarity

        if sample_weight is not None:
            diversity = diversity * sample_weight

        self.diversity_sum.assign_add(tf.reduce_sum(diversity))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self) -> tf.Tensor:
        return self.diversity_sum / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.diversity_sum.assign(0.0)
        self.count.assign(0.0)


class MRR(keras.metrics.Metric):
    """
    Mean Reciprocal Rank (MRR) metric.

    Focuses on the rank of the first relevant item.
    MRR = (1/|Q|) * sum(1/rank_i) where rank_i is the rank of the first relevant item.
    """

    def __init__(self, name: str = "mrr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mrr_sum = self.add_weight(
            name="mrr_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        """
        Update MRR metric.

        Args:
            y_true: Binary relevance labels (batch_size, num_items)
            y_pred: Predicted scores (batch_size, num_items)
            sample_weight: Optional sample weights
        """
        batch_size = tf.shape(y_true)[0]
        num_items = tf.shape(y_true)[1]

        # Get ranking of items by predicted score (descending)
        _, sorted_indices = tf.math.top_k(y_pred, k=num_items)

        # Reorder true labels according to predicted ranking
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), axis=1),
            num_items,
            axis=1
        )
        indices = tf.stack([batch_indices, sorted_indices], axis=2)
        sorted_labels = tf.gather_nd(y_true, indices)

        # Find rank of first relevant item
        # Create position tensor (1-indexed)
        positions = tf.cast(tf.range(1, num_items + 1), tf.float32)

        # Find first relevant item position for each sample
        is_relevant = tf.cast(sorted_labels > 0, tf.float32)

        # Calculate reciprocal rank
        # Set non-relevant positions to a large number
        masked_positions = tf.where(
            is_relevant > 0,
            positions,
            tf.fill([batch_size, num_items], float('inf'))
        )

        first_relevant_rank = tf.reduce_min(masked_positions, axis=1)
        reciprocal_rank = 1.0 / first_relevant_rank

        # Handle cases with no relevant items
        reciprocal_rank = tf.where(
            tf.math.is_inf(reciprocal_rank),
            tf.zeros_like(reciprocal_rank),
            reciprocal_rank
        )

        if sample_weight is not None:
            reciprocal_rank = reciprocal_rank * sample_weight

        self.mrr_sum.assign_add(tf.reduce_sum(reciprocal_rank))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self) -> tf.Tensor:
        return self.mrr_sum / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.mrr_sum.assign(0.0)
        self.count.assign(0.0)


class NDCG(keras.metrics.Metric):
    """
    Normalized Discounted Cumulative Gain (nDCG) metric.

    Measures ranking quality when relevance scores can be graded (non-binary).
    nDCG@k = DCG@k / IDCG@k
    DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
    """

    def __init__(self, k: int = 10, name: str = None, **kwargs):
        super().__init__(name=name or f"ndcg_at_{k}", **kwargs)
        self.k = k
        self.ndcg_sum = self.add_weight(
            name="ndcg_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )

    def _dcg(self, relevance: tf.Tensor) -> tf.Tensor:
        """Calculate DCG for relevance scores."""
        positions = tf.cast(tf.range(1, self.k + 1), tf.float32)
        discounts = tf.math.log(positions + 1.0) / tf.math.log(2.0)
        return tf.reduce_sum(relevance / discounts, axis=1)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        """
        Update nDCG metric.

        Args:
            y_true: Relevance scores (can be graded) (batch_size, num_items)
            y_pred: Predicted scores (batch_size, num_items)
            sample_weight: Optional sample weights
        """
        batch_size = tf.shape(y_true)[0]
        num_items = tf.shape(y_true)[1]

        # Get top-k indices based on predictions
        _, pred_top_k_indices = tf.math.top_k(y_pred, k=self.k)

        # Gather relevance for predicted top-k
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), axis=1),
            self.k,
            axis=1
        )
        indices = tf.stack([batch_indices, pred_top_k_indices], axis=2)
        pred_relevance = tf.cast(tf.gather_nd(y_true, indices), tf.float32)

        # Calculate DCG
        dcg = self._dcg(pred_relevance)

        # Calculate IDCG (ideal DCG with perfect ranking)
        _, ideal_top_k_indices = tf.math.top_k(y_true, k=self.k)
        ideal_indices = tf.stack([batch_indices, ideal_top_k_indices], axis=2)
        ideal_relevance = tf.cast(tf.gather_nd(y_true, ideal_indices), tf.float32)
        idcg = self._dcg(ideal_relevance)

        # Calculate nDCG
        ndcg = dcg / (idcg + keras.backend.epsilon())

        # Handle cases where IDCG is 0
        ndcg = tf.where(
            idcg > 0,
            ndcg,
            tf.zeros_like(ndcg)
        )

        if sample_weight is not None:
            ndcg = ndcg * sample_weight

        self.ndcg_sum.assign_add(tf.reduce_sum(ndcg))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self) -> tf.Tensor:
        return self.ndcg_sum / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.ndcg_sum.assign(0.0)
        self.count.assign(0.0)


class MAP(keras.metrics.Metric):
    """
    Mean Average Precision (mAP) metric.

    Works with binary relevance scores.
    AP = sum(Precision@k * rel_k) / number_of_relevant_items
    mAP = mean of AP across all queries
    """

    def __init__(self, k: int = None, name: str = "map", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k  # If None, use all items
        self.map_sum = self.add_weight(
            name="map_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        """
        Update mAP metric.

        Args:
            y_true: Binary relevance labels (batch_size, num_items)
            y_pred: Predicted scores (batch_size, num_items)
            sample_weight: Optional sample weights
        """
        batch_size = tf.shape(y_true)[0]
        num_items = tf.shape(y_true)[1]

        # Determine k
        k = self.k if self.k is not None else num_items

        # Get ranking of items by predicted score
        _, sorted_indices = tf.math.top_k(y_pred, k=k)

        # Reorder true labels according to predicted ranking
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), axis=1),
            k,
            axis=1
        )
        indices = tf.stack([batch_indices, sorted_indices], axis=2)
        sorted_labels = tf.cast(tf.gather_nd(y_true, indices), tf.float32)

        # Calculate cumulative sum of relevant items at each position
        cumsum_relevant = tf.cumsum(sorted_labels, axis=1)

        # Calculate precision at each position
        positions = tf.cast(tf.range(1, k + 1), tf.float32)
        precision_at_k = cumsum_relevant / positions

        # Multiply by relevance indicator (only count precision at relevant positions)
        precision_times_rel = precision_at_k * sorted_labels

        # Calculate AP for each sample
        num_relevant = tf.reduce_sum(sorted_labels, axis=1)
        ap = tf.reduce_sum(precision_times_rel, axis=1) / (num_relevant + keras.backend.epsilon())

        # Handle cases with no relevant items
        ap = tf.where(
            num_relevant > 0,
            ap,
            tf.zeros_like(ap)
        )

        if sample_weight is not None:
            ap = ap * sample_weight

        self.map_sum.assign_add(tf.reduce_sum(ap))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self) -> tf.Tensor:
        return self.map_sum / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.map_sum.assign(0.0)
        self.count.assign(0.0)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10
) -> float:
    """
    Numpy implementation of Precision@k for evaluation.

    Args:
        y_true: Binary relevance labels (num_samples, num_items)
        y_pred: Predicted scores (num_samples, num_items)
        k: Number of top items to consider

    Returns:
        Average precision@k across all samples
    """
    num_samples = y_true.shape[0]
    precisions = []

    for i in range(num_samples):
        # Get top-k indices
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]

        # Calculate precision
        relevant_in_top_k = np.sum(y_true[i][top_k_indices])
        precision = relevant_in_top_k / k
        precisions.append(precision)

    return np.mean(precisions)


def compute_diversity(
    embeddings: np.ndarray,
    predictions: np.ndarray,
    k: int = 10
) -> float:
    """
    Numpy implementation of diversity metric for evaluation.

    Args:
        embeddings: Video embeddings (num_videos, embedding_dim)
        predictions: Predicted scores (num_samples, num_videos)
        k: Number of top items to consider

    Returns:
        Average diversity across all samples
    """
    num_samples = predictions.shape[0]
    diversities = []

    for i in range(num_samples):
        # Get top-k indices
        top_k_indices = np.argsort(predictions[i])[::-1][:k]

        # Get embeddings for top-k videos
        top_k_emb = embeddings[top_k_indices]

        # Normalize embeddings
        norms = np.linalg.norm(top_k_emb, axis=1, keepdims=True)
        top_k_normalized = top_k_emb / (norms + 1e-8)

        # Compute pairwise similarity
        similarity = np.dot(top_k_normalized, top_k_normalized.T)

        # Calculate average pairwise similarity (excluding diagonal)
        mask = 1.0 - np.eye(k)
        avg_similarity = np.sum(similarity * mask) / (k * (k - 1))

        # Diversity = 1 - average_similarity
        diversities.append(1.0 - avg_similarity)

    return np.mean(diversities)


def compute_mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numpy implementation of MRR for evaluation.

    Args:
        y_true: Binary relevance labels (num_samples, num_items)
        y_pred: Predicted scores (num_samples, num_items)

    Returns:
        Mean Reciprocal Rank
    """
    num_samples = y_true.shape[0]
    reciprocal_ranks = []

    for i in range(num_samples):
        # Get ranking
        sorted_indices = np.argsort(y_pred[i])[::-1]
        sorted_labels = y_true[i][sorted_indices]

        # Find first relevant item
        relevant_indices = np.where(sorted_labels > 0)[0]

        if len(relevant_indices) > 0:
            first_relevant_rank = relevant_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Numpy implementation of nDCG@k for evaluation.

    Args:
        y_true: Relevance scores (num_samples, num_items)
        y_pred: Predicted scores (num_samples, num_items)
        k: Number of top items to consider

    Returns:
        Average nDCG@k
    """
    def dcg(relevance):
        positions = np.arange(1, len(relevance) + 1)
        discounts = np.log2(positions + 1)
        return np.sum(relevance / discounts)

    num_samples = y_true.shape[0]
    ndcgs = []

    for i in range(num_samples):
        # Get predicted top-k
        pred_top_k = np.argsort(y_pred[i])[::-1][:k]
        pred_relevance = y_true[i][pred_top_k]

        # Get ideal top-k
        ideal_top_k = np.argsort(y_true[i])[::-1][:k]
        ideal_relevance = y_true[i][ideal_top_k]

        # Calculate DCG and IDCG
        dcg_score = dcg(pred_relevance)
        idcg_score = dcg(ideal_relevance)

        if idcg_score > 0:
            ndcgs.append(dcg_score / idcg_score)
        else:
            ndcgs.append(0.0)

    return np.mean(ndcgs)


def compute_map(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """
    Numpy implementation of mAP for evaluation.

    Args:
        y_true: Binary relevance labels (num_samples, num_items)
        y_pred: Predicted scores (num_samples, num_items)
        k: Number of top items to consider (None = all items)

    Returns:
        Mean Average Precision
    """
    num_samples = y_true.shape[0]
    average_precisions = []

    for i in range(num_samples):
        # Get ranking
        num_items = y_true.shape[1] if k is None else min(k, y_true.shape[1])
        sorted_indices = np.argsort(y_pred[i])[::-1][:num_items]
        sorted_labels = y_true[i][sorted_indices]

        # Calculate AP
        num_relevant = np.sum(sorted_labels)
        if num_relevant == 0:
            average_precisions.append(0.0)
            continue

        cumsum_relevant = np.cumsum(sorted_labels)
        positions = np.arange(1, num_items + 1)
        precision_at_k = cumsum_relevant / positions

        ap = np.sum(precision_at_k * sorted_labels) / num_relevant
        average_precisions.append(ap)

    return np.mean(average_precisions)
