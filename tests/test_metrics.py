"""
Unit tests for recommendation metrics.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.models.metrics import (
    PrecisionAtK,
    DiversityMetric,
    MRR,
    NDCG,
    MAP,
    compute_precision_at_k,
    compute_diversity,
    compute_mrr,
    compute_ndcg,
    compute_map,
)


class TestPrecisionAtK:
    """Tests for Precision@K metric."""

    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect predictions."""
        metric = PrecisionAtK(k=3)

        y_true = tf.constant([
            [1, 1, 1, 0, 0],  # First 3 are relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Correctly ranks first 3 highest
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        assert result.numpy() == pytest.approx(1.0, rel=0.01)

    def test_precision_at_k_partial(self):
        """Test precision@k with partial correct predictions."""
        metric = PrecisionAtK(k=3)

        y_true = tf.constant([
            [1, 0, 1, 0, 1],  # Items 0, 2, 4 are relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Predicts 0, 1, 2 as top-3
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # 2 out of 3 are relevant
        assert result.numpy() == pytest.approx(2/3, rel=0.01)

    def test_precision_at_k_multiple_batches(self):
        """Test precision@k accumulation across batches."""
        metric = PrecisionAtK(k=2)

        # Batch 1: 2/2 correct
        y_true1 = tf.constant([[1, 1, 0, 0]], dtype=tf.float32)
        y_pred1 = tf.constant([[0.9, 0.8, 0.2, 0.1]])
        metric.update_state(y_true1, y_pred1)

        # Batch 2: 1/2 correct
        y_true2 = tf.constant([[0, 1, 0, 1]], dtype=tf.float32)
        y_pred2 = tf.constant([[0.9, 0.8, 0.2, 0.1]])
        metric.update_state(y_true2, y_pred2)

        result = metric.result()

        # Average: (1.0 + 0.5) / 2 = 0.75
        assert result.numpy() == pytest.approx(0.75, rel=0.01)


class TestDiversityMetric:
    """Tests for Diversity metric."""

    def test_diversity_identical_items(self):
        """Test diversity with identical item embeddings."""
        metric = DiversityMetric(k=3)

        # All embeddings are the same
        embeddings = tf.constant([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        predictions = tf.constant([
            [0.9, 0.8, 0.7, 0.1],
        ])

        metric.update_state(embeddings, predictions)
        result = metric.result()

        # Diversity should be 0 (all items are identical)
        assert result.numpy() == pytest.approx(0.0, abs=0.01)

    def test_diversity_orthogonal_items(self):
        """Test diversity with orthogonal item embeddings."""
        metric = DiversityMetric(k=3)

        # Orthogonal embeddings
        embeddings = tf.constant([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])

        predictions = tf.constant([
            [0.9, 0.8, 0.7, 0.1],
        ])

        metric.update_state(embeddings, predictions)
        result = metric.result()

        # Diversity should be 1 (all items are orthogonal)
        assert result.numpy() == pytest.approx(1.0, abs=0.01)


class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""

    def test_mrr_first_position(self):
        """Test MRR when relevant item is first."""
        metric = MRR()

        y_true = tf.constant([
            [1, 0, 0, 0, 0],
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # RR = 1/1 = 1.0
        assert result.numpy() == pytest.approx(1.0, rel=0.01)

    def test_mrr_third_position(self):
        """Test MRR when relevant item is third."""
        metric = MRR()

        y_true = tf.constant([
            [0, 0, 1, 0, 0],  # Item 2 is relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Item 2 is ranked 3rd
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # RR = 1/3
        assert result.numpy() == pytest.approx(1/3, rel=0.01)


class TestNDCG:
    """Tests for Normalized DCG metric."""

    def test_ndcg_perfect_ranking(self):
        """Test nDCG with perfect ranking."""
        metric = NDCG(k=3)

        y_true = tf.constant([
            [3, 2, 1, 0, 0],  # Graded relevance
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Perfect ranking
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Perfect ranking should give nDCG = 1.0
        assert result.numpy() == pytest.approx(1.0, rel=0.01)

    def test_ndcg_reversed_ranking(self):
        """Test nDCG with reversed ranking."""
        metric = NDCG(k=3)

        y_true = tf.constant([
            [0, 0, 1, 2, 3],  # Last items are most relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Predicts opposite
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should be less than 1.0
        assert result.numpy() < 1.0


class TestMAP:
    """Tests for Mean Average Precision metric."""

    def test_map_perfect_ranking(self):
        """Test mAP with perfect ranking."""
        metric = MAP()

        y_true = tf.constant([
            [1, 1, 0, 0, 0],  # First 2 are relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Perfect ranking
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Perfect ranking: AP = (1/1 + 2/2) / 2 = 1.0
        assert result.numpy() == pytest.approx(1.0, rel=0.01)

    def test_map_partial_ranking(self):
        """Test mAP with partial correct ranking."""
        metric = MAP()

        y_true = tf.constant([
            [1, 0, 1, 0, 0],  # Items 0 and 2 are relevant
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.9, 0.8, 0.7, 0.2, 0.1],  # Ranks: 0 (correct), 1, 2 (correct)
        ])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        assert result.numpy() == pytest.approx(5/6, rel=0.01)


class TestNumpyMetrics:
    """Tests for numpy-based metric implementations."""

    @pytest.fixture
    def sample_data(self):
        return {
            "y_true": np.array([
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
            ]),
            "y_pred": np.array([
                [0.9, 0.8, 0.7, 0.2, 0.1],
                [0.3, 0.9, 0.7, 0.2, 0.1],
            ]),
        }

    def test_numpy_precision_at_k(self, sample_data):
        """Test numpy precision@k."""
        precision = compute_precision_at_k(
            sample_data["y_true"],
            sample_data["y_pred"],
            k=3
        )

        # Sample 1: 2/3 in top-3
        # Sample 2: 2/3 in top-3
        # Average: 2/3
        assert precision == pytest.approx(2/3, rel=0.01)

    def test_numpy_mrr(self, sample_data):
        """Test numpy MRR."""
        mrr = compute_mrr(
            sample_data["y_true"],
            sample_data["y_pred"]
        )

        # Sample 1: first relevant at position 1, RR = 1
        # Sample 2: first relevant at position 1, RR = 1
        # MRR = 1.0
        assert mrr == pytest.approx(1.0, rel=0.01)

    def test_numpy_ndcg(self, sample_data):
        """Test numpy nDCG."""
        ndcg = compute_ndcg(
            sample_data["y_true"],
            sample_data["y_pred"],
            k=3
        )

        assert 0 <= ndcg <= 1

    def test_numpy_map(self, sample_data):
        """Test numpy mAP."""
        map_score = compute_map(
            sample_data["y_true"],
            sample_data["y_pred"]
        )

        assert 0 <= map_score <= 1

    def test_numpy_diversity(self):
        """Test numpy diversity."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ])

        predictions = np.array([
            [0.9, 0.8, 0.7, 0.1],
        ])

        diversity = compute_diversity(embeddings, predictions, k=3)

        assert 0 <= diversity <= 1
