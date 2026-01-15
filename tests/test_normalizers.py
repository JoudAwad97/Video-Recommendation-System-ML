"""Tests for numeric normalizers."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.preprocessing.normalizers import (
    StandardNormalizer,
    MinMaxNormalizer,
    LogTransformer,
    BucketTransformer,
    CyclicalEncoder,
)


class TestStandardNormalizer:
    """Tests for StandardNormalizer."""

    def test_fit_computes_mean_std(self):
        """Test that fit computes mean and std."""
        normalizer = StandardNormalizer(name="test")
        normalizer.fit([10, 20, 30, 40, 50])

        assert normalizer.mean == 30.0
        assert normalizer.std == pytest.approx(14.14, rel=0.01)

    def test_transform_normalizes(self):
        """Test that transform produces z-scores."""
        normalizer = StandardNormalizer(name="test")
        normalizer.fit([0, 10])  # mean=5, std=5

        result = normalizer.transform([5])  # Should be 0 (at mean)
        assert result[0] == pytest.approx(0.0, abs=0.01)

        result = normalizer.transform([10])  # Should be 1 (one std above mean)
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_inverse_transform(self):
        """Test that inverse_transform recovers original values."""
        normalizer = StandardNormalizer(name="test")
        normalizer.fit([10, 20, 30])

        original = [15, 25]
        normalized = normalizer.transform(original)
        recovered = normalizer.inverse_transform(normalized)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_save_and_load(self):
        """Test saving and loading normalizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "normalizer.json"

            # Fit and save
            norm1 = StandardNormalizer(name="test")
            norm1.fit([10, 20, 30, 40, 50])
            norm1.save(filepath)

            # Load
            norm2 = StandardNormalizer(name="test")
            norm2.load(filepath)

            # Should have same stats
            assert norm1.mean == norm2.mean
            assert norm1.std == norm2.std


class TestMinMaxNormalizer:
    """Tests for MinMaxNormalizer."""

    def test_fit_computes_min_max(self):
        """Test that fit computes min and max."""
        normalizer = MinMaxNormalizer(name="test")
        normalizer.fit([10, 20, 30, 40, 50])

        assert normalizer.min_val == 10.0
        assert normalizer.max_val == 50.0

    def test_transform_scales_to_range(self):
        """Test that transform scales to [0, 1]."""
        normalizer = MinMaxNormalizer(name="test")
        normalizer.fit([0, 100])

        result = normalizer.transform([0, 50, 100])
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_custom_range(self):
        """Test custom feature range."""
        normalizer = MinMaxNormalizer(name="test", feature_range=(-1, 1))
        normalizer.fit([0, 100])

        result = normalizer.transform([0, 50, 100])
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_transform_applies_log1p(self):
        """Test that transform applies log1p."""
        transformer = LogTransformer(name="test")

        result = transformer.transform([0, 1, 9, 99])
        expected = np.log1p([0, 1, 9, 99])

        np.testing.assert_array_almost_equal(result, expected)

    def test_inverse_transform(self):
        """Test that inverse_transform recovers original values."""
        transformer = LogTransformer(name="test")

        original = [1, 10, 100, 1000]
        log_vals = transformer.transform(original)
        recovered = transformer.inverse_transform(log_vals)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_handles_zero(self):
        """Test that log1p handles zero values."""
        transformer = LogTransformer(name="test")
        result = transformer.transform([0])
        assert result[0] == 0.0  # log1p(0) = 0


class TestBucketTransformer:
    """Tests for BucketTransformer."""

    def test_fixed_boundaries(self):
        """Test bucketing with fixed boundaries."""
        transformer = BucketTransformer(
            name="test",
            boundaries=[10, 20, 30]
        )

        result = transformer.transform([5, 15, 25, 35])
        # 5 < 10 -> bucket 0
        # 10 <= 15 < 20 -> bucket 1
        # 20 <= 25 < 30 -> bucket 2
        # 30 <= 35 -> bucket 3
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_fit_computes_boundaries(self):
        """Test that fit computes quantile-based boundaries."""
        transformer = BucketTransformer(name="test", num_buckets=4)
        transformer.fit([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        assert transformer._is_fitted
        assert len(transformer.boundaries) == 3  # 4 buckets need 3 boundaries

    def test_num_output_buckets(self):
        """Test num_output_buckets property."""
        transformer = BucketTransformer(name="test", boundaries=[10, 20, 30])
        assert transformer.num_output_buckets == 4

    def test_bucket_labels(self):
        """Test getting bucket labels."""
        transformer = BucketTransformer(name="test", boundaries=[10, 20])
        labels = transformer.get_bucket_labels()

        assert len(labels) == 3
        assert "< 10" in labels[0]


class TestCyclicalEncoder:
    """Tests for CyclicalEncoder."""

    def test_hour_encoding(self):
        """Test encoding hours with period 24."""
        encoder = CyclicalEncoder(name="hour", period=24)

        # Hour 0 and 24 should be the same
        result_0 = encoder.transform([0])
        result_24 = encoder.transform([24])
        np.testing.assert_array_almost_equal(result_0, result_24)

    def test_output_shape(self):
        """Test that output has sin and cos components."""
        encoder = CyclicalEncoder(name="hour", period=24)
        result = encoder.transform([0, 6, 12, 18])

        assert result.shape == (4, 2)  # 4 values, 2 components (sin, cos)

    def test_sin_cos_range(self):
        """Test that sin/cos values are in [-1, 1]."""
        encoder = CyclicalEncoder(name="hour", period=24)
        result = encoder.transform(list(range(24)))

        assert np.all(result >= -1)
        assert np.all(result <= 1)

    def test_hour_6_values(self):
        """Test specific hour values."""
        encoder = CyclicalEncoder(name="hour", period=24)

        # Hour 6: sin should be 1, cos should be 0 (quarter period)
        result = encoder.transform([6])
        assert result[0, 0] == pytest.approx(1.0, abs=0.01)  # sin
        assert result[0, 1] == pytest.approx(0.0, abs=0.01)  # cos
