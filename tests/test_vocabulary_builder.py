"""Tests for vocabulary builders."""

import pytest
import tempfile
from pathlib import Path

from src.preprocessing.vocabulary_builder import (
    StringLookupVocabulary,
    IntegerLookupVocabulary,
    HashBucketVocabulary,
)


class TestStringLookupVocabulary:
    """Tests for StringLookupVocabulary."""

    def test_build_basic(self):
        """Test basic vocabulary building."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana", "cherry"])

        assert vocab.vocab_size == 5  # 3 values + PAD + UNK
        assert vocab._is_built

    def test_lookup_known_values(self):
        """Test looking up known values."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana", "cherry"])

        # Known values should return indices > 1 (0=PAD, 1=UNK)
        assert vocab.lookup("apple") >= 2
        assert vocab.lookup("banana") >= 2
        assert vocab.lookup("cherry") >= 2

    def test_lookup_unknown_value(self):
        """Test looking up unknown values returns UNK index."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana"])

        # Unknown value should return UNK index (1)
        assert vocab.lookup("unknown_fruit") == 1

    def test_lookup_none_value(self):
        """Test looking up None returns PAD index."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana"])

        # None should return PAD index (0)
        assert vocab.lookup(None) == 0

    def test_mask_value(self):
        """Test mask value handling."""
        vocab = StringLookupVocabulary(
            name="test",
            include_start=True,
            mask_value="-"
        )
        vocab.build(["apple", "banana", "-"])

        # Mask value should map to START token
        start_idx = vocab.lookup("-")
        assert start_idx == 2  # 0=PAD, 1=UNK, 2=START

    def test_save_and_load(self):
        """Test saving and loading vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "vocab.json"

            # Build and save
            vocab1 = StringLookupVocabulary(name="test")
            vocab1.build(["apple", "banana", "cherry"])
            vocab1.save(filepath)

            # Load
            vocab2 = StringLookupVocabulary(name="test")
            vocab2.load(filepath)

            # Should have same vocab
            assert vocab1.vocab_size == vocab2.vocab_size
            assert vocab1.lookup("apple") == vocab2.lookup("apple")

    def test_inverse_lookup(self):
        """Test inverse lookup."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana"])

        idx = vocab.lookup("apple")
        assert vocab.inverse_lookup(idx) == "apple"

    def test_lookup_batch(self):
        """Test batch lookup."""
        vocab = StringLookupVocabulary(name="test")
        vocab.build(["apple", "banana", "cherry"])

        indices = vocab.lookup_batch(["apple", "banana", "unknown"])
        assert len(indices) == 3
        assert indices[2] == 1  # UNK


class TestIntegerLookupVocabulary:
    """Tests for IntegerLookupVocabulary."""

    def test_build_basic(self):
        """Test basic vocabulary building."""
        vocab = IntegerLookupVocabulary(name="test")
        vocab.build([100, 200, 300])

        assert vocab.vocab_size == 5  # 3 values + PAD + UNK
        assert vocab._is_built

    def test_lookup_known_values(self):
        """Test looking up known values."""
        vocab = IntegerLookupVocabulary(name="test")
        vocab.build([100, 200, 300])

        assert vocab.lookup(100) >= 2
        assert vocab.lookup(200) >= 2
        assert vocab.lookup(300) >= 2

    def test_lookup_unknown_value(self):
        """Test looking up unknown values."""
        vocab = IntegerLookupVocabulary(name="test")
        vocab.build([100, 200])

        # Unknown value should return UNK index (1)
        assert vocab.lookup(999) == 1

    def test_save_and_load(self):
        """Test saving and loading vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "vocab.json"

            # Build and save
            vocab1 = IntegerLookupVocabulary(name="test")
            vocab1.build([100, 200, 300])
            vocab1.save(filepath)

            # Load
            vocab2 = IntegerLookupVocabulary(name="test")
            vocab2.load(filepath)

            # Should have same vocab
            assert vocab1.vocab_size == vocab2.vocab_size
            assert vocab1.lookup(100) == vocab2.lookup(100)


class TestHashBucketVocabulary:
    """Tests for HashBucketVocabulary."""

    def test_lookup_returns_bucket(self):
        """Test that lookup returns valid bucket index."""
        vocab = HashBucketVocabulary(name="test", num_buckets=100)

        idx = vocab.lookup("any_value")
        assert 0 <= idx < 100

    def test_consistent_hashing(self):
        """Test that same value returns same bucket."""
        vocab = HashBucketVocabulary(name="test", num_buckets=100)

        idx1 = vocab.lookup("test_value")
        idx2 = vocab.lookup("test_value")
        assert idx1 == idx2

    def test_vocab_size(self):
        """Test vocab_size returns num_buckets."""
        vocab = HashBucketVocabulary(name="test", num_buckets=1000)
        assert vocab.vocab_size == 1000
